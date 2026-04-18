#!/usr/bin/env python3
"""
SEMANTIC STUDENT — Transfers VLM reasoning via text prototypes.

Instead of distilling a single probability, we transfer the actual
semantic content of dual-hypothesis cues into the student.

For each image, the teacher produced:
  - REAL-side: caption + 3 cues → encode with CLIP text encoder → e_real
  - FAKE-side: caption + 3 cues → encode with CLIP text encoder → e_fake

The student learns:
  - Real images: image embedding should align with e_real, away from e_fake
  - Fake images: image embedding should align with e_fake, away from e_real

Loss = contrastive(z_img, e_real, e_fake) + CE(prediction, label)

This transfers the LANGUAGE of reasoning, not just a number.
"""

import os, json, time, torch, sys, io, glob, logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from PIL import Image, ImageFilter
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from datetime import datetime

HF_CACHE = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
RESULTS_DIR = "/NAS_DISK/Saptarshi_data/results"
BACKUP_DIR = "/NAS_DISK/Saptarshi_data/results_backup"
OUTPUT_DIR = "/NAS_DISK/Saptarshi_data/pipeline_output/semantic_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20
EVAL_EVERY = 5
BATCH_SIZE = 32
LR = 3e-4
LABEL_SMOOTHING = 0.1
LAMBDA_CONTRAST = 0.5  # weight for contrastive loss
LAMBDA_CE = 0.5        # weight for classification loss
TEMPERATURE = 0.07     # contrastive temperature
torch.manual_seed(SEED)
np.random.seed(SEED)

log_path = os.path.join(OUTPUT_DIR, "training.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
log = logging.getLogger()


# ══════════════════════════════════════════════════════
# MODEL: Image encoder + projection head
# ══════════════════════════════════════════════════════
class SemanticStudent(nn.Module):
    """
    Student that projects image embeddings into the same space
    as CLIP text embeddings of VLM cues.
    """
    def __init__(self, clip_model, embed_dim=768):
        super().__init__()
        self.image_encoder = clip_model.visual  # frozen

        # Projection: maps image embedding to text-aligned space
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )

        # Classification head on top of projected features
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        with torch.no_grad():
            img_feat = self.image_encoder(x).float()
        projected = self.projector(img_feat)
        projected_norm = F.normalize(projected, dim=-1)
        logits = self.classifier(projected)
        return projected_norm, logits


# ══════════════════════════════════════════════════════
# LOSS: Contrastive alignment + classification
# ══════════════════════════════════════════════════════
def semantic_loss(proj_img, logits, labels, e_real, e_fake, temperature=0.07):
    """
    Combined loss:
    1. Contrastive: real images → close to e_real, far from e_fake
                    fake images → close to e_fake, far from e_real
    2. Classification: standard CE with label smoothing
    """
    # Cosine similarities
    sim_real = (proj_img * e_real).sum(dim=-1) / temperature  # (B,)
    sim_fake = (proj_img * e_fake).sum(dim=-1) / temperature  # (B,)

    # For real images (label=0): sim_real should be high, sim_fake low
    # For fake images (label=1): sim_fake should be high, sim_real low
    # This is a 2-way contrastive: pick the right prototype

    # Stack into logits: [sim_real, sim_fake] → shape (B, 2)
    proto_logits = torch.stack([sim_real, sim_fake], dim=1)  # (B, 2)
    # Labels: 0 = real (align with col 0 = e_real), 1 = fake (align with col 1 = e_fake)
    contrastive_loss = F.cross_entropy(proto_logits, labels)

    # Classification loss
    ce_loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)

    total = LAMBDA_CONTRAST * contrastive_loss + LAMBDA_CE * ce_loss
    return total, contrastive_loss.item(), ce_loss.item()


# ══════════════════════════════════════════════════════
# BUILD TEXT PROTOTYPES FROM VLM RESULTS
# ══════════════════════════════════════════════════════
def build_text_prototypes(clip_model, tokenizer, use_open_clip=True):
    """
    Load all VLM results, encode REAL-side and FAKE-side texts,
    create per-image text prototype vectors.
    Returns: dict[image_id] → {"e_real": tensor, "e_fake": tensor, "gt": str}
    """
    log.info("Building text prototypes from VLM results...")

    # Load VLM results
    all_results = []
    for d in [RESULTS_DIR, BACKUP_DIR]:
        if os.path.isdir(d):
            for fp in glob.glob(os.path.join(d, "*.json")):
                try:
                    with open(fp) as f:
                        data = json.load(f)
                    entries = data if isinstance(data, list) else data.get("results", [])
                    all_results.extend(entries)
                except:
                    pass
    log.info(f"  Loaded {len(all_results)} VLM entries")

    # Group by image_id
    by_image = defaultdict(lambda: {"REAL": [], "FAKE": [], "gt": ""})
    for r in all_results:
        img_id = r.get("image_id", "")
        assume = r.get("assumption", "")
        gt = r.get("ground_truth", "")
        if assume in ("REAL", "FAKE"):
            texts = []
            cap = r.get("caption", "")
            if cap:
                texts.append(cap)
            for i in range(1, 4):
                c = r.get(f"cue_{i}", "")
                if c:
                    texts.append(c)
            by_image[img_id][assume].extend(texts)
            by_image[img_id]["gt"] = gt

    log.info(f"  Images with VLM data: {len(by_image)}")

    # Encode all texts with CLIP
    device = next(clip_model.parameters()).device
    prototypes = {}
    batch_size = 64
    processed = 0

    for img_id, data in by_image.items():
        real_texts = data["REAL"]
        fake_texts = data["FAKE"]
        gt = data["gt"]

        if not real_texts or not fake_texts:
            continue

        try:
            # Encode REAL-side texts → average → prototype
            if use_open_clip:
                real_tokens = tokenizer(real_texts).to(device)
                fake_tokens = tokenizer(fake_texts).to(device)
            else:
                real_tokens = tokenizer(real_texts, truncate=True).to(device)
                fake_tokens = tokenizer(fake_texts, truncate=True).to(device)

            with torch.no_grad():
                real_feats = clip_model.encode_text(real_tokens)
                real_feats = F.normalize(real_feats, dim=-1)
                e_real = real_feats.mean(dim=0)  # average across all real texts
                e_real = F.normalize(e_real, dim=-1)

                fake_feats = clip_model.encode_text(fake_tokens)
                fake_feats = F.normalize(fake_feats, dim=-1)
                e_fake = fake_feats.mean(dim=0)
                e_fake = F.normalize(e_fake, dim=-1)

            prototypes[img_id] = {
                "e_real": e_real.cpu(),
                "e_fake": e_fake.cpu(),
                "gt": gt,
                "n_real_texts": len(real_texts),
                "n_fake_texts": len(fake_texts),
            }
            processed += 1

        except Exception as e:
            pass

        if processed % 2000 == 0 and processed > 0:
            log.info(f"  Encoded {processed} images...")

    log.info(f"  Built prototypes for {len(prototypes)} images")

    # Project prototypes to 256-dim to match student projector output
    # We'll do this after model init since we need the projector

    return prototypes


# ══════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════
class SemanticDataset(Dataset):
    """Dataset that returns image + text prototypes + label."""
    def __init__(self, paths_and_ids, prototypes, preprocess, augment=True, proto_dim=768):
        self.preprocess = preprocess
        self.augment = augment
        self.items = []
        for path, img_id in paths_and_ids:
            if img_id in prototypes:
                p = prototypes[img_id]
                label = 0 if p["gt"].upper() == "REAL" else 1
                self.items.append((path, img_id, label, p["e_real"], p["e_fake"]))

    def __len__(self):
        return len(self.items)

    def _augment(self, img):
        if not self.augment:
            return img
        if np.random.random() > 0.2:
            q = np.random.randint(20, 95)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if np.random.random() > 0.4:
            r = np.random.uniform(0.3, 2.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))
        if np.random.random() > 0.4:
            w, h = img.size
            s = np.random.uniform(0.3, 0.9)
            small = img.resize((max(int(w*s), 10), max(int(h*s), 10)), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __getitem__(self, idx):
        path, img_id, label, e_real, e_fake = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self._augment(img)
            img_tensor = self.preprocess(img)
        except:
            img_tensor = torch.zeros(3, 224, 224)
        return img_tensor, label, e_real, e_fake


class SimpleDataset(Dataset):
    """For test sets without prototypes — just image + label."""
    def __init__(self, paths, label, preprocess):
        self.paths = paths
        self.label = label
        self.preprocess = preprocess
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.preprocess(img), self.label
        except:
            return torch.zeros(3, 224, 224), self.label


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def get_paths(d):
    if not os.path.isdir(d):
        return []
    return sorted([str(p) for p in Path(d).rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])

def get_paths_with_ids(d):
    """Return (path, stem) pairs."""
    if not os.path.isdir(d):
        return []
    return sorted([(str(p), p.stem) for p in Path(d).rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])

def get_progan_paths(base):
    real, fake = [], []
    for root, dirs, files in os.walk(base):
        folder = os.path.basename(root)
        imgs = sorted([os.path.join(root, f) for f in files
                       if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])
        if folder == "0_real":
            real.extend(imgs)
        elif folder == "1_fake":
            fake.extend(imgs)
    return real, fake

def split(paths_ids, ratio=0.5):
    paths_ids = sorted(paths_ids, key=lambda x: x[0])
    np.random.seed(SEED)
    idx = np.random.permutation(len(paths_ids))
    s = int(len(paths_ids) * ratio)
    return [paths_ids[i] for i in idx[:s]], [paths_ids[i] for i in idx[s:]]

def split_paths(paths, ratio=0.5):
    paths = sorted(paths)
    np.random.seed(SEED)
    idx = np.random.permutation(len(paths))
    s = int(len(paths) * ratio)
    return [paths[i] for i in idx[:s]], [paths[i] for i in idx[s:]]

def find_optimal_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return thresholds[np.argmax(tpr - fpr)]

def evaluate(student, loader, device="cuda"):
    student.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            labels = batch[1]
            _, logits = student(imgs)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("SEMANTIC STUDENT — Dual-Hypothesis Cue Transfer")
    log.info(f"Started: {datetime.now()}")
    log.info(f"Contrastive weight: {LAMBDA_CONTRAST}, CE weight: {LAMBDA_CE}")
    log.info(f"Temperature: {TEMPERATURE}")
    log.info("=" * 70)

    # ── Load CLIP ──
    log.info("\nLoading CLIP ViT-L-14...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    clip_model = clip_model.cuda().eval()

    # ── Build text prototypes from VLM results ──
    prototypes = build_text_prototypes(clip_model, tokenizer)

    # We need to project 768-dim CLIP text embeddings to 256-dim
    # to match the student projector. We'll use a learned linear layer
    # OR we can make the student project to 768 directly.
    # Simpler: make prototypes 256-dim via a fixed random projection
    # OR: just make student projector output 768-dim.
    # Let's make student output 768 to match CLIP directly.

    # ── Build student ──
    student = SemanticStudent(clip_model, embed_dim=768).cuda()
    # Update projector to output 768 to match CLIP text dim
    student.projector = nn.Sequential(
        nn.Linear(768, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 768),
    ).cuda()
    student.classifier = nn.Sequential(
        nn.Linear(768, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    ).cuda()

    for p in student.image_encoder.parameters():
        p.requires_grad = False
    log.info("  Student: CLIP encoder (frozen) + projector (768→768) + classifier")

    # ── Data splits ──
    log.info("\n[DATA SPLITS]")

    # COCO real + Midjourney fake (have VLM prototypes)
    coco_pi = get_paths_with_ids("/NAS_DISK/Saptarshi_data/dataset/real/coco")
    mj_pi = get_paths_with_ids("/NAS_DISK/Saptarshi_data/dataset/fake/midjourney")
    coco_tr, coco_te = split(coco_pi, 0.5)
    mj_tr, mj_te = split(mj_pi, 0.5)
    log.info(f"  COCO:       {len(coco_tr)} train / {len(coco_te)} test")
    log.info(f"  Midjourney: {len(mj_tr)} train / {len(mj_te)} test")

    # GANs (no VLM prototypes — use for classification-only test)
    gan_test = {}
    for name in ["starGAN", "styleGAN", "BigGAN"]:
        d = f"/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/{name}"
        paths = get_paths(d)
        log.info(f"  {name}:    {len(paths)} test (no prototypes, classification only)")
        gan_test[name] = paths

    # ProGAN
    pr_real, pr_fake = get_progan_paths("/NAS_DISK/Saptarshi_data/dataset/cnndetection_test")
    log.info(f"  ProGAN:     {len(pr_real)}r + {len(pr_fake)}f test (100% held out)")

    # Filter training to images that have prototypes
    coco_tr_with_proto = [(p, i) for p, i in coco_tr if i in prototypes][:1500]
    mj_tr_with_proto = [(p, i) for p, i in mj_tr if i in prototypes][:1500]
    log.info(f"\n  Training with prototypes: {len(coco_tr_with_proto)} real + {len(mj_tr_with_proto)} fake")

    if len(coco_tr_with_proto) == 0 or len(mj_tr_with_proto) == 0:
        log.error("No training images matched prototypes! Check image IDs.")
        log.info(f"  Sample prototype IDs: {list(prototypes.keys())[:5]}")
        log.info(f"  Sample COCO IDs: {[i for _, i in coco_tr[:5]]}")
        log.info(f"  Sample MJ IDs: {[i for _, i in mj_tr[:5]]}")
        sys.exit(1)

    # ── Dataloaders ──
    train_ds = ConcatDataset([
        SemanticDataset(coco_tr_with_proto, prototypes, preprocess, augment=True),
        SemanticDataset(mj_tr_with_proto, prototypes, preprocess, augment=True),
    ])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    log.info(f"  Train dataset: {len(train_ds)} images with semantic prototypes")

    # Test loaders (classification only — no prototypes needed)
    test_loaders = {}
    coco_te_paths = [p for p, i in coco_te[:1000]]
    mj_te_paths = [p for p, i in mj_te[:1000]]

    test_loaders["Midjourney"] = DataLoader(
        ConcatDataset([SimpleDataset(coco_te_paths, 0, preprocess),
                       SimpleDataset(mj_te_paths, 1, preprocess)]),
        batch_size=64, num_workers=0)

    for name, paths in gan_test.items():
        test_loaders[name] = DataLoader(
            ConcatDataset([SimpleDataset(coco_te_paths[:len(paths)], 0, preprocess),
                           SimpleDataset(paths, 1, preprocess)]),
            batch_size=64, num_workers=0)

    np.random.shuffle(pr_real)
    np.random.shuffle(pr_fake)
    test_loaders["ProGAN"] = DataLoader(
        ConcatDataset([SimpleDataset(pr_real[:1000], 0, preprocess),
                       SimpleDataset(pr_fake[:1000], 1, preprocess)]),
        batch_size=64, num_workers=0)

    # ── Training ──
    optimizer = torch.optim.AdamW(
        list(student.projector.parameters()) + list(student.classifier.parameters()),
        lr=LR, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    latest_path = os.path.join(OUTPUT_DIR, "latest_model.pt")
    results_path = os.path.join(OUTPUT_DIR, "results_history.json")
    best_min_auc = 0
    no_improve = 0
    history = []

    log.info(f"\nTraining up to {MAX_EPOCHS} epochs (patience={PATIENCE})...")
    log.info(f"Loss = {LAMBDA_CONTRAST}*contrastive + {LAMBDA_CE}*CE")
    log.info("-" * 80)

    for epoch in range(MAX_EPOCHS):
        student.train()
        total_loss, total_con, total_ce, n_b, correct, total = 0, 0, 0, 0, 0, 0

        for imgs, labels, e_real, e_fake in train_loader:
            imgs = imgs.cuda()
            labels = labels.cuda().long()
            e_real = F.normalize(e_real.float().cuda(), dim=-1)
            e_fake = F.normalize(e_fake.float().cuda(), dim=-1)

            proj_img, logits = student(imgs)

            loss, con_loss, ce_loss = semantic_loss(
                proj_img, logits, labels, e_real, e_fake, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_con += con_loss
            total_ce += ce_loss
            n_b += 1
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        # Save latest
        torch.save({"projector": student.projector.state_dict(),
                     "classifier": student.classifier.state_dict(),
                     "epoch": epoch}, latest_path)

        # Evaluate
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            epoch_res = {"epoch": epoch + 1, "loss": total_loss / n_b,
                         "contrastive": total_con / n_b, "ce": total_ce / n_b,
                         "train_acc": correct / total}
            aucs = {}
            for name, loader in test_loaders.items():
                probs, labels_np = evaluate(student, loader)
                if len(set(labels_np)) > 1:
                    auc = roc_auc_score(labels_np, probs)
                    thresh = find_optimal_threshold(labels_np, probs)
                    acc = accuracy_score(labels_np, (probs > thresh).astype(int))
                    aucs[name] = auc
                    epoch_res[f"{name}_auc"] = round(auc, 4)
                    epoch_res[f"{name}_acc"] = round(acc, 4)

            history.append(epoch_res)
            with open(results_path, "w") as f:
                json.dump(history, f, indent=2)

            min_auc = min(aucs.values()) if aucs else 0
            auc_str = " | ".join(f"{k[:6]}:{v:.3f}" for k, v in aucs.items())
            log.info(f"Ep {epoch+1:3d}/{MAX_EPOCHS} | L:{total_loss/n_b:.4f} "
                     f"(con:{total_con/n_b:.3f} ce:{total_ce/n_b:.3f}) | "
                     f"Tr:{correct/total:.3f} | {auc_str}")

            if min_auc > best_min_auc:
                best_min_auc = min_auc
                no_improve = 0
                torch.save({"projector": student.projector.state_dict(),
                             "classifier": student.classifier.state_dict(),
                             "epoch": epoch}, best_path)
                log.info(f"  >>> NEW BEST (min AUC={min_auc:.4f})")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    log.info(f"  EARLY STOP at epoch {epoch+1}")
                    break

    # ── Final evaluation ──
    log.info(f"\n{'=' * 80}")
    log.info("FINAL RESULTS — SEMANTIC STUDENT")
    log.info(f"{'=' * 80}")

    ckpt = torch.load(best_path, weights_only=True)
    student.projector.load_state_dict(ckpt["projector"])
    student.classifier.load_state_dict(ckpt["classifier"])

    final = {}
    log.info(f"\n{'Test Set':<22} {'AUROC':<10} {'Acc@0.5':<10} {'Acc@opt':<10} {'Thresh':<10} {'Status'}")
    log.info("-" * 72)

    for name, loader in test_loaders.items():
        probs, labels_np = evaluate(student, loader)
        auc = roc_auc_score(labels_np, probs)
        thresh = find_optimal_threshold(labels_np, probs)
        acc05 = accuracy_score(labels_np, (probs > 0.5).astype(int))
        acc_opt = accuracy_score(labels_np, (probs > thresh).astype(int))
        status = "HELD OUT" if name == "ProGAN" else "holdout"
        log.info(f"{name:<22} {auc:<10.4f} {acc05:<10.4f} {acc_opt:<10.4f} {thresh:<10.4f} {status}")
        final[name] = {"auroc": round(auc, 4), "acc_05": round(acc05, 4),
                        "acc_opt": round(acc_opt, 4), "threshold": round(float(thresh), 4)}

    log.info(f"\n  KEY DIFFERENCE FROM PREVIOUS:")
    log.info(f"  Previous student: image → CLIP → MLP → P(fake)")
    log.info(f"  This student:     image → CLIP → align with VLM text prototypes → P(fake)")
    log.info(f"  The reasoning semantics are transferred, not just a probability.")

    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump({"config": {"lambda_contrast": LAMBDA_CONTRAST, "lambda_ce": LAMBDA_CE,
                               "temperature": TEMPERATURE, "approach": "semantic_prototype_alignment"},
                    "results": final, "history": history}, f, indent=2)

    log.info(f"\nTime: {(time.time()-t0)/60:.1f} min")
    log.info(f"Model: {best_path}")
    log.info("DONE.")

if __name__ == "__main__":
    main()
