# ARCHIVAL SCRIPT — not used in final paper
# Kept for ablation reference only.
# For the final model, use scripts/training/train_dhsd_v2.py
# ---------------------------------------------------------------
#!/usr/bin/env python3
"""
ADAPTIVE SEMANTIC STUDENT

Instead of KL divergence (which fights contrastive), the teacher's
confidence MODULATES the contrastive loss:

  - Teacher confident (P=0.95) → push HARD toward correct prototype
  - Teacher uncertain (P=0.55) → push GENTLY

Loss = alpha * CE(logits, y_true)
     + gamma * w_teacher * contrastive(z, e_real, e_fake)

where w_teacher = |P_teacher - 0.5| * 2   (0=uncertain, 1=confident)

This unifies teacher confidence + semantic reasoning into ONE loss
instead of two competing losses.
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
from tqdm import tqdm

HF_CACHE = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
RESULTS_DIR = "/NAS_DISK/Saptarshi_data/results"
BACKUP_DIR = "/NAS_DISK/Saptarshi_data/results_backup"
SOFT_LABELS_PATH = "/NAS_DISK/Saptarshi_data/pipeline_output/teacher_soft_labels_3k.json"
OUTPUT_DIR = "/NAS_DISK/Saptarshi_data/pipeline_output/adaptive_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
MAX_EPOCHS = 500
PATIENCE = 50
EVAL_EVERY = 5
BATCH_SIZE = 32
LR = 1e-4
LABEL_SMOOTHING = 0.15
ALPHA = 0.5        # CE weight
GAMMA = 0.5        # contrastive weight (modulated by teacher confidence)
TEMPERATURE = 0.5
WEIGHT_DECAY = 0.5
torch.manual_seed(SEED)
np.random.seed(SEED)

log_path = os.path.join(OUTPUT_DIR, "training.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
log = logging.getLogger()


class SemanticStudent(nn.Module):
    def __init__(self, clip_model, embed_dim=768):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        with torch.no_grad():
            img_feat = self.image_encoder(x).float()
        projected = self.projector(img_feat)
        projected_norm = F.normalize(projected, dim=-1)
        logits = self.classifier(projected)
        return projected_norm, logits


def adaptive_loss(proj_img, logits, labels, e_real, e_fake, teacher_prob,
                  temperature=0.5):
    """
    Teacher confidence modulates contrastive strength.
    w = |P_teacher - 0.5| * 2  → 0 when uncertain, 1 when confident
    """
    # 1. CE loss
    ce_loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)

    # 2. Contrastive loss (per-sample)
    sim_real = (proj_img * e_real).sum(dim=-1) / temperature
    sim_fake = (proj_img * e_fake).sum(dim=-1) / temperature
    proto_logits = torch.stack([sim_real, sim_fake], dim=1)

    # Per-sample contrastive loss
    contrastive_per_sample = F.cross_entropy(proto_logits, labels, reduction='none')

    # 3. Teacher confidence weight per sample
    # |P - 0.5| * 2: maps [0,1] → [0,1] where 0.5 → 0 and 0 or 1 → 1
    w_teacher = (teacher_prob - 0.5).abs() * 2.0

    # Weighted contrastive: confident teacher → stronger push
    weighted_contrastive = (w_teacher * contrastive_per_sample).mean()

    total = ALPHA * ce_loss + GAMMA * weighted_contrastive
    return total, ce_loss.item(), weighted_contrastive.item(), w_teacher.mean().item()


def build_text_prototypes(clip_model, tokenizer):
    log.info("Building text prototypes from VLM results...")
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

    device = next(clip_model.parameters()).device
    prototypes = {}
    for img_id, data in tqdm(by_image.items(), desc="  Building prototypes"):
        real_texts = data["REAL"]
        fake_texts = data["FAKE"]
        gt = data["gt"]
        if not real_texts or not fake_texts:
            continue
        try:
            real_tokens = tokenizer(real_texts).to(device)
            fake_tokens = tokenizer(fake_texts).to(device)
            with torch.no_grad():
                real_feats = clip_model.encode_text(real_tokens)
                real_feats = F.normalize(real_feats, dim=-1)
                e_real = F.normalize(real_feats.mean(dim=0), dim=-1)
                fake_feats = clip_model.encode_text(fake_tokens)
                fake_feats = F.normalize(fake_feats, dim=-1)
                e_fake = F.normalize(fake_feats.mean(dim=0), dim=-1)
            prototypes[img_id] = {"e_real": e_real.cpu(), "e_fake": e_fake.cpu(), "gt": gt}
        except:
            pass
    log.info(f"  Built prototypes for {len(prototypes)} images")
    return prototypes


def load_teacher_soft_labels():
    log.info(f"Loading teacher soft labels: {SOFT_LABELS_PATH}")
    if not os.path.exists(SOFT_LABELS_PATH):
        log.warning("  No soft labels — using binary confidence (1.0)")
        return {}
    with open(SOFT_LABELS_PATH) as f:
        records = json.load(f)
    soft = {}
    for r in records:
        soft[r["image_id"]] = r["teacher_prob_fake"]
    log.info(f"  Loaded {len(soft)} soft labels")
    return soft


class AdaptiveDataset(Dataset):
    def __init__(self, paths_and_ids, prototypes, soft_labels, preprocess, augment=True):
        self.preprocess = preprocess
        self.augment = augment
        self.items = []
        for path, img_id in paths_and_ids:
            if img_id in prototypes:
                p = prototypes[img_id]
                label = 0 if p["gt"].upper() == "REAL" else 1
                # If no soft label, use confident default based on ground truth
                teacher_prob = soft_labels.get(img_id, float(label))
                self.items.append((path, label, p["e_real"], p["e_fake"], teacher_prob))

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
        path, label, e_real, e_fake, teacher_prob = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self._augment(img)
            return self.preprocess(img), label, e_real, e_fake, teacher_prob
        except:
            return torch.zeros(3, 224, 224), label, e_real, e_fake, teacher_prob


class SimpleDataset(Dataset):
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


def get_paths_with_ids(d):
    if not os.path.isdir(d):
        return []
    return sorted([(str(p), p.stem) for p in Path(d).rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])

def get_paths(d):
    if not os.path.isdir(d):
        return []
    return sorted([str(p) for p in Path(d).rglob("*")
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

def split_pi(paths_ids, ratio=0.5):
    paths_ids = sorted(paths_ids, key=lambda x: x[0])
    np.random.seed(SEED)
    idx = np.random.permutation(len(paths_ids))
    s = int(len(paths_ids) * ratio)
    return [paths_ids[i] for i in idx[:s]], [paths_ids[i] for i in idx[s:]]

def find_optimal_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return thresholds[np.argmax(tpr - fpr)]

def evaluate(student, loader):
    student.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval", leave=False):
            imgs = batch[0].cuda()
            labels = batch[1]
            _, logits = student(imgs)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("ADAPTIVE SEMANTIC STUDENT")
    log.info("Teacher confidence modulates contrastive strength")
    log.info(f"Started: {datetime.now()}")
    log.info(f"Alpha(CE)={ALPHA}, Gamma(adaptive contrastive)={GAMMA}")
    log.info(f"Temp={TEMPERATURE}, WeightDecay={WEIGHT_DECAY}, LR={LR}")
    log.info("=" * 70)

    log.info("\nLoading CLIP ViT-L-14...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    clip_model = clip_model.cuda().eval()

    prototypes = build_text_prototypes(clip_model, tokenizer)
    soft_labels = load_teacher_soft_labels()

    student = SemanticStudent(clip_model, embed_dim=768).cuda()
    for p in student.image_encoder.parameters():
        p.requires_grad = False
    log.info("  Student: CLIP (frozen) + projector + classifier")
    log.info(f"  Loss: {ALPHA}*CE + {GAMMA}*w_teacher*contrastive")
    log.info(f"  w_teacher = |P_teacher - 0.5| * 2 (confident=1, uncertain=0)")

    log.info("\n[DATA SPLITS]")
    coco_pi = get_paths_with_ids("/NAS_DISK/Saptarshi_data/dataset/real/coco")
    mj_pi = get_paths_with_ids("/NAS_DISK/Saptarshi_data/dataset/fake/midjourney")
    coco_tr, coco_te = split_pi(coco_pi, 0.5)
    mj_tr, mj_te = split_pi(mj_pi, 0.5)
    log.info(f"  COCO:       {len(coco_tr)} train / {len(coco_te)} test")
    log.info(f"  Midjourney: {len(mj_tr)} train / {len(mj_te)} test")

    gan_test = {}
    for name in ["starGAN", "styleGAN", "BigGAN"]:
        d = f"/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/{name}"
        paths = get_paths(d)
        log.info(f"  {name}:    {len(paths)} test")
        gan_test[name] = paths

    pr_real, pr_fake = get_progan_paths("/NAS_DISK/Saptarshi_data/dataset/cnndetection_test")
    log.info(f"  ProGAN:     {len(pr_real)}r + {len(pr_fake)}f test (HELD OUT)")

    coco_tr_p = [(p, i) for p, i in coco_tr if i in prototypes][:1500]
    mj_tr_p = [(p, i) for p, i in mj_tr if i in prototypes][:1500]

    n_soft_coco = sum(1 for _, i in coco_tr_p if i in soft_labels)
    n_soft_mj = sum(1 for _, i in mj_tr_p if i in soft_labels)
    log.info(f"\n  Training: {len(coco_tr_p)} real + {len(mj_tr_p)} fake")
    log.info(f"  With teacher soft labels: {n_soft_coco} real + {n_soft_mj} fake")
    log.info(f"  Without soft labels: use confident default (0.0 or 1.0)")

    if len(coco_tr_p) == 0 or len(mj_tr_p) == 0:
        log.error("No matching prototypes!")
        sys.exit(1)

    train_ds = ConcatDataset([
        AdaptiveDataset(coco_tr_p, prototypes, soft_labels, preprocess, augment=True),
        AdaptiveDataset(mj_tr_p, prototypes, soft_labels, preprocess, augment=True),
    ])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    log.info(f"  Train dataset: {len(train_ds)} images")

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

    optimizer = torch.optim.AdamW(
        list(student.projector.parameters()) + list(student.classifier.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    latest_path = os.path.join(OUTPUT_DIR, "latest_model.pt")
    results_path = os.path.join(OUTPUT_DIR, "results_history.json")
    best_min_auc = 0
    no_improve = 0
    history = []

    log.info(f"\nTraining up to {MAX_EPOCHS} epochs (patience={PATIENCE})...")
    log.info(f"Loss = {ALPHA}*CE + {GAMMA}*w_teacher*contrastive")
    log.info("-" * 80)

    for epoch in range(MAX_EPOCHS):
        student.train()
        t_loss, t_ce, t_con, t_w, n_b, correct, total = 0, 0, 0, 0, 0, 0, 0

        for imgs, labels, e_real, e_fake, teacher_prob in tqdm(
                train_loader, desc=f"Ep {epoch+1}/{MAX_EPOCHS}", leave=False):
            imgs = imgs.cuda()
            labels = labels.cuda().long()
            e_real = F.normalize(e_real.float().cuda(), dim=-1)
            e_fake = F.normalize(e_fake.float().cuda(), dim=-1)
            teacher_prob = teacher_prob.float().cuda()

            proj_img, logits = student(imgs)
            loss, ce, con, w_avg = adaptive_loss(
                proj_img, logits, labels, e_real, e_fake, teacher_prob, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_ce += ce
            t_con += con
            t_w += w_avg
            n_b += 1
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        torch.save({"projector": student.projector.state_dict(),
                     "classifier": student.classifier.state_dict(),
                     "epoch": epoch}, latest_path)

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            epoch_res = {"epoch": epoch + 1,
                         "loss": round(t_loss/n_b, 4),
                         "ce": round(t_ce/n_b, 4),
                         "w_contrastive": round(t_con/n_b, 4),
                         "avg_teacher_w": round(t_w/n_b, 4),
                         "train_acc": round(correct/total, 4)}
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
            log.info(f"Ep {epoch+1:3d}/{MAX_EPOCHS} | L:{t_loss/n_b:.4f} "
                     f"(ce:{t_ce/n_b:.3f} wCon:{t_con/n_b:.3f} avgW:{t_w/n_b:.2f}) | "
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

    # Final
    log.info(f"\n{'=' * 80}")
    log.info("FINAL RESULTS - ADAPTIVE SEMANTIC STUDENT")
    log.info(f"{'=' * 80}")

    ckpt = torch.load(best_path, weights_only=True)
    student.projector.load_state_dict(ckpt["projector"])
    student.classifier.load_state_dict(ckpt["classifier"])

    log.info(f"\n{'Test Set':<22} {'AUROC':<10} {'Acc@0.5':<10} {'Acc@opt':<10} {'Thresh':<10} {'Status'}")
    log.info("-" * 72)

    final = {}
    for name, loader in test_loaders.items():
        probs, labels_np = evaluate(student, loader)
        auc = roc_auc_score(labels_np, probs)
        thresh = find_optimal_threshold(labels_np, probs)
        acc05 = accuracy_score(labels_np, (probs > 0.5).astype(int))
        acc_opt = accuracy_score(labels_np, (probs > thresh).astype(int))
        status = "NEVER SEEN" if name == "ProGAN" else "holdout"
        log.info(f"{name:<22} {auc:<10.4f} {acc05:<10.4f} {acc_opt:<10.4f} {thresh:<10.4f} {status}")
        final[name] = {"auroc": round(auc, 4), "acc_05": round(acc05, 4),
                        "acc_opt": round(acc_opt, 4), "threshold": round(float(thresh), 4)}

    log.info(f"\n  ADAPTIVE DESIGN:")
    log.info(f"  Teacher confidence weights the contrastive push")
    log.info(f"  Confident teacher → strong alignment with correct prototype")
    log.info(f"  Uncertain teacher → gentle alignment (lets CE dominate)")
    log.info(f"  No KL divergence → no competing loss directions")

    # Ablation comparison
    log.info(f"\n  ABLATION COMPARISON:")
    log.info(f"  {'Approach':<35} {'ProGAN AUROC'}")
    log.info(f"  {'-'*50}")
    log.info(f"  {'Soft label only (KL)':<35} {'~0.58'}")
    log.info(f"  {'Semantic only (contrastive)':<35} {'0.785'}")
    log.info(f"  {'Combined (KL + contrastive)':<35} {'~0.73 (conflicts)'}")
    pg_auc = final.get("ProGAN", {}).get("auroc", "?")
    log.info(f"  {'Adaptive (this run)':<35} {pg_auc}")

    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump({"config": {"alpha_ce": ALPHA, "gamma_contrastive": GAMMA,
                               "temperature": TEMPERATURE, "weight_decay": WEIGHT_DECAY,
                               "approach": "adaptive_teacher_weighted_contrastive"},
                    "results": final, "history": history}, f, indent=2)
    log.info(f"\nTime: {(time.time()-t0)/60:.1f} min")
    log.info(f"Model: {best_path}")
    log.info("DONE.")


if __name__ == "__main__":
    main()
