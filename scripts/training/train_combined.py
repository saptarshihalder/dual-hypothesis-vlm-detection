#!/usr/bin/env python3
"""
FULL COMBINED PIPELINE - matches thesis document exactly.
Loss = alpha * CE + beta * KL(teacher_soft) + gamma * contrastive(prototypes)
"""

import os, json, time, torch, sys, io, re, glob, logging
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
OUTPUT_DIR = "/NAS_DISK/Saptarshi_data/pipeline_output/combined_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
MAX_EPOCHS = 500
PATIENCE = 50
EVAL_EVERY = 5
BATCH_SIZE = 32
LR = 1e-4
LABEL_SMOOTHING = 0.15
ALPHA = 0.4    # CE weight
BETA = 0.3     # KL teacher soft label weight
GAMMA = 0.3    # contrastive prototype weight
TEMPERATURE = 0.5
KD_TEMPERATURE = 3.0
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


def combined_loss(proj_img, logits, labels, e_real, e_fake, teacher_prob,
                  temperature=0.5, kd_temperature=3.0):
    # 1. CE loss with ground truth
    ce_loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)

    # 2. KL divergence with teacher soft labels
    teacher_dist = torch.stack([1 - teacher_prob, teacher_prob], dim=1)
    student_log_soft = F.log_softmax(logits / kd_temperature, dim=1)
    teacher_soft = (teacher_dist + 1e-8).pow(1.0 / kd_temperature)
    teacher_soft = teacher_soft / teacher_soft.sum(dim=1, keepdim=True)
    kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (kd_temperature ** 2)

    # 3. Contrastive alignment with semantic prototypes
    sim_real = (proj_img * e_real).sum(dim=-1) / temperature
    sim_fake = (proj_img * e_fake).sum(dim=-1) / temperature
    proto_logits = torch.stack([sim_real, sim_fake], dim=1)
    contrastive_loss = F.cross_entropy(proto_logits, labels)

    total = ALPHA * ce_loss + BETA * kl_loss + GAMMA * contrastive_loss
    return total, ce_loss.item(), kl_loss.item(), contrastive_loss.item()


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
        log.warning("  Soft labels file not found! Running without KL loss.")
        return {}
    with open(SOFT_LABELS_PATH) as f:
        records = json.load(f)
    soft = {}
    for r in records:
        soft[r["image_id"]] = r["teacher_prob_fake"]
    log.info(f"  Loaded {len(soft)} soft labels")
    return soft


class CombinedDataset(Dataset):
    def __init__(self, paths_and_ids, prototypes, soft_labels, preprocess, augment=True):
        self.preprocess = preprocess
        self.augment = augment
        self.items = []
        n_with_soft = 0
        for path, img_id in paths_and_ids:
            if img_id in prototypes:
                p = prototypes[img_id]
                label = 0 if p["gt"].upper() == "REAL" else 1
                teacher_prob = soft_labels.get(img_id, float(label))
                if img_id in soft_labels:
                    n_with_soft += 1
                self.items.append((path, label, p["e_real"], p["e_fake"], teacher_prob))
        log.info(f"    {len(self.items)} images with prototypes, {n_with_soft} with teacher soft labels")

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
    log.info("COMBINED PIPELINE — CE + KL(teacher) + Contrastive(prototypes)")
    log.info(f"Started: {datetime.now()}")
    log.info(f"Alpha(CE)={ALPHA}, Beta(KL)={BETA}, Gamma(Contrast)={GAMMA}")
    log.info(f"Temp={TEMPERATURE}, KD_Temp={KD_TEMPERATURE}")
    log.info(f"Weight decay={WEIGHT_DECAY}, LR={LR}")
    log.info("=" * 70)

    # Load CLIP
    log.info("\nLoading CLIP ViT-L-14...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    clip_model = clip_model.cuda().eval()

    # Build prototypes
    prototypes = build_text_prototypes(clip_model, tokenizer)

    # Load teacher soft labels
    soft_labels = load_teacher_soft_labels()

    # Build student
    student = SemanticStudent(clip_model, embed_dim=768).cuda()
    for p in student.image_encoder.parameters():
        p.requires_grad = False
    log.info("  Student: CLIP (frozen) + projector + classifier")
    log.info(f"  Loss: {ALPHA}*CE + {BETA}*KL(teacher) + {GAMMA}*contrastive(prototypes)")

    # Data splits
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

    # Filter to images with prototypes
    coco_tr_p = [(p, i) for p, i in coco_tr if i in prototypes][:1500]
    mj_tr_p = [(p, i) for p, i in mj_tr if i in prototypes][:1500]
    log.info(f"\n  Training: {len(coco_tr_p)} real + {len(mj_tr_p)} fake with prototypes")

    n_soft = sum(1 for _, i in coco_tr_p + mj_tr_p if i in soft_labels)
    log.info(f"  Of which {n_soft} also have teacher soft labels")

    if len(coco_tr_p) == 0 or len(mj_tr_p) == 0:
        log.error("No matching prototypes!")
        sys.exit(1)

    # Dataloaders
    train_ds = ConcatDataset([
        CombinedDataset(coco_tr_p, prototypes, soft_labels, preprocess, augment=True),
        CombinedDataset(mj_tr_p, prototypes, soft_labels, preprocess, augment=True),
    ])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    log.info(f"  Train dataset: {len(train_ds)} images")

    # Test loaders
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

    # Optimizer
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
    log.info(f"Loss = {ALPHA}*CE + {BETA}*KL(teacher) + {GAMMA}*contrastive")
    log.info("-" * 80)

    for epoch in range(MAX_EPOCHS):
        student.train()
        t_loss, t_ce, t_kl, t_con, n_b, correct, total = 0, 0, 0, 0, 0, 0, 0

        for imgs, labels, e_real, e_fake, teacher_prob in tqdm(
                train_loader, desc=f"Ep {epoch+1}/{MAX_EPOCHS}", leave=False):
            imgs = imgs.cuda()
            labels = labels.cuda().long()
            e_real = F.normalize(e_real.float().cuda(), dim=-1)
            e_fake = F.normalize(e_fake.float().cuda(), dim=-1)
            teacher_prob = teacher_prob.float().cuda()

            proj_img, logits = student(imgs)
            loss, ce, kl, con = combined_loss(
                proj_img, logits, labels, e_real, e_fake, teacher_prob,
                TEMPERATURE, KD_TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_ce += ce
            t_kl += kl
            t_con += con
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
                         "kl": round(t_kl/n_b, 4),
                         "con": round(t_con/n_b, 4),
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
                     f"(ce:{t_ce/n_b:.3f} kl:{t_kl/n_b:.3f} con:{t_con/n_b:.3f}) | "
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

    # Final eval
    log.info(f"\n{'=' * 80}")
    log.info("FINAL RESULTS - COMBINED PIPELINE")
    log.info(f"Loss = {ALPHA}*CE + {BETA}*KL(teacher) + {GAMMA}*contrastive")
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

    log.info(f"\n  COMBINED TRANSFER:")
    log.info(f"  CE: learns from ground truth labels")
    log.info(f"  KL: learns from teacher GBM confidence")
    log.info(f"  Contrastive: aligns with VLM reasoning semantics")
    log.info(f"  All three supervision signals active")

    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump({"config": {"alpha_ce": ALPHA, "beta_kl": BETA, "gamma_contrastive": GAMMA,
                               "temperature": TEMPERATURE, "kd_temperature": KD_TEMPERATURE,
                               "weight_decay": WEIGHT_DECAY, "lr": LR,
                               "approach": "CE_plus_KL_teacher_plus_contrastive_prototypes"},
                    "results": final, "history": history}, f, indent=2)
    log.info(f"\nTime: {(time.time()-t0)/60:.1f} min")
    log.info(f"Model: {best_path}")
    log.info("DONE.")


if __name__ == "__main__":
    main()
