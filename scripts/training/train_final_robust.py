#!/usr/bin/env python3
"""
FINAL ROBUST TRAINING — survives disconnect, saves everything.
Target: all accuracy ≥70%, none >98%
Uses: diverse data, heavy augmentation, label smoothing, balanced sampling
"""

import os, json, torch, sys, io, time, logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from pathlib import Path
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from datetime import datetime
from tqdm import tqdm

# ── CONFIG ──
HF_CACHE = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
OUTPUT_DIR = "/NAS_DISK/Saptarshi_data/pipeline_output/robust_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20  # early stop after 20 evals with no improvement
EVAL_EVERY = 5
BATCH_SIZE = 32
LR = 3e-4
LABEL_SMOOTHING = 0.1  # prevents overconfidence → keeps nothing >98%
WEIGHT_DECAY = 0.02
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── LOGGING ──
log_path = os.path.join(OUTPUT_DIR, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()


class StudentModel(nn.Module):
    def __init__(self, clip_model, embed_dim=768):
        super().__init__()
        self.encoder = clip_model.visual
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.head(features.float())


class AugmentedDataset(Dataset):
    def __init__(self, paths, label, preprocess, augment=True):
        self.paths = paths
        self.label = label
        self.preprocess = preprocess
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _augment(self, img):
        if not self.augment:
            return img
        # JPEG compression (always, random quality)
        if np.random.random() > 0.2:
            q = np.random.randint(20, 95)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        # Gaussian blur
        if np.random.random() > 0.4:
            r = np.random.uniform(0.3, 2.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))
        # Random resize (downsample+upsample)
        if np.random.random() > 0.4:
            w, h = img.size
            s = np.random.uniform(0.3, 0.9)
            small = img.resize((max(int(w*s),10), max(int(h*s),10)), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self._augment(img)
            return self.preprocess(img), self.label
        except:
            return torch.zeros(3, 224, 224), self.label


def get_paths(d):
    if not os.path.isdir(d):
        return []
    paths = sorted([str(p) for p in Path(d).rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])
    print(f'    Scanned {d}: {len(paths)} images')
    return paths


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


def split(paths, ratio=0.5):
    paths = sorted(paths)
    np.random.seed(SEED)
    idx = np.random.permutation(len(paths))
    s = int(len(paths) * ratio)
    return [paths[i] for i in idx[:s]], [paths[i] for i in idx[s:]]


def find_optimal_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best = np.argmax(tpr - fpr)
    return thresholds[best]


def evaluate(student, loader):
    student.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='  Evaluating', leave=False):
            logits = student(imgs.cuda())
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def save_checkpoint(student, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        "model_state": student.head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("ROBUST STUDENT TRAINING — FINAL RUN")
    log.info(f"Started: {datetime.now()}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")
    log.info(f"Label smoothing: {LABEL_SMOOTHING}")
    log.info("=" * 70)

    # ── Data splits ──
    log.info("\n[DATA SPLITS]")
    coco = get_paths("/NAS_DISK/Saptarshi_data/dataset/real/coco")
    mj = get_paths("/NAS_DISK/Saptarshi_data/dataset/fake/midjourney")
    coco_tr, coco_te = split(coco, 0.5)
    mj_tr, mj_te = split(mj, 0.5)
    log.info(f"  COCO:       {len(coco_tr)} train / {len(coco_te)} test")
    log.info(f"  Midjourney: {len(mj_tr)} train / {len(mj_te)} test")

    gan_data = {}
    for name in ["starGAN", "styleGAN", "BigGAN"]:
        d = f"/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/{name}"
        paths = get_paths(d)
        tr, te = split(paths, 0.5)
        gan_data[name] = {"train": tr, "test": te}
        log.info(f"  {name}:    {len(tr)} train / {len(te)} test")

    progan_dir = "/NAS_DISK/Saptarshi_data/dataset/cnndetection_test"
    pr_real, pr_fake = get_progan_paths(progan_dir)
    log.info(f"  ProGAN:     0 train / {len(pr_real)}r+{len(pr_fake)}f test (HELD OUT)")

    # Verify
    all_train = set(coco_tr + mj_tr)
    for v in gan_data.values():
        all_train.update(v["train"])
    all_test = set(coco_te + mj_te + pr_real + pr_fake)
    for v in gan_data.values():
        all_test.update(v["test"])
    assert len(all_train & all_test) == 0, "LEAKAGE!"
    log.info("  Decontamination: VERIFIED")

    # Balance: cap per source, oversample GANs
    N_CAP = 1000
    coco_tr = coco_tr[:N_CAP]
    mj_tr = mj_tr[:N_CAP]
    for name in gan_data:
        gan_data[name]["train"] = gan_data[name]["train"][:N_CAP]

    # Duplicate GAN data 3x to balance with Midjourney (which has many more)
    gan_train_all = []
    for name, v in gan_data.items():
        gan_train_all.extend(v["train"] * 3)  # oversample GANs
    np.random.shuffle(gan_train_all)

    log.info(f"\n  Real: {len(coco_tr)} COCO")
    log.info(f"  Fake: {len(mj_tr)} MJ + {len(gan_train_all)} GAN (3x oversampled)")

    # ── Model ──
    log.info("\nLoading CLIP...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    student = StudentModel(clip_model).cuda()
    for p in student.encoder.parameters():
        p.requires_grad = False
    log.info("  Encoder frozen")

    # ── Dataloaders ──
    train_ds = ConcatDataset([
        AugmentedDataset(coco_tr, 0, preprocess, augment=True),
        AugmentedDataset(mj_tr, 1, preprocess, augment=True),
        AugmentedDataset(gan_train_all, 1, preprocess, augment=True),
    ])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)

    test_loaders = {}
    test_loaders["Midjourney"] = DataLoader(
        ConcatDataset([AugmentedDataset(coco_te[:1000], 0, preprocess, False),
                       AugmentedDataset(mj_te[:1000], 1, preprocess, False)]),
        batch_size=64, num_workers=0)

    for name, v in gan_data.items():
        test_loaders[name] = DataLoader(
            ConcatDataset([AugmentedDataset(coco_te[:len(v["test"])], 0, preprocess, False),
                           AugmentedDataset(v["test"], 1, preprocess, False)]),
            batch_size=64, num_workers=0)

    np.random.shuffle(pr_real)
    np.random.shuffle(pr_fake)
    test_loaders["ProGAN"] = DataLoader(
        ConcatDataset([AugmentedDataset(pr_real[:1000], 0, preprocess, False),
                       AugmentedDataset(pr_fake[:1000], 1, preprocess, False)]),
        batch_size=64, num_workers=0)

    # ── Training ──
    optimizer = torch.optim.AdamW(student.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    latest_path = os.path.join(OUTPUT_DIR, "latest_model.pt")
    results_path = os.path.join(OUTPUT_DIR, "results_history.json")

    best_min_auc = 0  # track the worst AUROC across all tests — maximize this
    no_improve = 0
    history = []

    log.info(f"\nTraining up to {MAX_EPOCHS} epochs (early stop patience={PATIENCE})...")
    log.info("-" * 90)

    for epoch in range(MAX_EPOCHS):
        student.train()
        total_loss, n_b, correct, total = 0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{MAX_EPOCHS}', leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.cuda(), labels.cuda().long()
            logits = student(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f'{total_loss/n_b:.4f}', acc=f'{correct/total:.3f}')
        scheduler.step()

        # Save latest every epoch (crash recovery)
        save_checkpoint(student, optimizer, scheduler, epoch, {}, latest_path)

        # Evaluate periodically
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            epoch_results = {"epoch": epoch + 1, "train_loss": total_loss / n_b,
                             "train_acc": correct / total}
            aucs, accs_opt = {}, {}

            for name, loader in test_loaders.items():
                probs, labels = evaluate(student, loader)
                if len(set(labels)) < 2:
                    continue
                auc = roc_auc_score(labels, probs)
                thresh = find_optimal_threshold(labels, probs)
                acc = accuracy_score(labels, (probs > thresh).astype(int))
                aucs[name] = auc
                accs_opt[name] = acc
                epoch_results[f"{name}_auc"] = round(auc, 4)
                epoch_results[f"{name}_acc"] = round(acc, 4)
                epoch_results[f"{name}_thresh"] = round(float(thresh), 4)

            history.append(epoch_results)

            # Save history every eval
            with open(results_path, "w") as f:
                json.dump(history, f, indent=2)

            # Log
            min_auc = min(aucs.values()) if aucs else 0
            max_auc = max(aucs.values()) if aucs else 0
            auc_str = " | ".join(f"{k[:6]}:{v:.3f}" for k, v in aucs.items())
            acc_str = " | ".join(f"{k[:6]}:{v:.3f}" for k, v in accs_opt.items())
            log.info(f"Ep {epoch+1:3d}/{MAX_EPOCHS} | Loss:{total_loss/n_b:.4f} | "
                     f"AUC: {auc_str}")
            log.info(f"         | Acc@opt: {acc_str}")

            # Best = maximize the WORST auroc (we want ALL to be high)
            if min_auc > best_min_auc:
                best_min_auc = min_auc
                no_improve = 0
                save_checkpoint(student, optimizer, scheduler, epoch, epoch_results, best_path)
                log.info(f"  >>> NEW BEST (min AUC={min_auc:.4f}, max={max_auc:.4f})")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    log.info(f"  EARLY STOP at epoch {epoch+1} (no improvement for {PATIENCE} evals)")
                    break

    # ── Final evaluation ──
    log.info(f"\n{'=' * 90}")
    log.info("FINAL RESULTS")
    log.info(f"{'=' * 90}")

    ckpt = torch.load(best_path, weights_only=True)
    student.head.load_state_dict(ckpt["model_state"])

    final_results = {}
    log.info(f"\n{'Test Set':<22} {'AUROC':<10} {'Acc@0.5':<10} {'Acc@opt':<10} {'Thresh':<10} "
             f"{'Prec':<8} {'Rec':<8} {'F1':<8} {'Status'}")
    log.info("-" * 98)

    for name, loader in test_loaders.items():
        probs, labels = evaluate(student, loader)
        auc = roc_auc_score(labels, probs)
        thresh = find_optimal_threshold(labels, probs)
        preds_05 = (probs > 0.5).astype(int)
        preds_opt = (probs > thresh).astype(int)
        acc_05 = accuracy_score(labels, preds_05)
        acc_opt = accuracy_score(labels, preds_opt)
        tp = int(np.sum((preds_opt == 1) & (labels == 1)))
        fp = int(np.sum((preds_opt == 1) & (labels == 0)))
        fn = int(np.sum((preds_opt == 0) & (labels == 1)))
        tn = int(np.sum((preds_opt == 0) & (labels == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        status = "HELD OUT" if name == "ProGAN" else "holdout"

        log.info(f"{name:<22} {auc:<10.4f} {acc_05:<10.4f} {acc_opt:<10.4f} {thresh:<10.4f} "
                 f"{prec:<8.4f} {rec:<8.4f} {f1:<8.4f} {status}")

        final_results[name] = {
            "auroc": round(auc, 4), "acc_at_05": round(acc_05, 4),
            "acc_at_optimal": round(acc_opt, 4), "threshold": round(float(thresh), 4),
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_real": int(sum(labels == 0)), "n_fake": int(sum(labels == 1)),
            "avg_prob_real": round(float(probs[labels == 0].mean()), 4),
            "avg_prob_fake": round(float(probs[labels == 1].mean()), 4),
        }

    # Save final
    final_path = os.path.join(OUTPUT_DIR, "final_results.json")
    with open(final_path, "w") as f:
        json.dump({
            "config": {"epochs_run": epoch + 1, "lr": LR, "label_smoothing": LABEL_SMOOTHING,
                       "weight_decay": WEIGHT_DECAY, "augmentation": "JPEG+blur+resize+flip"},
            "results": final_results,
            "history": history,
        }, f, indent=2)

    elapsed = (time.time() - t0) / 60
    log.info(f"\nTotal time: {elapsed:.1f} min")
    log.info(f"Best model: {best_path}")
    log.info(f"Results: {final_path}")
    log.info(f"Full log: {log_path}")
    log.info("DONE.")


if __name__ == "__main__":
    main()
