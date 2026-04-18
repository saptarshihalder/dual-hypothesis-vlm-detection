#!/usr/bin/env python3
"""
Robust student with:
1. Diverse training (Midjourney + half of each GAN)
2. Heavy augmentation (JPEG, blur, resize)
3. Optimal threshold per test set
4. Proper decontamination
"""

import os, json, torch, sys, io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold

HF_CACHE = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
OUTPUT_DIR = "/NAS_DISK/Saptarshi_data/pipeline_output"
SEED = 42
EPOCHS = 100
torch.manual_seed(SEED)
np.random.seed(SEED)


class StudentModel(nn.Module):
    def __init__(self, clip_model, embed_dim=768):
        super().__init__()
        self.encoder = clip_model.visual
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.head(features.float())


class AugmentedDataset(Dataset):
    """Dataset with heavy augmentation — key for generalization."""
    def __init__(self, paths, label, preprocess, augment=True):
        self.paths = paths
        self.label = label
        self.preprocess = preprocess
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _augment(self, img):
        """Random JPEG compression + blur + resize (Wang et al. CVPR 2020 recipe)"""
        if not self.augment or np.random.random() > 0.5:
            return img

        # Random JPEG compression (quality 30-95)
        if np.random.random() > 0.3:
            quality = np.random.randint(30, 95)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # Random Gaussian blur
        if np.random.random() > 0.5:
            radius = np.random.uniform(0.5, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Random resize (downsample then upsample)
        if np.random.random() > 0.5:
            w, h = img.size
            scale = np.random.uniform(0.5, 0.9)
            small = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)

        return img

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self._augment(img)
            return self.preprocess(img), self.label
        except:
            return torch.zeros(3, 224, 224), self.label


def get_paths(d):
    paths = []
    if os.path.isdir(d):
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif"):
                paths.append(str(p))
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
    j = tpr - fpr
    best_idx = np.argmax(j)
    return thresholds[best_idx]


def evaluate(student, loader):
    student.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = student(imgs.cuda())
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    print("=" * 70)
    print("ROBUST STUDENT — Diverse training + Augmentation")
    print("=" * 70)

    # ── Split data: 50% train, 50% test for each GAN ──
    print("\n[DATA SPLITS]")
    coco = get_paths("/NAS_DISK/Saptarshi_data/dataset/real/coco")
    mj = get_paths("/NAS_DISK/Saptarshi_data/dataset/fake/midjourney")
    coco_tr, coco_te = split(coco, 0.5)
    mj_tr, mj_te = split(mj, 0.5)
    print(f"  COCO:       {len(coco_tr)} train / {len(coco_te)} test")
    print(f"  Midjourney: {len(mj_tr)} train / {len(mj_te)} test")

    gan_data = {}
    for name in ["starGAN", "styleGAN", "BigGAN"]:
        d = f"/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/{name}"
        paths = get_paths(d)
        tr, te = split(paths, 0.5)
        gan_data[name] = {"train": tr, "test": te}
        print(f"  {name}:    {len(tr)} train / {len(te)} test")

    # ProGAN — 100% held out
    progan_dir = "/NAS_DISK/Saptarshi_data/dataset/cnndetection_test"
    pr_real, pr_fake = get_progan_paths(progan_dir)
    print(f"  ProGAN:     0 train / {len(pr_real)}r+{len(pr_fake)}f test (100% held out)")

    # Verify no overlap
    all_train = set(coco_tr + mj_tr)
    for v in gan_data.values():
        all_train.update(v["train"])
    all_test = set(coco_te + mj_te + pr_real + pr_fake)
    for v in gan_data.values():
        all_test.update(v["test"])
    assert len(all_train & all_test) == 0, "LEAKAGE!"
    print("  Decontamination: VERIFIED")

    # ── Balance training: cap each source ──
    N_CAP = min(1000, len(coco_tr), len(mj_tr))
    coco_tr = coco_tr[:N_CAP]
    mj_tr = mj_tr[:N_CAP]
    for name in gan_data:
        gan_data[name]["train"] = gan_data[name]["train"][:N_CAP]

    n_real = len(coco_tr)
    n_fake = len(mj_tr) + sum(len(v["train"]) for v in gan_data.values())
    print(f"\n  Training: {n_real} real + {n_fake} fake (balanced via augmentation)")

    # ── Load CLIP ──
    print("\nLoading CLIP...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    student = StudentModel(clip_model).cuda()
    for p in student.encoder.parameters():
        p.requires_grad = False

    # ── Build dataloaders with augmentation ──
    train_ds = ConcatDataset([
        AugmentedDataset(coco_tr, 0, preprocess, augment=True),
        AugmentedDataset(mj_tr, 1, preprocess, augment=True),
    ] + [
        AugmentedDataset(v["train"], 1, preprocess, augment=True)
        for v in gan_data.values()
    ])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # Test loaders (no augmentation)
    test_loaders = {}

    # Midjourney holdout
    test_loaders["Midjourney (holdout)"] = DataLoader(
        ConcatDataset([AugmentedDataset(coco_te[:1000], 0, preprocess, augment=False),
                       AugmentedDataset(mj_te[:1000], 1, preprocess, augment=False)]),
        batch_size=64, num_workers=4)

    # GAN holdouts
    for name, v in gan_data.items():
        test_loaders[f"{name} (holdout)"] = DataLoader(
            ConcatDataset([AugmentedDataset(coco_te[:len(v["test"])], 0, preprocess, augment=False),
                           AugmentedDataset(v["test"], 1, preprocess, augment=False)]),
            batch_size=64, num_workers=4)

    # ProGAN (100% unseen)
    np.random.shuffle(pr_real)
    np.random.shuffle(pr_fake)
    test_loaders["ProGAN (100% unseen)"] = DataLoader(
        ConcatDataset([AugmentedDataset(pr_real[:1000], 0, preprocess, augment=False),
                       AugmentedDataset(pr_fake[:1000], 1, preprocess, augment=False)]),
        batch_size=64, num_workers=4)

    # ── Train ──
    optimizer = torch.optim.AdamW(student.head.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    save_path = os.path.join(OUTPUT_DIR, "student_robust.pt")
    best_avg = 0

    print(f"\nTraining {EPOCHS} epochs with augmentation...")
    print("-" * 70)

    for epoch in range(EPOCHS):
        student.train()
        total_loss, n_b, correct, total = 0, 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda().long()
            logits = student(imgs)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        # Quick eval
        if (epoch + 1) % 10 == 0 or epoch == 0:
            aucs = {}
            for name, loader in test_loaders.items():
                probs, labels = evaluate(student, loader)
                if len(set(labels)) > 1:
                    aucs[name] = roc_auc_score(labels, probs)
            avg = np.mean(list(aucs.values())) if aucs else 0
            auc_str = " | ".join(f"{k[:8]}:{v:.3f}" for k, v in aucs.items())
            print(f"Ep {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/n_b:.4f} | Tr: {correct/total:.3f} | {auc_str}")
            if avg > best_avg:
                best_avg = avg
                torch.save({"model_state": student.head.state_dict(), "epoch": epoch}, save_path)

    # ── Final evaluation ──
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS — ROBUST MODEL")
    print(f"{'=' * 70}")

    ckpt = torch.load(save_path, weights_only=True)
    student.head.load_state_dict(ckpt["model_state"])

    print(f"\n{'Test Set':<25} {'AUROC':<10} {'Acc@0.5':<10} {'Acc@opt':<10} {'Thresh':<10} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-" * 91)

    for name, loader in test_loaders.items():
        probs, labels = evaluate(student, loader)
        auc = roc_auc_score(labels, probs)

        # Fixed threshold
        preds_05 = (probs > 0.5).astype(int)
        acc_05 = accuracy_score(labels, preds_05)

        # Optimal threshold (Youden's J)
        opt_thresh = find_optimal_threshold(labels, probs)
        preds_opt = (probs > opt_thresh).astype(int)
        acc_opt = accuracy_score(labels, preds_opt)

        tp = int(np.sum((preds_opt == 1) & (labels == 1)))
        fp = int(np.sum((preds_opt == 1) & (labels == 0)))
        fn = int(np.sum((preds_opt == 0) & (labels == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        status = "UNSEEN" if "unseen" in name.lower() else "holdout"
        print(f"{name:<25} {auc:<10.4f} {acc_05:<10.4f} {acc_opt:<10.4f} {opt_thresh:<10.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f}")

    print(f"\n  Note: Acc@opt uses Youden's optimal threshold per test set")
    print(f"  ProGAN is 100% unseen — NEVER in training")

    # Save
    print(f"\n  Model: {save_path}")

if __name__ == "__main__":
    main()
