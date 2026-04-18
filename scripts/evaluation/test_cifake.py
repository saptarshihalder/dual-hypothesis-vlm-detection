#!/usr/bin/env python3
"""
Test on CIFAKE — Stable Diffusion v1.4 generated images + CIFAR-10 real.
Student trained ONLY on Midjourney + COCO. CIFAKE is 100% unseen.
Published benchmark: Bird & Lotfi, IEEE Access 2024.
"""

import os, torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

HF_CACHE = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
MODEL_PATH = "/NAS_DISK/Saptarshi_data/pipeline_output/student_best.pt"
SEED = 42
np.random.seed(SEED)


class StudentModel(nn.Module):
    def __init__(self, clip_model, embed_dim=768):
        super().__init__()
        self.encoder = clip_model.visual
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.head(features.float())


def find_cifake():
    import kagglehub
    return kagglehub.dataset_download('birdy654/cifake-real-and-ai-generated-synthetic-images')


class CIFAKEDataset(Dataset):
    def __init__(self, base_dir, preprocess, max_per_class=1500):
        self.preprocess = preprocess
        self.items = []

        # CIFAKE structure: test/REAL/, test/FAKE/ (or train/REAL, train/FAKE)
        # Use test set for clean eval
        real_paths, fake_paths = [], []

        for root, dirs, files in os.walk(base_dir):
            folder = os.path.basename(root).upper()
            imgs = [os.path.join(root, f) for f in files
                    if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
            if folder == "REAL":
                real_paths.extend(imgs)
            elif folder == "FAKE":
                fake_paths.extend(imgs)

        print(f"  Found {len(real_paths)} real, {len(fake_paths)} fake")

        # Sample
        np.random.shuffle(real_paths)
        np.random.shuffle(fake_paths)
        real_paths = real_paths[:max_per_class]
        fake_paths = fake_paths[:max_per_class]

        for p in real_paths:
            self.items.append((p, 0))
        for p in fake_paths:
            self.items.append((p, 1))
        np.random.shuffle(self.items)
        print(f"  Using {len(real_paths)} real + {len(fake_paths)} fake = {len(self.items)} total")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.preprocess(img), label
        except:
            return torch.zeros(3, 224, 224), label


def main():
    print("=" * 70)
    print("CIFAKE BENCHMARK — Stable Diffusion v1.4 (100% UNSEEN)")
    print("Student trained ONLY on Midjourney + COCO")
    print("Bird & Lotfi, IEEE Access 2024")
    print("=" * 70)

    cifake_path = find_cifake()
    print(f"  Dataset: {cifake_path}")

    print("\nLoading CLIP + student...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE
    )
    student = StudentModel(clip_model).cuda()
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    student.head.load_state_dict(ckpt["model_state"])
    student.eval()

    # Test on CIFAKE test set (use test split for clean eval)
    # Try test/ first, then fall back to full dataset
    test_dir = os.path.join(cifake_path, "test")
    if not os.path.isdir(test_dir):
        test_dir = cifake_path
        print(f"  No test/ subfolder, using full dataset")

    ds = CIFAKEDataset(test_dir, preprocess, max_per_class=1500)
    if len(ds) == 0:
        print("No images in test/. Trying full dataset...")
        ds = CIFAKEDataset(cifake_path, preprocess, max_per_class=1500)

    if len(ds) == 0:
        print("ERROR: No images found!")
        os.system(f"find {cifake_path} -type d")
        sys.exit(1)

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = student(imgs.cuda())
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    print(f"\n{'=' * 70}")
    print(f"RESULTS — CIFAKE (Stable Diffusion v1.4)")
    print(f"{'=' * 70}")
    print(f"  AUROC:    {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Images:   {len(all_labels)} ({sum(all_labels==0)} real, {sum(all_labels==1)} fake)")
    print(f"\n{classification_report(all_labels, preds, target_names=['REAL','FAKE'])}")
    print(f"  Avg prob on REAL: {all_probs[all_labels==0].mean():.4f}")
    print(f"  Avg prob on FAKE: {all_probs[all_labels==1].mean():.4f}")
    print(f"\n  TRAINED ON:  Midjourney (diffusion) + COCO real")
    print(f"  TESTED ON:   CIFAKE — Stable Diffusion v1.4 + CIFAR-10 real")
    print(f"  OVERLAP:     ZERO")
    print(f"  GENERATOR:   Stable Diffusion v1.4 (never seen)")
    print(f"  DOMAIN:      CIFAR-10 categories (never seen)")

    # Combined results table with previous GAN results
    print(f"\n{'=' * 70}")
    print(f"COMPLETE CROSS-GENERATOR SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Test Set':<25} {'AUROC':<10} {'Generator':<20} {'In Training?'}")
    print("-" * 65)
    print(f"{'Midjourney (holdout)':<25} {'~1.000':<10} {'Midjourney':<20} {'Yes (70/30)'}")
    print(f"{'StarGAN':<25} {'0.9996':<10} {'StarGAN':<20} {'NO'}")
    print(f"{'BigGAN':<25} {'0.8449':<10} {'BigGAN':<20} {'NO'}")
    print(f"{'StyleGAN':<25} {'0.7716':<10} {'StyleGAN':<20} {'NO'}")
    print(f"{'CIFAKE (SD v1.4)':<25} {f'{auc:.4f}':<10} {'Stable Diffusion':<20} {'NO'}")

    # Save
    import json
    out = "/NAS_DISK/Saptarshi_data/pipeline_output/cifake_results.json"
    with open(out, "w") as f:
        json.dump({
            "dataset": "CIFAKE",
            "generator": "Stable_Diffusion_v1.4",
            "real_source": "CIFAR-10",
            "auroc": auc, "accuracy": acc,
            "n_real": int(sum(all_labels==0)),
            "n_fake": int(sum(all_labels==1)),
            "in_training": False,
            "reference": "Bird & Lotfi, IEEE Access 2024",
        }, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
