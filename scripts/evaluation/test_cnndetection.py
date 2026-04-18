#!/usr/bin/env python3
"""
Test on CNNDetection ProGAN testset (Wang et al., CVPR 2020)
THE standard GAN detection benchmark. 20 categories, diverse content.
Student trained ONLY on Midjourney + COCO. This is 100% unseen GAN data.
"""
import os, torch, sys, json
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
BASE_DIR = "/NAS_DISK/Saptarshi_data/dataset/cnndetection_test"
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

class CNNDetDataset(Dataset):
    def __init__(self, base_dir, preprocess, max_per_class=1500):
        self.preprocess = preprocess
        self.items = []
        real_paths, fake_paths = [], []
        for root, dirs, files in os.walk(base_dir):
            folder = os.path.basename(root)
            imgs = [os.path.join(root, f) for f in files
                    if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")]
            if folder == "0_real":
                real_paths.extend(imgs)
            elif folder == "1_fake":
                fake_paths.extend(imgs)
        print(f"  Found {len(real_paths)} real, {len(fake_paths)} fake")
        np.random.shuffle(real_paths)
        np.random.shuffle(fake_paths)
        real_paths = real_paths[:max_per_class]
        fake_paths = fake_paths[:max_per_class]
        for p in real_paths:
            self.items.append((p, 0))
        for p in fake_paths:
            self.items.append((p, 1))
        np.random.shuffle(self.items)
        print(f"  Using {len(real_paths)} real + {len(fake_paths)} fake")
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
    print("CNNDetection BENCHMARK — ProGAN (Wang et al., CVPR 2020)")
    print("THE standard GAN detection benchmark")
    print("Student trained ONLY on Midjourney (diffusion) + COCO real")
    print("ProGAN: 100% unseen GAN, diverse categories")
    print("=" * 70)

    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    student = StudentModel(clip_model).cuda()
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    student.head.load_state_dict(ckpt["model_state"])
    student.eval()

    ds = CNNDetDataset(BASE_DIR, preprocess, max_per_class=1500)
    if len(ds) == 0:
        print("No images found! Checking structure...")
        os.system(f"find {BASE_DIR} -type d | head -30")
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
    print(f"RESULTS — CNNDetection ProGAN (CVPR 2020 Benchmark)")
    print(f"{'=' * 70}")
    print(f"  AUROC:    {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Images:   {len(all_labels)} ({sum(all_labels==0)} real, {sum(all_labels==1)} fake)")
    print(f"\n{classification_report(all_labels, preds, target_names=['REAL','FAKE'])}")
    print(f"  Avg prob REAL: {all_probs[all_labels==0].mean():.4f}")
    print(f"  Avg prob FAKE: {all_probs[all_labels==1].mean():.4f}")

    print(f"\n{'=' * 70}")
    print(f"COMPLETE RESULTS TABLE")
    print(f"{'=' * 70}")
    print(f"{'Test Set':<28} {'AUROC':<10} {'Type':<12} {'External?'}")
    print("-" * 62)
    print(f"{'Midjourney (holdout)':<28} {'~1.000':<10} {'Diffusion':<12} {'No'}")
    print(f"{'StarGAN (own test)':<28} {'0.9996':<10} {'GAN':<12} {'No'}")
    print(f"{'BigGAN (own test)':<28} {'0.8449':<10} {'GAN':<12} {'No'}")
    print(f"{'StyleGAN (own test)':<28} {'0.7716':<10} {'GAN':<12} {'No'}")
    print(f"{'ProGAN (CNNDetection)':<28} {f'{auc:.4f}':<10} {'GAN':<12} {'YES - CVPR2020'}")

    out = "/NAS_DISK/Saptarshi_data/pipeline_output/cnndetection_results.json"
    with open(out, "w") as f:
        json.dump({"dataset": "CNNDetection_ProGAN", "reference": "Wang et al. CVPR 2020",
                    "generator": "ProGAN", "content": "20 LSUN categories (diverse)",
                    "auroc": auc, "accuracy": acc, "in_training": False}, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
