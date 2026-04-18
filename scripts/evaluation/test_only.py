#!/usr/bin/env python3
"""Quick cross-generator test — loads saved student, tests on mixed real+fake"""

import os, json, torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

HF_CACHE = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
MODEL_PATH = "/NAS_DISK/Saptarshi_data/pipeline_output/student_best.pt"
REAL_DIR = "/NAS_DISK/Saptarshi_data/dataset/real/coco"

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

class MixedDataset(Dataset):
    def __init__(self, fake_dir, real_dir, preprocess, n=500):
        self.preprocess = preprocess
        self.items = []
        # Grab fake images
        fake_imgs = sorted(Path(fake_dir).rglob("*"))
        fake_imgs = [p for p in fake_imgs if p.suffix.lower() in (".jpg",".jpeg",".png",".webp")][:n]
        for p in fake_imgs:
            self.items.append((str(p), 1))
        # Grab same number of real images
        real_imgs = sorted(Path(real_dir).rglob("*"))
        real_imgs = [p for p in real_imgs if p.suffix.lower() in (".jpg",".jpeg",".png",".webp")][:len(fake_imgs)]
        for p in real_imgs:
            self.items.append((str(p), 0))
        np.random.seed(42)
        np.random.shuffle(self.items)
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
    fake_dirs = {
        "starGAN": "/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/starGAN",
        "styleGAN": "/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/styleGAN",
        "BigGAN": "/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/BigGAN",
        "midjourney": "/NAS_DISK/Saptarshi_data/dataset/fake/midjourney",
    }

    print("Loading CLIP...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE
    )
    student = StudentModel(clip_model).cuda()

    print(f"Loading saved model: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    student.head.load_state_dict(ckpt["model_state"])
    student.eval()
    print(f"  (trained to val_auc={ckpt['val_auc']:.4f} at epoch {ckpt['epoch']+1})")

    print("\n" + "="*60)
    print(f"{'Generator':<15} {'AUROC':<10} {'Accuracy':<10} {'N_fake':<8} {'N_real':<8}")
    print("="*60)

    results = {}
    for name, fake_dir in fake_dirs.items():
        if not os.path.isdir(fake_dir):
            print(f"{name:<15} SKIPPED (dir not found)")
            continue
        ds = MixedDataset(fake_dir, REAL_DIR, preprocess, n=500)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                logits = student(imgs.cuda())
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
        n_fake = sum(all_labels)
        n_real = len(all_labels) - n_fake
        print(f"{name:<15} {auc:<10.4f} {acc:<10.4f} {n_fake:<8} {n_real:<8}")
        results[name] = {"auroc": auc, "accuracy": acc, "n_fake": int(n_fake), "n_real": int(n_real)}

    # Save
    out = "/NAS_DISK/Saptarshi_data/pipeline_output/crossgen_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
