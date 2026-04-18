#!/usr/bin/env python3
"""Run student on ALL available test data, generate full stats + plots."""

import os, json, torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
from collections import defaultdict

HF_CACHE = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
MODEL_PATH = "/NAS_DISK/Saptarshi_data/pipeline_output/student_best.pt"
OUTPUT_DIR = "/NAS_DISK/Saptarshi_data/pipeline_output"
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

class ImageFolderDataset(Dataset):
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
    print("COMPLETE EVALUATION — ALL GENERATORS")
    print("Trained on: Midjourney (diffusion) + COCO real")
    print("=" * 70)

    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    student = StudentModel(clip_model).cuda()
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    student.head.load_state_dict(ckpt["model_state"])
    student.eval()
    print("Model loaded.\n")

    REAL_DIR = "/NAS_DISK/Saptarshi_data/dataset/real/coco"
    real_all = get_paths(REAL_DIR)
    np.random.shuffle(real_all)

    tests = {}

    # Own GAN tests
    for name in ["starGAN", "styleGAN", "BigGAN"]:
        d = f"/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/{name}"
        if os.path.isdir(d):
            fake = get_paths(d)
            real = real_all[:len(fake)]
            tests[name] = {"fake": fake, "real": real, "type": "GAN", "external": "No"}

    # Midjourney
    mj = get_paths("/NAS_DISK/Saptarshi_data/dataset/fake/midjourney")
    np.random.shuffle(mj)
    tests["Midjourney"] = {"fake": mj[:1000], "real": real_all[:1000], "type": "Diffusion", "external": "No (train)"}

    # ProGAN (CNNDetection)
    progan_dir = "/NAS_DISK/Saptarshi_data/dataset/cnndetection_test"
    if os.path.isdir(progan_dir):
        pr_real, pr_fake = get_progan_paths(progan_dir)
        np.random.shuffle(pr_real)
        np.random.shuffle(pr_fake)
        tests["ProGAN (CNNDet)"] = {"fake": pr_fake[:1000], "real": pr_real[:1000], "type": "GAN", "external": "Yes"}

    # Run all
    all_results = {}
    print(f"{'Test Set':<22} {'AUROC':<10} {'Acc':<10} {'Prec':<8} {'Rec':<8} {'F1':<8} {'N':<8} {'Type':<12} {'External'}")
    print("-" * 98)

    for name, data in tests.items():
        fake_ds = ImageFolderDataset(data["fake"], 1, preprocess)
        real_ds = ImageFolderDataset(data["real"], 0, preprocess)
        loader = DataLoader(ConcatDataset([real_ds, fake_ds]), batch_size=64, num_workers=4)

        probs, labels = evaluate(student, loader)
        preds = (probs > 0.5).astype(int)

        n_real = int(sum(labels == 0))
        n_fake = int(sum(labels == 1))

        if n_real > 0 and n_fake > 0:
            auc = roc_auc_score(labels, probs)
            acc = accuracy_score(labels, preds)
            # Per-class stats
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            tn = int(np.sum((preds == 0) & (labels == 0)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            print(f"{name:<22} {auc:<10.4f} {acc:<10.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f} {n_real+n_fake:<8} {data['type']:<12} {data['external']}")

            # ROC curve data
            fpr, tpr, thresholds = roc_curve(labels, probs)

            all_results[name] = {
                "auroc": float(auc), "accuracy": float(acc),
                "precision": float(prec), "recall": float(rec), "f1": float(f1),
                "n_real": n_real, "n_fake": n_fake,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "generator_type": data["type"],
                "external": data["external"],
                "avg_prob_real": float(probs[labels == 0].mean()),
                "avg_prob_fake": float(probs[labels == 1].mean()),
                "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            }

    # Summary stats
    print(f"\n{'=' * 70}")
    print("DETAILED STATS")
    print(f"{'=' * 70}")
    for name, r in all_results.items():
        print(f"\n  {name} ({r['generator_type']}, External: {r['external']})")
        print(f"    AUROC: {r['auroc']:.4f}  |  Accuracy: {r['accuracy']:.4f}")
        print(f"    Precision: {r['precision']:.4f}  |  Recall: {r['recall']:.4f}  |  F1: {r['f1']:.4f}")
        print(f"    TP: {r['tp']}  FP: {r['fp']}  TN: {r['tn']}  FN: {r['fn']}")
        print(f"    Avg P(fake) on REAL: {r['avg_prob_real']:.4f}")
        print(f"    Avg P(fake) on FAKE: {r['avg_prob_fake']:.4f}")

    # GAN vs Diffusion summary
    gan_aucs = [r["auroc"] for n, r in all_results.items() if r["generator_type"] == "GAN" and r["external"] == "No"]
    print(f"\n{'=' * 70}")
    print("THESIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Training data:     Midjourney (diffusion) + COCO real")
    print(f"  Training VLMs:     5 (InternVL, Qwen, GLM-4V, Pixtral, Phi-4)")
    print(f"  Student model:     CLIP ViT-L-14 encoder + 3-layer MLP head")
    print(f"  Distillation:      CE + KL divergence (alpha=0.5, beta=0.5)")
    if gan_aucs:
        print(f"\n  Cross-generator (Diffusion → GAN):")
        print(f"    Mean GAN AUROC:  {np.mean(gan_aucs):.4f}")
        print(f"    Best GAN AUROC:  {max(gan_aucs):.4f}")
        print(f"    Worst GAN AUROC: {min(gan_aucs):.4f}")

    # Save
    out = os.path.join(OUTPUT_DIR, "complete_results.json")
    # Remove fpr/tpr for clean JSON (too large)
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
    with open(out, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved: {out}")

if __name__ == "__main__":
    main()
