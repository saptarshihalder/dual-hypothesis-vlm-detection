#!/usr/bin/env python3
"""
Student Model Training via Knowledge Distillation (Steps 6-8)
+ Inference on unseen images (Steps 9-11)
"""

import os, json, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--soft_labels", default="/NAS_DISK/Saptarshi_data/pipeline_output/teacher_soft_labels_3k.json")
parser.add_argument("--real_dir", default="/NAS_DISK/Saptarshi_data/dataset/real/coco")
parser.add_argument("--fake_dir", default="/NAS_DISK/Saptarshi_data/dataset/fake/midjourney")
parser.add_argument("--test_dir", default="", help="Dir with unseen test images")
parser.add_argument("--test_label", default="FAKE")
parser.add_argument("--hf_cache", default="/NAS_DISK/Saptarshi_data/hf_cache")
parser.add_argument("--output_dir", default="/NAS_DISK/Saptarshi_data/pipeline_output")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--temperature", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.environ["HF_HOME"] = args.hf_cache
torch.manual_seed(args.seed)
np.random.seed(args.seed)


class DistillationDataset(Dataset):
    def __init__(self, records, real_dir, fake_dir, preprocess):
        self.records = records
        self.preprocess = preprocess
        self.path_cache = {}
        for d in [real_dir, fake_dir]:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    self.path_cache[Path(f).stem] = os.path.join(d, f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_id = rec["image_id"]
        gt = 0 if rec["ground_truth"] == "REAL" else 1
        teacher_prob = rec["teacher_prob_fake"]
        path = self.path_cache.get(img_id)
        if path is None:
            return torch.zeros(3, 224, 224), gt, teacher_prob
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = self.preprocess(img)
        except:
            img_tensor = torch.zeros(3, 224, 224)
        return img_tensor, gt, teacher_prob


class TestDataset(Dataset):
    def __init__(self, image_dir, label, preprocess):
        self.preprocess = preprocess
        self.label = 0 if label == "REAL" else 1
        self.paths = []
        for p in sorted(Path(image_dir).rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                self.paths.append(str(p))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.preprocess(img), self.label, self.paths[idx]
        except:
            return torch.zeros(3, 224, 224), self.label, self.paths[idx]


class StudentModel(nn.Module):
    def __init__(self, clip_model, embed_dim=768):
        super().__init__()
        self.encoder = clip_model.visual
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        if hasattr(features, "float"):
            features = features.float()
        logits = self.head(features)
        return logits


def distillation_loss(student_logits, labels, teacher_probs, alpha, beta, temperature):
    ce_loss = F.cross_entropy(student_logits, labels)
    teacher_dist = torch.stack([1 - teacher_probs, teacher_probs], dim=1)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = (teacher_dist + 1e-8).pow(1.0 / temperature)
    teacher_soft = teacher_soft / teacher_soft.sum(dim=1, keepdim=True)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    return alpha * ce_loss + beta * kd_loss, ce_loss.item(), kd_loss.item()


def main():
    print("="*60)
    print("STUDENT MODEL TRAINING (Knowledge Distillation)")
    print("="*60)

    print(f"\nLoading soft labels: {args.soft_labels}")
    with open(args.soft_labels) as f:
        records = json.load(f)
    print(f"  {len(records)} records")

    np.random.shuffle(records)
    split = int(len(records) * 0.8)
    train_recs = records[:split]
    val_recs = records[split:]
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    print("\nLoading CLIP encoder...")
    try:
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", cache_dir=args.hf_cache
        )
        embed_dim = 768
    except ImportError:
        import clip
        clip_model, preprocess = clip.load("ViT-L-14", device="cpu")
        embed_dim = 768

    student = StudentModel(clip_model, embed_dim).cuda()
    for param in student.encoder.parameters():
        param.requires_grad = False
    print("  Encoder frozen - training head only")

    optimizer = torch.optim.Adam(student.head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_ds = DistillationDataset(train_recs, args.real_dir, args.fake_dir, preprocess)
    val_ds = DistillationDataset(val_recs, args.real_dir, args.fake_dir, preprocess)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_auc = 0
    save_path = os.path.join(args.output_dir, "student_best.pt")
    print(f"\nTraining for {args.epochs} epochs (a={args.alpha}, b={args.beta}, T={args.temperature})")
    print("-"*60)

    for epoch in range(args.epochs):
        student.train()
        total_loss, total_ce, total_kd, n_batches = 0, 0, 0, 0
        for imgs, labels, teacher_probs in train_loader:
            imgs = imgs.cuda()
            labels = labels.cuda().long()
            teacher_probs = teacher_probs.cuda().float()
            logits = student(imgs)
            loss, ce, kd = distillation_loss(logits, labels, teacher_probs, args.alpha, args.beta, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_ce += ce
            total_kd += kd
            n_batches += 1
        scheduler.step()

        student.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.cuda()
                logits = student(imgs)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        val_auc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {avg_loss:.4f} (CE:{total_ce/n_batches:.4f} KD:{total_kd/n_batches:.4f}) | Val AUC: {val_auc:.4f} Acc: {val_acc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({"model_state": student.head.state_dict(), "epoch": epoch, "val_auc": val_auc, "val_acc": val_acc}, save_path)

    print(f"\nBest Val AUC: {best_auc:.4f}")
    print(f"Model saved: {save_path}")

    if args.test_dir and os.path.isdir(args.test_dir):
        print(f"\n{'='*60}")
        print(f"CROSS-GENERATOR INFERENCE")
        print(f"Test dir: {args.test_dir} (label: {args.test_label})")
        print(f"{'='*60}")
        ckpt = torch.load(save_path, weights_only=True)
        student.head.load_state_dict(ckpt["model_state"])
        student.eval()
        test_ds = TestDataset(args.test_dir, args.test_label, preprocess)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"  Test images: {len(test_ds)}")
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels, paths in test_loader:
                imgs = imgs.cuda()
                logits = student(imgs)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        test_auc = roc_auc_score(all_labels, all_probs)
        test_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
        print(f"\n  CROSS-GENERATOR RESULTS:")
        print(f"  AUROC: {test_auc:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  (Trained on Midjourney, tested on {Path(args.test_dir).name})")
        result_path = os.path.join(args.output_dir, "crossgen_student_results.json")
        with open(result_path, "w") as f:
            json.dump({
                "train_source": "midjourney", "test_source": Path(args.test_dir).name,
                "test_label": args.test_label, "n_test": len(test_ds),
                "auroc": test_auc, "accuracy": test_acc, "best_val_auc": best_auc,
            }, f, indent=2)
        print(f"  Saved: {result_path}")
    else:
        print("\nNo --test_dir provided, skipping cross-gen test.")


if __name__ == "__main__":
    main()
