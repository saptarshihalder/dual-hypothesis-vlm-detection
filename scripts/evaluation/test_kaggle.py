#!/usr/bin/env python3
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
KAGGLE_DIR = "/NAS_DISK/Saptarshi_data/dataset/kaggle_faces"

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

class KaggleFaceDataset(Dataset):
    def __init__(self, base_dir, preprocess):
        self.preprocess = preprocess
        self.items = []
        found_real, found_fake = [], []
        for root, dirs, files in os.walk(base_dir):
            folder_name = os.path.basename(root).lower()
            img_files = [f for f in files if Path(f).suffix.lower() in (".jpg",".jpeg",".png",".webp")]
            if not img_files:
                continue
            if folder_name == "training_real":
                for f in img_files:
                    found_real.append(os.path.join(root, f))
            elif folder_name == "training_fake":
                for f in img_files:
                    found_fake.append(os.path.join(root, f))
        print(f"  Found {len(found_real)} real, {len(found_fake)} fake images")
        np.random.seed(42)
        if len(found_real) > 1000:
            found_real = list(np.random.choice(found_real, 1000, replace=False))
        if len(found_fake) > 1000:
            found_fake = list(np.random.choice(found_fake, 1000, replace=False))
        for p in found_real:
            self.items.append((p, 0))
        for p in found_fake:
            self.items.append((p, 1))
        np.random.shuffle(self.items)
        print(f"  Using {sum(1 for x in self.items if x[1]==0)} real + {sum(1 for x in self.items if x[1]==1)} fake")
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
    print("="*60)
    print("CROSS-DOMAIN TEST: Kaggle Real & Fake Faces")
    print("Trained on: Midjourney (objects/scenes)")
    print("Testing on: GAN faces (Yonsei University)")
    print("="*60)
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    student = StudentModel(clip_model).cuda()
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    student.head.load_state_dict(ckpt["model_state"])
    student.eval()
    ds = KaggleFaceDataset(KAGGLE_DIR, preprocess)
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
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  AUROC:    {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Images:   {len(all_labels)} ({sum(all_labels==0)} real, {sum(all_labels==1)} fake)")
    print(f"\n{classification_report(all_labels, preds, target_names=['REAL','FAKE'])}")
    print(f"  Avg prob on REAL: {all_probs[all_labels==0].mean():.4f}")
    print(f"  Avg prob on FAKE: {all_probs[all_labels==1].mean():.4f}")
    print(f"\n  TRAINED ON: Midjourney diffusion (objects/scenes)")
    print(f"  TESTED ON:  GAN faces (completely unseen domain + generator)")

if __name__ == "__main__":
    main()
