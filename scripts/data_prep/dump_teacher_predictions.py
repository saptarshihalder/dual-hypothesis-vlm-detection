#!/usr/bin/env python3
"""
Load the trained 22-feature hybrid MLP and emit its predictions on all
21,305 MJ+COCO images as soft labels for student distillation.
"""
import json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path

NPZ_FEATS  = "/NAS_DISK/Saptarshi_data/teacher_soft_labels_percue_no_phi4.npz"
CLIP_JSON  = "/NAS_DISK/Saptarshi_data/clip_scores.json"
CKPT       = "/NAS_DISK/Saptarshi_data/adaptive_semantic/final_model.pt"
OUT        = "/NAS_DISK/Saptarshi_data/hybrid_teacher_soft_labels.npz"

class MLP(nn.Module):
    def __init__(self, d_in, h=64, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h//2, 2),
        )
    def forward(self, x): return self.net(x)

print("Loading features + checkpoint...")
d = np.load(NPZ_FEATS, allow_pickle=True)
ids    = np.array([str(x) for x in d["ids"]])
sem    = d["feats"].astype(np.float32)
with open(CLIP_JSON) as f: clip_d = json.load(f)
clip_map = {s["image_id"]: (s["real_similarity"], s["fake_similarity"]) for s in clip_d["scores"]}
direct = np.array([clip_map[i] for i in ids], dtype=np.float32)
X = np.concatenate([sem, direct], axis=1).astype(np.float32)

ck = torch.load(CKPT, map_location="cuda", weights_only=False)
m = ck["norm_mean"]; s = ck["norm_std"]
Xn = (X - m) / s

model = MLP(X.shape[1]).cuda().eval()
model.load_state_dict(ck["state_dict"])

with torch.no_grad():
    probs = F.softmax(model(torch.tensor(Xn, device="cuda")), -1).cpu().numpy()

# Sanity: must match reported test AUROC roughly
labels = d["labels"]
from sklearn.metrics import roc_auc_score
print(f"AUROC on all images: {roc_auc_score(labels, probs[:,1]):.4f}  "
      f"(expected ~0.94)")
print(f"Confidence distribution on real images:  mean P(fake) = {probs[labels==0,1].mean():.3f}")
print(f"Confidence distribution on fake images:  mean P(fake) = {probs[labels==1,1].mean():.3f}")

np.savez_compressed(OUT, ids=ids, probs=probs.astype(np.float32), labels=labels)
print(f"Wrote {OUT}")
