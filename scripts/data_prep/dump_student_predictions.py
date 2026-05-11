#!/usr/bin/env python3
"""
Loads best.pt from the v2 run, reproduces the cross-gen test split,
runs Student inference (no blend), dumps per-image P(fake) to .npz.
"""
import os, sys, json, random, importlib.util
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

OUT_DIR     = Path("/NAS_DISK/Saptarshi_data/dhsd_v2_20260427_133533")
CKPT        = OUT_DIR / "best.pt"
CNN_ROOT    = Path("/NAS_DISK/Saptarshi_data/dataset/cnndetection_test")
HF_CACHE    = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

DEVICE = "cuda"
SEED   = 42
CROSSGEN_VAL_PER_GEN = 250
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Import DHSDv2 from the actual training script (no re-derivation)
spec = importlib.util.spec_from_file_location("train_dhsd_v2",
        "/home/tbvl_akshay/train_dhsd_v2.py")
mod = importlib.util.module_from_spec(spec)
# We don't run main(); we only want the class + helpers
sys.modules["train_dhsd_v2"] = mod
# Read the file, strip out the main() call to avoid running training
src = open("/home/tbvl_akshay/train_dhsd_v2.py").read()
src = src.replace('if __name__ == "__main__":', 'if False:')
exec(compile(src, "train_dhsd_v2.py", "exec"), mod.__dict__)
DHSDv2 = mod.DHSDv2
load_clip = mod.load_clip
clip_forward = mod.clip_forward

# Load CLIP + register hooks (uses module's own state)
print("[1/4] Loading CLIP ViT-L/14...")
clip_model, preprocess, tokenizer, tap_blocks = load_clip()

# Load Student
print("[2/4] Loading Student checkpoint...")
ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
student = DHSDv2().to(DEVICE).eval()
student.load_state_dict(ck["state"])
print(f"  best epoch: {ck['epoch']}, crossgen_macro: {ck['crossgen_macro']:.4f}")

# Reproduce cross-gen test split — same seed/logic as training
print("[3/4] Building cross-gen test split (same seeds as training)...")
from collections import defaultdict
crossgen_all = defaultdict(list)
for gen_dir in sorted(CNN_ROOT.iterdir()):
    if not gen_dir.is_dir(): continue
    for sub, lbl in [("0_real", 0), ("1_fake", 1)]:
        d = gen_dir / sub
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in IMG_EXT:
                    crossgen_all[gen_dir.name].append((str(p), lbl, p.stem))
    for sub in gen_dir.iterdir():
        if not sub.is_dir() or sub.name in ("0_real","1_fake"): continue
        for inner in ("0_real","1_fake"):
            d = sub / inner
            if d.exists():
                lbl = 0 if inner=="0_real" else 1
                for p in d.iterdir():
                    if p.suffix.lower() in IMG_EXT:
                        crossgen_all[gen_dir.name].append((str(p), lbl, p.stem))

crossgen_test = {}
rng = random.Random(SEED + 100)
for gen, items in crossgen_all.items():
    if set(l for _,l,_ in items) != {0,1}: continue
    real = [x for x in items if x[1]==0]
    fake = [x for x in items if x[1]==1]
    rng.shuffle(real); rng.shuffle(fake)
    nv = min(CROSSGEN_VAL_PER_GEN, len(real)//2, len(fake)//2)
    crossgen_test[gen] = real[nv:] + fake[nv:]

class EvalDS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        pth, y, sid = self.items[i]
        try: x = preprocess(Image.open(pth).convert("RGB"))
        except: x = torch.zeros(3, 224, 224)
        return x, int(y), sid, pth

# Run inference per generator
print("[4/4] Running inference...")
all_results = {}  # gen -> {paths, labels, probs}
for gen, items in crossgen_test.items():
    ld = DataLoader(EvalDS(items), batch_size=96, shuffle=False,
                    num_workers=6, pin_memory=True)
    probs, labels, paths = [], [], []
    for x, y, sids, pths in ld:
        x = x.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            feat, taps = clip_forward(clip_model, x)
            logit, _ = student(feat, taps)
            p = torch.sigmoid(logit).cpu().numpy()
        probs.extend(p); labels.extend(y.numpy()); paths.extend(pths)
    probs = np.array(probs); labels = np.array(labels)
    auroc = roc_auc_score(labels, probs)
    all_results[gen] = {"probs": probs, "labels": labels, "paths": paths, "auroc": float(auroc)}
    print(f"  {gen:<22s} n={len(probs):>5,}  AUROC={auroc:.4f}")

macro = float(np.mean([r["auroc"] for r in all_results.values()]))
print(f"\nStudent standalone macro AUROC: {macro:.4f}")

# Save
out = OUT_DIR / "crossgen_test_predictions.npz"
np.savez_compressed(out,
    **{f"{g}_probs":  all_results[g]["probs"]  for g in all_results},
    **{f"{g}_labels": all_results[g]["labels"] for g in all_results},
    **{f"{g}_paths":  np.array(all_results[g]["paths"], dtype=object) for g in all_results},
    macro=macro,
    generators=np.array(list(all_results.keys()), dtype=object),
)
print(f"\nWrote {out}")
