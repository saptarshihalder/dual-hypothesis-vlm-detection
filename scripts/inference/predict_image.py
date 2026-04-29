#!/usr/bin/env python3
"""
Run the trained student on a single image or folder of images.

Usage:
    python3 scripts/inference/predict_image.py \
        --ckpt   results/student/best.pt \
        --input  /path/to/image.jpg

    python3 scripts/inference/predict_image.py \
        --ckpt   results/student/best.pt \
        --input  /path/to/folder/ \
        --threshold 0.5
"""

import argparse, sys, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt",      required=True, help="path to best.pt")
ap.add_argument("--input",     required=True, help="image file or folder")
ap.add_argument("--threshold", type=float, default=0.5,
                help="decision threshold (default 0.5)")
ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

DEVICE = args.device

# ── Load backbone ─────────────────────────────────────────────────
try:
    import open_clip
except ImportError:
    sys.exit("open_clip not found. pip install open-clip-torch")

print("Loading CLIP ViT-L/14...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai", device=DEVICE)
clip_model.eval()

# ── Load student head ─────────────────────────────────────────────
# Add training script directory to path so we can import helpers
SCRIPT_DIR = Path(__file__).parent.parent / "training"
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from train_dhsd_v2 import DHSDv2Head as HeadClass, clip_forward, load_clip
except ImportError:
    # Fallback: find head class name dynamically
    import importlib.util, inspect, torch.nn as nn
    spec = importlib.util.spec_from_file_location(
        "train_dhsd_v2", SCRIPT_DIR / "train_dhsd_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    clip_forward = mod.clip_forward
    load_clip    = mod.load_clip
    # Find the head class (subclass of nn.Module that isn't nn.Module itself)
    candidates = [v for k, v in vars(mod).items()
                  if isinstance(v, type) and issubclass(v, nn.Module)
                  and v is not nn.Module and "Head" in k]
    if not candidates:
        sys.exit("Could not find Head class in train_dhsd_v2.py")
    HeadClass = candidates[0]
    print(f"Using head class: {HeadClass.__name__}")

print("Loading student head...")
clip_model_full, _, _, tap_blocks = load_clip()

with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
    _, taps = clip_forward(clip_model_full, dummy)
    tap_dims = [t.shape[-1] for t in taps]
    cls_dim  = clip_model_full.visual.output_dim

head = HeadClass(d_cls=cls_dim, tap_dims=tap_dims).to(DEVICE)
ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
head.load_state_dict(ckpt["state"])
head.eval()

# ── Collect images ────────────────────────────────────────────────
input_path = Path(args.input)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
if input_path.is_file():
    images = [input_path]
elif input_path.is_dir():
    images = [p for p in input_path.rglob("*") if p.suffix.lower() in EXTS]
else:
    sys.exit(f"Input not found: {input_path}")

print(f"\nRunning inference on {len(images)} image(s)...")
print(f"{'Image':<50}  {'P(fake)':>8}  {'Label':>8}")
print("-" * 70)

results = []
for img_path in sorted(images):
    img = Image.open(img_path).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat, taps_out = clip_forward(clip_model_full, x)
        logit, _       = head(feat, taps_out)
        p_fake         = torch.sigmoid(logit).item()
    label = "FAKE" if p_fake >= args.threshold else "REAL"
    name  = img_path.name[:48]
    print(f"{name:<50}  {p_fake:>8.4f}  {label:>8}")
    results.append({"file": str(img_path), "p_fake": round(p_fake, 6),
                    "label": label, "threshold": args.threshold})

if len(results) > 1:
    p_vals = [r["p_fake"] for r in results]
    n_fake = sum(1 for r in results if r["label"] == "FAKE")
    print("-" * 70)
    print(f"Total: {len(results)} images  |  "
          f"Predicted FAKE: {n_fake}  REAL: {len(results)-n_fake}  |  "
          f"Mean P(fake): {np.mean(p_vals):.4f}")
