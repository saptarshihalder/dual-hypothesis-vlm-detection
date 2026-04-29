#!/usr/bin/env python3
"""
Run the trained DHSD v2 student on a single image or folder.

Usage:
    # Single image
    python3 scripts/inference/predict_image.py \
        --ckpt results/student/best.pt \
        --input /path/to/image.jpg

    # Folder of images
    python3 scripts/inference/predict_image.py \
        --ckpt results/student/best.pt \
        --input /path/to/folder/ \
        --threshold 0.5 \
        --out results.json
"""

import argparse, sys, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      required=True,
                    help="path to best.pt")
    ap.add_argument("--input",     required=True,
                    help="image file or folder")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="decision threshold (default 0.5)")
    ap.add_argument("--out",       default=None,
                    help="save results to this JSON file")
    ap.add_argument("--device",    default=None,
                    help="cuda or cpu (default: auto)")
    return ap.parse_args()

def load_model(ckpt_path, device):
    try:
        import open_clip
    except ImportError:
        sys.exit("Missing: pip install open-clip-torch")

    print(f"[1/3] Loading CLIP ViT-L/14 on {device}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    clip_model.eval()

    print("[2/3] Loading student head...")
    train_dir = Path(__file__).parent.parent / "training"
    sys.path.insert(0, str(train_dir))

    import importlib.util, torch.nn as nn
    spec = importlib.util.spec_from_file_location(
        "train_dhsd_v2", train_dir / "train_dhsd_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    clip_forward = mod.clip_forward
    load_clip    = mod.load_clip

    # Find head class dynamically
    candidates = [v for k, v in vars(mod).items()
                  if isinstance(v, type)
                  and issubclass(v, nn.Module)
                  and v is not nn.Module
                  and "Head" in k]
    if not candidates:
        sys.exit("Cannot find Head class in train_dhsd_v2.py")
    HeadClass = candidates[0]

    clip_full, _, _, _ = load_clip()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        _, taps = clip_forward(clip_full, dummy)
        tap_dims = [t.shape[-1] for t in taps]
        cls_dim  = clip_full.visual.output_dim

    head = HeadClass(d_cls=cls_dim, tap_dims=tap_dims).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["state"])
    head.eval()
    print("[3/3] Model ready.")
    return clip_full, clip_forward, head, preprocess

def predict(img_path, clip_model, clip_forward, head, preprocess, device):
    img = Image.open(img_path).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat, taps = clip_forward(clip_model, x)
        logit, _   = head(feat, taps)
        p_fake     = torch.sigmoid(logit).item()
    return p_fake

def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, clip_forward, head, preprocess = load_model(args.ckpt, device)

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    inp  = Path(args.input)
    if inp.is_file():
        images = [inp]
    elif inp.is_dir():
        images = sorted(p for p in inp.rglob("*") if p.suffix.lower() in EXTS)
    else:
        sys.exit(f"Not found: {inp}")

    print(f"\nRunning on {len(images)} image(s) | threshold={args.threshold}\n")
    print(f"{'File':<55}  {'P(fake)':>8}  {'Label':>6}")
    print("-" * 74)

    records = []
    for img_path in images:
        p = predict(img_path, clip_model, clip_forward, head, preprocess, device)
        label = "FAKE" if p >= args.threshold else "REAL"
        name  = str(img_path)[-53:] if len(str(img_path)) > 53 else str(img_path)
        print(f"{name:<55}  {p:>8.4f}  {label:>6}")
        records.append({"file": str(img_path), "p_fake": round(p, 6),
                         "label": label})

    if len(records) > 1:
        vals   = [r["p_fake"] for r in records]
        n_fake = sum(1 for r in records if r["label"] == "FAKE")
        print("-" * 74)
        print(f"Total {len(records)} | FAKE {n_fake} | REAL {len(records)-n_fake} "
              f"| mean P(fake) {np.mean(vals):.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
