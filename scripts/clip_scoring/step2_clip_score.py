#!/usr/bin/env python3
"""Step 2: CLIP-score VLM captions+cues against images.

For each image+model+assumption, computes CLIP cosine similarity between
the image and the combined text (caption + 3 cues).

Output: merged JSON with clip_score added to each entry.
"""

import json, os, time, torch
import numpy as np
from pathlib import Path
from PIL import Image

# ── CONFIG ──
MERGED_INPUT = "/NAS_DISK/Saptarshi_data/merged_5vlm.json"
OUTPUT = "/NAS_DISK/Saptarshi_data/merged_5vlm_clipped.json"

REAL_DIR = "/NAS_DISK/Saptarshi_data/dataset/real/coco"
FAKE_DIR = "/NAS_DISK/Saptarshi_data/dataset/fake/midjourney"

CLIP_MODEL = "ViT-L-14"       # better than ViT-B/32
CLIP_PRETRAINED = "openai"
BATCH_SIZE = 64                # text batch size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── INSTALL open_clip if needed ──
try:
    import open_clip
except ImportError:
    os.system("pip install open_clip_torch --break-system-packages -q")
    import open_clip

# ── LOAD CLIP ──
print(f"Loading CLIP {CLIP_MODEL}...")
model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=DEVICE
)
tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
model.eval()
print(f"CLIP loaded on {DEVICE}")

# ── LOAD DATA ──
print("Loading merged VLM results...")
with open(MERGED_INPUT) as f:
    data = json.load(f)
results = data["results"]
print(f"Total entries: {len(results)}")

# ── BUILD IMAGE PATH MAP ──
def find_image(image_id, ground_truth):
    """Find image file given ID and ground truth label."""
    if ground_truth == "REAL":
        base = REAL_DIR
    else:
        base = FAKE_DIR
    # Try common extensions
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG", ".PNG"]:
        p = Path(base) / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    # Try globbing
    matches = list(Path(base).glob(f"{image_id}.*"))
    if matches:
        return str(matches[0])
    return None

# ── ENCODE IMAGE (cached per image_id) ──
image_cache = {}

def get_image_features(image_id, ground_truth):
    if image_id in image_cache:
        return image_cache[image_id]
    path = find_image(image_id, ground_truth)
    if path is None:
        image_cache[image_id] = None
        return None
    try:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        image_cache[image_id] = feat
        return feat
    except Exception as e:
        print(f"  Image error {image_id}: {e}")
        image_cache[image_id] = None
        return None

# ── ENCODE TEXT ──
def get_text_features(text):
    tokens = tokenizer([text]).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat

# ── MAIN SCORING LOOP ──
print("\nScoring...")
t0 = time.time()
scored = 0
errors = 0

for i, r in enumerate(results):
    # Build combined text: caption + cues
    parts = [r.get("caption", "")]
    for k in ["cue_1", "cue_2", "cue_3"]:
        if r.get(k):
            parts.append(r[k])
    combined_text = ". ".join(p for p in parts if p)

    # Get image features
    img_feat = get_image_features(r["image_id"], r["ground_truth"])
    if img_feat is None:
        r["clip_score"] = None
        errors += 1
        continue

    # Get text features and compute similarity
    try:
        txt_feat = get_text_features(combined_text)
        sim = (img_feat @ txt_feat.T).item()
        r["clip_score"] = float(sim)
        scored += 1
    except Exception as e:
        r["clip_score"] = None
        errors += 1

    if (i + 1) % 5000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(results) - i - 1) / rate
        print(f"  [{i+1}/{len(results)}] {rate:.0f} entries/sec, "
              f"ETA: {eta/60:.1f}min, errors: {errors}")

    # Clear image cache periodically to save memory
    if len(image_cache) > 5000:
        image_cache.clear()
        torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nDone! Scored {scored}, errors {errors}, time {elapsed/60:.1f}min")

# ── SAVE ──
data["results"] = results
data["metadata"]["clip_model"] = CLIP_MODEL
with open(OUTPUT, "w") as f:
    json.dump(data, f)
print(f"Saved to {OUTPUT} ({os.path.getsize(OUTPUT)/1e6:.1f}MB)")
