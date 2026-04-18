#!/usr/bin/env python3
"""Step 1: Merge all 5 VLM result files into a single JSON."""

import json, os
from pathlib import Path
from collections import Counter

# ── CONFIG ──
RESULT_DIR = "/NAS_DISK/Saptarshi_data/results"
OUTPUT = "/NAS_DISK/Saptarshi_data/merged_5vlm.json"

FILES = {
    "InternVL2.5-8B-MPO": f"{RESULT_DIR}/internvl_results.json",
    "GLM-4V-9B":          f"{RESULT_DIR}/glm4v_results.json",
    "Qwen2.5-VL-7B":      f"{RESULT_DIR}/qwen3vl_results.json",
    "Pixtral-12B":        f"{RESULT_DIR}/pixtral_results.json",
    "Phi-4-multimodal":   f"{RESULT_DIR}/phi4_results.json",
}

# ── MERGE ──
all_results = []
for model_name, fpath in FILES.items():
    if not os.path.exists(fpath):
        print(f"  MISSING: {fpath}")
        continue
    with open(fpath) as f:
        data = json.load(f)
    results = data.get("results", data if isinstance(data, list) else [])
    # Only keep successful entries
    good = [r for r in results if r.get("success", True)]
    all_results.extend(good)
    print(f"  {model_name}: {len(good)} entries loaded")

# Find common images across all models
img_per_model = {}
for r in all_results:
    model = r["model"]
    img_id = r["image_id"]
    if model not in img_per_model:
        img_per_model[model] = set()
    img_per_model[model].add(img_id)

common_imgs = set.intersection(*img_per_model.values())
print(f"\nCommon images across all {len(img_per_model)} models: {len(common_imgs)}")

# Filter to common images only
filtered = [r for r in all_results if r["image_id"] in common_imgs]
print(f"Filtered entries: {len(filtered)}")

# Stats
gt_counts = Counter(r["ground_truth"] for r in filtered)
model_counts = Counter(r["model"] for r in filtered)
print(f"Ground truth: {dict(gt_counts)}")
print(f"Per model: {dict(model_counts)}")

# Save
merged = {
    "metadata": {
        "total_entries": len(filtered),
        "unique_images": len(common_imgs),
        "models": list(img_per_model.keys()),
    },
    "results": filtered,
}
with open(OUTPUT, "w") as f:
    json.dump(merged, f)
print(f"\nSaved to {OUTPUT} ({os.path.getsize(OUTPUT)/1e6:.1f}MB)")
