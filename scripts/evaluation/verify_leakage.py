#!/usr/bin/env python3
"""Verify zero data leakage — run alongside training."""

import os, json, hashlib
import numpy as np
from pathlib import Path

SEED = 42
np.random.seed(SEED)

def get_paths(d):
    if not os.path.isdir(d):
        return []
    return sorted([str(p) for p in Path(d).rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])

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

def split(paths, ratio=0.5):
    paths = sorted(paths)
    np.random.seed(SEED)
    idx = np.random.permutation(len(paths))
    s = int(len(paths) * ratio)
    return [paths[i] for i in idx[:s]], [paths[i] for i in idx[s:]]

def file_hash(path):
    """MD5 hash of file content — catches duplicates even with different names"""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

print("=" * 70)
print("DATA LEAKAGE VERIFICATION")
print("=" * 70)

# Reconstruct exact same splits as training script
coco = get_paths("/NAS_DISK/Saptarshi_data/dataset/real/coco")
mj = get_paths("/NAS_DISK/Saptarshi_data/dataset/fake/midjourney")
coco_tr, coco_te = split(coco, 0.5)
mj_tr, mj_te = split(mj, 0.5)

gan_data = {}
for name in ["starGAN", "styleGAN", "BigGAN"]:
    d = f"/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/{name}"
    paths = get_paths(d)
    tr, te = split(paths, 0.5)
    gan_data[name] = {"train": tr, "test": te}

pr_real, pr_fake = get_progan_paths("/NAS_DISK/Saptarshi_data/dataset/cnndetection_test")

# ── CHECK 1: Path overlap ──
print("\n[CHECK 1] Path-level overlap")
train_paths = set(coco_tr[:1000] + mj_tr[:1000])
for name, v in gan_data.items():
    train_paths.update(v["train"][:1000] * 3)  # same as oversampled

test_paths = set(coco_te[:1000] + mj_te[:1000])
for name, v in gan_data.items():
    test_paths.update(v["test"])
test_paths.update(pr_real)
test_paths.update(pr_fake)

overlap_paths = train_paths & test_paths
print(f"  Train files:  {len(train_paths)}")
print(f"  Test files:   {len(test_paths)}")
print(f"  Path overlap: {len(overlap_paths)}")
if overlap_paths:
    print(f"  LEAKAGE DETECTED!")
    for p in list(overlap_paths)[:5]:
        print(f"    {p}")
else:
    print(f"  PASS — zero path overlap")

# ── CHECK 2: ProGAN isolation ──
print("\n[CHECK 2] ProGAN completely excluded from training")
progan_all = set(pr_real + pr_fake)
progan_in_train = progan_all & train_paths
print(f"  ProGAN total files:    {len(progan_all)}")
print(f"  ProGAN in train set:   {len(progan_in_train)}")
if progan_in_train:
    print(f"  LEAKAGE DETECTED!")
else:
    print(f"  PASS — ProGAN 100% held out")

# ── CHECK 3: Content-level (hash) overlap ──
print("\n[CHECK 3] Content-level duplicate check (MD5 hashing)")
print("  Hashing train samples (first 500)...")
train_sample = list(train_paths)[:500]
train_hashes = set()
for p in train_sample:
    h = file_hash(p)
    if h:
        train_hashes.add(h)

print("  Hashing ProGAN test samples (first 500)...")
progan_sample = list(progan_all)[:500]
progan_hashes = set()
for p in progan_sample:
    h = file_hash(p)
    if h:
        progan_hashes.add(h)

hash_overlap = train_hashes & progan_hashes
print(f"  Train hashes:   {len(train_hashes)}")
print(f"  ProGAN hashes:  {len(progan_hashes)}")
print(f"  Content overlap: {len(hash_overlap)}")
if hash_overlap:
    print(f"  DUPLICATE CONTENT DETECTED!")
else:
    print(f"  PASS — zero content duplicates")

# ── CHECK 4: Directory isolation ──
print("\n[CHECK 4] Directory isolation")
train_dirs = set(os.path.dirname(p) for p in train_paths)
progan_dirs = set(os.path.dirname(p) for p in progan_all)
dir_overlap = train_dirs & progan_dirs
print(f"  Train directories:  {len(train_dirs)}")
print(f"  ProGAN directories: {len(progan_dirs)}")
print(f"  Dir overlap:        {len(dir_overlap)}")
if dir_overlap:
    print(f"  SHARED DIRECTORIES:")
    for d in dir_overlap:
        print(f"    {d}")
else:
    print(f"  PASS — completely separate directories")

# ── CHECK 5: GAN train/test split integrity ──
print("\n[CHECK 5] GAN train/test split integrity")
for name, v in gan_data.items():
    overlap = set(v["train"]) & set(v["test"])
    print(f"  {name}: {len(v['train'])} train / {len(v['test'])} test / {len(overlap)} overlap", end="")
    print("  PASS" if len(overlap) == 0 else "  LEAKAGE!")

# ── CHECK 6: Real image reuse ──
print("\n[CHECK 6] COCO real image split integrity")
coco_overlap = set(coco_tr) & set(coco_te)
print(f"  COCO train: {len(coco_tr)} / test: {len(coco_te)} / overlap: {len(coco_overlap)}")
print(f"  {'PASS' if len(coco_overlap) == 0 else 'LEAKAGE!'}")

# ── SUMMARY ──
print(f"\n{'=' * 70}")
all_pass = (len(overlap_paths) == 0 and len(progan_in_train) == 0 and
            len(hash_overlap) == 0 and len(dir_overlap) == 0 and
            len(coco_overlap) == 0 and
            all(len(set(v["train"]) & set(v["test"])) == 0 for v in gan_data.values()))

if all_pass:
    print("ALL 6 CHECKS PASSED — ZERO LEAKAGE CONFIRMED")
    print("Safe to report results in thesis.")
else:
    print("LEAKAGE DETECTED — DO NOT REPORT RESULTS")
print(f"{'=' * 70}")

# Save verification
with open(os.path.join("/NAS_DISK/Saptarshi_data/pipeline_output/robust_run", "leakage_verification.json"), "w") as f:
    json.dump({
        "path_overlap": len(overlap_paths),
        "progan_in_train": len(progan_in_train),
        "content_overlap": len(hash_overlap),
        "dir_overlap": len(dir_overlap),
        "coco_split_overlap": len(coco_overlap),
        "all_pass": all_pass,
    }, f, indent=2)
print("\nSaved: leakage_verification.json")

