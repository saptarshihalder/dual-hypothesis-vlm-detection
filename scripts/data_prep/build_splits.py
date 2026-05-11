#!/usr/bin/env python3
"""
Build splits.json for the student trainer.

Layout:
  train        — 80% of COCO real + 80% of Midjourney fake (teacher labels exist)
  val_seen     — 10% of COCO + 10% of Midjourney (teacher labels exist, same distribution)
  test_seen    — remaining 10% of COCO + 10% of Midjourney (held out, same distribution)
  val_holdout  — fake/gan_test/{starGAN,styleGAN,BigGAN} + matched COCO real
                 (used for early-stopping — DIFFERENT generators, same real)
  test_crossgen— cnndetection_test/* — 13 unseen generators, real OOD evaluation

Only images that have teacher labels go into train/val_seen/test_seen.
Holdout + cross-gen eval use student-only (no teacher labels needed;
the dataloader falls back to one-hot labels for these).
"""
import json, numpy as np, random
from pathlib import Path

SEED = 42
random.seed(SEED); np.random.seed(SEED)

DATASET  = Path("/NAS_DISK/Saptarshi_data/dataset")
NPZ      = "/NAS_DISK/Saptarshi_data/teacher_soft_labels_no_phi4.npz"
OUT      = "/NAS_DISK/Saptarshi_data/splits.json"

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def scan(dir_path, label, gen_name, expected_id_stem=None):
    """Return list of entries {id, path, gen, label}. id = filename stem."""
    p = Path(dir_path)
    if not p.exists():
        print(f"  [skip] {p} (missing)")
        return []
    entries = []
    for f in sorted(p.iterdir()):
        if f.suffix.lower() not in IMG_EXT: continue
        entries.append({
            "id":    f.stem,
            "path":  str(f),
            "gen":   gen_name,
            "label": label,
        })
    print(f"  {p}: {len(entries):,} images (gen={gen_name}, label={label})")
    return entries

def split_fraction(items, fractions, seed=SEED):
    """Deterministic shuffle then slice by fractions (must sum to 1.0)."""
    rng = random.Random(seed)
    items = list(items); rng.shuffle(items)
    n = len(items); cuts = []
    acc = 0.0
    for f in fractions[:-1]:
        acc += f
        cuts.append(int(n * acc))
    return [items[a:b] for a, b in zip([0]+cuts, cuts+[n])]

def main():
    # Load teacher label index to know which images are labeled
    npz = np.load(NPZ, allow_pickle=True)
    labeled_ids = set(str(x) for x in npz["ids"])
    print(f"Labeled images: {len(labeled_ids):,}")

    # --- In-distribution data (COCO real + Midjourney fake) ---
    print("\n[in-distribution]")
    coco = scan(DATASET / "real/coco",        label=0, gen_name="coco_real")
    mj   = scan(DATASET / "fake/midjourney",  label=1, gen_name="midjourney")
    coco_lab = [e for e in coco if e["id"] in labeled_ids]
    mj_lab   = [e for e in mj   if e["id"] in labeled_ids]
    print(f"  COCO w/ labels:       {len(coco_lab):,} / {len(coco):,}")
    print(f"  Midjourney w/ labels: {len(mj_lab):,} / {len(mj):,}")

    # 80/10/10 split per class
    coco_tr, coco_va, coco_te = split_fraction(coco_lab, [0.8, 0.1, 0.1])
    mj_tr,   mj_va,   mj_te   = split_fraction(mj_lab,   [0.8, 0.1, 0.1])

    train     = coco_tr + mj_tr
    val_seen  = coco_va + mj_va
    test_seen = coco_te + mj_te
    random.Random(SEED+1).shuffle(train)

    # --- In-dataset holdout generators (same NAS folder, different generator) ---
    print("\n[val_holdout: gan_test generators]")
    holdout = []
    for sub in ("starGAN", "styleGAN", "BigGAN"):
        holdout += scan(DATASET / "fake/gan_test" / sub, label=1, gen_name=f"gan_test_{sub}")
    # Matched real: sample from COCO test partition (unused in train)
    holdout_real_n = min(2000, len(coco_te))
    holdout_real = random.Random(SEED+2).sample(coco_te, holdout_real_n)
    # Re-tag so gen makes sense in eval
    holdout_real = [dict(e, gen="coco_real_holdout") for e in holdout_real]
    val_holdout = holdout + holdout_real
    # IMPORTANT: these COCO test items also stay in test_seen — they're ~2K images
    # reused for a companion real slice, OK since holdout measures generator shift.

    # --- Cross-generator test (CNNDetection, 13 generators) ---
    print("\n[test_crossgen: cnndetection_test]")
    cnn = DATASET / "cnndetection_test"
    test_crossgen = []
    # These follow a "0_real / 1_fake" layout
    for sub in ("deepfake", "san", "crn", "imle", "gaugan", "whichfaceisreal",
                "biggan", "seeingdark", "stargan"):
        test_crossgen += scan(cnn / sub / "0_real", label=0, gen_name=f"cnn_{sub}_real")
        test_crossgen += scan(cnn / sub / "1_fake", label=1, gen_name=f"cnn_{sub}")
    # cyclegan / stylegan / stylegan2 / progan have per-class subdirs (no real/fake split under that layer)
    # In CNNDetection these are all fake images; matched real comes from 0_real elsewhere
    for parent, name in [("progan","progan"), ("cyclegan","cyclegan"),
                         ("stylegan","stylegan"), ("stylegan2","stylegan2")]:
        pdir = cnn / parent
        if not pdir.exists(): continue
        for cls in sorted(pdir.iterdir()):
            if not cls.is_dir(): continue
            test_crossgen += scan(cls, label=1, gen_name=f"cnn_{name}_{cls.name}")

    # --- Write ---
    out = {
        "meta": {
            "seed": SEED,
            "teacher_labels": NPZ,
            "counts": {
                "train":         len(train),
                "val_seen":      len(val_seen),
                "test_seen":     len(test_seen),
                "val_holdout":   len(val_holdout),
                "test_crossgen": len(test_crossgen),
            },
        },
        "train":         train,
        "val_seen":      val_seen,
        "test_seen":     test_seen,
        "val_holdout":   val_holdout,
        "test_crossgen": test_crossgen,
    }
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\n{'='*60}\nSPLITS SUMMARY\n{'='*60}")
    for k, v in out["meta"]["counts"].items():
        print(f"  {k:16s}: {v:,}")
    print(f"\n  val_holdout generators: {sorted(set(e['gen'] for e in val_holdout))}")
    print(f"  test_crossgen generators: {len(set(e['gen'] for e in test_crossgen))} unique")
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
