#!/usr/bin/env python3
"""
Build teacher soft labels from merged_5vlm_clipped.json.

Pipeline per image:
  1. For each VLM, compute discrepancy = clip_score(FAKE) - clip_score(REAL).
     Large positive -> FAKE assumption fits better -> image is likely AI.
     Small / negative -> REAL assumption fits better -> image is likely real.
  2. Aggregate across 5 VLMs (mean + std).
  3. Calibrate mean_discrepancy -> P(FAKE) via logistic regression against
     ground truth on a training split, giving well-scaled soft labels.
  4. Emit teacher_soft_labels.npz with (ids, probs[N,2]=[P(real), P(fake)]).

Two variants written:
  teacher_soft_labels.npz       — all 5 VLMs
  teacher_soft_labels_no_phi4.npz — Phi-4 excluded (noisy cues per ref doc)
"""
import json, numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SRC     = "/NAS_DISK/Saptarshi_data/merged_5vlm_clipped.json"
OUT_DIR = Path("/NAS_DISK/Saptarshi_data")

def build(exclude_models=None, tag=""):
    exclude_models = set(exclude_models or [])
    print(f"\n{'='*60}\nBuilding teacher labels — tag='{tag}'  exclude={exclude_models}\n{'='*60}")

    with open(SRC) as f:
        d = json.load(f)
    results = d["results"]

    # img_id -> {vlm -> {"REAL": score, "FAKE": score}}
    by_img = defaultdict(lambda: defaultdict(dict))
    gt = {}
    for r in results:
        if r["model"] in exclude_models: continue
        if not r.get("success", True): continue
        by_img[r["image_id"]][r["model"]][r["assumption"]] = r["clip_score"]
        gt[r["image_id"]] = r["ground_truth"]

    # Per-image: list of per-VLM discrepancies
    img_ids, discrepancies, labels = [], [], []
    skipped = 0
    for iid, vlm_dict in by_img.items():
        diffs = []
        for vlm, asm in vlm_dict.items():
            if "REAL" in asm and "FAKE" in asm:
                diffs.append(asm["FAKE"] - asm["REAL"])
        if len(diffs) < 2:     # need at least 2 VLMs
            skipped += 1
            continue
        img_ids.append(iid)
        discrepancies.append(diffs)     # variable length (if some VLMs missing an entry)
        labels.append(1 if gt[iid] == "FAKE" else 0)

    print(f"  Images w/ usable entries: {len(img_ids):,}  (skipped: {skipped})")

    # Feature: [mean, std, min, max, agreement] — fixed-length per image
    feats = np.zeros((len(img_ids), 5), dtype=np.float32)
    for i, diffs in enumerate(discrepancies):
        arr = np.array(diffs, dtype=np.float32)
        feats[i, 0] = arr.mean()
        feats[i, 1] = arr.std() if len(arr) > 1 else 0.0
        feats[i, 2] = arr.min()
        feats[i, 3] = arr.max()
        feats[i, 4] = (arr > 0).mean()        # fraction of VLMs voting "fake"
    labels = np.array(labels, dtype=np.int32)

    print(f"  Features shape: {feats.shape}")
    print(f"  Label balance: REAL={int((labels==0).sum())}  FAKE={int((labels==1).sum())}")
    print(f"  Mean discrepancy: REAL={feats[labels==0,0].mean():+.4f}  "
          f"FAKE={feats[labels==1,0].mean():+.4f}")
    print(f"  (gap between classes: {feats[labels==1,0].mean()-feats[labels==0,0].mean():+.4f})")

    # Calibrate: logistic regression mean_disc -> P(fake). Use all images
    # (we're producing soft labels, not testing generalization here).
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(feats, labels)
    probs_fake = clf.predict_proba(feats)[:, 1]

    # Sanity: how well does the mean-discrepancy teacher separate classes?
    auroc_single = roc_auc_score(labels, feats[:, 0])
    auroc_multi  = roc_auc_score(labels, probs_fake)
    print(f"  Teacher AUROC (mean_disc only): {auroc_single:.4f}")
    print(f"  Teacher AUROC (5-feat logistic): {auroc_multi:.4f}")
    print(f"  Calibrated P(fake) distribution:")
    for q in [0, 10, 25, 50, 75, 90, 100]:
        print(f"    p{q:3d}: {np.percentile(probs_fake, q):.3f}")

    # Confidence sanity: is the teacher over-confident?
    conf = np.maximum(probs_fake, 1.0 - probs_fake)
    print(f"  Teacher confidence (max prob): mean={conf.mean():.3f}  "
          f">0.9 frac={(conf>0.9).mean():.2%}   >0.95 frac={(conf>0.95).mean():.2%}")
    print(f"  If >0.9 frac is >80%, raise T_CALIB in training script.")

    # Pack into 2-col simplex
    probs = np.stack([1.0 - probs_fake, probs_fake], axis=1).astype(np.float32)
    ids = np.array(img_ids)

    out_path = OUT_DIR / f"teacher_soft_labels{'_'+tag if tag else ''}.npz"
    np.savez_compressed(out_path, ids=ids, probs=probs,
                         feats=feats, labels=labels,
                         logreg_coef=clf.coef_, logreg_intercept=clf.intercept_)
    print(f"  Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return auroc_multi

if __name__ == "__main__":
    # Variant A: all 5 VLMs
    a = build(exclude_models=None, tag="")
    # Variant B: exclude Phi-4 (known bad cues)
    b = build(exclude_models=["Phi-4-multimodal"], tag="no_phi4")
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"  Teacher AUROC — all 5 VLMs: {a:.4f}")
    print(f"  Teacher AUROC — no Phi-4:   {b:.4f}")
    print(f"  Recommended for training: {'no_phi4' if b > a else 'all 5'}")
