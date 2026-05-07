#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS = {
    "CNNSpot":  "/NAS_DISK/Saptarshi_data/baselines_v1/cnnspot_crossgen_predictions.npz",
    "NPR":      "/NAS_DISK/Saptarshi_data/baselines_v1/npr_crossgen_predictions.npz",
    "UniFD":    "/NAS_DISK/Saptarshi_data/baselines_v1/univfd_crossgen_predictions.npz",
    "DHSDv2":   "/NAS_DISK/Saptarshi_data/dhsd_v2_20260427_133533/crossgen_test_predictions.npz",
}
N_BOOT = 1000
SEED = 42
rng = np.random.default_rng(SEED)

def load_per_gen(path):
    d = np.load(path, allow_pickle=True)
    gens = [str(g) for g in d["generators"]]
    return {g: (np.asarray(d[f"{g}_labels"]).astype(int),
                np.asarray(d[f"{g}_probs"]).astype(float)) for g in gens}

def macro(pg, fn):
    return np.mean([fn(yt, ys) for yt, ys in pg.values()])

def boot_ci(pg, fn):
    boots = []
    for _ in range(N_BOOT):
        rs = {}
        for g, (yt, ys) in pg.items():
            idx = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[idx])) < 2:
                continue
            rs[g] = (yt[idx], ys[idx])
        if len(rs) == len(pg):
            boots.append(macro(rs, fn))
    return np.percentile(np.array(boots), [2.5, 97.5])

print(f"\n{'Method':<10} {'macro AUROC [95% CI]':<32} {'macro AP [95% CI]':<32}")
print("-" * 76)
for name, path in RESULTS.items():
    try:
        pg = load_per_gen(path)
        au_pt, ap_pt = macro(pg, roc_auc_score), macro(pg, average_precision_score)
        au_ci, ap_ci = boot_ci(pg, roc_auc_score), boot_ci(pg, average_precision_score)
        print(f"{name:<10} {au_pt:.3f} [{au_ci[0]:.3f}, {au_ci[1]:.3f}]          "
              f"{ap_pt:.3f} [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]")
    except Exception as e:
        print(f"{name:<10} ERROR: {e}")

print("\nPer-generator AUROC CIs (DHSDv2):")
pg = load_per_gen(RESULTS["DHSDv2"])
for g, (yt, ys) in pg.items():
    pt = roc_auc_score(yt, ys)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[idx])) < 2:
            continue
        boots.append(roc_auc_score(yt[idx], ys[idx]))
    ci = np.percentile(boots, [2.5, 97.5])
    print(f"  {g:<22} {pt:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
