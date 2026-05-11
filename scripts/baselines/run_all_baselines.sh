#!/bin/bash
# Master runner: trains/evaluates all baselines + summarizes vs DHSDv2.
# Usage:
#   /home/tbvl_akshay/run_all.sh
#   /home/tbvl_akshay/run_all.sh --eval-only

set -e
cd /home/tbvl_akshay

OUT_DIR=/NAS_DISK/Saptarshi_data/baselines_v1
mkdir -p "$OUT_DIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate phi4

LOG="$OUT_DIR/run_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "  RUN_ALL - full baseline pipeline"
echo "  Started: $(date)"
echo "  Logging to: $LOG"
echo "=========================================================="

# Step 1: syntax check
echo ""
echo "[STEP 1/3] Syntax check..."
python3 -c "import ast; ast.parse(open('/home/tbvl_akshay/train_baselines.py').read()); print('SYNTAX OK')"

# Step 2: train + eval all baselines
echo ""
echo "[STEP 2/3] Training and evaluating all baselines..."
if [ "$1" = "--eval-only" ]; then
    python3 train_baselines.py --method all --eval-only
else
    python3 train_baselines.py --method all
fi

# Step 3: unified comparison table
echo ""
echo "[STEP 3/3] Generating unified comparison table..."
python3 << 'PYEND'
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score, accuracy_score
)

OUT = Path("/NAS_DISK/Saptarshi_data/baselines_v1")
DHSD_PRED = Path("/NAS_DISK/Saptarshi_data/dhsd_v2_20260427_133533/crossgen_test_predictions.npz")

# DHSD numbers from saved predictions
print("\n" + "=" * 78)
print("  Computing DHSDv2 metrics from saved predictions")
print("=" * 78)
d = np.load(DHSD_PRED, allow_pickle=True)
gens = sorted(list(d["generators"]))
dhsd_per_gen = {}
auroc_l, ap_l = [], []
for g in gens:
    y = d[f"{g}_labels"]; p = d[f"{g}_probs"]
    auroc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    dhsd_per_gen[g] = {"auroc": auroc, "ap": ap}
    auroc_l.append(auroc); ap_l.append(ap)
dhsd_macro_auroc = float(np.mean(auroc_l))
dhsd_macro_ap = float(np.mean(ap_l))
print(f"  DHSDv2 macro AUROC: {dhsd_macro_auroc:.4f}")
print(f"  DHSDv2 macro AP:    {dhsd_macro_ap:.4f}")

# Load baseline results
results = {"DHSDv2 (ours)": {"auroc": dhsd_macro_auroc, "ap": dhsd_macro_ap, "per_gen": dhsd_per_gen}}
for method in ["cnnspot", "npr", "univfd"]:
    fpath = OUT / f"{method}_crossgen_results.json"
    if not fpath.exists():
        print(f"  [warn] {fpath} missing, skipping {method}")
        continue
    with open(fpath) as f:
        r = json.load(f)
    results[method.upper()] = {
        "auroc": r["macro_auroc"],
        "ap": r["macro_ap"],
        "per_gen": {g: {"auroc": v["auroc"], "ap": v["ap"]}
                    for g, v in r["per_gen"].items()},
    }

# Headline table
print("\n" + "=" * 78)
print("  HEADLINE COMPARISON (all methods, matched training+test)")
print("=" * 78)
print(f"  {'Method':<18}  {'macro AUROC':>12}  {'macro AP':>10}")
print(f"  {'-'*18}  {'-'*12}  {'-'*10}")
order = ["CNNSPOT", "NPR", "UNIVFD", "DHSDv2 (ours)"]
for m in order:
    if m in results:
        r = results[m]
        print(f"  {m:<18}  {r['auroc']:>12.4f}  {r['ap']:>10.4f}")

# Per-generator table
print("\n" + "=" * 78)
print("  PER-GENERATOR AUROC (rows=generator, cols=method)")
print("=" * 78)
all_gens = sorted({g for r in results.values() for g in r["per_gen"]})
present = [m for m in order if m in results]
hdr = f"  {'Generator':<22}  " + "  ".join(f"{m:>10}" for m in present)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for g in all_gens:
    row = f"  {g:<22}"
    best_method = max(present, key=lambda m: results[m]["per_gen"].get(g, {"auroc": -1})["auroc"])
    for m in present:
        v = results[m]["per_gen"].get(g, {}).get("auroc")
        if v is None:
            row += f"  {'--':>10}"
        elif m == best_method:
            row += f"  {('*' + f'{v:.4f}'):>10}"
        else:
            row += f"  {v:>10.4f}"
    print(row)
print("  (* marks best per generator)")

# Save merged summary
merged = {m: {"macro_auroc": r["auroc"], "macro_ap": r["ap"]} for m, r in results.items()}
with open(OUT / "all_methods_summary.json", "w") as f:
    json.dump(merged, f, indent=2)
print(f"\nSaved {OUT}/all_methods_summary.json")
PYEND

echo ""
echo "=========================================================="
echo "  Finished: $(date)"
echo "=========================================================="
