#!/bin/bash
# Step 3: Evaluate pre-trained student on CNNDetection (13 generators)
# Reproduces both the 13-generator result (AUROC=0.8288) and
# the 8-generator ForenSynths/NPR subset (AP=84.1%)

set -e
CKPT=${CKPT:-"results/student/best.pt"}
DATA_ROOT=${DATA_ROOT:-"/path/to/dataset"}

echo "=== 13-generator eval (AUROC) ==="
python3 scripts/evaluation/test_cnndetection.py \
    --ckpt      "$CKPT" \
    --data_root "$DATA_ROOT/cnndetection_test"

echo ""
echo "=== 8-generator ForenSynths/NPR subset (AP) ==="
python3 scripts/evaluation/compare_npr.py

echo ""
echo "=== Leakage audit ==="
python3 scripts/evaluation/leakage_check.py
