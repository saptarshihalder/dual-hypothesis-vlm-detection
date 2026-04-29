#!/bin/bash
# Step 1: Generate teacher soft-label probabilities
# Teacher = ProGAN-trained CNNDetection detector
# Output: teacher_probs.npz with keys: ids, probs

set -e
DATA_ROOT=${DATA_ROOT:-"/path/to/dataset"}
OUT=${OUT:-"./teacher_probs.npz"}

echo "Generating teacher P(fake) for training images..."
echo "DATA_ROOT: $DATA_ROOT"
echo "Output:    $OUT"

python3 scripts/evaluation/teacher_progan.py \
    --data_root "$DATA_ROOT" \
    --out       "$OUT"

echo "Done. Set TEACHER_PROB=$OUT before training."
