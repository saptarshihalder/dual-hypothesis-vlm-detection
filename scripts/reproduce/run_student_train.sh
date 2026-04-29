#!/bin/bash
# Step 2: Train the student and reproduce 0.8288 crossgen AUROC
# Prerequisites: teacher_probs.npz must exist (run run_teacher_features.sh first)

set -e
DATA_ROOT=${DATA_ROOT:-"/path/to/dataset"}
TEACHER_PROB=${TEACHER_PROB:-"./teacher_probs.npz"}

if [ ! -f "$TEACHER_PROB" ]; then
    echo "ERROR: $TEACHER_PROB not found. Run run_teacher_features.sh first."
    exit 1
fi

echo "Training student..."
echo "DATA_ROOT:    $DATA_ROOT"
echo "TEACHER_PROB: $TEACHER_PROB"

DATA_ROOT=$DATA_ROOT TEACHER_PROB=$TEACHER_PROB \
    python3 scripts/training/train_dhsd_v2.py

echo "Done. Check output dir for final_results.json"
