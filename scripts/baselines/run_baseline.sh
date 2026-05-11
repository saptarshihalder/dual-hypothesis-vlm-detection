#!/bin/bash
set -e
cd /home/tbvl_akshay
mkdir -p /NAS_DISK/Saptarshi_data/baselines_v1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate phi4
echo "=== Syntax check ==="
python3 -c "import ast; ast.parse(open('/home/tbvl_akshay/train_baselines.py').read()); print('SYNTAX OK')"
echo "=== Launching training ==="
python3 train_baselines.py --method univfd 2>&1 | tee /NAS_DISK/Saptarshi_data/baselines_v1/univfd_run.log
echo "=== Done ==="
