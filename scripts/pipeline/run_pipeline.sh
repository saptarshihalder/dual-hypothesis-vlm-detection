#!/bin/bash
# ═══════════════════════════════════════════════════════════
# FULL PIPELINE — Run on server (172.30.2.52)
# ssh tbvl_akshay@172.30.2.52
# Password: <PASSWORD_REDACTED>
# ═══════════════════════════════════════════════════════════

# ── STEP 0: Upload scripts to server ──
# Run these FROM YOUR LOCAL MACHINE (not on server):

scp step1_merge.py step2_clip_score.py step3_train_classifier.py tbvl_akshay@172.30.2.52:/home/tbvl_akshay/

# ── STEP 0.5: Make sure result files are on server ──
# If they're not already in /NAS_DISK/Saptarshi_data/results/, upload them:

scp internvl_results.json glm4v_results.json qwen3vl_results.json pixtral_results.json phi4_results.json tbvl_akshay@172.30.2.52:/NAS_DISK/Saptarshi_data/results/


# ═══════════════════════════════════════════════════════════
# NOW SSH INTO SERVER AND RUN:
# ═══════════════════════════════════════════════════════════

ssh tbvl_akshay@172.30.2.52

# ── Activate working env ──
source ~/miniforge3/envs/phi4/bin/activate

# ── Install dependencies (one-time) ──
pip install open_clip_torch scikit-learn pandas xgboost --break-system-packages -q

# ── STEP 1: Merge all VLM results (~10 seconds) ──
cd /home/tbvl_akshay
python3 step1_merge.py

# ── STEP 2: CLIP score VLM outputs (~20-30 min) ──
python3 step2_clip_score.py

# ── STEP 3: Train classifier + ablations (~2 min) ──
python3 step3_train_classifier.py

# ═══════════════════════════════════════════════════════════
# OUTPUT FILES:
#   /NAS_DISK/Saptarshi_data/merged_5vlm.json          (merged VLM data)
#   /NAS_DISK/Saptarshi_data/merged_5vlm_clipped.json   (with CLIP scores)
#   /NAS_DISK/Saptarshi_data/classifier_results/
#     ├── classifier_results.json   (accuracy, AUROC, F1)
#     └── feature_matrix.csv        (features for further analysis)
# ═══════════════════════════════════════════════════════════

# ── OPTIONAL: Run all 3 steps in one go (background, with logging) ──
nohup bash -c "python3 step1_merge.py && python3 step2_clip_score.py && python3 step3_train_classifier.py" > pipeline_log.txt 2>&1 &
tail -f pipeline_log.txt
