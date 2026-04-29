# Dual-Hypothesis Semantic Fake Image Detection

A lightweight student model (0.71M params) trained on Midjourney+COCO that
generalises to unseen GAN generators without any GAN training data.

> **This is a method paper**, not a SOTA benchmark claim.
> The contribution is showing that semantic distillation from a vision-language
> model transfers to GAN-domain detection without GAN training data.

**Result: crossgen test macro AUROC = 0.8288 across 13 unseen generators
(CNNDetection benchmark, standalone student, no score blending, verified leakage-free)**

---

## Install

```bash
git clone https://github.com/saptarshihalder/dual-hypothesis-vlm-detection.git
cd dual-hypothesis-vlm-detection

# Install Git LFS (needed to pull model weights)
git lfs install
git lfs pull

pip install -r requirements.txt
```

requirements.txt must include:
`torch torchvision open-clip-torch scikit-learn numpy Pillow pyyaml`

---

## Quick start — run on one image

```bash
python3 scripts/inference/predict_image.py \
    --ckpt   results/student/best.pt \
    --input  /path/to/image.jpg
```

Output:
Run on a folder:
```bash
python3 scripts/inference/predict_image.py \
    --ckpt      results/student/best.pt \
    --input     /path/to/folder/ \
    --threshold 0.5 \
    --out       results.json
```

---

## Reproduce the 0.8288 result

### Step 1 — Set paths

```bash
export DATA_ROOT=/path/to/dataset
export TEACHER_PROB=/path/to/teacher_probs.npz
```

See DATA.md for the required folder structure.

### Step 2 — Generate teacher soft labels

```bash
bash scripts/reproduce/run_teacher_features.sh
```

### Step 3 — Train

```bash
bash scripts/reproduce/run_student_train.sh

# Or with explicit config:
python3 scripts/training/train_dhsd_v2.py \
    --config       configs/final_dhsd_v2.yaml \
    --data_root    $DATA_ROOT \
    --teacher_prob $TEACHER_PROB
```

Expected: best checkpoint at epoch 3, crossgen val macro ~0.83.
Training time: ~3.5 hours on one A100.

### Step 4 — Evaluate

```bash
# 13-generator result (AUROC)
bash scripts/reproduce/run_student_eval.sh

# Or manually:
python3 scripts/evaluation/test_cnndetection.py \
    --ckpt      results/student/best.pt \
    --data_root $DATA_ROOT/cnndetection_test
```

### Step 5 — Leakage audit

```bash
python3 scripts/evaluation/leakage_check.py
```

Expected output: 17/17 checks pass, no leakage detected.

---

## Results

### 13-generator CNNDetection (our full eval)

| Generator | AUROC | AP% |
|-----------|-------|-----|
| stargan | 0.9979 | 99.82 |
| cyclegan | 0.9777 | 98.01 |
| progan | 0.9631 | 96.60 |
| gaugan | 0.9525 | 95.66 |
| biggan | 0.9451 | 95.06 |
| crn | 0.9091 | 91.51 |
| imle | 0.8573 | 87.11 |
| whichfaceisreal | 0.8308 | 84.66 |
| deepfake | 0.7249 | 77.88 |
| stylegan | 0.6952 | 74.03 |
| seeingdark | 0.6674 | 68.99 |
| stylegan2 | 0.6536 | 64.76 |
| san | 0.5998 | 57.09 |
| **Macro** | **0.8288** | **83.94** |

### Comparison with NPR (Tan et al. CVPR 2024) on ForenSynths (8 generators)

| Training data | GAN test (mean AP%) | Midjourney test (AP%) |
|---|---|---|
| Ours (Midjourney+COCO) | 84.1 | ~100 |
| NPR (ProGAN) | 96.1 | 81.9 |

NPR wins on GAN generators because it trains on GAN data.
Our student wins on Midjourney. Neither generalises fully across training domains.
The finding: semantic distillation closes the GAN gap to -12 AP points
without any GAN training data.

NPR numbers from Table 1 and Table 5 of [arxiv:2312.10461](https://arxiv.org/abs/2312.10461).

---

## Model weights (Git LFS)

```bash
git lfs install
git lfs pull
ls -lh results/student/best.pt   # should be ~2.8MB
```

See MODEL_CARD.md for input format, preprocessing, and limitations.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/training/train_dhsd_v2.py` | Main training script (final model) |
| `scripts/inference/predict_image.py` | Run on one image or folder |
| `scripts/evaluation/test_cnndetection.py` | Eval on CNNDetection 13 generators |
| `scripts/evaluation/compare_npr.py` | 2x2 comparison vs NPR |
| `scripts/evaluation/leakage_check.py` | Full leakage audit |
| `scripts/reproduce/run_student_train.sh` | Reproduce training |
| `scripts/reproduce/run_student_eval.sh` | Reproduce evaluation |
| `configs/final_dhsd_v2.yaml` | Exact hyperparameters used |

Scripts in `scripts/training/` other than `train_dhsd_v2.py` are
archival ablation experiments and are not used in the final paper.

---

## Leakage verification

No score blending at inference. No test data used during training.
Full audit: `python3 scripts/evaluation/leakage_check.py`
Provenance record: `results/student/provenance.json`
