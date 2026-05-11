# Dual-Hypothesis Vision-Language Reasoning for AI-Generated Image Detection

Code, model weights, and reproducibility materials for the NeurIPS 2026 submission **Dual-Hypothesis Vision-Language Reasoning for AI-Generated Image Detection**.

A lightweight image-only detector with approximately **705K trainable parameters** on a frozen **CLIP ViT-L/14** backbone. The model is trained on **Midjourney + COCO** and generalises to **13 unseen generators** in the CNNDetection benchmark without using any GAN training data.

## Headline Result

**Cross-generator macro AUROC:** `0.829 [0.815, 0.843]` with a 95% bootstrap confidence interval across 13 held-out generators.

This is a **+0.186 improvement** over the matched UniFD baseline that uses the same frozen CLIP backbone.

This is a **method paper**. The central contribution is showing that **dual-hypothesis VLM reasoning** — querying multiple vision-language models under opposing real/fake priors and measuring semantic discrepancy — provides supervision that transfers across generator families when distilled into a compact image-only student.

---

## How It Works

### Teacher: Offline, Training Only

For each training image, five vision-language models are queried twice:

- **Real hypothesis:** describe the image assuming it is a real photograph and give three supporting cues.
- **Fake hypothesis:** describe the image assuming it is AI-generated and give three supporting cues.

The five VLMs are:

1. InternVL2.5
2. Qwen2.5-VL
3. GLM-4V
4. Pixtral
5. Phi-4

Each image therefore produces:

```text
caption + 3 cues × 2 hypotheses × 5 VLMs = 40 text outputs
```

Each text output is scored against the image using CLIP. The resulting 40 similarity scores are aggregated into a **22-dimensional semantic feature vector**.

A small MLP teacher then maps this semantic feature vector to a soft fake probability:

```text
22 → 64 → 32 → 2
```

The teacher produces:

```text
p_T(fake)
```

### Student: Online, Deployed

The deployed student is an image-only detector. It does **not** call any VLMs at inference time.

The student uses:

- A frozen CLIP ViT-L/14 backbone.
- A Transformer Intermediate Embedding module, called **TIE**.
- CLS tokens extracted from CLIP blocks `6`, `12`, `18`, and `24`.
- Attention pooling over intermediate CLS tokens.
- A linear projection of the final CLIP embedding.
- A fused 256-dimensional representation.
- An MLP classifier.

Only the adaptor and classifier are trainable. The CLIP backbone is never updated.

The student is trained using:

- Ground-truth binary cross-entropy loss.
- Adaptive distillation from the teacher.
- A CLIP zero-shot prior.

At deployment, no VLMs are called. The student runs in approximately **9 ms per image** at batch size 1 on an RTX 6000 Ada, which is comparable to UniFD.

For full method details, see the paper.

---

## Installation

```bash
git clone https://github.com/saptarshihalder/dual-hypothesis-vlm-detection.git
cd dual-hypothesis-vlm-detection
```

Pull model weights using Git LFS:

```bash
git lfs install
git lfs pull
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

The main dependencies are:

```text
torch
torchvision
open-clip-torch
scikit-learn
numpy
Pillow
pyyaml
matplotlib
```

---

## Quick Start

### Run on a Single Image

```bash
python3 scripts/inference/predict_image.py \
  --ckpt  results/student/best.pt \
  --input /path/to/image.jpg
```

### Run on a Folder

```bash
python3 scripts/inference/predict_image.py \
  --ckpt     results/student/best.pt \
  --input    /path/to/folder/ \
  --threshold 0.5 \
  --out      results.json
```

The output is:

```text
p_S(fake) ∈ [0, 1]
```

A value above `0.5` indicates that the image is likely AI-generated under the default threshold.

---

## Reproduce the Paper Results

### 1. Verify the Headline Numbers

No training is required for this step.

The trained student checkpoint and prediction NPZ files are included in the repository. The bootstrap 95% confidence intervals in Table 3 of the paper can be reproduced directly:

```bash
python3 scripts/bootstrap_cis.py
```

Expected output:

```text
Method     macro AUROC [95% CI]              macro AP [95% CI]

CNNSpot    0.479 [0.471, 0.488]              0.485 [0.477, 0.497]
NPR        0.539 [0.529, 0.548]              0.535 [0.526, 0.548]
UniFD      0.643 [0.633, 0.652]              0.624 [0.614, 0.636]
DHSDv2     0.825 [0.816, 0.834]              0.839 [0.829, 0.851]
```

Bootstrap uses 1,000 stratified resamples within each of the 13 generators, with macro scores re-aggregated per resample.

The point-estimate macro AUROC reported in the paper, `0.829`, is computed as the unweighted mean of per-generator AUROCs from the same prediction NPZ files.

### 2. Re-evaluate the Student on CNNDetection

```bash
export DATA_ROOT=/path/to/cnndetection_test
bash scripts/reproduce/run_student_eval.sh
```

The per-generator output should match Table 4 of the paper.

### 3. Retrain from Scratch

Set the dataset and teacher-probability paths:

```bash
export DATA_ROOT=/path/to/training/data
export TEACHER_PROB=results/teacher_labels/hybrid_teacher_soft_labels.npz
```

Run training:

```bash
bash scripts/reproduce/run_student_train.sh
```

Or run with explicit configuration:

```bash
python3 scripts/training/train_dhsd_v2.py \
  --config       configs/final_dhsd_v2.yaml \
  --data_root    $DATA_ROOT \
  --teacher_prob $TEACHER_PROB
```

Expected behaviour:

```text
Seed: 42
Best checkpoint: epoch 3
In-distribution validation macro AUROC: approximately 1.0
Cross-generator validation macro AUROC: approximately 0.82
Training time: approximately 3.5 hours on one A100, or approximately 6 hours on one RTX 6000 Ada
```

### 4. Regenerate Teacher Soft Labels

This step runs the full teacher pipeline and requires all five VLMs.

It is computationally expensive, taking approximately **100 seconds per image** across five VLMs.

```bash
bash scripts/reproduce/run_teacher_features.sh
```

See `DATA.md` for the required dataset structure.

---

## Repository Layout

```text
configs/
└── final_dhsd_v2.yaml
    Hyperparameters for the final student model

results/
├── student/
│   ├── best.pt
│   │   Trained student checkpoint, approximately 2.8 MB, via Git LFS
│   ├── final_results.json
│   │   Per-generator AUROC and AP
│   ├── provenance.json
│   │   Training environment and seed
│   └── predictions/
│       ├── dhsd_v2_crossgen_test_predictions.npz
│       │   Student predictions on 13 generators
│       ├── cnnspot_crossgen_predictions.npz
│       ├── npr_crossgen_predictions.npz
│       └── univfd_crossgen_predictions.npz
│
├── baselines/
│   ├── cnnspot_best.pt
│   ├── npr_best.pt
│   ├── crossgen_results.json
│   └── UniFD checkpoint excluded due to size; available separately at de-anonymization
│
├── teacher_labels/
│   ├── teacher_soft_labels_percue_5vlm.npz
│   │   5-VLM teacher MLP soft labels
│   ├── teacher_soft_labels_percue_4vlm.npz
│   │   4-VLM ablation labels, Phi-4 excluded
│   └── hybrid_teacher_soft_labels.npz
│       Labels used for student training
│
├── figures/
│   ├── fig.pdf
│   ├── fig_*.png
│   ├── gan_collage.pdf
│   └── gan_collage.png
│
└── diffusion/
    ├── internvl_results.json
    ├── qwen3vl_results.json
    ├── glm4v_results.json
    ├── pixtral_results.json
    └── phi4_results.json
        Raw VLM caption and cue outputs, approximately 42,760 entries each

scripts/
├── training/
│   └── train_dhsd_v2.py
│       Main training script used for the paper configuration
│
├── inference/
│   └── predict_image.py
│       Single-image or folder inference
│
├── evaluation/
│   ├── test_cnndetection.py
│   │   13-generator CNNDetection evaluation
│   ├── compare_npr.py
│   │   Comparison against NPR, Tan et al. CVPR 2024
│   ├── leakage_check.py
│   └── verify_leakage.py
│
├── data_prep/
│   ├── build_teacher_labels.py
│   │   Builds 5-feature teacher labels
│   ├── final_analysis.py
│   │   Builds 22-dimensional MLP teacher labels, paper configuration
│   ├── rescore_per_cue.py
│   │   Per-cue CLIP scoring
│   ├── build_splits.py
│   │   Train/validation/test split construction
│   ├── dump_student_predictions.py
│   └── dump_teacher_predictions.py
│
├── baselines/
│   └── train_baselines.py
│       Trains CNNSpot, NPR, and UniFD on matched Midjourney + COCO
│
├── figures/
│   ├── make_all_figures.py
│   │   Regenerates the 11 main paper figures
│   └── make_gan_collage.py
│       Regenerates Figure 3
│
└── reproduce/
    ├── run_student_train.sh
    ├── run_student_eval.sh
    ├── run_teacher_features.sh
    └── bootstrap_cis.py
        Reproduces Table 3 bootstrap confidence intervals

DATA.md
MODEL_CARD.md
requirements.txt
```

Other training scripts in `scripts/training/`, such as `train_adaptive.py` and `train_combined.py`, are archival ablation experiments and are not used in the final paper.

The single script used for all reported student results is:

```text
scripts/training/train_dhsd_v2.py
```

---

## Cross-Generator Results

Per-generator AUROC on CNNDetection with 13 held-out generators. All methods are trained on the matched Midjourney + COCO source pool.

| Generator | CNNSpot | NPR | UniFD | DHSDv2/Ours |
|---|---:|---:|---:|---:|
| StarGAN | 0.799 | 0.961 | 0.921 | **0.998** |
| CycleGAN | 0.418 | 0.298 | 0.780 | **0.978** |
| ProGAN | 0.534 | 0.621 | 0.715 | **0.963** |
| GauGAN | 0.426 | 0.410 | 0.791 | **0.953** |
| BigGAN | 0.481 | 0.623 | 0.798 | **0.945** |
| CRN | 0.379 | 0.409 | 0.393 | **0.909** |
| IMLE | 0.554 | 0.426 | 0.532 | **0.857** |
| WhichFaceIsReal | 0.363 | 0.510 | 0.573 | **0.831** |
| DeepFake | 0.453 | 0.720 | 0.677 | **0.725** |
| StyleGAN | 0.511 | 0.581 | 0.548 | **0.695** |
| SeeingDark | 0.261 | 0.384 | 0.648 | **0.667** |
| StyleGAN2 | 0.505 | 0.577 | 0.495 | **0.654** |
| SAN | 0.548 | 0.482 | 0.483 | **0.600** |
| **Macro** | **0.479** | **0.539** | **0.643** | **0.829** |

All methods use:

- The same Midjourney + COCO source pool.
- The same held-out CNNDetection test split.

CNNSpot and NPR drop below chance on multiple generators because their pixel-fingerprint signals do not transfer from diffusion training to GAN test data.

UniFD partially generalises through its CLIP backbone but lacks explicit semantic discrepancy supervision.

DHSDv2 remains above chance on all 13 generators.

For comparison against NPR's reported numbers on its native ProGAN-trained setup, see:

```text
scripts/evaluation/compare_npr.py
```

NPR outperforms DHSDv2 when trained and tested both on GAN data, as expected. DHSDv2's contribution is reaching usable cross-domain performance without GAN training data.

---

## Leakage Verification

All evaluation generators in CNNDetection are held out during training.

There is no score blending at inference.

To run the full leakage audit:

```bash
python3 scripts/evaluation/leakage_check.py
```

Expected result:

```text
17/17 checks pass
```

The audit verifies that:

- No test images appear in the training pool.
- No per-generator validation samples appear in the test split.
- The cross-generator validation split used for early stopping is disjoint from the held-out test split.

The provenance record is available at:

```text
results/student/provenance.json
```

---

## Model Weights via Git LFS

```bash
git lfs install
git lfs pull
ls -lh results/student/best.pt
```

Expected checkpoint size:

```text
approximately 2.8 MB
```

The student checkpoint contains the trainable adaptor and classifier weights only, corresponding to approximately **705K trainable parameters**.

The frozen CLIP ViT-L/14 backbone is loaded from `open-clip-torch` at inference time.

See `MODEL_CARD.md` for:

- Input preprocessing.
- Expected output range.
- Known limitations.

---

## Reproducibility

### Seed

All experiments use:

```text
SEED=42
```

Three independent training runs at this seed produced identical cross-generator AUROC to four decimal places.

### Hardware

Training and evaluation were performed on:

```text
NVIDIA RTX 6000 Ada, 48 GB
```

### Software

The main software environment was:

```text
Python 3.10
PyTorch 2.7.0
CUDA 12.6
transformers 5.4.0
```

See `requirements.txt` for the full pinned dependency list.

### Bootstrap Confidence Intervals

Reported intervals use:

- 1,000 stratified resamples.
- Resampling within each of the 13 generators.
- Macro re-aggregation for every resample.

The intervals can be reproduced using:

```bash
python3 scripts/bootstrap_cis.py
```

---

## Citation

```bibtex
@inproceedings{halder2026dhsd,
  title     = {Dual-Hypothesis Vision-Language Reasoning for AI-Generated Image Detection},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

Code:

```text
MIT License
```

Model weights:

```text
MIT License with attribution
```

Dataset access is subject to the original dataset licenses, including:

- CNNDetection
- WildFake
- MS-COCO
- Midjourney Terms of Service

See `DATA.md` for details.

---

## Acknowledgements

The frozen CLIP backbone is from `open-clip-torch`, using OpenAI ViT-L/14 under the MIT License.

Vision-language models used for offline teacher generation:

| Model | License |
|---|---|
| InternVL2.5-8B-MPO | MIT |
| Qwen2.5-VL-7B-Instruct | Apache 2.0 |
| GLM-4V-9B | GLM-4 License |
| Pixtral-12B | Apache 2.0 |
| Phi-4-Multimodal-Instruct | MIT |

Datasets:

| Dataset | Source / License |
|---|---|
| MS-COCO | CC BY 4.0 annotations |
| CNNDetection | Wang et al., CVPR 2020 |
| WildFake | Hong et al., AAAI 2025 |
