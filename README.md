Dual-Hypothesis Vision-Language Reasoning for AI-Generated Image Detection

Code, model weights, and reproducibility materials for the NeurIPS 2026 submission Dual-Hypothesis Vision-Language Reasoning for AI-Generated Image Detection.

A lightweight image-only detector (~705K trainable parameters on a frozen CLIP ViT-L/14 backbone) trained on Midjourney + COCO that generalises to 13 unseen generators in the CNNDetection benchmark without any GAN training data.

Headline result. Cross-generator macro AUROC of 0.829 [0.815, 0.843] (95% bootstrap CI) across 13 held-out generators, a +0.186 improvement over the matched UniFD baseline that uses the same frozen CLIP backbone.

This is a method paper. The contribution is showing that dual-hypothesis VLM reasoning — querying multiple vision-language models under opposing real/fake priors and measuring the semantic discrepancy — provides supervision that transfers across generator families when distilled into a compact image-only student.

How it works

Teacher (offline, training only). For each training image, five vision-language models (InternVL2.5, Qwen2.5-VL, GLM-4V, Pixtral, Phi-4) are queried twice: once under a real hypothesis ("describe this assuming it is a real photograph and give three supporting cues"), once under a fake hypothesis ("describe this assuming it is AI-generated and give three supporting cues"). Each pair of outputs (caption + 3 cues × 2 hypotheses × 5 VLMs = 40 texts) is scored against the image with CLIP. The 40 similarity scores are aggregated into a 22-dimensional semantic feature, and a small MLP teacher (22→64→32→2) produces a soft fake probability p_T(fake).

Student (online, deployed). A frozen CLIP ViT-L/14 with a small adaptor — a Transformer Intermediate Embedding (TIE) module that attention-pools CLS tokens from blocks 6/12/18/24, plus a linear projection of the final CLIP embedding — fused into a 256-d feature and passed through an MLP classifier. Trained against ground-truth labels (BCE) plus adaptive distillation from the teacher and a CLIP zero-shot prior. Only the adaptor and classifier are trainable (~705K parameters); the backbone is never updated.

At deployment, no VLMs are called. The student runs in ~9 ms per image at batch size 1 on an RTX 6000 Ada — comparable to UniFD.

For full method details, see the paper.

Install

```bash
git clone https://github.com/saptarshihalder/dual-hypothesis-vlm-detection.git
cd dual-hypothesis-vlm-detection

Pull model weights (Git LFS)

git lfs install
git lfs pull

Install Python dependencies

pip install -r requirements.txt
```
Requirements: torch, torchvision, open-clip-torch, scikit-learn, numpy, Pillow, pyyaml, matplotlib.

Quick start

Run on a single image:
```bash
python3 scripts/inference/predict_image.py 

--ckpt   results/student/best.pt 

--input  /path/to/image.jpg
```

Run on a folder:
```bash
python3 scripts/inference/predict_image.py 

--ckpt      results/student/best.pt 

--input     /path/to/folder/ 

--threshold 0.5 

--out       results.json
```
Output: p_S(fake) in [0, 1]. A value above 0.5 indicates likely AI-generated under the default threshold.

Reproduce the paper results

1. Verify the headline numbers (no training needed)

The trained student checkpoint and the prediction NPZs are included in the repo. The bootstrap 95% confidence intervals in Table 3 of the paper can be reproduced directly:

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
(Bootstrap uses 1,000 stratified resamples within each of 13 generators, with macro re-aggregated per resample. The point-estimate macro reported in the paper, 0.829, is computed as the unweighted mean of per-generator AUROCs from the same prediction NPZs.)

2. Re-evaluate the student on CNNDetection

```bash
export DATA_ROOT=/path/to/cnndetection_test
bash scripts/reproduce/run_student_eval.sh
```
Per-generator output should match Table 4 of the paper.

3. Retrain from scratch

Set paths and run:
```bash
export DATA_ROOT=/path/to/training/data
export TEACHER_PROB=results/teacher_labels/hybrid_teacher_soft_labels.npz

bash scripts/reproduce/run_student_train.sh
```
Or with explicit configuration:
```bash
python3 scripts/training/train_dhsd_v2.py 

--config       configs/final_dhsd_v2.yaml 

--data_root    $DATA_ROOT 

--teacher_prob $TEACHER_PROB
```
Training is deterministic at SEED=42. Expected: best checkpoint at epoch 3, in-distribution validation macro AUROC ~1.0, cross-generator validation macro AUROC ~0.82. Training time: ~3.5 hours on one A100, ~6 hours on one RTX 6000 Ada.

4. Regenerate teacher soft labels (full pipeline, requires 5 VLMs)

This step is computationally expensive (~100 sec/image across 5 VLMs):
```bash
bash scripts/reproduce/run_teacher_features.sh
```
See DATA.md for the dataset structure required.

Repository layout

```text
configs/
final_dhsd_v2.yaml           Hyperparameters for the final student model

results/
student/
best.pt                    Trained student checkpoint (~2.8 MB, via Git LFS)
final_results.json         Per-generator AUROC and AP
provenance.json            Training environment and seed
predictions/
dhsd_v2_crossgen_test_predictions.npz       Student predictions on 13 generators
cnnspot_crossgen_predictions.npz            Baseline predictions for bootstrap CIs
npr_crossgen_predictions.npz
univfd_crossgen_predictions.npz
baselines/
cnnspot_best.pt, npr_best.pt                Baseline checkpoints (univfd
excluded due to size; available
separately at de-anonymization)
crossgen_results.json                     Per-baseline evaluation summaries
teacher_labels/
teacher_soft_labels_percue_5vlm.npz         5-VLM teacher MLP soft labels
teacher_soft_labels_percue_4vlm.npz         4-VLM ablation labels (Phi-4 excluded)
hybrid_teacher_soft_labels.npz              Labels used for student training
figures/
fig.pdf / fig_*.png                       Paper figures
gan_collage.pdf / gan_collage.png           Dataset overview (Figure 3)
diffusion/
{internvl,qwen3vl,glm4v,pixtral,phi4}_results.json
Raw VLM caption + cue outputs
(~42,760 entries each)

scripts/
training/
train_dhsd_v2.py           Main training script (paper config)
inference/
predict_image.py           Single-image or folder inference
evaluation/
test_cnndetection.py       13-generator CNNDetection evaluation
compare_npr.py             Comparison against NPR (Tan et al. CVPR 2024)
leakage_check.py           Train/test leakage audit
verify_leakage.py
data_prep/
build_teacher_labels.py    Builds 5-feature teacher labels
final_analysis.py          Builds 22-d MLP teacher labels (paper config)
rescore_per_cue.py         Per-cue CLIP scoring
build_splits.py            Train/val/test split construction
dump_student_predictions.py
dump_teacher_predictions.py
baselines/
train_baselines.py         Trains CNNSpot, NPR, UniFD on matched MJ+COCO
figures/
make_all_figures.py        Regenerates the 11 main paper figures
make_gan_collage.py        Regenerates Figure 3
reproduce/
run_student_train.sh
run_student_eval.sh
run_teacher_features.sh
bootstrap_cis.py             Reproduces Table 3 bootstrap CIs

DATA.md                        Dataset structure and download instructions
MODEL_CARD.md                  Model card: inputs, preprocessing, limitations
requirements.txt
```
Other training scripts in scripts/training/ (train_adaptive.py, train_combined.py, etc.) are archival ablation experiments not used in the final paper. The single script used for all reported student results is train_dhsd_v2.py.

Cross-generator results

Per-generator AUROC on CNNDetection (13 held-out generators), trained on matched Midjourney + COCO source pool:

Generator

CNNSpot

NPR

UniFD

DHSD (ours)

StarGAN

0.799

0.961

0.921

0.998

CycleGAN

0.418

0.298

0.780

0.978

ProGAN

0.534

0.621

0.715

0.963

GauGAN

0.426

0.410

0.791

0.953

BigGAN

0.481

0.623

0.798

0.945

CRN

0.379

0.409

0.393

0.909

IMLE

0.554

0.426

0.532

0.857

WhichFaceIsReal

0.363

0.510

0.573

0.831

DeepFake

0.453

0.720

0.677

0.725

StyleGAN

0.511

0.581

0.548

0.695

SeeingDark

0.261

0.384

0.648

0.667

StyleGAN2

0.505

0.577

0.495

0.654

SAN

0.548

0.482

0.483

0.600

Macro

0.479

0.539

0.643

0.829

All methods use the same Midjourney + COCO source pool and the same held-out CNNDetection test split. CNNSpot and NPR drop below chance on multiple generators because their pixel-fingerprint signals do not transfer from diffusion training to GAN test data. UniFD partially generalises through its CLIP backbone but lacks explicit semantic discrepancy supervision. DHSD remains above chance on all 13 generators.

For comparison against NPR's reported numbers on its native ProGAN-trained setup, see scripts/evaluation/compare_npr.py. NPR outperforms DHSD when trained and tested both on GAN data, as expected; DHSD's contribution is reaching usable cross-domain performance without GAN training data.

Leakage verification

All evaluation generators in CNNDetection are held out during training. There is no score blending at inference. To run the full audit:

```bash
python3 scripts/evaluation/leakage_check.py
```
Expected: 17/17 checks pass. The audit verifies that no test images appear in the training pool, no per-generator validation samples appear in the test split, and that the cross-generator validation split used for early stopping is disjoint from the held-out test split.

Provenance record: results/student/provenance.json.

Model weights via Git LFS

```bash
git lfs install
git lfs pull
ls -lh results/student/best.pt   # ~2.8 MB
```
The student checkpoint contains the trainable adaptor and classifier weights only (~705K parameters). The frozen CLIP ViT-L/14 backbone is loaded from open-clip-torch at inference time.

See MODEL_CARD.md for input preprocessing, expected output range, and known limitations (single training source, evaluation restricted to CNNDetection 2020).

Reproducibility

Seed. All experiments use SEED=42. Three independent training runs at this seed produced identical cross-generator AUROC to four decimal places.

Hardware. Training and evaluation on a single NVIDIA RTX 6000 Ada (48 GB).

Software. Python 3.10, PyTorch 2.7.0, CUDA 12.6, transformers 5.4.0. See requirements.txt for the full pin list.

Bootstrap CIs. Reported intervals use 1,000 stratified resamples within each of 13 generators, with macro re-aggregated per resample. Reproducible from scripts/bootstrap_cis.py.

Citation

```bibtex
@inproceedings{halder2026dhsd,
title     = {Dual-Hypothesis Vision-Language Reasoning for AI-Generated Image Detection},
author    = {Anonymous},
booktitle = {Advances in Neural Information Processing Systems},
year      = {2026},
note      = {Under review}
}
```

License

Code: MIT License (see LICENSE).

Model weights: MIT License with attribution.

Dataset access subject to original dataset licenses (CNNDetection, WildFake, MS-COCO, Midjourney Terms of Service). See DATA.md.

Acknowledgements

Frozen CLIP backbone from open-clip-torch (OpenAI ViT-L/14, MIT License). Vision-language models used for offline teacher generation: InternVL2.5-8B-MPO (MIT), Qwen2.5-VL-7B-Instruct (Apache 2.0), GLM-4V-9B (GLM-4 License), Pixtral-12B (Apache 2.0), Phi-4-Multimodal-Instruct (MIT). Datasets: MS-COCO (CC BY 4.0 annotations), CNNDetection (Wang et al. CVPR 2020), WildFake (Hong et al. AAAI 2025).
