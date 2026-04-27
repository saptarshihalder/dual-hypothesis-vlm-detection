# Dual-Hypothesis Semantic Fake Image Detection

We trained a small AI model to detect fake images. It learns from real and AI-generated photos, and can spot fakes from generators it has never seen before.

Result: 0.8288 AUROC across 13 unseen generators (CNNDetection benchmark)

---

## What this does

Most fake image detectors are trained on one type of fake (e.g. ProGAN faces) and fail on others. Our model is trained only on Midjourney images but still generalises to BigGAN, CycleGAN, StyleGAN, and 10 other generators it was never shown.

It works by using CLIP — a large vision-language model — as a backbone. CLIP already understands what real photos look like from its training on billions of internet images. We attach a small detection head on top and train it to say real or fake.

---

## What you need

- A machine with a GPU (tested on A100)
- Python 3.10+
- The CNNDetection dataset for evaluation
- Midjourney + COCO images for training

Install dependencies:

    pip install -r requirements.txt

---

## Step by step

### Step 1 — Get the data

Put your images in this folder structure:

    dataset/
      fake/midjourney/          <- AI-generated images for training
      real/coco/                <- Real photos for training
      cnndetection_test/
        biggan/
          val/0_real/  val/1_fake/
          test/0_real/ test/1_fake/
        crn/ cyclegan/ deepfake/ ...

Then open scripts/training/train_dhsd_v2.py and set DATA_ROOT on line 47 to your dataset path.

### Step 2 — Get teacher probabilities

We use a ProGAN-trained teacher model to give extra training signal. Run this once:

    python3 scripts/evaluation/teacher_progan.py \
        --data_root /your/dataset \
        --out /your/path/teacher_probs.npz

Then set TEACHER_PROB on line 49 of the training script to that path.

### Step 3 — Train

    python3 scripts/training/train_dhsd_v2.py

Training takes about 3-4 hours on one GPU. The best model is saved automatically.
You will see per-generator AUROC scores printed after each epoch.

### Step 4 — Check your results

At the end of training, a final_results.json file is written to the output directory.
It contains per-generator AUROC scores and the overall macro average.

To use our pre-trained model instead of training from scratch:

    python3 scripts/evaluation/test_cnndetection.py \
        --ckpt results/student/best.pt \
        --data_root /your/dataset/cnndetection_test

---

## Results

All 13 generators below were unseen during training.

    stargan          0.9979
    cyclegan         0.9777
    progan           0.9631
    gaugan           0.9525
    biggan           0.9451
    crn              0.9091
    imle             0.8573
    whichfaceisreal  0.8308
    deepfake         0.7249
    stylegan         0.6952
    seeingdark       0.6674
    stylegan2        0.6536
    san              0.5998
    -------------------------
    Macro average    0.8288

---

## Is this cheating?

No. We verified this carefully:

- No test data was used during training
- No scores are blended at inference — only the student head runs
- CLIP zero-shot and teacher signals are used only as training loss terms, not at evaluation
- A full leakage audit is at scripts/evaluation/verify_leakage.py

---

## Files in this repo

    scripts/training/train_dhsd_v2.py              main training script
    scripts/evaluation/test_cnndetection.py        evaluate on CNNDetection
    scripts/evaluation/verify_leakage.py           leakage audit
    scripts/plotting/plot_student_results_cvpr.py  generate paper figures
    results/student/best.pt                        pre-trained model weights
    results/student/final_results.json             per-generator results
    results/student/provenance.json                verification record
