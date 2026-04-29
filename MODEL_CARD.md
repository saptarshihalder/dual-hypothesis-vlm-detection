# Model Card — DHSD v2 Student (best.pt)

## Overview

A 0.71M parameter detection head trained on top of a frozen CLIP ViT-L/14 backbone
to classify images as real or AI-generated.

## File

    results/student/best.pt

    Keys in checkpoint:
      state      -> OrderedDict  (head weights only, not CLIP backbone)
      epoch      -> 3            (best epoch selected by early stopping)

## Input preprocessing

CLIP ViT-L/14 default preprocessing (OpenAI):
  - Resize shortest edge to 224
  - Center crop to 224x224
  - Normalize: mean=[0.48145466, 0.4578275, 0.40821073]
                std=[0.26862954, 0.26130258, 0.27577711]

Do NOT use ImageNet normalization — use open_clip's preprocess transform.

## Output

    sigmoid(logit) in [0, 1]
    > 0.5  → predicted FAKE
    ≤ 0.5  → predicted REAL

## Performance

    In-distribution (Midjourney+COCO test):
      AUROC = 1.0000   Acc = 0.9991

    Cross-generator (CNNDetection, 13 generators, never seen in training):
      AUROC macro = 0.8288
      AP    macro = 83.94%

    Per-generator AUROC:
      stargan         0.9979     cyclegan    0.9777
      progan          0.9631     gaugan      0.9525
      biggan          0.9451     crn         0.9091
      imle            0.8573     whichfaceisreal 0.8308
      deepfake        0.7249     stylegan    0.6952
      seeingdark      0.6674     stylegan2   0.6536
      san             0.5998

## Training setup

    Backbone  : CLIP ViT-L/14 (frozen, OpenAI weights)
    Head      : TIE + classifier (0.71M trainable params)
    Training  : Midjourney V5/V6 (fake) + COCO 2017 (real)
    KD        : Teacher ProGAN (alpha=0.5) + CLIP-ZS (beta=1.5)
    Epochs    : 15 (early stop at epoch 3)
    LR        : 1.5e-4 (OneCycleLR)

## Limitations

- Weak on face-domain GAN generators (StyleGAN2 AUROC=0.65, StyleGAN=0.70)
- Strong on non-face diffusion and GAN generators
- Not tested on heavily compressed or resized images
- Performance may degrade on generators released after Midjourney V6

## Intended use

Research only. Not validated for forensic or legal use.

## Standalone verification

This model produces its predictions using only:
    Image → CLIP ViT-L/14 (frozen) → student head → sigmoid

No score blending at inference. Verified by leakage audit:
    python3 scripts/evaluation/leakage_check.py
