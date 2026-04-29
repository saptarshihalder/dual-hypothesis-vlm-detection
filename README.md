# Dual-Hypothesis Semantic Fake Image Detection

We trained a small AI model to detect fake images. It learns from real and AI-generated photos, and can spot fakes from generators it has never seen before.

**Result: 0.8288 AUROC across 13 unseen generators (CNNDetection benchmark)**

---

## What this does

Most fake image detectors are trained on one type of fake (e.g. ProGAN faces) and fail on others. Our model is trained only on Midjourney images but still generalises to BigGAN, CycleGAN, StyleGAN, and 10 other generators it was never shown.

It works by using CLIP — a large vision-language model — as a backbone. CLIP already understands what real photos look like from its training on billions of internet images. We attach a small detection head on top and train it to say "real" or "fake".

---

## What you need

- A machine with a GPU (tested on A100)
- Python 3.10+
- The CNNDetection dataset for evaluation
- Midjourney + COCO images for training

Install dependencies:


---

## Comparison with NPR (CVPR 2024)

NPR (Tan et al., CVPR 2024) is the state-of-the-art artifact-based detector trained on ProGAN.

| | GAN test (mean AP%) | Midjourney test (AP%) |
|---|---|---|
| **Student (ours, MJ-trained)** | 84.1 | ~100 |
| **NPR (ProGAN-trained)** | 96.1 | 81.9 |

NPR wins on GAN generators because it trains on GAN data.
Our student wins on Midjourney because it trains on Midjourney data.
Neither generalises fully — training domain is the bottleneck.

The key finding: our model achieves 84.1% AP on 8 GAN generators **without ever seeing GAN training data**, using only semantic distillation from CLIP.

NPR numbers from Table 1 and Table 5 of [arxiv:2312.10461](https://arxiv.org/abs/2312.10461).
Full comparison: `scripts/evaluation/compare_npr.py`
