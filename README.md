# Dual-Hypothesis Vision-Language Reasoning for AI-Generated Image Detection

BS Thesis — **Saptarshi Halder (22283)**, IISER Bhopal
Supervisor: Dr. Akshay Aggarwal
Department of Data Science and Engineering

## Idea
For every image, prompt 5 VLMs under two opposing assumptions — *"assume REAL"* and *"assume FAKE"* — and measure the semantic discrepancy between the two responses using CLIP. Large gap ⇒ AI-generated; small gap ⇒ real. Generator-agnostic by construction.

## Pipeline
1. **Dual-prompt** 5 VLMs (InternVL2.5, Qwen2.5-VL, GLM-4V, Pixtral, Phi-4)
2. **CLIP-score** each caption + 3 cues against the image
3. **Discrepancy** = fake_score − real_score, per VLM
4. **Aggregate** mean/std across VLMs
5. **Classify** with a fusion MLP / XGBoost

## Repo layout
```
scripts/
  vlm_runners/     # one runner per VLM (5 models)
  clip_scoring/    # CLIP feature extraction + BLIP-2 sanity check
  benchmarks/      # evaluation scripts
results/
  diffusion/       # 5 VLMs × ~21k images × 2 assumptions ≈ 210k entries
  gan/             # earlier GAN-only run (if present)
docs/              # thesis docs, presentation, methodology update
```

## Datasets
- **WildFake** (Hong et al., AAAI 2025): 10K real + fake images, GAN + Diffusion
- **GenImage** (Zhu et al., NeurIPS 2024): cross-generator benchmark
- COCO + Midjourney Advanced for the diffusion run (~21K pairs)

## Scale of outputs
~213,630 reasoning entries and ~640,890 visual cues across 5 VLMs.

## Hardware
NVIDIA RTX 6000 Ada Generation (48 GB)

## Methodology update
See [docs/Update](docs/Update) for the shift from an explicit reasoning pipeline at inference time to a **teacher → student distillation** framework, where the full VLM ensemble acts as the teacher that supervises a compact image-only student model.

## References
1. Zhu et al., *GenImage*, NeurIPS 2024
2. Hong et al., *WildFake*, AAAI 2025
3. Yu et al., *LVLM-DFD*, 2025
4. Li et al., *FakeVLM*, 2025
5. Jia et al., CVPR Workshop 2024
