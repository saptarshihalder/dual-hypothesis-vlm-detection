#!/usr/bin/env python3
"""Generate a labeled collage showing real COCO + all GAN-family fakes from CNNDetection."""
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

random.seed(123)

# All GAN-family generators in CNNDetection (7 total)
GENERATORS = ["progan", "biggan", "cyclegan", "gaugan", "stargan", "stylegan", "stylegan2"]
DATASET_ROOT = "/NAS_DISK/Saptarshi_data/dataset/cnndetection_test"
COCO_ROOT = "/NAS_DISK/Saptarshi_data/dataset/real/coco"
OUT_PATH = "/NAS_DISK/Saptarshi_data/dhsd_figures_final/gan_collage.png"

def find_fake_images(generator, n=2):
    base = Path(DATASET_ROOT) / generator
    if not base.exists():
        print(f"  Warning: {base} not found")
        return []
    candidates = list(base.rglob("*.png")) + list(base.rglob("*.jpg")) + list(base.rglob("*.jpeg"))
    fake_candidates = [p for p in candidates if "1_fake" in str(p) or "fake" in str(p).lower()]
    if fake_candidates:
        candidates = fake_candidates
    if not candidates:
        print(f"  Warning: no images in {base}")
        return []
    return random.sample(candidates, min(n, len(candidates)))

def find_real_images(n=2):
    base = Path(COCO_ROOT)
    candidates = list(base.rglob("*.jpg")) + list(base.rglob("*.png"))
    if not candidates:
        return []
    return random.sample(candidates, min(n, len(candidates)))

def load_and_crop_square(path, size=224):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
    return img

# Build figure: 8 columns × 2 rows = 16 images
n_cols = 1 + len(GENERATORS)  # 1 for COCO + 7 GANs = 8
fig, axes = plt.subplots(2, n_cols, figsize=(2.0 * n_cols + 1, 5.5))
fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.02, wspace=0.05, hspace=0.18)

# Column 0: COCO real
print("Loading COCO real images...")
real_imgs = find_real_images(2)
for row, p in enumerate(real_imgs):
    img = load_and_crop_square(p)
    axes[row, 0].imshow(img)
    axes[row, 0].axis("off")
    if row == 0:
        axes[row, 0].set_title("COCO\n(real)", fontsize=11, fontweight="bold", color="#1d9e75")

# Columns 1-7: Generators
for col, gen in enumerate(GENERATORS, start=1):
    print(f"Loading {gen}...")
    imgs = find_fake_images(gen, 2)
    for row in range(2):
        if row < len(imgs):
            img = load_and_crop_square(imgs[row])
            axes[row, col].imshow(img)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(f"{gen}\n(fake)", fontsize=11, fontweight="bold", color="#b85450")

# Visual separator between real and fake
sep_x = 1.0 / n_cols + 0.005
fig.add_artist(plt.Line2D([sep_x, sep_x], [0.07, 0.88], transform=fig.transFigure,
                          color="#cccccc", linewidth=1.5, linestyle="--"))

# Suptitle and caption

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PATH.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT_PATH}")
print(f"Saved: {OUT_PATH.replace('.png', '.pdf')}")
