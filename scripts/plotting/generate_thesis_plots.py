#!/usr/bin/env python3
"""
THESIS PLOTS GENERATOR
Generates all plots needed for the thesis from saved results.

Plots produced:
1. Training curves (loss components over epochs)
2. Test AUROC over epochs (per generator)
3. Final AUROC bar chart (per generator)
4. ROC curves (per generator on one plot)
5. Confusion matrices (per generator)
6. Ablation comparison (soft-only vs semantic vs adaptive)
7. Benchmark comparison (ours vs literature)
8. Cross-generator transfer heatmap
9. Training vs test accuracy (overfitting check)
10. Score distribution per generator
11. Threshold sweep analysis
12. Probability distributions (real vs fake)
"""

import os, json, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Headless
from pathlib import Path

# Style config — thesis-quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Consistent color scheme
COLORS = {
    'Midjourney': '#4E79A7',  # blue
    'starGAN':    '#59A14F',  # green
    'styleGAN':   '#F28E2B',  # orange
    'BigGAN':     '#E15759',  # red
    'ProGAN':     '#B07AA1',  # purple
    'teacher':    '#76B7B2',  # teal
    'student':    '#FF9DA7',  # pink
}

OUTPUT_BASE = "/NAS_DISK/Saptarshi_data/pipeline_output"
PLOTS_DIR = os.path.join(OUTPUT_BASE, "thesis_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(fig, name, dpi=300):
    """Save both PDF (for LaTeX) and PNG (for preview)."""
    pdf = os.path.join(PLOTS_DIR, f"{name}.pdf")
    png = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(pdf, bbox_inches='tight', dpi=dpi)
    fig.savefig(png, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {name}.pdf + .png")


def load_history(run_name):
    """Load results_history.json from a run."""
    path = os.path.join(OUTPUT_BASE, run_name, "results_history.json")
    if not os.path.exists(path):
        print(f"  Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_final(run_name):
    """Load final_results.json from a run."""
    path = os.path.join(OUTPUT_BASE, run_name, "final_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════
# PLOT 1: Training loss curves
# ════════════════════════════════════════════
def plot_training_curves(history, run_name, title_suffix=""):
    """Plot loss components over epochs."""
    if not history:
        return

    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total + components
    if "loss" in history[0]:
        axes[0].plot(epochs, [r["loss"] for r in history], label="Total loss",
                     color='black', linewidth=2)
    if "ce" in history[0]:
        axes[0].plot(epochs, [r["ce"] for r in history], label="CE (hard labels)",
                     color=COLORS['starGAN'], linestyle='--')
    if "kl" in history[0]:
        axes[0].plot(epochs, [r["kl"] for r in history], label="KL (teacher soft)",
                     color=COLORS['BigGAN'], linestyle='--')
    if "con" in history[0] or "contrastive" in history[0]:
        key = "contrastive" if "contrastive" in history[0] else "con"
        axes[0].plot(epochs, [r[key] for r in history], label="Contrastive (prototypes)",
                     color=COLORS['Midjourney'], linestyle='--')
    if "w_contrastive" in history[0]:
        axes[0].plot(epochs, [r["w_contrastive"] for r in history], label="Weighted contrastive",
                     color=COLORS['Midjourney'], linestyle='--')

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Training loss components{title_suffix}")
    axes[0].legend(loc='best')

    # Training accuracy
    if "train_acc" in history[0]:
        axes[1].plot(epochs, [r["train_acc"] for r in history],
                     label="Train accuracy", color='black')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Training accuracy{title_suffix}")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    savefig(fig, f"01_training_curves_{run_name}")


# ════════════════════════════════════════════
# PLOT 2: Test AUROC over epochs
# ════════════════════════════════════════════
def plot_test_auroc_curves(history, run_name):
    """Plot test AUROC per generator across epochs."""
    if not history:
        return

    generators = ["Midjourney", "starGAN", "styleGAN", "BigGAN", "ProGAN"]
    epochs = [r["epoch"] for r in history]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for gen in generators:
        key = f"{gen}_auc"
        if key not in history[0]:
            continue
        values = [r.get(key, np.nan) for r in history]
        style = '-' if gen != "ProGAN" else '-'
        marker = 'o' if gen == "ProGAN" else None
        lw = 2.5 if gen == "ProGAN" else 1.5
        ax.plot(epochs, values, label=f"{gen}{'  (held out)' if gen=='ProGAN' else ''}",
                color=COLORS[gen], linestyle=style, marker=marker,
                markersize=4, linewidth=lw, markevery=max(1, len(epochs)//20))

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random chance')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_title("Cross-generator AUROC during training")
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, f"02_test_auroc_curves_{run_name}")


# ════════════════════════════════════════════
# PLOT 3: Final AUROC bar chart
# ════════════════════════════════════════════
def plot_final_auroc(final_results, run_name):
    """Bar chart of final AUROC per generator."""
    if not final_results or "results" not in final_results:
        return

    results = final_results["results"]
    generators = list(results.keys())
    aucs = [results[g]["auroc"] for g in generators]
    colors = [COLORS.get(g, '#888') for g in generators]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(generators, aucs, color=colors, edgecolor='black', linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Highlight ProGAN
    for i, g in enumerate(generators):
        if g == "ProGAN":
            bars[i].set_hatch('///')
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1.5)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, label='0.9 threshold')
    ax.set_ylabel("AUROC")
    ax.set_title("Final AUROC by generator (// = never seen)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')

    plt.tight_layout()
    savefig(fig, f"03_final_auroc_{run_name}")


# ════════════════════════════════════════════
# PLOT 4: ROC curves
# ════════════════════════════════════════════
def plot_roc_curves(run_name):
    """Generate ROC curves for each generator."""
    import torch
    from sklearn.metrics import roc_curve, auc as sk_auc
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
    from PIL import Image
    import torch.nn as nn

    model_path = os.path.join(OUTPUT_BASE, run_name, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"  No model: {model_path}")
        return

    print(f"  Loading CLIP + model from {run_name}...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir="/NAS_DISK/Saptarshi_data/hf_cache")
    clip_model = clip_model.cuda().eval()

    class SemanticStudent(nn.Module):
        def __init__(self, clip_model, embed_dim=768):
            super().__init__()
            self.image_encoder = clip_model.visual
            self.projector = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.BatchNorm1d(512), nn.GELU(),
                nn.Dropout(0.3), nn.Linear(512, embed_dim))
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 2))
        def forward(self, x):
            with torch.no_grad():
                f = self.image_encoder(x).float()
            p = F.normalize(self.projector(f), dim=-1)
            return p, self.classifier(self.projector(f))

    student = SemanticStudent(clip_model).cuda()
    ckpt = torch.load(model_path, weights_only=True)
    student.projector.load_state_dict(ckpt["projector"])
    student.classifier.load_state_dict(ckpt["classifier"])
    student.eval()

    class SimpleDataset(Dataset):
        def __init__(self, paths, label, preprocess):
            self.paths, self.label, self.preprocess = paths, label, preprocess
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.preprocess(img), self.label
            except:
                return torch.zeros(3, 224, 224), self.label

    def get_paths(d):
        if not os.path.isdir(d): return []
        return sorted([str(p) for p in Path(d).rglob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])

    def get_progan_paths(base):
        real, fake = [], []
        for root, _, files in os.walk(base):
            folder = os.path.basename(root)
            imgs = sorted([os.path.join(root, f) for f in files
                           if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])
            if folder == "0_real": real.extend(imgs)
            elif folder == "1_fake": fake.extend(imgs)
        return real, fake

    np.random.seed(42)
    coco = get_paths("/NAS_DISK/Saptarshi_data/dataset/real/coco")
    np.random.shuffle(coco)
    coco_te = coco[5000:6000]
    mj = get_paths("/NAS_DISK/Saptarshi_data/dataset/fake/midjourney")
    np.random.shuffle(mj)
    mj_te = mj[7376:8376]

    test_sets = {"Midjourney": (mj_te, coco_te)}
    for name in ["starGAN", "styleGAN", "BigGAN"]:
        paths = get_paths(f"/NAS_DISK/Saptarshi_data/dataset/fake/gan_test/{name}")
        test_sets[name] = (paths, coco_te[:len(paths)])

    pr_real, pr_fake = get_progan_paths("/NAS_DISK/Saptarshi_data/dataset/cnndetection_test")
    np.random.shuffle(pr_real)
    np.random.shuffle(pr_fake)
    test_sets["ProGAN"] = (pr_fake[:1000], pr_real[:1000])

    # Compute ROC per generator
    fig, ax = plt.subplots(figsize=(7, 7))

    for name, (fake_p, real_p) in test_sets.items():
        loader = DataLoader(
            ConcatDataset([SimpleDataset(real_p, 0, preprocess),
                           SimpleDataset(fake_p, 1, preprocess)]),
            batch_size=64, num_workers=0)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                _, logits = student(imgs.cuda())
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        all_probs, all_labels = np.array(all_probs), np.array(all_labels)

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = sk_auc(fpr, tpr)

        marker = '--' if name == "ProGAN" else '-'
        lw = 2.5 if name == "ProGAN" else 1.5
        ax.plot(fpr, tpr, color=COLORS[name], linestyle=marker, linewidth=lw,
                label=f'{name} (AUC = {roc_auc:.3f})' +
                      ('  [held out]' if name == "ProGAN" else ''))

    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curves across generators')
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    savefig(fig, f"04_roc_curves_{run_name}")


# ════════════════════════════════════════════
# PLOT 5: Confusion matrices
# ════════════════════════════════════════════
def plot_confusion_matrices(final_results, run_name):
    """Plot confusion matrix for each generator."""
    if not final_results or "results" not in final_results:
        return

    results = final_results["results"]
    generators = list(results.keys())
    n = len(generators)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, gen in zip(axes, generators):
        r = results[gen]
        tp = r.get("tp", 0)
        fp = r.get("fp", 0)
        tn = r.get("tn", 0)
        fn = r.get("fn", 0)

        # If not in results, compute from accuracy
        if all(v == 0 for v in [tp, fp, tn, fn]):
            ax.text(0.5, 0.5, "No CM data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gen)
            continue

        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()
        cm_norm = cm / total

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=0.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred real', 'Pred fake'])
        ax.set_yticklabels(['True real', 'True fake'])
        ax.set_title(f'{gen}\n(acc={r.get("acc_opt", r.get("accuracy", 0)):.3f})')

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = cm_norm[i, j] * 100
                ax.text(j, i, f'{val}\n({pct:.1f}%)',
                        ha='center', va='center',
                        color='white' if cm_norm[i, j] > 0.25 else 'black',
                        fontweight='bold')

    fig.suptitle(f'Confusion matrices ({run_name})', y=1.02)
    plt.tight_layout()
    savefig(fig, f"05_confusion_matrices_{run_name}")


# ════════════════════════════════════════════
# PLOT 6: Ablation comparison
# ════════════════════════════════════════════
def plot_ablation_comparison():
    """Compare approaches: soft-only, semantic, combined, adaptive."""
    approaches = {
        "Image-only\n(soft label KL)": {"Midjourney": 1.00, "starGAN": 0.9996, "styleGAN": 0.77,
                                        "BigGAN": 0.84, "ProGAN": 0.58},
        "Robust\n(diverse + aug)":       {"Midjourney": 1.00, "starGAN": 0.9994, "styleGAN": 0.77,
                                        "BigGAN": 0.83, "ProGAN": 0.91},
        "Semantic only\n(contrastive)":  {"Midjourney": 1.00, "starGAN": 1.0000, "styleGAN": 0.84,
                                        "BigGAN": 0.94, "ProGAN": 0.785},
        "Combined\n(CE+KL+contrastive)": {"Midjourney": 1.00, "starGAN": 1.0000, "styleGAN": 0.65,
                                        "BigGAN": 0.80, "ProGAN": 0.73},
        "Adaptive\n(weighted contrastive)": {"Midjourney": 1.00, "starGAN": 1.0000, "styleGAN": 0.84,
                                        "BigGAN": 0.94, "ProGAN": 0.782},
    }

    generators = ["Midjourney", "starGAN", "styleGAN", "BigGAN", "ProGAN"]
    x = np.arange(len(generators))
    width = 0.16

    fig, ax = plt.subplots(figsize=(14, 6))

    palette = ['#B8B8B8', '#7AA5D2', '#4E79A7', '#F28E2B', '#59A14F']

    for i, (name, results) in enumerate(approaches.items()):
        values = [results[g] for g in generators]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=palette[i], edgecolor='black', linewidth=0.4)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.008,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(generators)
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation: impact of each distillation component")
    ax.legend(loc='lower left', ncol=2, framealpha=0.95)
    ax.set_ylim(0.4, 1.08)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight ProGAN with shaded box
    ax.axvspan(3.55, 4.45, alpha=0.08, color='red', zorder=0)
    ax.text(4, 0.44, 'Never seen\nduring training', ha='center', fontsize=8,
            color='darkred', style='italic')

    plt.tight_layout()
    savefig(fig, "06_ablation_comparison")


# ════════════════════════════════════════════
# PLOT 7: Benchmark comparison
# ════════════════════════════════════════════
def plot_benchmark_comparison():
    """Compare against prior work on ProGAN."""
    methods = [
        ("ResNet-50 baseline",      0.55, "Pixel", "Trained on ProGAN"),
        ("Wang et al. CVPR'20",     0.93, "Pixel+aug", "Trained on ProGAN"),
        ("Gram-Net",                0.60, "Pixel", "Trained on ProGAN"),
        ("LGrad",                   0.58, "Pixel", "Trained on ProGAN"),
        ("UnivFakeDetect",          0.92, "CLIP", "Trained on ProGAN"),
        ("FatFormer CVPR'24",       0.98, "CLIP+adapter", "Trained on ProGAN"),
        ("Ours (semantic)",         0.785, "VLM reasoning", "NO ProGAN in training"),
        ("Ours (adaptive)",         0.782, "VLM reasoning", "NO ProGAN in training"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    names = [m[0] for m in methods]
    aucs = [m[1] for m in methods]
    types = [m[2] for m in methods]

    colors = []
    for m in methods:
        if "Ours" in m[0]:
            colors.append('#59A14F')
        elif "CLIP" in m[2]:
            colors.append('#4E79A7')
        else:
            colors.append('#B8B8B8')

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor='black', linewidth=0.6)

    for bar, val, note in zip(bars, aucs, [m[3] for m in methods]):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f' {val:.3f}',
                ha='left', va='center', fontweight='bold', fontsize=9)
        ax.text(0.02, bar.get_y() + bar.get_height()/2.,
                note, ha='left', va='center', fontsize=8,
                style='italic', color='white' if width > 0.3 else 'black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('AUROC on ProGAN')
    ax.set_title('Comparison with prior work on CNNDetection benchmark (ProGAN)')
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Legend patches
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#B8B8B8', edgecolor='black', label='Pixel-based'),
        Patch(facecolor='#4E79A7', edgecolor='black', label='CLIP-based'),
        Patch(facecolor='#59A14F', edgecolor='black', label='Ours (VLM reasoning)'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', framealpha=0.95)

    plt.tight_layout()
    savefig(fig, "07_benchmark_comparison")


# ════════════════════════════════════════════
# PLOT 8: Dataset composition
# ════════════════════════════════════════════
def plot_dataset_composition():
    """Show training / test split composition."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Training
    train_data = {
        'COCO real': 1500,
        'Midjourney fake': 1500,
    }
    colors_t = ['#59A14F', '#E15759']
    axes[0].pie(train_data.values(), labels=train_data.keys(), colors=colors_t,
                autopct='%1.0f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title('Training set (3,000 images\nwith VLM semantic prototypes)')

    # Test
    test_data = {
        'COCO': 1000, 'Midjourney': 1000,
        'starGAN': 1000, 'styleGAN': 1000, 'BigGAN': 1000,
        'ProGAN real': 1000, 'ProGAN fake': 1000,
    }
    colors_te = ['#59A14F', '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#B07AA1', '#B07AA1']
    axes[1].pie(test_data.values(), labels=test_data.keys(), colors=colors_te,
                autopct='%1.0f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Test sets (7,000 images,\nProGAN held out)')

    plt.tight_layout()
    savefig(fig, "08_dataset_composition")


# ════════════════════════════════════════════
# PLOT 9: VLM entries breakdown
# ════════════════════════════════════════════
def plot_vlm_data_stats():
    """Bar chart of VLM entries generated."""
    vlms = {
        "InternVL2.5-8B": 42760,
        "Qwen2.5-VL-7B":  42760,
        "GLM-4V-9B":      42759,
        "Pixtral-12B":    42632,
        "Phi-4-MM":       42719,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(vlms.keys())
    counts = list(vlms.values())
    bars = ax.bar(names, counts, color='#4E79A7', edgecolor='black', linewidth=0.8)

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                f'{c:,}', ha='center', va='bottom', fontweight='bold')

    total = sum(counts)
    ax.axhline(y=total/5, color='red', linestyle='--', alpha=0.5, label=f'Mean: {total/5:.0f}')
    ax.set_ylabel("Reasoning entries")
    ax.set_title(f"Teacher VLM reasoning entries (total = {total:,})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    savefig(fig, "09_vlm_data_stats")


# ════════════════════════════════════════════
# PLOT 10: Accuracy matrix heatmap
# ════════════════════════════════════════════
def plot_accuracy_heatmap():
    """Heatmap of approach vs generator."""
    approaches = ["Soft-KD", "Robust", "Semantic", "Combined", "Adaptive"]
    generators = ["Midjourney", "starGAN", "styleGAN", "BigGAN", "ProGAN"]

    data = np.array([
        [1.00, 0.9996, 0.77, 0.84, 0.58],   # Soft-KD
        [1.00, 0.9994, 0.77, 0.83, 0.91],   # Robust
        [1.00, 1.0000, 0.84, 0.94, 0.785],  # Semantic
        [1.00, 1.0000, 0.65, 0.80, 0.73],   # Combined
        [1.00, 1.0000, 0.84, 0.94, 0.782],  # Adaptive
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

    ax.set_xticks(np.arange(len(generators)))
    ax.set_yticks(np.arange(len(approaches)))
    ax.set_xticklabels(generators)
    ax.set_yticklabels(approaches)

    for i in range(len(approaches)):
        for j in range(len(generators)):
            val = data[i, j]
            color = 'white' if val < 0.75 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=color, fontweight='bold', fontsize=10)

    ax.set_title("AUROC heatmap: approach × generator")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUROC')

    # Highlight ProGAN
    ax.axvline(x=3.5, color='red', linewidth=2)
    ax.text(4, -0.8, 'Held out', ha='center', color='darkred', fontweight='bold', fontsize=10)

    plt.tight_layout()
    savefig(fig, "10_accuracy_heatmap")


# ════════════════════════════════════════════
# PLOT 11: CLIPScore VLM quality benchmark
# ════════════════════════════════════════════
def plot_clipscore_benchmark():
    """Reproduce slide 7 benchmark: VLM caption quality vs BLIP-2."""
    vlms = [
        ("Qwen3-VL-30B", 0.272),
        ("Gemma-3-27B", 0.264),
        ("InternVL2.5-8B", 0.263),
        ("Phi-4-MM", 0.258),
        ("BLIP-2 (SOTA)", 0.253),
        ("GLM-4.6V", 0.240),
        ("Pixtral-12B", 0.238),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [v[0] for v in vlms]
    scores = [v[1] for v in vlms]
    colors = ['#59A14F' if s > 0.253 else ('#E15759' if 'BLIP' in n else '#B8B8B8')
              for n, s in vlms]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.6)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2.,
                f'{s:.3f}', va='center', fontweight='bold', fontsize=9)

    ax.axvline(x=0.253, color='red', linestyle='--', alpha=0.6, label='BLIP-2 baseline')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('CLIPScore (image-caption alignment)')
    ax.set_title('VLM caption quality (sanity check: 4/6 beat BLIP-2)')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0.2, 0.29)

    plt.tight_layout()
    savefig(fig, "11_clipscore_benchmark")


# ════════════════════════════════════════════
# PLOT 12: Teacher vs student speed comparison
# ════════════════════════════════════════════
def plot_speed_comparison():
    """Speed comparison: teacher vs student."""
    fig, ax = plt.subplots(figsize=(9, 5))

    components = ["Teacher\n(5 VLMs × 2 prompts\n+ CLIP + GBM)",
                  "Student\n(CLIP + MLP)"]
    times_s = [300, 0.001]  # 5 min = 300s, 1 ms = 0.001s

    bars = ax.bar(components, times_s, color=['#E15759', '#59A14F'],
                  edgecolor='black', linewidth=0.8)
    ax.set_yscale('log')
    ax.set_ylabel('Inference time per image (seconds, log scale)')
    ax.set_title(f'Teacher vs Student inference speed ({300/0.001:.0f}× speedup)')

    for bar, t in zip(bars, times_s):
        label = f'{t*1000:.0f} ms' if t < 1 else f'{t:.0f} s = {t/60:.1f} min'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.5,
                label, ha='center', va='bottom', fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y', which='both')
    plt.tight_layout()
    savefig(fig, "12_speed_comparison")


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
def main():
    print("=" * 70)
    print("THESIS PLOTS GENERATOR")
    print(f"Output: {PLOTS_DIR}")
    print("=" * 70)

    runs = ["semantic_grok", "combined_run", "adaptive_run", "semantic_run", "robust_run"]

    print("\n[1/12] Training curves")
    for run in runs:
        hist = load_history(run)
        if hist:
            plot_training_curves(hist, run, f" ({run})")

    print("\n[2/12] Test AUROC over epochs")
    for run in runs:
        hist = load_history(run)
        if hist:
            plot_test_auroc_curves(hist, run)

    print("\n[3/12] Final AUROC bars")
    for run in runs:
        final = load_final(run)
        if final:
            plot_final_auroc(final, run)

    print("\n[4/12] ROC curves")
    try:
        plot_roc_curves("adaptive_run")
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n[5/12] Confusion matrices")
    for run in runs:
        final = load_final(run)
        if final:
            plot_confusion_matrices(final, run)

    print("\n[6/12] Ablation comparison")
    plot_ablation_comparison()

    print("\n[7/12] Benchmark comparison")
    plot_benchmark_comparison()

    print("\n[8/12] Dataset composition")
    plot_dataset_composition()

    print("\n[9/12] VLM data stats")
    plot_vlm_data_stats()

    print("\n[10/12] Accuracy heatmap")
    plot_accuracy_heatmap()

    print("\n[11/12] CLIPScore benchmark")
    plot_clipscore_benchmark()

    print("\n[12/12] Speed comparison")
    plot_speed_comparison()

    print("\n" + "=" * 70)
    print(f"ALL PLOTS SAVED to: {PLOTS_DIR}")
    print("Both .pdf (for LaTeX) and .png (for preview) generated.")
    print("=" * 70)
    files = sorted(os.listdir(PLOTS_DIR))
    print(f"\nGenerated {len(files)} files:")
    for f in files:
        size = os.path.getsize(os.path.join(PLOTS_DIR, f)) / 1024
        print(f"  {f}  ({size:.1f} KB)")


if __name__ == "__main__":
    main()
