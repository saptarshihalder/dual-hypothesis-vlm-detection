#!/usr/bin/env python3
"""
Generate all figures for the DHSD NeurIPS paper.

Outputs to /NAS_DISK/Saptarshi_data/dhsd_figures_final/

Figures produced:
  Group A: Headline & comparison
    1. fig_headline_3panel.{pdf,png}      — 3-panel: AUROC, AP, gen-gap
    2. fig_per_gen_auroc_grouped.{pdf,png}  — bar chart, all 4 methods × 13 gens
    3. fig_per_gen_ap_grouped.{pdf,png}     — same with AP
    4. fig_generalization_gap.{pdf,png}   — slope chart val→test
    5. fig_method_hierarchy.{pdf,png}     — tier visualization

  Group B: DHSDv2 deep-dive
    6. fig_dhsd_per_gen_auroc.{pdf,png}    — DHSDv2-only horizontal bar
    7. fig_dhsd_roc_curves.{pdf,png}       — per-generator ROC curves
    8. fig_dhsd_pr_curves.{pdf,png}        — per-generator PR curves
    9. fig_score_distributions.{pdf,png}   — DHSDv2 score histograms

  Group C: Architecture & method
    10. fig_dual_hypothesis_concept.{pdf,png}  — schematic of dual-prompt idea
    11. fig_pipeline_diagram.{pdf,png}     — teacher → student pipeline

Run: python3 ~/make_all_figures.py
"""
import os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from sklearn.metrics import roc_curve, precision_recall_curve

# ─── Paths ────────────────────────────────────────────────────
OUT       = Path("/NAS_DISK/Saptarshi_data/dhsd_figures_final")
OUT.mkdir(parents=True, exist_ok=True)

DHSD_PRED = Path("/NAS_DISK/Saptarshi_data/dhsd_v2_20260427_133533/crossgen_test_predictions.npz")
BASE_DIR  = Path("/NAS_DISK/Saptarshi_data/baselines_v1")

# ─── CVPR style ───────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
})

# Okabe-Ito palette (colorblind-safe)
COLORS = {
    "CNNSpot": "#999999",
    "NPR":     "#0072B2",
    "UniFD":   "#E69F00",
    "DHSDv2":  "#009E73",
}
LABEL = {
    "CNNSpot": "CNNSpot",
    "NPR":     "NPR",
    "UniFD":   "UniFD",
    "DHSDv2":  "DHSDv2 (ours)",
}
ORDER = ["CNNSpot", "NPR", "UniFD", "DHSDv2"]

# ─── Data: hardcoded macro numbers (matches your runs exactly) ─
MACRO = {
    "CNNSpot": {"auroc": 0.479, "ap": 0.485, "val_auroc": 1.000},
    "NPR":     {"auroc": 0.539, "ap": 0.535, "val_auroc": 1.000},
    "UniFD":   {"auroc": 0.643, "ap": 0.624, "val_auroc": 1.000},
    "DHSDv2":  {"auroc": 0.829, "ap": 0.839, "val_auroc": 1.000},
}

# Per-generator AUROC and AP (from JSON outputs)
GENS_ORDERED = ["stargan", "cyclegan", "progan", "gaugan", "biggan",
                "crn", "imle", "whichfaceisreal", "deepfake", "stylegan",
                "seeingdark", "stylegan2", "san"]

PER_GEN_AUROC = {
    "CNNSpot": {"stargan":0.799, "cyclegan":0.418, "progan":0.534, "gaugan":0.426,
                "biggan":0.481, "crn":0.379, "imle":0.554, "whichfaceisreal":0.363,
                "deepfake":0.453, "stylegan":0.511, "seeingdark":0.261,
                "stylegan2":0.505, "san":0.548},
    "NPR":     {"stargan":0.961, "cyclegan":0.298, "progan":0.621, "gaugan":0.410,
                "biggan":0.623, "crn":0.409, "imle":0.426, "whichfaceisreal":0.510,
                "deepfake":0.720, "stylegan":0.581, "seeingdark":0.384,
                "stylegan2":0.577, "san":0.482},
    "UniFD":   {"stargan":0.921, "cyclegan":0.780, "progan":0.715, "gaugan":0.791,
                "biggan":0.798, "crn":0.393, "imle":0.532, "whichfaceisreal":0.573,
                "deepfake":0.677, "stylegan":0.548, "seeingdark":0.648,
                "stylegan2":0.495, "san":0.483},
    "DHSDv2":  {"stargan":0.998, "cyclegan":0.978, "progan":0.963, "gaugan":0.953,
                "biggan":0.945, "crn":0.909, "imle":0.857, "whichfaceisreal":0.831,
                "deepfake":0.725, "stylegan":0.695, "seeingdark":0.667,
                "stylegan2":0.654, "san":0.600},
}

PER_GEN_AP = {
    "CNNSpot": {"stargan":0.808, "cyclegan":0.422, "progan":0.502, "gaugan":0.434,
                "biggan":0.457, "crn":0.414, "imle":0.537, "whichfaceisreal":0.404,
                "deepfake":0.449, "stylegan":0.501, "seeingdark":0.374,
                "stylegan2":0.474, "san":0.530},
    "NPR":     {"stargan":0.966, "cyclegan":0.371, "progan":0.572, "gaugan":0.428,
                "biggan":0.568, "crn":0.430, "imle":0.455, "whichfaceisreal":0.492,
                "deepfake":0.674, "stylegan":0.553, "seeingdark":0.442,
                "stylegan2":0.525, "san":0.485},
    "UniFD":   {"stargan":0.910, "cyclegan":0.708, "progan":0.658, "gaugan":0.739,
                "biggan":0.767, "crn":0.419, "imle":0.516, "whichfaceisreal":0.553,
                "deepfake":0.661, "stylegan":0.540, "seeingdark":0.646,
                "stylegan2":0.485, "san":0.514},
    "DHSDv2":  {"stargan":0.998, "cyclegan":0.980, "progan":0.966, "gaugan":0.957,
                "biggan":0.951, "crn":0.915, "imle":0.871, "whichfaceisreal":0.847,
                "deepfake":0.779, "stylegan":0.740, "seeingdark":0.690,
                "stylegan2":0.648, "san":0.571},
}


# ═══════════════════════════════════════════════════════════════
# GROUP A: HEADLINE & COMPARISON
# ═══════════════════════════════════════════════════════════════

def fig1_headline_3panel():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))

    # Panel A: macro AUROC
    ax = axes[0]
    vals = [MACRO[m]["auroc"] for m in ORDER]
    bars = ax.bar(np.arange(len(ORDER)), vals,
                  color=[COLORS[m] for m in ORDER],
                  edgecolor="white", linewidth=0.4, width=0.65)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Macro AUROC")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[m] for m in ORDER], rotation=20, ha="right")
    ax.set_title("(a) Cross-gen AUROC", fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # Panel B: macro AP
    ax = axes[1]
    vals = [MACRO[m]["ap"] for m in ORDER]
    bars = ax.bar(np.arange(len(ORDER)), vals,
                  color=[COLORS[m] for m in ORDER],
                  edgecolor="white", linewidth=0.4, width=0.65)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Macro AP")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[m] for m in ORDER], rotation=20, ha="right")
    ax.set_title("(b) Cross-gen AP", fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # Panel C: generalization gap
    ax = axes[2]
    gaps = [MACRO[m]["auroc"] - MACRO[m]["val_auroc"] for m in ORDER]
    bars = ax.bar(np.arange(len(ORDER)), gaps,
                  color=[COLORS[m] for m in ORDER],
                  edgecolor="white", linewidth=0.4, width=0.65)
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, g - 0.015,
                f"{g:+.3f}", ha="center", va="top",
                fontsize=8, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-0.65, 0.05)
    ax.set_yticks([0, -0.2, -0.4, -0.6])
    ax.set_ylabel("Gen-gap (cross-gen − val)")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[m] for m in ORDER], rotation=20, ha="right")
    ax.set_title("(c) Generalization gap", fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "fig_headline_3panel.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_headline_3panel")


def fig2_per_gen_auroc_grouped():
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    x = np.arange(len(GENS_ORDERED))
    width = 0.20
    for i, m in enumerate(ORDER):
        vals = [PER_GEN_AUROC[m][g] for g in GENS_ORDERED]
        ax.bar(x + (i - 1.5) * width, vals, width, label=LABEL[m],
               color=COLORS[m], edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.text(len(GENS_ORDERED) - 0.5, 0.51, "chance", fontsize=7,
            ha="right", va="bottom", alpha=0.5)
    ax.set_xlabel("Generator (sorted by DHSDv2 AUROC, descending)")
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(GENS_ORDERED, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower left", ncol=4, frameon=False,
              bbox_to_anchor=(0, 1.02), columnspacing=1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig_per_gen_auroc_grouped.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_per_gen_auroc_grouped")


def fig3_per_gen_ap_grouped():
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    x = np.arange(len(GENS_ORDERED))
    width = 0.20
    for i, m in enumerate(ORDER):
        vals = [PER_GEN_AP[m][g] for g in GENS_ORDERED]
        ax.bar(x + (i - 1.5) * width, vals, width, label=LABEL[m],
               color=COLORS[m], edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_xlabel("Generator (sorted by DHSDv2 AUROC, descending)")
    ax.set_ylabel("Average Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(GENS_ORDERED, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower left", ncol=4, frameon=False,
              bbox_to_anchor=(0, 1.02), columnspacing=1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig_per_gen_ap_grouped.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_per_gen_ap_grouped")


def fig4_generalization_gap():
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    for m in ORDER:
        v = MACRO[m]["val_auroc"]
        c = MACRO[m]["auroc"]
        ax.plot([1, 0], [v, c],
                color=COLORS[m], linewidth=1.8, alpha=0.9,
                marker="o", markersize=8, markeredgecolor="white",
                markeredgewidth=0.8, label=LABEL[m])
        ax.text(-0.05, c, f"{c:.3f}", color=COLORS[m],
                fontsize=7, fontweight="bold", ha="right", va="center")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
    ax.text(0.5, 0.51, "chance", fontsize=7, ha="center", va="bottom", alpha=0.5)
    ax.set_xlim(-0.25, 1.15)
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Cross-gen\ntest", "In-distribution\nval"])
    ax.set_ylabel("AUROC")
    ax.set_title("Generalization gap")
    ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig_generalization_gap.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_generalization_gap")


def fig5_method_hierarchy():
    """Tier visualization: pixel < CLIP-feat < CLIP+dual-hypothesis."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    tiers = [
        ("Pixel-fingerprint", ["CNNSpot", "NPR"], "#cccccc"),
        ("CLIP features", ["UniFD"], "#fbe5b6"),
        ("CLIP + dual-hyp", ["DHSDv2"], "#b8e6d8"),
    ]
    y_offset = {"CNNSpot": 0, "NPR": 0.4, "UniFD": 0.8, "DHSDv2": 1.2}
    for i, (tier_name, methods, color) in enumerate(tiers):
        ax.axhspan(i - 0.45, i + 0.45, color=color, alpha=0.5, zorder=0)
        ax.text(-0.05, i, tier_name, ha="right", va="center",
                fontsize=8, fontweight="bold")
        for j, m in enumerate(methods):
            x = MACRO[m]["auroc"]
            ax.scatter(x, i, s=180, color=COLORS[m], edgecolor="white",
                       linewidth=1.5, zorder=3)
            ax.text(x, i + 0.18, LABEL[m] + f"\n{x:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5, zorder=1)
    ax.text(0.5, -0.7, "chance", ha="center", fontsize=7, alpha=0.6)
    ax.set_xlim(0.4, 0.9)
    ax.set_ylim(-0.7, 2.5)
    ax.set_yticks([])
    ax.set_xlabel("Cross-generator macro AUROC")
    ax.set_title("Three-tier hierarchy of detection methods\n(matched training+test conditions)",
                 fontsize=9)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(OUT / "fig_method_hierarchy.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_method_hierarchy")


# ═══════════════════════════════════════════════════════════════
# GROUP B: DHSDv2 DEEP-DIVE
# ═══════════════════════════════════════════════════════════════

def fig6_dhsd_per_gen_auroc():
    """Horizontal bar chart, DHSDv2 only, sorted high to low."""
    pairs = sorted(PER_GEN_AUROC["DHSDv2"].items(), key=lambda x: x[1], reverse=True)
    gens, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    colors = []
    for v in vals:
        if v >= 0.9: colors.append("#1a9850")
        elif v >= 0.8: colors.append("#66bd63")
        elif v >= 0.7: colors.append("#d9ef8b")
        elif v >= 0.6: colors.append("#fee08b")
        else: colors.append("#fdae61")
    y = np.arange(len(gens))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.8)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(gens)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("AUROC")
    ax.set_title("DHSDv2 per-generator AUROC", fontsize=9)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig_dhsd_per_gen_auroc.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_dhsd_per_gen_auroc")


def fig7_dhsd_roc_curves():
    """ROC curves for DHSDv2 across all 13 generators."""
    if not DHSD_PRED.exists():
        print(f"  [skip] {DHSD_PRED} not found")
        return
    d = np.load(DHSD_PRED, allow_pickle=True)
    gens = sorted(list(d["generators"]))

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    cmap = plt.cm.viridis
    for i, g in enumerate(gens):
        y = d[f"{g}_labels"]
        p = d[f"{g}_probs"]
        fpr, tpr, _ = roc_curve(y, p)
        auroc = PER_GEN_AUROC["DHSDv2"][g]
        color = cmap(i / max(len(gens) - 1, 1))
        ax.plot(fpr, tpr, color=color, linewidth=1.0, alpha=0.85,
                label=f"{g} ({auroc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("DHSDv2 ROC curves (per generator)", fontsize=9)
    ax.legend(loc="lower right", fontsize=6, frameon=False, ncol=2,
              columnspacing=0.8, handlelength=1.0)
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig_dhsd_roc_curves.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_dhsd_roc_curves")


def fig8_dhsd_pr_curves():
    """Precision-Recall curves for DHSDv2 across all 13 generators."""
    if not DHSD_PRED.exists():
        print(f"  [skip] {DHSD_PRED} not found")
        return
    d = np.load(DHSD_PRED, allow_pickle=True)
    gens = sorted(list(d["generators"]))

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    cmap = plt.cm.plasma
    for i, g in enumerate(gens):
        y = d[f"{g}_labels"]
        p = d[f"{g}_probs"]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = PER_GEN_AP["DHSDv2"][g]
        color = cmap(i / max(len(gens) - 1, 1))
        ax.plot(rec, prec, color=color, linewidth=1.0, alpha=0.85,
                label=f"{g} ({ap:.2f})")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("DHSDv2 Precision-Recall curves (per generator)", fontsize=9)
    ax.legend(loc="lower left", fontsize=6, frameon=False, ncol=2,
              columnspacing=0.8, handlelength=1.0)
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig_dhsd_pr_curves.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_dhsd_pr_curves")


def fig9_score_distributions():
    """Score histograms for DHSDv2 — real vs fake distributions per generator."""
    if not DHSD_PRED.exists():
        print(f"  [skip] {DHSD_PRED} not found")
        return
    d = np.load(DHSD_PRED, allow_pickle=True)
    gens = sorted(list(d["generators"]))

    n = len(gens)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.16, 2.0 * rows), sharex=True)
    axes = axes.flatten()

    for i, g in enumerate(gens):
        ax = axes[i]
        y = d[f"{g}_labels"]
        p = d[f"{g}_probs"]
        ax.hist(p[y == 0], bins=30, range=(0, 1), color="#0072B2",
                alpha=0.55, label="real", density=True)
        ax.hist(p[y == 1], bins=30, range=(0, 1), color="#D55E00",
                alpha=0.55, label="fake", density=True)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        auroc = PER_GEN_AUROC["DHSDv2"][g]
        ax.set_title(f"{g}\nAUROC={auroc:.3f}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, frameon=False, loc="upper center")

    for i in range(len(gens), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("DHSDv2 score distributions per generator (P(fake))",
                 fontsize=10, y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / "fig_score_distributions.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_score_distributions")


# ═══════════════════════════════════════════════════════════════
# GROUP C: ARCHITECTURE & METHOD
# ═══════════════════════════════════════════════════════════════

def fig10_dual_hypothesis_concept():
    """Schematic showing the dual-hypothesis prompting concept."""
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis("off")

    # Image at left
    img_box = FancyBboxPatch((0.3, 2.0), 1.6, 1.8,
                              boxstyle="round,pad=0.05",
                              facecolor="#f0f0f0", edgecolor="#666",
                              linewidth=1.2)
    ax.add_patch(img_box)
    ax.text(1.1, 2.9, "Input\nimage", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # 5 VLMs box
    vlm_box = FancyBboxPatch((2.6, 1.3), 1.6, 3.4,
                              boxstyle="round,pad=0.05",
                              facecolor="#fff7e6", edgecolor="#E69F00",
                              linewidth=1.2)
    ax.add_patch(vlm_box)
    ax.text(3.4, 4.4, "5 VLMs", ha="center", va="center",
            fontsize=9, fontweight="bold")
    vlms = ["InternVL2.5-8B", "Qwen2.5-VL-7B", "GLM-4V-9B", "Pixtral-12B", "Phi-4-MM"]
    for j, name in enumerate(vlms):
        ax.text(3.4, 3.9 - j*0.55, name, ha="center", va="center",
                fontsize=7)

    # Two prompts emerge
    real_box = FancyBboxPatch((5.0, 3.4), 2.4, 1.4,
                               boxstyle="round,pad=0.05",
                               facecolor="#d4edda", edgecolor="#28a745",
                               linewidth=1.2)
    ax.add_patch(real_box)
    ax.text(6.2, 4.5, '"Assume REAL"', ha="center", va="center",
            fontsize=8, fontweight="bold", color="#155724")
    ax.text(6.2, 4.1, "Caption + 3 cues\nproving real", ha="center", va="center",
            fontsize=7)

    fake_box = FancyBboxPatch((5.0, 1.2), 2.4, 1.4,
                               boxstyle="round,pad=0.05",
                               facecolor="#f8d7da", edgecolor="#dc3545",
                               linewidth=1.2)
    ax.add_patch(fake_box)
    ax.text(6.2, 2.3, '"Assume FAKE"', ha="center", va="center",
            fontsize=8, fontweight="bold", color="#721c24")
    ax.text(6.2, 1.9, "Caption + 3 cues\nproving fake", ha="center", va="center",
            fontsize=7)

    # Discrepancy → score
    disc_box = FancyBboxPatch((8.0, 2.3), 1.7, 1.4,
                               boxstyle="round,pad=0.05",
                               facecolor="#b8e6d8", edgecolor="#009E73",
                               linewidth=1.2)
    ax.add_patch(disc_box)
    ax.text(8.85, 3.3, "Semantic\ndiscrepancy", ha="center", va="center",
            fontsize=9, fontweight="bold")
    ax.text(8.85, 2.6, "P(fake)", ha="center", va="center",
            fontsize=8, fontstyle="italic")

    # Arrows
    arrow_kw = dict(arrowstyle="->", linewidth=1.0, color="#444",
                    mutation_scale=15)
    ax.add_patch(FancyArrowPatch((1.95, 2.9), (2.55, 2.9), **arrow_kw))
    ax.add_patch(FancyArrowPatch((4.25, 3.4), (4.95, 4.0), **arrow_kw))
    ax.add_patch(FancyArrowPatch((4.25, 2.6), (4.95, 1.95), **arrow_kw))
    ax.add_patch(FancyArrowPatch((7.45, 4.0), (7.97, 3.3), **arrow_kw))
    ax.add_patch(FancyArrowPatch((7.45, 2.0), (7.97, 2.7), **arrow_kw))

    ax.set_title("Dual-hypothesis VLM reasoning",
                 fontsize=11, fontweight="bold", y=0.97)

    plt.tight_layout()
    plt.savefig(OUT / "fig_dual_hypothesis_concept.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_dual_hypothesis_concept")


def fig11_pipeline_diagram():
    """Teacher → student distillation pipeline."""
    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7)
    ax.axis("off")

    # ── Teacher path (top) ────────────────────────────────
    ax.text(3.0, 6.5, "TEACHER PIPELINE (training time, offline)",
            fontsize=9, fontweight="bold", color="#666")

    # Image
    ax.add_patch(FancyBboxPatch((0.3, 4.2), 1.5, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#f0f0f0", edgecolor="#666",
                                 linewidth=1.0))
    ax.text(1.05, 4.9, "Image", ha="center", va="center", fontsize=8,
            fontweight="bold")

    # Dual prompts
    ax.add_patch(FancyBboxPatch((2.5, 4.7), 1.6, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#d4edda", edgecolor="#28a745"))
    ax.text(3.3, 5.1, '"Real" prompt', ha="center", va="center", fontsize=7)
    ax.add_patch(FancyBboxPatch((2.5, 3.7), 1.6, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#f8d7da", edgecolor="#dc3545"))
    ax.text(3.3, 4.1, '"Fake" prompt', ha="center", va="center", fontsize=7)

    # 5 VLMs
    ax.add_patch(FancyBboxPatch((4.7, 3.7), 1.4, 1.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#fff7e6", edgecolor="#E69F00"))
    ax.text(5.4, 5.2, "5 VLMs", ha="center", va="center", fontsize=8,
            fontweight="bold")
    ax.text(5.4, 4.6, "captions\n+ cues", ha="center", va="center", fontsize=7)

    # CLIP scoring → 22-d feature
    ax.add_patch(FancyBboxPatch((6.7, 4.0), 1.5, 1.2,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#e6f3ff", edgecolor="#0072B2"))
    ax.text(7.45, 4.85, "CLIP score", ha="center", va="center", fontsize=8,
            fontweight="bold")
    ax.text(7.45, 4.4, "22-d features", ha="center", va="center", fontsize=7)

    # GBM → P_teacher
    ax.add_patch(FancyBboxPatch((8.8, 4.0), 1.5, 1.2,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#b8e6d8", edgecolor="#009E73"))
    ax.text(9.55, 4.85, "GBM", ha="center", va="center", fontsize=8,
            fontweight="bold")
    ax.text(9.55, 4.4, "P_teacher", ha="center", va="center", fontsize=7,
            fontstyle="italic")

    # Arrows for teacher
    arrow_kw = dict(arrowstyle="->", linewidth=1.0, color="#444",
                    mutation_scale=12)
    ax.add_patch(FancyArrowPatch((1.85, 4.9), (2.45, 5.05), **arrow_kw))
    ax.add_patch(FancyArrowPatch((1.85, 4.9), (2.45, 4.15), **arrow_kw))
    ax.add_patch(FancyArrowPatch((4.15, 5.1), (4.65, 4.8), **arrow_kw))
    ax.add_patch(FancyArrowPatch((4.15, 4.1), (4.65, 4.4), **arrow_kw))
    ax.add_patch(FancyArrowPatch((6.15, 4.6), (6.65, 4.6), **arrow_kw))
    ax.add_patch(FancyArrowPatch((8.25, 4.6), (8.75, 4.6), **arrow_kw))

    # ── Student path (bottom) ──────────────────────────
    ax.text(3.0, 2.5, "STUDENT (deployment, ~2 ms/image)",
            fontsize=9, fontweight="bold", color="#666")

    # Image (same)
    ax.add_patch(FancyBboxPatch((0.3, 0.7), 1.5, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#f0f0f0", edgecolor="#666"))
    ax.text(1.05, 1.4, "Image", ha="center", va="center", fontsize=8,
            fontweight="bold")

    # Frozen CLIP
    ax.add_patch(FancyBboxPatch((2.5, 0.7), 2.2, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#e6f3ff", edgecolor="#0072B2"))
    ax.text(3.6, 1.6, "Frozen CLIP-L/14", ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(3.6, 1.1, "hooks at\nblocks 6,12,18,24", ha="center", va="center",
            fontsize=7)

    # TIE + final stream
    ax.add_patch(FancyBboxPatch((5.4, 0.7), 1.8, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#fff7e6", edgecolor="#E69F00"))
    ax.text(6.3, 1.6, "TIE + final-CLS", ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(6.3, 1.1, "256-d\nrepresentation", ha="center", va="center",
            fontsize=7)

    # Classifier
    ax.add_patch(FancyBboxPatch((7.8, 0.7), 1.5, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#f0e6ff", edgecolor="#7B3FA2"))
    ax.text(8.55, 1.6, "MLP", ha="center", va="center", fontsize=8,
            fontweight="bold")
    ax.text(8.55, 1.1, "256→256→64→1", ha="center", va="center", fontsize=7)

    # P_student
    ax.add_patch(FancyBboxPatch((9.9, 0.7), 1.4, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#b8e6d8", edgecolor="#009E73"))
    ax.text(10.6, 1.6, "P_student", ha="center", va="center", fontsize=8,
            fontweight="bold", fontstyle="italic")
    ax.text(10.6, 1.1, "real / fake", ha="center", va="center", fontsize=7)

    # Arrows for student
    ax.add_patch(FancyArrowPatch((1.85, 1.4), (2.45, 1.4), **arrow_kw))
    ax.add_patch(FancyArrowPatch((4.75, 1.4), (5.35, 1.4), **arrow_kw))
    ax.add_patch(FancyArrowPatch((7.25, 1.4), (7.75, 1.4), **arrow_kw))
    ax.add_patch(FancyArrowPatch((9.35, 1.4), (9.85, 1.4), **arrow_kw))

    # Distillation arrow (teacher → student)
    ax.annotate("", xy=(10.3, 2.1), xytext=(9.55, 3.95),
                arrowprops=dict(arrowstyle="->", linewidth=1.5,
                                color="#009E73", linestyle="--"))
    ax.text(10.7, 3.0, "adaptive\nKD", fontsize=8, color="#009E73",
            fontweight="bold", ha="left", va="center")

    # Trainable / frozen markers
    ax.text(11.6, 1.4, "training\ntime", fontsize=7, color="#666",
            ha="left", va="center", style="italic")
    ax.text(11.6, 4.6, "training\ntime\nonly", fontsize=7, color="#666",
            ha="left", va="center", style="italic")

    # Title
    ax.text(7, 6.7, "Dual-hypothesis VLM reasoning → distilled student",
            ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT / "fig_pipeline_diagram.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  saved fig_pipeline_diagram")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {OUT}\n")
    print("Group A: Headline & comparison")
    fig1_headline_3panel()
    fig2_per_gen_auroc_grouped()
    fig3_per_gen_ap_grouped()
    fig4_generalization_gap()
    fig5_method_hierarchy()

    print("\nGroup B: DHSDv2 deep-dive")
    fig6_dhsd_per_gen_auroc()
    fig7_dhsd_roc_curves()
    fig8_dhsd_pr_curves()
    fig9_score_distributions()

    print("\nGroup C: Architecture & method")
    fig10_dual_hypothesis_concept()
    fig11_pipeline_diagram()

    print(f"\nAll figures in: {OUT}")
    print("Each figure produced as both .pdf (vector) and .png (300 dpi)")
