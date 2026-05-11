#!/usr/bin/env python3
"""
Generates all CVPR-styled thesis figures from the saved predictions.
"""
import json, sys, random
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix
from PIL import Image

# ============================================================
# CVPR/IEEE style
# ============================================================
matplotlib.rcParams.update({
    "text.usetex": False, "mathtext.default": "regular",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.4,
    "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COL_W   = 3.45
PAGE_W  = 7.16
C_BLUE   = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN  = "#009E73"
C_GRAY   = "#666666"
C_FEAT   = "#E69F00"
C_LOSS   = "#D55E00"
C_NPR    = "#CC79A7"

# ============================================================
# Paths
# ============================================================
RUN_DIR = Path("/NAS_DISK/Saptarshi_data/dhsd_v2_20260427_133533")
OUT     = RUN_DIR / "figures_cvpr"
OUT.mkdir(parents=True, exist_ok=True)

PRED_NPZ    = RUN_DIR / "crossgen_test_predictions.npz"
RESULTS_JSN = RUN_DIR / "final_results.json"
LOG_JSONL   = RUN_DIR / "train_log.jsonl"

# Direct-CLIP zero-shot baselines (from your scrollback)
CLIP_ZS_BASELINE = {
    "biggan": 0.9544, "crn": 0.9734, "deepfake": 0.7136, "gaugan": 0.9815,
    "imle": 0.8844, "san": 0.7378, "seeingdark": 0.8284, "stargan": 0.9874,
    "whichfaceisreal": 0.9466,
}

# ============================================================
# Load
# ============================================================
R = json.load(open(RESULTS_JSN))
preds = np.load(PRED_NPZ, allow_pickle=True)
GENS = list(preds["generators"])
print(f"[load] generators with predictions: {GENS}")

# Per-generator from final_results.json (already computed, standalone)
pergen_auroc = {g: v["auroc"] for g, v in R["per_gen_test"].items()}
test_macro = R.get("crossgen_test_macro",
                   float(np.mean(list(pergen_auroc.values()))))
mj_test    = R.get("mj_test_auroc", 0.0)
val_macro  = R.get("crossgen_val_macro", 0.0)
best_epoch = R.get("best_epoch", -1)

# Training log
log_rows = []
if LOG_JSONL.exists():
    for line in open(LOG_JSONL):
        line = line.strip()
        if line:
            try: log_rows.append(json.loads(line))
            except: pass

# ============================================================
# Fig 1 — Per-generator AUROC bar chart
# ============================================================
gens_sorted = sorted(pergen_auroc.keys(), key=lambda g: pergen_auroc[g])
aurocs = [pergen_auroc[g] for g in gens_sorted]
fig, ax = plt.subplots(figsize=(COL_W, COL_W * 1.05))
ax.barh(np.arange(len(gens_sorted)), aurocs, color=C_BLUE, alpha=0.85,
        edgecolor="black", linewidth=0.4, height=0.7)
for i, a in enumerate(aurocs):
    if a >= 0.20:
        ax.text(a - 0.015, i, f"{a:.3f}", va="center", ha="right",
                fontsize=6.5, color="white", fontweight="medium")
    else:
        ax.text(a + 0.012, i, f"{a:.3f}", va="center", ha="left",
                fontsize=6.5, color="black")
ax.axvline(0.5, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.7)
ax.text(0.5, len(gens_sorted) - 0.3, "chance",
        ha="center", va="bottom", fontsize=6.5, color=C_GRAY)
ax.axvline(test_macro, color=C_ORANGE, linestyle="-", linewidth=0.9)
ax.text(test_macro, -0.7, f"macro = {test_macro:.3f}",
        ha="center", va="top", fontsize=6.8, color=C_ORANGE)
ax.set_yticks(np.arange(len(gens_sorted)))
ax.set_yticklabels(gens_sorted, fontsize=7.5)
ax.set_xlim(0, 1.02); ax.set_xlabel("AUROC")
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.savefig(OUT / "fig01_per_generator_auroc.pdf")
plt.savefig(OUT / "fig01_per_generator_auroc.png", dpi=600)
plt.close()
print(f"[ok] fig01_per_generator_auroc")

# ============================================================
# Fig 2 — Training curve
# ============================================================
if log_rows:
    eps = [r["epoch"] for r in log_rows]
    mj  = [r.get("mj_val", np.nan) for r in log_rows]
    cg  = [r.get("crossgen_macro", r.get("crossgen_val_macro", np.nan))
           for r in log_rows]
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))
    ax.plot(eps, mj, color=C_BLUE, linewidth=1.2, marker="o", markersize=3.5,
            markerfacecolor="white", markeredgewidth=0.9,
            label="In-distribution MJ val.")
    ax.plot(eps, cg, color=C_ORANGE, linewidth=1.2, marker="s", markersize=3.5,
            markerfacecolor="white", markeredgewidth=0.9,
            label="Cross-generator val. (macro)")
    if best_epoch is not None and 0 <= best_epoch < len(cg):
        ax.axvline(best_epoch, color=C_GRAY, linestyle=":", linewidth=0.6)
        ax.annotate(f"best (ep. {best_epoch})",
                    xy=(best_epoch, cg[best_epoch]),
                    xytext=(best_epoch + 0.3, cg[best_epoch] - 0.07),
                    fontsize=6.5, color=C_GRAY,
                    arrowprops=dict(arrowstyle="-", color=C_GRAY, linewidth=0.5))
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUROC")
    ax.set_xlim(min(eps) - 0.2, max(eps) + 0.2); ax.set_ylim(0.5, 1.02)
    ax.set_xticks(eps); ax.legend(loc="lower right", handlelength=1.5)
    plt.savefig(OUT / "fig02_training_curve.pdf")
    plt.savefig(OUT / "fig02_training_curve.png", dpi=600)
    plt.close()
    print(f"[ok] fig02_training_curve")

# ============================================================
# Fig 3 — Architecture diagram (CORRECTED for actual DHSDv2)
# ============================================================
fig, ax = plt.subplots(figsize=(PAGE_W, 4.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

def box(x, y, w, h, label, color, *, fa=0.10, fs=7.5, fw="normal", italic=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.5",
        facecolor=color, alpha=fa, edgecolor=color, linewidth=0.9))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center",
            fontsize=fs, fontweight=fw, style="italic" if italic else "normal")
    return (x, y, w, h)

def arr(p1, p2, color="black", lw=0.9, dashed=False, mut=10):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
        mutation_scale=mut, color=color, lw=lw,
        linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))

def lab(p1, p2, text, color="black", dx=0, dy=0.7, fs=6.5):
    mx = (p1[0]+p2[0])/2 + dx; my = (p1[1]+p2[1])/2 + dy
    ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.7))

re = lambda b: (b[0]+b[2], b[1]+b[3]/2)
le = lambda b: (b[0],      b[1]+b[3]/2)
te = lambda b: (b[0]+b[2]/2, b[1]+b[3])
be = lambda b: (b[0]+b[2]/2, b[1])

b_input = box(2, 26, 9, 7, "Input image\n$224 \\times 224 \\times 3$", C_GRAY, fs=7.5)

# Frozen CLIP
clip_x, clip_y, clip_w, clip_h = 15, 8, 22, 47
ax.add_patch(FancyBboxPatch((clip_x, clip_y), clip_w, clip_h,
    boxstyle="round,pad=0.5,rounding_size=0.6", facecolor=C_BLUE,
    alpha=0.05, edgecolor=C_BLUE, linewidth=1.0))
ax.text(clip_x + clip_w/2, clip_y + clip_h - 1.6,
        "CLIP ViT-L/14   (frozen, 304 M params)",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_BLUE)

n_blocks = 6; blk_x, blk_w, blk_h = clip_x + 2.5, 8, 3.4
blk_top = clip_y + clip_h - 4
tap_idx = {1, 2, 3, 4}; tap_centers = []
for i in range(n_blocks):
    by = blk_top - i*(blk_h + 0.6) - blk_h
    is_tap = i in tap_idx
    box(blk_x, by, blk_w, blk_h, "ViT block", C_BLUE,
        fa=0.18 if is_tap else 0.07, fs=7)
    if is_tap: tap_centers.append((blk_x + blk_w, by + blk_h/2))
ax.text(blk_x + blk_w/2, blk_top - n_blocks*(blk_h + 0.6) - 0.8,
        "(24 blocks total)", ha="center", va="center",
        fontsize=6.8, color=C_BLUE, style="italic")

b_finalcls = box(clip_x + 2.5, clip_y + 0.6, blk_w, 2.5,
                 "Final CLS  (768-d)", C_FEAT, fa=0.30, fs=7, fw="bold")

# Two trainable branches
mid_x = 46
b_tie = box(mid_x, 38, 22, 9,
            "TIE Aggregator\n"
            "$4 \\times \\mathrm{Linear}(1024 \\to 128)$ + GELU\n"
            "softmax-attention pool $\\to$ L2-norm",
            C_GREEN, fs=7.2)

b_finproj = box(mid_x, 22, 22, 8,
                "Final-CLS Projection\n"
                "$\\mathrm{Linear}(768 \\to 128)$ + GELU\n"
                "$\\to$ L2-norm",
                C_GREEN, fs=7.2)

# Wiring
arr(re(b_input), (clip_x, clip_y + clip_h - 8), color=C_GRAY)

rail_x = clip_x + clip_w + 1.5
for (tx, ty) in tap_centers:
    ax.add_patch(FancyArrowPatch((tx, ty), (rail_x, ty),
        arrowstyle="-", mutation_scale=8, color=C_BLUE, lw=0.7, linestyle=":"))
    arr((rail_x, ty), (mid_x, b_tie[1] + b_tie[3]/2),
        color=C_BLUE, lw=0.7, mut=8)
ax.text(rail_x + 5, b_tie[1] + b_tie[3] + 1.2,
        "4 intermediate CLS tokens (1024-d each)",
        ha="left", va="center", fontsize=6.5, color=C_BLUE, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))

p1 = re(b_finalcls); p2 = le(b_finproj)
arr(p1, p2, color=C_BLUE)
lab(p1, p2, "768-d", color=C_BLUE, dy=-1.0, fs=6.5)

# Concat + classifier
b_concat = box(72, 30, 8, 6, "concat\n256-d", C_FEAT, fa=0.30, fs=7, fw="bold")
p1 = re(b_tie);     arr(p1, le(b_concat), color=C_GREEN); lab(p1, le(b_concat), "128-d", color=C_GREEN, dy=0.7, fs=6.3)
p1 = re(b_finproj); arr(p1, le(b_concat), color=C_GREEN); lab(p1, le(b_concat), "128-d", color=C_GREEN, dy=0.7, fs=6.3)

b_clf = box(72, 16, 16, 9,
            "Classifier MLP\n"
            "Lin($256{\\to}256$) + GELU + Drop\n"
            "Lin($256{\\to}64$) + GELU + Drop\n"
            "$\\to$ Lin($64{\\to}1$)",
            C_GREEN, fa=0.18, fs=6.5, fw="bold")
arr(be(b_concat), te(b_clf), color=C_GREEN)
lab(be(b_concat), te(b_clf), "256-d", color=C_GREEN, dx=1.4, dy=0, fs=6.3)

b_out = box(72, 6, 16, 6,
            "$\\sigma(\\mathrm{logit})$\n$P(\\mathrm{fake})$",
            C_GRAY, fa=0.12, fs=7.5, fw="bold")
arr(be(b_clf), te(b_out), color="black", lw=1.0)
lab(be(b_clf), te(b_out), "logit", dx=1.2, dy=0, fs=6.3)

# Training-only loss box
ax.text(50, 2.5,
        "Training: $\\mathcal{L} = "
        "\\mathrm{BCE}(\\hat{y}, y) + "
        "\\alpha \\cdot \\mathcal{L}_{\\mathrm{KD}}^{\\mathrm{teacher}} + "
        "\\beta \\cdot \\mathcal{L}_{\\mathrm{KD}}^{\\mathrm{CLIP\\text{-}zs}}$  "
        "(adaptive sample-weighted)",
        ha="center", va="center", fontsize=7.5, color=C_LOSS,
        bbox=dict(facecolor="white", edgecolor=C_LOSS, lw=0.6,
                  boxstyle="round,pad=0.4"))

plt.savefig(OUT / "fig03_student_architecture.pdf")
plt.savefig(OUT / "fig03_student_architecture.png", dpi=600)
plt.close()
print(f"[ok] fig03_student_architecture (corrected to real DHSDv2)")

# ============================================================
# Fig 4 — ROC curves (per-generator + macro)
# ============================================================
fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))
all_fprs, all_tprs = [], []
gen_colors = plt.cm.tab20(np.linspace(0, 1, len(GENS)))
for g, c in zip(GENS, gen_colors):
    p = preds[f"{g}_probs"]; y = preds[f"{g}_labels"]
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    ax.plot(fpr, tpr, color=c, linewidth=0.7, alpha=0.5,
            label=f"{g} ({auc:.2f})")

# Macro-averaged ROC
all_y = np.concatenate([preds[f"{g}_labels"] for g in GENS])
all_p = np.concatenate([preds[f"{g}_probs"]  for g in GENS])
fpr_m, tpr_m, _ = roc_curve(all_y, all_p)
ax.plot(fpr_m, tpr_m, color="black", linewidth=1.5, label=f"micro avg")

ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle="--", linewidth=0.6)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="lower right", fontsize=5.5, ncol=2, columnspacing=0.5,
          handlelength=1.0, borderpad=0.4)
plt.savefig(OUT / "fig04_roc_curves.pdf")
plt.savefig(OUT / "fig04_roc_curves.png", dpi=600)
plt.close()
print(f"[ok] fig04_roc_curves")

# ============================================================
# Fig 5 — Confusion matrix at threshold 0.5 (aggregate over all gens)
# ============================================================
y_true = np.concatenate([preds[f"{g}_labels"] for g in GENS])
y_pred = (np.concatenate([preds[f"{g}_probs"] for g in GENS]) > 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(COL_W * 0.85, COL_W * 0.85))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        txt_color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]:.2f})",
                ha="center", va="center", color=txt_color, fontsize=8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["pred Real", "pred Fake"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["true Real", "true Fake"])
ax.set_xlabel(""); ax.set_ylabel("")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.grid(False)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=6.5)
plt.savefig(OUT / "fig05_confusion_matrix.pdf")
plt.savefig(OUT / "fig05_confusion_matrix.png", dpi=600)
plt.close()
print(f"[ok] fig05_confusion_matrix")

# ============================================================
# Fig 6 — Threshold curve (precision / recall / F1 vs threshold)
# ============================================================
y_true = np.concatenate([preds[f"{g}_labels"] for g in GENS])
y_prob = np.concatenate([preds[f"{g}_probs"]  for g in GENS])
thresholds = np.linspace(0.05, 0.95, 91)
precs, recs, f1s = [], [], []
for t in thresholds:
    yp = (y_prob > t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, yp,
                                                  average="binary",
                                                  zero_division=0)
    precs.append(p); recs.append(r); f1s.append(f)
fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))
ax.plot(thresholds, precs, color=C_BLUE,   linewidth=1.2, label="precision")
ax.plot(thresholds, recs,  color=C_GREEN,  linewidth=1.2, label="recall")
ax.plot(thresholds, f1s,   color=C_ORANGE, linewidth=1.4, label="F1")
best_t = thresholds[int(np.argmax(f1s))]
ax.axvline(best_t, color=C_GRAY, linestyle=":", linewidth=0.6)
ax.annotate(f"best F1\n@ t={best_t:.2f}",
            xy=(best_t, max(f1s)), xytext=(best_t + 0.06, max(f1s) - 0.10),
            fontsize=6.5, color=C_GRAY,
            arrowprops=dict(arrowstyle="-", color=C_GRAY, linewidth=0.5))
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="lower center", ncol=3, handlelength=1.5)
plt.savefig(OUT / "fig06_threshold_curve.pdf")
plt.savefig(OUT / "fig06_threshold_curve.png", dpi=600)
plt.close()
print(f"[ok] fig06_threshold_curve")

# ============================================================
# Fig 7 — Student vs CLIP zero-shot per-generator comparison
# ============================================================
common_gens = [g for g in CLIP_ZS_BASELINE if g in pergen_auroc]
common_gens = sorted(common_gens, key=lambda g: pergen_auroc[g])
student_vals = [pergen_auroc[g]      for g in common_gens]
clip_vals    = [CLIP_ZS_BASELINE[g] for g in common_gens]

fig, ax = plt.subplots(figsize=(COL_W, COL_W * 1.05))
y = np.arange(len(common_gens)); h = 0.38
ax.barh(y - h/2, student_vals, h, color=C_BLUE,   alpha=0.85,
        edgecolor="black", linewidth=0.4, label="Student (ours)")
ax.barh(y + h/2, clip_vals,    h, color=C_ORANGE, alpha=0.85,
        edgecolor="black", linewidth=0.4, label="CLIP zero-shot")
for i, (s, c) in enumerate(zip(student_vals, clip_vals)):
    ax.text(s + 0.01, i - h/2, f"{s:.2f}", va="center", ha="left",
            fontsize=6, color=C_BLUE)
    ax.text(c + 0.01, i + h/2, f"{c:.2f}", va="center", ha="left",
            fontsize=6, color=C_ORANGE)
ax.axvline(0.5, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.7)
ax.set_yticks(y); ax.set_yticklabels(common_gens, fontsize=7)
ax.set_xlim(0, 1.10); ax.set_xlabel("AUROC")
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc="lower right", handlelength=1.5)
plt.savefig(OUT / "fig07_student_vs_clip_zs.pdf")
plt.savefig(OUT / "fig07_student_vs_clip_zs.png", dpi=600)
plt.close()
print(f"[ok] fig07_student_vs_clip_zs")

# ============================================================
# Fig 8 — Qualitative examples
# ============================================================
np_rng = np.random.default_rng(SEED := 7)
panels = []
chosen_gens = ["stargan", "gaugan", "deepfake", "stylegan2"]   # 2 strong + 2 weak
for g in chosen_gens:
    if g not in GENS: continue
    paths  = preds[f"{g}_paths"]
    probs  = preds[f"{g}_probs"]
    labels = preds[f"{g}_labels"]
    # Pick one real (label 0) + one fake (label 1) — prefer median-confidence sample
    for target_lbl in [0, 1]:
        mask = labels == target_lbl
        if not mask.any(): continue
        idxs = np.where(mask)[0]
        # Sort by closeness to median prob within the class — gives "typical" examples
        med = np.median(probs[idxs])
        idx = idxs[np.argmin(np.abs(probs[idxs] - med))]
        panels.append({"gen": g, "path": str(paths[idx]),
                       "label": int(target_lbl), "prob": float(probs[idx])})

n = len(panels)
ncols = 4; nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_W, PAGE_W * 0.55))
axes = np.atleast_2d(axes).reshape(nrows, ncols)
for ax in axes.flatten(): ax.axis("off")

for k, p in enumerate(panels):
    r, c = divmod(k, ncols)
    ax = axes[r, c]
    try:
        img = Image.open(p["path"]).convert("RGB")
        # Center crop to square
        w, h = img.size; s = min(w, h)
        img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
        img = img.resize((180, 180))
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f"[err: {e}]", ha="center", va="center")

    correct = (p["prob"] > 0.5) == bool(p["label"])
    title_color = C_GREEN if correct else C_LOSS
    title_lbl  = "fake" if p["label"] == 1 else "real"
    ax.set_title(f"{p['gen']}  (true: {title_lbl})\n"
                 f"$P(\\mathrm{{fake}}) = {p['prob']:.3f}$",
                 fontsize=7, color=title_color, pad=2)

plt.tight_layout()
plt.savefig(OUT / "fig08_qualitative_examples.pdf")
plt.savefig(OUT / "fig08_qualitative_examples.png", dpi=600)
plt.close()
print(f"[ok] fig08_qualitative_examples")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"All CVPR-styled figures saved to:")
print(f"  {OUT}")
print("=" * 60)
import subprocess
subprocess.run(["ls", "-lh", str(OUT)])
