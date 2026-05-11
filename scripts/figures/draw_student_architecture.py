#!/usr/bin/env python3
"""CVPR-styled Student architecture diagram (full text width)."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

matplotlib.rcParams.update({
    "text.usetex": False, "mathtext.default": "regular",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.0, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_FROZEN, C_TRAIN, C_FEAT = "#0072B2", "#009E73", "#E69F00"
C_INPUT, C_LOSS, C_NPR    = "#666666", "#D55E00", "#CC79A7"
LW_BOX, LW_ARROW, ALPHA_FILL = 0.9, 0.9, 0.10

fig, ax = plt.subplots(figsize=(7.16, 6.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 75); ax.axis("off")

def box(x, y, w, h, label, color, *, fill_alpha=ALPHA_FILL, fontsize=7.5,
        fontweight="normal", italic=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.5",
        facecolor=color, alpha=fill_alpha, edgecolor=color, linewidth=LW_BOX))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            style="italic" if italic else "normal")
    return (x, y, w, h)

def arrow(p1, p2, *, color="black", lw=LW_ARROW, dashed=False, mutation=10):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
        mutation_scale=mutation, color=color, lw=lw,
        linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))

def label_on(p1, p2, text, *, color="black", pos=0.5, dx=0, dy=0.7, fs=6.5):
    mx = p1[0] + (p2[0]-p1[0])*pos + dx; my = p1[1] + (p2[1]-p1[1])*pos + dy
    ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.7))

right_edge = lambda b: (b[0]+b[2], b[1]+b[3]/2)
left_edge  = lambda b: (b[0],      b[1]+b[3]/2)
top_edge   = lambda b: (b[0]+b[2]/2, b[1]+b[3])
bot_edge   = lambda b: (b[0]+b[2]/2, b[1])

b_input = box(2, 38, 9, 7, "Input image\n$224 \\times 224 \\times 3$", C_INPUT, fontsize=7.5)

clip_x, clip_y, clip_w, clip_h = 15, 21, 22, 47
ax.add_patch(FancyBboxPatch((clip_x, clip_y), clip_w, clip_h,
    boxstyle="round,pad=0.5,rounding_size=0.6", facecolor=C_FROZEN,
    alpha=0.05, edgecolor=C_FROZEN, linewidth=1.0))
ax.text(clip_x + clip_w/2, clip_y + clip_h - 1.6,
        "CLIP ViT-L/14   (frozen, 304 M params)",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_FROZEN)

n_blocks = 6; blk_x, blk_w, blk_h = clip_x + 2.5, 8, 3.4
blk_top = clip_y + clip_h - 4
tap_idx = {1, 2, 3, 4}; tap_centers = []
for i in range(n_blocks):
    by = blk_top - i * (blk_h + 0.6) - blk_h
    is_tap = i in tap_idx
    box(blk_x, by, blk_w, blk_h, "ViT block", C_FROZEN,
        fill_alpha=0.18 if is_tap else 0.07, fontsize=7)
    if is_tap: tap_centers.append((blk_x + blk_w, by + blk_h/2))
ax.text(blk_x + blk_w/2, blk_top - n_blocks*(blk_h + 0.6) - 0.8,
        "(24 transformer blocks total)", ha="center", va="center",
        fontsize=6.8, color=C_FROZEN, style="italic")

b_finalcls = box(clip_x + 2.5, clip_y + 0.6, blk_w, 2.5,
                 "Final CLS  (768-d)", C_FEAT, fill_alpha=0.30,
                 fontsize=7, fontweight="bold")

b_npr = box(15, 8.5, 22, 5.5,
            "NPR transform   $x - \\mathrm{Up}(\\mathrm{Down}(x))$\n"
            "captures up-sampling artifacts",
            C_NPR, fill_alpha=0.10, fontsize=7.2)

mid_x = 46
b_tie = box(mid_x, 53, 22, 9,
            "Multi-depth Aggregator (TIE)\n"
            "$4 \\times \\mathrm{Linear}(1024 \\to 128)$ + softmax gate\n"
            "$\\to$ 128-d  CLIP feature",
            C_TRAIN, fontsize=7.2)
b_sem = box(mid_x, 36.5, 22, 9,
            "Semantic Head\n$768 \\to 256 \\to 22$\n"
            "$\\to$ 22-d  simulated dual-hypothesis vector",
            C_TRAIN, fontsize=7.2)
b_npr_br = box(mid_x, 20, 22, 9,
               "NPR Branch\n5-layer CNN on NPR map\n"
               "$\\to$ 128-d  artifact feature",
               C_TRAIN, fontsize=7.2)

arrow(right_edge(b_input), (clip_x, clip_y + clip_h - 8), color=C_INPUT)
arrow(bot_edge(b_input), (b_npr[0] + 4, b_npr[1] + b_npr[3]), color=C_INPUT)

rail_x = clip_x + clip_w + 1.5
for (tx, ty) in tap_centers:
    ax.add_patch(FancyArrowPatch((tx, ty), (rail_x, ty),
        arrowstyle="-", mutation_scale=8, color=C_FROZEN, lw=0.7, linestyle=":"))
    arrow((rail_x, ty), (mid_x, b_tie[1] + b_tie[3]/2),
          color=C_FROZEN, lw=0.7, mutation=8)
ax.text(rail_x + 5, b_tie[1] + b_tie[3] + 1.2,
        "4 intermediate CLS tokens (1024-d each)",
        ha="left", va="center", fontsize=6.5, color=C_FROZEN, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))

p1 = right_edge(b_finalcls); p2 = left_edge(b_sem)
arrow(p1, p2, color=C_FROZEN); label_on(p1, p2, "768-d", color=C_FROZEN, dy=-1.2, fs=6.5)
p1 = right_edge(b_npr); p2 = left_edge(b_npr_br)
arrow(p1, p2, color=C_NPR); label_on(p1, p2, "NPR map", color=C_NPR, dy=0.8, fs=6.5)

b_fuse = box(76, 36, 14, 10,
             "Gated Fusion\nLinear proj. +\nsoftmax gate\n$\\to$ 256-d",
             C_TRAIN, fill_alpha=0.18, fontsize=7, fontweight="bold")

join_top = (b_fuse[0], b_fuse[1] + b_fuse[3] - 1.5)
join_mid = (b_fuse[0], b_fuse[1] + b_fuse[3]/2)
join_bot = (b_fuse[0], b_fuse[1] + 1.5)
p1 = right_edge(b_tie);    arrow(p1, join_top, color=C_TRAIN); label_on(p1, join_top, "128-d", color=C_TRAIN, dy=0.8, fs=6.5)
p1 = right_edge(b_sem);    arrow(p1, join_mid, color=C_TRAIN); label_on(p1, join_mid, "22-d",  color=C_TRAIN, dy=0.8, fs=6.5)
p1 = right_edge(b_npr_br); arrow(p1, join_bot, color=C_TRAIN); label_on(p1, join_bot, "128-d", color=C_TRAIN, dy=0.8, fs=6.5)

b_clf = box(76, 21, 14, 9,
            "Classifier\nLin($256{\\to}128$)\n+ GELU + Dropout\n"
            "$\\to$ Lin($128{\\to}2$)",
            C_TRAIN, fontsize=7)
arrow(bot_edge(b_fuse), top_edge(b_clf), color=C_TRAIN)
label_on(bot_edge(b_fuse), top_edge(b_clf), "256-d", color=C_TRAIN, dx=1.2, dy=0, fs=6.5)

b_out = box(76, 9, 14, 6, "softmax\n$\\to P(\\mathrm{fake})$",
            C_INPUT, fill_alpha=0.12, fontsize=7.5, fontweight="bold")
arrow(bot_edge(b_clf), top_edge(b_out), color="black", lw=1.0)
label_on(bot_edge(b_clf), top_edge(b_out), "logits (2)", dx=1.4, dy=0, fs=6.5)

b_tgt = box(46, 65, 22, 6,
            "Teacher 22-d vector  (Section 4.2)\n"
            "MSE regression target — training only",
            C_LOSS, fill_alpha=0.06, fontsize=7, italic=True)
p1 = bot_edge(b_tgt); p2 = top_edge(b_sem)
ax.add_patch(FancyArrowPatch((p1[0]+8, p1[1]), (p2[0]+8, p2[1]),
    arrowstyle="-|>", mutation_scale=10, color=C_LOSS, lw=0.8,
    linestyle="--", shrinkA=2, shrinkB=2))
ax.text(p2[0]+9, (p1[1]+p2[1])/2,
        "$\\mathcal{L}_{\\mathrm{MSE}}$", color=C_LOSS, fontsize=7,
        ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))

ax.text(50, 2.5,
        "$\\mathcal{L} = \\mathcal{L}_{\\mathrm{BCE}} + "
        "\\lambda_1 \\mathcal{L}_{\\mathrm{MSE}} + "
        "\\lambda_2 \\mathcal{L}_{\\mathrm{KL}} + "
        "\\lambda_3 \\mathcal{L}_{\\mathrm{AU}}$",
        ha="center", va="center", fontsize=8, color=C_LOSS,
        bbox=dict(facecolor="white", edgecolor=C_LOSS, lw=0.6, alpha=1.0,
                  boxstyle="round,pad=0.4"))

plt.savefig("fig_student_architecture.pdf")
plt.savefig("fig_student_architecture.png", dpi=600)
print("Wrote fig_student_architecture.pdf and .png")
