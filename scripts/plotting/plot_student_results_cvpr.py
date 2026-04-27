#!/usr/bin/env python3
"""CVPR-styled Student evaluation figures (single column)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "text.usetex": False, "mathtext.default": "regular",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.0,  "axes.labelsize": 8.0,  "axes.titlesize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.4,
    "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COL_W = 3.45
C_BLUE, C_ORANGE, C_GRAY = "#0072B2", "#D55E00", "#666666"

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="final_results.json")
ap.add_argument("--log",     default="train_log.jsonl")
ap.add_argument("--out",     default=".")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

if not Path(args.results).exists():
    sys.exit(f"[error] {args.results} not found")
R = json.load(open(args.results))

def find_pergen(d):
    if isinstance(d, dict):
        if d and all(isinstance(v, dict) and "auroc" in v for v in d.values()):
            return d
        for v in d.values():
            r = find_pergen(v)
            if r is not None: return r
    return None

pergen = find_pergen(R)
if not pergen:
    sys.exit(f"[error] no per-generator dict. Keys: {list(R.keys())}")

mj_test    = R.get("mj_test_auroc", 0.0)
test_macro = (R.get("crossgen_test_macro") or R.get("test_macro")
              or float(np.mean([v["auroc"] for v in pergen.values()])))
val_macro  = R.get("crossgen_val_macro", R.get("best_crossgen_val_macro", 0.0))
best_epoch = R.get("best_epoch", -1)

log_rows = []
if Path(args.log).exists():
    for line in open(args.log):
        line = line.strip()
        if not line: continue
        try: log_rows.append(json.loads(line))
        except json.JSONDecodeError: pass

# Figure 1 — per-generator
gens   = sorted(pergen.keys(), key=lambda g: pergen[g]["auroc"])
aurocs = [pergen[g]["auroc"] for g in gens]

fig, ax = plt.subplots(figsize=(COL_W, COL_W * 1.05))
ax.barh(np.arange(len(gens)), aurocs, color=C_BLUE, alpha=0.85,
        edgecolor="black", linewidth=0.4, height=0.7)
for i, a in enumerate(aurocs):
    if a >= 0.20:
        ax.text(a - 0.015, i, f"{a:.3f}", va="center", ha="right",
                fontsize=6.5, color="white", fontweight="medium")
    else:
        ax.text(a + 0.012, i, f"{a:.3f}", va="center", ha="left",
                fontsize=6.5, color="black")
ax.axvline(0.5, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.7)
ax.text(0.5, len(gens) - 0.3, "chance", ha="center", va="bottom",
        fontsize=6.5, color=C_GRAY)
ax.axvline(test_macro, color=C_ORANGE, linestyle="-", linewidth=0.9)
ax.text(test_macro, -0.7, f"macro = {test_macro:.3f}",
        ha="center", va="top", fontsize=6.8, color=C_ORANGE)
ax.set_yticks(np.arange(len(gens))); ax.set_yticklabels(gens, fontsize=7.5)
ax.set_xlim(0, 1.02); ax.set_xlabel("AUROC")
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis="x", pad=2); ax.tick_params(axis="y", pad=1)
plt.savefig(OUT / "fig_per_generator_auroc.pdf")
plt.savefig(OUT / "fig_per_generator_auroc.png", dpi=600)
plt.close()
print(f"[ok] {OUT}/fig_per_generator_auroc.pdf")

# Figure 2 — training curve
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
    plt.savefig(OUT / "fig_training_curve.pdf")
    plt.savefig(OUT / "fig_training_curve.png", dpi=600)
    plt.close()
    print(f"[ok] {OUT}/fig_training_curve.pdf")

# LaTeX table
print()
print("─" * 60); print("LaTeX table source"); print("─" * 60)
print(r"\begin{table}[t]"); print(r"\centering")
print(r"\caption{Per-generator AUROC of the Student on the CNNDetection test set, "
      r"sorted in descending order. The model is trained on Midjourney+COCO only "
      r"and evaluated without any cross-generator fine-tuning.}")
print(r"\label{tab:per_generator}"); print(r"\begin{tabular}{lc}")
print(r"\toprule"); print(r"Generator & AUROC \\"); print(r"\midrule")
for g, v in sorted(pergen.items(), key=lambda kv: -kv[1]["auroc"]):
    print(f"{g} & {v['auroc']:.4f} \\\\")
print(r"\midrule")
print(f"\\textbf{{Macro avg.}} & \\textbf{{{test_macro:.4f}}} \\\\")
print(r"\bottomrule"); print(r"\end{tabular}"); print(r"\end{table}")
print("─" * 60)
print(f"In-distribution MJ test : {mj_test:.4f}")
print(f"Cross-generator macro    : {test_macro:.4f}")
print(f"Best val macro at sel.   : {val_macro:.4f}  (ep. {best_epoch})")
