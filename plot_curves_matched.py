"""Figure 3 (revision): learning curves on all four corpora, every model at a matched batch size.

One panel per corpus, fold-level mean ± SD across independent folds (seeds averaged
within fold first). Series and colours are fixed across panels (identity, not rank):
ACTSNet, EEGNet, BigCNN (size-matched), TapNet (AC branch -> LSTM). Chance level drawn
as a recessive dashed line. Outputs paper/figures/curves_matched.{pdf,png,svg}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "results"
FR = [0.10, 0.25, 0.50, 1.00]
TAGS = ["f010", "f025", "f050", "f100"]

# fixed categorical order + Okabe-Ito hues (validated, CVD-safe); same hues as Fig. 1
SERIES = [("ACTSNet", "#0072B2", "o"), ("EEGNet", "#D55E00", "s"),
          ("BigCNN (size-matched)", "#009E73", "^"), ("TapNet (AC → LSTM)", "#9E4B8A", "D")]
INK, MUTED, GRID = "#1f1f1f", "#5c5c5c", "#e6e6e6"


def fold_stats(path):
    p = f"{R}/{path}/per_fold.json"
    if not os.path.exists(p):
        return None
    acc = {}
    for x in json.load(open(p)):
        acc.setdefault(x["fold"], []).append(x["balanced_accuracy"])
    v = np.array([np.mean(a) for _, a in sorted(acc.items())])
    return v.mean(), v.std(), len(v)


# where each (corpus, model, fraction) lives — matched-batch runs where they exist
def loc(ds, model, tag):
    if model == "ACTSNet":
        if ds == "mumtaz":
            return f"h1_recheck/mumtaz_v1_{tag}"
        full = {"seed_iv": "seed_iv", "tuab": "tuab", "cavanagh": "cavanagh_v1"}[ds]
        return full if tag == "f100" else f"h1/{ds}_v1_{tag}"
    key = {"EEGNet": "eegnet", "BigCNN (size-matched)": "bigcnn", "TapNet (AC → LSTM)": "tapnet"}[model]
    if ds == "mumtaz" and key in ("eegnet", "bigcnn"):
        return f"h1_recheck/mumtaz_{key}_{tag}"
    return f"planB/{ds}_{key}_{tag}"


PANELS = [("seed_iv", "SEED-IV", "4-class emotion · 15 subjects", 0.25, (0.22, 0.40)),
          ("mumtaz", "Mumtaz MDD", "58 subjects", 0.50, (0.45, 1.00)),
          ("cavanagh", "Cavanagh", "BDI labels · 119 subjects", 0.50, (0.45, 0.70)),
          ("tuab", "TUAB", "2329 subjects", 0.50, (0.58, 0.82))]

plt.rcParams.update({"font.family": ["Nimbus Sans", "FreeSans", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8})
fig, axes = plt.subplots(1, 4, figsize=(7.09, 2.5), dpi=300)   # 180 mm wide
missing = []
for ax, (ds, title, sub, chance, ylim) in zip(axes, PANELS):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    if ylim[0] <= chance <= ylim[1]:          # TUAB: chance (0.50) lies below the axis; said in caption
        ax.axhline(chance, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.text(0.98, chance, "chance", va="bottom", ha="right", fontsize=6.5, color=MUTED,
                transform=ax.get_yaxis_transform())
    for name, col, mk in SERIES:
        xs, ys, es = [], [], []
        for fr, tag in zip(FR, TAGS):
            st = fold_stats(loc(ds, name, tag))
            if st is None:
                missing.append((ds, name, tag)); continue
            xs.append(fr); ys.append(st[0]); es.append(st[1])
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, markersize=3.6, linewidth=1.4,
                    capsize=0, elinewidth=0.7, alpha=0.95, zorder=3, label=name,
                    markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xscale("log")
    ax.set_xticks(FR); ax.set_xticklabels(["10", "25", "50", "100"], fontsize=7.5)
    ax.minorticks_off()
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=10, fontweight="semibold")
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="left", va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, labelsize=7.5)
axes[0].set_ylabel("Balanced accuracy (fold mean ± SD)", fontsize=8, color=INK)
fig.text(0.5, 0.02, "Training subjects (% of the training fold; log scale)", ha="center", fontsize=8, color=INK)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.05, 1, 0.90), w_pad=1.2)
os.makedirs("paper/figures", exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"paper/figures/curves_matched.{ext}", dpi=300, bbox_inches="tight")
print("wrote paper/figures/curves_matched.{pdf,png,svg}")
if missing:
    print("missing points (left blank):", missing)
