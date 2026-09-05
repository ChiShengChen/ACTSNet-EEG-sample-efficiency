"""ACTSNet architecture schematic -> results/figures/arch.png (+ paper/figures/)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.2, 5.6)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 12)

C_in, C_ms, C_ac, C_fuse, C_head = "#e8eef7", "#dbe7d8", "#f7e6d5", "#e6dcef", "#f5d9de"

def box(x, y, w, h, text, fc, fs=8.5):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="#333", lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=5)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=12, lw=1.1, color="#333", shrinkA=2, shrinkB=2))

# input
box(5, 11.3, 4.4, 0.8, r"Input  $\mathbf{X}\in\mathbb{R}^{C\times S\times T}$  (channels $\times$ sub-bands $\times$ time)", C_in, 9)
# split
arrow(5, 10.9, 2.6, 10.2); arrow(5, 10.9, 7.4, 10.2)

# left: multi-scale branch
box(2.6, 9.7, 3.8, 0.85, "Multi-scale encoder\nrandom permutation $\\to$ $G$ groups", C_ms)
box(2.6, 8.3, 3.8, 0.8, "shared Conv1D+BN+ReLU", C_ms)
box(2.6, 7.0, 3.8, 0.7, r"GAP $\to$ concat  $\mathbf{m}$", C_ms)
arrow(2.6, 9.25, 2.6, 8.7); arrow(2.6, 7.9, 2.6, 7.35)

# right: AC branch
box(7.4, 9.7, 3.8, 0.8, r"$1{\times}1$ Conv projection", C_ac)
box(7.4, 8.5, 3.8, 0.8, r"$3\times$ (Conv1D+IN+PReLU)", C_ac)
box(7.4, 7.3, 3.8, 0.85, r"AC: $\mathrm{softmax}(\mathbf{H})\odot\mathbf{H}$", C_ac)
box(7.4, 6.05, 3.8, 0.85, r"FC+$\sigma$, IN, GAP $\to$ $\mathbf{a}$", C_ac)
arrow(7.4, 9.3, 7.4, 8.9); arrow(7.4, 8.1, 7.4, 7.72); arrow(7.4, 6.87, 7.4, 6.47)

# fuse
box(5, 5.0, 5.2, 0.8, r"Concat $[\mathbf{m};\mathbf{a}]$  $\to$  FC  $\to$  embedding $f(\mathbf{X})\in\mathbb{R}^{d}$", C_fuse, 8.5)
arrow(2.6, 6.65, 4.2, 5.4); arrow(7.4, 5.62, 5.9, 5.4)

# prototypical head
box(5, 3.5, 6.0, 0.9, "Supervised attentional prototypical head\n"
    r"$\mathbf{p}_k=\sum_i \alpha_{k,i} f_i$", C_head, 8.5)
box(5, 2.1, 6.0, 0.8, r"classify: $\mathrm{softmax}_k(-\|f-\mathbf{p}_k\|^2)$", C_head, 9)
arrow(5, 4.55, 5, 3.95); arrow(5, 3.05, 5, 2.5)

ax.text(2.6, 10.35, "Branch 1", ha="center", fontsize=8, style="italic", color="#557")
ax.text(7.4, 10.35, "Branch 2 (AC)", ha="center", fontsize=8, style="italic", color="#755")

fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/arch.png", dpi=160, bbox_inches="tight")
print("saved results/figures/arch.png")
