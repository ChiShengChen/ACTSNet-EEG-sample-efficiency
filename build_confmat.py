"""Confusion matrix + per-class recall from a dump_preds run (pools all LOSO folds).
Usage: python build_confmat.py results/seed_iv_v1_preds "neutral,sad,fear,happy"
-> prints normalized CM + per-class recall, saves results/figures/confmat_<dir>.png"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

d = sys.argv[1]
names = sys.argv[2].split(",") if len(sys.argv) > 2 else None
rows = json.load(open(f"{d}/per_fold.json"))
y = np.array([v for r in rows for v in r["labels"]])
p = np.array([v for r in rows for v in r["preds"]])
K = int(max(y.max(), p.max()) + 1)
if names is None: names = [str(i) for i in range(K)]

cm = confusion_matrix(y, p, labels=range(K))
cmn = cm / cm.sum(1, keepdims=True)              # row-normalized (recall)
print(f"=== {os.path.basename(d)}  (N={len(y)}, BACC={balanced_accuracy_score(y,p):.4f}) ===")
print("per-class recall:", {names[i]: round(cmn[i, i], 3) for i in range(K)})
print("row-normalized confusion matrix (true→pred):")
print("        " + "  ".join(f"{n[:5]:>6}" for n in names))
for i in range(K):
    print(f"{names[i][:7]:>7} " + "  ".join(f"{cmn[i,j]:6.2f}" for j in range(K)))

fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"ACTSNet v1 — SEED-IV emotion (recall)")
for i in range(K):
    for j in range(K):
        ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
out = f"results/figures/confmat_{os.path.basename(d)}.png"
fig.savefig(out, dpi=150); print("saved", out)
