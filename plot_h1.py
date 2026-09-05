"""Plot H1 learning curves (v1 vs EEGNet) for a dataset -> results/figures/h1_<ds>.png."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds = sys.argv[1] if len(sys.argv) > 1 else "seed_iv"
chance = {"seed_iv": 0.25, "mumtaz": 0.50}.get(ds, 0.5)
H1 = "results/h1"
v1full = f"results/{ds}_v1" if os.path.isdir(f"results/{ds}_v1") else f"results/{ds}"
POINTS = [
    (0.10, f"{H1}/{ds}_v1_f010", f"{H1}/{ds}_eegnet_f010"),
    (0.25, f"{H1}/{ds}_v1_f025", f"{H1}/{ds}_eegnet_f025"),
    (0.50, f"{H1}/{ds}_v1_f050", f"{H1}/{ds}_eegnet_f050"),
    (1.00, v1full, f"results/{ds}_eegnet"),
]

def stat(d):
    m = json.load(open(f"{d}/summary.json"))["metrics"]["balanced_accuracy"]
    return m["mean"], m["std"]

BIG = {0.10: f"{H1}/{ds}_bigcnn_f010", 0.25: f"{H1}/{ds}_bigcnn_f025",
       0.50: f"{H1}/{ds}_bigcnn_f050", 1.00: f"results/{ds}_bigcnn"}

fr, vm, vs, em, es, bm, bs = [], [], [], [], [], [], []
for f, vd, ed in POINTS:
    if not (os.path.exists(f"{vd}/summary.json") and os.path.exists(f"{ed}/summary.json")):
        continue
    a = stat(vd); b = stat(ed)
    fr.append(f); vm.append(a[0]); vs.append(a[1]); em.append(b[0]); es.append(b[1])
    bd = BIG.get(f)
    if bd and os.path.exists(f"{bd}/summary.json"):
        g = stat(bd); bm.append(g[0]); bs.append(g[1])
    else:
        bm.append(np.nan); bs.append(np.nan)

fr = np.array(fr); bm = np.array(bm)
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.errorbar(fr, vm, yerr=vs, marker="o", capsize=3, label="ACTSNet", color="#1f77b4")
ax.errorbar(fr, em, yerr=es, marker="s", capsize=3, label="EEGNet", color="#ff7f0e")
if not np.all(np.isnan(bm)):
    ax.errorbar(fr, bm, yerr=bs, marker="^", capsize=3,
                label="BigCNN (size-matched)", color="#2ca02c")
ax.axhline(chance, ls="--", lw=1, color="gray", label=f"chance ({chance:g})")
ax.set_xscale("log"); ax.set_xticks(fr); ax.set_xticklabels([f"{int(f*100)}%" for f in fr])
ax.set_xlabel("Training-subject fraction"); ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Low-data learning curve — {ds}")
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
out = f"results/figures/h1_{ds}.png"
fig.savefig(out, dpi=150)
print("saved", out)
