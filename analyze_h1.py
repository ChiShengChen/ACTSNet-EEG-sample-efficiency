"""Assemble the H1 low-data learning curve for one dataset: v1 vs EEGNet across
train fractions, with per-fraction paired Wilcoxon and AULC.

AULC DEFINITION. The lowest training fraction actually evaluated is 0.10, so the
trapezoidal integral spans [0.10, 1.00] -- a width of 0.9, not 1.0. Left unnormalised it
therefore does not lie on the balanced-accuracy scale. Both quantities are printed below:
the raw integral and the same integral divided by the span of the sampled interval, which
puts AULC back on the BACC scale and makes it comparable across datasets. The relative
difference between models is unchanged by the normalisation -- the divisor is common to
both -- and the arithmetic is printed so every quoted percentage can be checked.

UNIT OF ANALYSIS: balanced accuracy is averaged across seeds first, so each fold
contributes one observation to the Wilcoxon test (see stats_paper1.py).
"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon

ds = sys.argv[1] if len(sys.argv) > 1 else "seed_iv"
H1 = "results/h1"
# (frac, v1_dir, eegnet_dir); frac=1.0 reuses the full-data runs
POINTS = [
    (0.10, f"{H1}/{ds}_v1_f010", f"{H1}/{ds}_eegnet_f010"),
    (0.25, f"{H1}/{ds}_v1_f025", f"{H1}/{ds}_eegnet_f025"),
    (0.50, f"{H1}/{ds}_v1_f050", f"{H1}/{ds}_eegnet_f050"),
    (1.00, f"results/{ds}_v1" if os.path.isdir(f"results/{ds}_v1") else f"results/{ds}",
           f"results/{ds}_eegnet"),
]

def per_fold(d):
    """One observation per fold: BACC averaged over seeds first."""
    acc = {}
    for x in json.load(open(f"{d}/per_fold.json")):
        acc.setdefault(x["fold"], []).append(x["balanced_accuracy"])
    return {f: float(np.mean(v)) for f, v in acc.items()}

fracs, v1m, egm = [], [], []
print(f"=== H1 learning curve: {ds} (BalAcc) ===")
print(f"{'frac':>6} {'v1':>16} {'EEGNet':>16} {'Δ':>8} {'v1 win':>8} {'Wilcoxon p':>12}")
for frac, v1d, egd in POINTS:
    if not (os.path.exists(f"{v1d}/per_fold.json") and os.path.exists(f"{egd}/per_fold.json")):
        print(f"{frac:>6}  (missing)"); continue
    a, b = per_fold(v1d), per_fold(egd)
    k = sorted(set(a) & set(b))
    av = np.array([a[i] for i in k]); bv = np.array([b[i] for i in k])
    try: _, p = wilcoxon(av, bv)
    except Exception: p = float("nan")
    fracs.append(frac); v1m.append(av.mean()); egm.append(bv.mean())
    print(f"{frac:>6} {av.mean():>7.4f}±{av.std():.4f} {bv.mean():>7.4f}±{bv.std():.4f} "
          f"{av.mean()-bv.mean():>+8.4f} {int((av>bv).sum()):>4}/{len(k):<3} {p:>12.4f}")

fracs, v1m, egm = np.array(fracs), np.array(v1m), np.array(egm)
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 2.x 移除了 trapz
raw_v1, raw_eg = trapz(v1m, fracs), trapz(egm, fracs)
span = fracs[-1] - fracs[0]                         # actual integration interval
print(f"\nAULC, trapezoidal over the sampled fractions {list(fracs)}")
print(f"  (note the unequal spacing of the sampled points; the trapezoid rule weights")
print(f"   each interval by its own width, so the 0.50-1.00 segment carries half the mass)")
print(f"  integration limits [{fracs[0]:.2f}, {fracs[-1]:.2f}], span {span:.2f}\n")
print(f"{'':20s}{'unnormalised':>14}{'normalised /span':>18}")
print(f"  {'v1':18s}{raw_v1:>14.4f}{raw_v1/span:>18.4f}")
print(f"  {'EEGNet':18s}{raw_eg:>14.4f}{raw_eg/span:>18.4f}")
print(f"  {'Δ':18s}{raw_v1-raw_eg:>+14.4f}{(raw_v1-raw_eg)/span:>+18.4f}")
print(f"\n  relative difference {100*(raw_v1-raw_eg)/raw_eg:+.1f}% over EEGNet "
      f"(identical either way: the span divides out)")
print(f"  arithmetic: ({raw_v1:.4f} - {raw_eg:.4f}) / {raw_eg:.4f} = {(raw_v1-raw_eg)/raw_eg:+.4f}")
print(f"\nGap vs data: " + "  ".join(f"f{f:.2f}:{(v-e):+.3f}" for f, v, e in zip(fracs, v1m, egm)))
