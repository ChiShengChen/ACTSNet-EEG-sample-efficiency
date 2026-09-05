"""Statistics for the ACTSNet comparisons: paired Wilcoxon + rank-biserial effect size
+ subject-level bootstrap 95% CI on the mean BACC difference, with Holm-Bonferroni
correction across the family of comparisons. Pure re-analysis of results/*/per_fold.json.

UNIT OF ANALYSIS. Three seeds evaluated on the same folds are not independent
observations, so treating every (seed, fold) pair as one inflates the effective sample
size threefold (45 "observations" from 15 independent folds on SEED-IV) and understates
uncertainty. The default here averages balanced accuracy across seeds first, so each fold
contributes exactly one observation and n equals the number of independent folds. Pass
--unit seedfold for the pooled (seed, fold) analysis used in the first version of this
work, which the fold-level analysis supersedes.
"""
import argparse, json, os
import numpy as np
from scipy.stats import wilcoxon

_AP = argparse.ArgumentParser()
_AP.add_argument("--unit", default="fold", choices=["fold", "seedfold"],
                 help="fold (default): average over seeds first, one observation per fold. "
                      "seedfold: every (seed,fold) treated as independent (inflates n).")
_AP.add_argument("--results", default="results", help="results root (e.g. results/planB)")
ARGS = _AP.parse_args()

RNG = np.random.RandomState(0)

def per_fold(d):
    """Observation dict keyed by the chosen unit of analysis (see module docstring)."""
    p = f"{d}/per_fold.json"
    if not os.path.exists(p): return None
    recs = json.load(open(p))
    if ARGS.unit == "seedfold":
        return {(x["seed"], x["fold"]): x["balanced_accuracy"] for x in recs}
    acc = {}
    for x in recs:
        acc.setdefault(x["fold"], []).append(x["balanced_accuracy"])
    return {f: float(np.mean(v)) for f, v in acc.items()}

def rank_biserial(a, b):
    """Matched-pairs rank-biserial = (#pos - #neg)/#nonzero of paired diffs."""
    d = a - b; nz = d[d != 0]
    if len(nz) == 0: return 0.0
    return (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz)

def bootstrap_ci(diff, n=5000):
    idx = np.arange(len(diff))
    means = [np.mean(diff[RNG.choice(idx, len(idx), replace=True)]) for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def compare(v1_dir, base_dir, label):
    a, b = per_fold(v1_dir), per_fold(base_dir)
    if a is None or b is None: return None
    k = sorted(set(a) & set(b))
    av = np.array([a[i] for i in k]); bv = np.array([b[i] for i in k])
    d = av - bv
    try: _, p = wilcoxon(av, bv)
    except ValueError: p = 1.0
    lo, hi = bootstrap_ci(d)
    return dict(label=label, n=len(k), v1=av.mean(), base=bv.mean(), delta=d.mean(),
                ci=(lo, hi), rb=rank_biserial(av, bv), p=p, wins=int((av > bv).sum()))

# --- family of comparisons (v1 vs EEGNet at full data + both learning curves + ablations) ---
C = []
C.append(compare("results/seed_iv", "results/seed_iv_eegnet", "SEED-IV full v1-vs-EEGNet"))
C.append(compare("results/tuab", "results/tuab_eegnet", "TUAB full v1-vs-EEGNet"))
C.append(compare("results/mumtaz_v1", "results/mumtaz_eegnet", "MDD full v1-vs-EEGNet"))
for f in ["f010", "f025", "f050"]:
    C.append(compare(f"results/h1/seed_iv_v1_{f}", f"results/h1/seed_iv_eegnet_{f}", f"SEED-IV {f} v1-vs-EEGNet"))
    C.append(compare(f"results/h1/mumtaz_v1_{f}", f"results/h1/mumtaz_eegnet_{f}", f"MDD {f} v1-vs-EEGNet"))
C.append(compare("results/seed_iv", "results/ablation/seed_iv_lstm_proto", "SEED-IV H2 AC-vs-LSTM"))
C.append(compare("results/seed_iv", "results/ablation/seed_iv_ac_softmax", "SEED-IV H3 proto-vs-softmax"))
# v1 vs 2nd baseline (ShallowConvNet), full data
C.append(compare("results/seed_iv", "results/seed_iv_shallowconv", "SEED-IV full v1-vs-ShallowConv"))
C.append(compare("results/tuab", "results/tuab_shallowconv", "TUAB full v1-vs-ShallowConv"))
C.append(compare("results/mumtaz_v1", "results/mumtaz_shallowconv", "MDD full v1-vs-ShallowConv"))
# capacity-matched control (BigCNN) at low data
C.append(compare("results/h1/mumtaz_v1_f010", "results/h1/mumtaz_bigcnn_f010", "MDD f010 v1-vs-BigCNN"))
C.append(compare("results/h1/mumtaz_v1_f025", "results/h1/mumtaz_bigcnn_f025", "MDD f025 v1-vs-BigCNN"))
C.append(compare("results/h1/seed_iv_v1_f010", "results/h1/seed_iv_bigcnn_f010", "SEED-IV f010 v1-vs-BigCNN"))
C.append(compare("results/h1/seed_iv_v1_f025", "results/h1/seed_iv_bigcnn_f025", "SEED-IV f025 v1-vs-BigCNN"))
# Cavanagh boundary cohort (BDI labels): expected non-significant at low data
C.append(compare("results/h1/cavanagh_v1_f010", "results/h1/cavanagh_bigcnn_f010", "Cavanagh f010 v1-vs-BigCNN"))
C.append(compare("results/h1/cavanagh_v1_f025", "results/h1/cavanagh_bigcnn_f025", "Cavanagh f025 v1-vs-BigCNN"))
C.append(compare("results/cavanagh_v1", "results/cavanagh_eegnet", "Cavanagh full v1-vs-EEGNet"))
C = [c for c in C if c]

# --- Holm-Bonferroni over the family ---
order = np.argsort([c["p"] for c in C]); m = len(C)
for rank, i in enumerate(order):
    C[i]["p_holm"] = min(1.0, C[i]["p"] * (m - rank))
# enforce monotonicity of Holm-adjusted p
prev = 0.0
for i in order:
    C[i]["p_holm"] = max(C[i]["p_holm"], prev); prev = C[i]["p_holm"]

print(f"{'comparison':<34}{'n':>3} {'v1':>7} {'base':>7} {'Δ':>7} {'95% CI':>17} {'r':>6} {'p':>9} {'p_holm':>9}")
for c in C:
    sig = "*" if c["p_holm"] < 0.05 else " "
    print(f"{c['label']:<34}{c['n']:>3} {c['v1']:>7.3f} {c['base']:>7.3f} {c['delta']:>+7.3f} "
          f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] {c['rb']:>+6.2f} {c['p']:>9.4f} {c['p_holm']:>8.4f}{sig}")
unit_note = ("one observation per fold (BACC averaged over the 3 seeds first)"
             if ARGS.unit == "fold" else
             "every (seed,fold) treated as independent -- inflates n threefold")
print(f"\nUnit of analysis: {unit_note}.")
print(f"Holm-Bonferroni over m={m} comparisons; * = significant at α=0.05 after correction.")
print("r = matched-pairs rank-biserial effect size; CI = subject/fold bootstrap 95% CI on mean Δ.")
