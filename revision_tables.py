"""Supplementary tables recomputed from the stored per-fold results.

Nothing here re-trains anything: every number is derived from the `per_fold.json`
files in `results/`, so each table is reproducible from this repository alone.

  --table auroc    AUROC for every model, at the fold level, on all four corpora
  --table robust   the full test-time robustness grid (both perturbation families,
                   both datasets), of which only part appears in the paper
  --table dup      diagnostic for the batch-size defect described in the README:
                   evidence that the Mumtaz low-fraction baselines never trained,
                   alongside the corrected curve
  --table all      (default)
"""
import argparse
import json
import os

import numpy as np

R = "results"


def load(d):
    p = f"{d}/per_fold.json"
    return json.load(open(p)) if os.path.exists(p) else None


def fold_mean(recs, key):
    """One observation per fold (averaged over seeds), then mean and SD across folds."""
    acc = {}
    for x in recs:
        v = x.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        acc.setdefault(x["fold"], []).append(v)
    if not acc:
        return None
    v = np.array([np.mean(a) for a in acc.values()])
    return v.mean(), v.std(), len(v)


def table_auroc():
    print("=" * 78)
    print("AUROC by model (window level, full training data)")
    print("=" * 78)
    print("Reported for every model, not only ACTSNet. These values were always present")
    print("in the `auc` field of per_fold.json; no retraining is needed to obtain them.\n")
    print("Note the level of analysis: these are window-level AUROCs. The subject-level")
    print("AUROC quoted for the Mumtaz cohort in the text aggregates windows per subject")
    print("first, and the two must not be placed in the same column.\n")
    DS = {"seed_iv": ("results/seed_iv", 4), "tuab": ("results/tuab", 2),
          "mumtaz": ("results/mumtaz_v1", 2), "cavanagh": ("results/cavanagh_v1", 2)}
    models = ["eegnet", "shallowconv", "bigcnn", "transformer", "tapnet"]
    print(f"{'dataset':10s}{'k':>4}  {'ACTSNet':>15}" + "".join(f"{m:>15}" for m in models))
    for ds, (v1dir, k) in DS.items():
        row = [f"{ds:10s}{k:>4}"]
        for d in [v1dir] + [f"results/{ds}_{m}" for m in models]:
            recs = load(d)
            r = fold_mean(recs, "auc") if recs else None
            row.append(f"{r[0]:>9.3f}+-{r[1]:.3f}" if r else f"{'--':>15}")
        print("  ".join(row))
    print("\nFor the 4-class SEED-IV task AUROC is not uniquely defined; `safe_auc` uses")
    print("one-vs-rest with macro averaging, and that scheme should be stated wherever")
    print("the value is quoted.")


def table_robust():
    print("=" * 78)
    print("Test-time robustness, complete grid")
    print("=" * 78)
    print("Both perturbation families (additive Gaussian noise and channel dropout) on")
    print("both datasets. All of it was produced by the same run; the paper reports only")
    print("the Gaussian-noise column on the MDD cohort.\n")
    conds = ["clean", "noise0.25", "noise0.50", "noise1.00", "chdrop0.2", "chdrop0.4"]
    print(f"{'dataset/model':22s}{'folds':>6}" + "".join(f"{c:>12}" for c in conds))
    for ds in ["mumtaz", "seed_iv"]:
        for m in ["v1", "eegnet"]:
            p = f"{R}/h5/{ds}_{m}/summary.json"
            if not os.path.exists(p):
                continue
            s = json.load(open(p))
            cells = "".join(f"{s['mean'][c]:>7.3f}+-{s['std'][c]:.2f}" if c in s["mean"]
                            else f"{'--':>12}" for c in conds)
            print(f"{ds + '/' + m:22s}{s.get('n_folds', '?'):>6}{cells}")
    print("\nTwo notes on reading this table:")
    print("  * The robustness runs use a single seed over all folds, and their clean-")
    print("    condition values are therefore not interchangeable with the main table's.")
    print("  * On the MDD cohort EEGNet is at least as robust as ACTSNet under every")
    print("    perturbation, so the attentional convolution does not buy test-time")
    print("    robustness; the sample-efficiency and robustness questions are separate.")


def table_dup():
    print("=" * 78)
    print("Diagnostic: the Mumtaz low-fraction baselines were never trained")
    print("=" * 78)
    print("At 10% and 25% of training subjects the baselines scored almost exactly 0.50")
    print("with negligible dispersion, which looks like a degenerate single-class")
    print("predictor. The real cause is worse and is visible in the stored results.\n")
    for m in ["eegnet", "bigcnn", "v1"]:
        a, b = load(f"{R}/h1/mumtaz_{m}_f010"), load(f"{R}/h1/mumtaz_{m}_f025")
        if not (a and b):
            continue
        da = {(x["seed"], x["fold"]): x["balanced_accuracy"] for x in a}
        db = {(x["seed"], x["fold"]): x["balanced_accuracy"] for x in b}
        k = sorted(set(da) & set(db))
        same = sum(1 for i in k if da[i] == db[i])
        flag = "   <- bit-identical; impossible for two independent training runs" \
            if same == len(k) else ""
        print(f"  mumtaz {m:8s} test BACC identical between f0.10 and f0.25: "
              f"{same}/{len(k)}{flag}")
    print("\nCause: the baselines were launched with --batch_size 256, but the Mumtaz cache")
    print("holds only ~20 windows per subject. At 10% and 25% of subjects the inner-training")
    print("set (80 and 200 windows) is smaller than a single batch, and with drop_last=True")
    print("each epoch yields zero batches. The models were evaluated at their random")
    print("initialisation, which is identical across the two fractions because the seed is")
    print("fixed -- hence identical predictions.\n")
    print("Corrected runs, all models at a matched batch size (results/h1_recheck/):")
    print(f"  {'frac':>6}{'ACTSNet':>16}{'EEGNet':>16}{'BigCNN':>16}")
    for tag, fr in [("f010", 0.10), ("f025", 0.25), ("f050", 0.50), ("f100", 1.00)]:
        row = [f"  {fr:>6.2f}"]
        for m in ["v1", "eegnet", "bigcnn"]:
            recs = load(f"{R}/h1_recheck/mumtaz_{m}_{tag}")
            r = fold_mean(recs, "balanced_accuracy") if recs else None
            row.append(f"{r[0]:>10.3f}+-{r[1]:.3f}" if r else f"{'(pending)':>16}")
        print("".join(row))
    print("\nOnce the baselines actually train, EEGNet reaches 0.641 at the 10% fraction")
    print("rather than chance, and no fraction shows a significant advantage for ACTSNet.")
    print("A trainability gate in run_loso.py now aborts any run that would yield zero")
    print("batches per epoch, and logs the batch count for every fold.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="all", choices=["all", "auroc", "robust", "dup"])
    a = ap.parse_args()
    for name, fn in [("dup", table_dup), ("robust", table_robust), ("auroc", table_auroc)]:
        if a.table in ("all", name):
            fn(); print()
