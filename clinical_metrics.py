"""Subject-level clinical metrics for a binary depression run with --dump_preds.
Aggregates window-level P(MDD) to a per-subject mean probability, then reports
sensitivity/specificity at the Youden-optimal and 0.5 operating points, AUROC with
subject bootstrap 95% CI, and accuracy. Usage: python clinical_metrics.py results/mumtaz_v1_preds"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

RNG = np.random.RandomState(0)
d = sys.argv[1]
rows = json.load(open(f"{d}/per_fold.json"))

# pool windows across folds/seeds, aggregate to subject-level mean P(class1)
subj_p, subj_y = {}, {}
for r in rows:
    if "prob1" not in r or "subjects" not in r:
        continue
    for s, p, y in zip(r["subjects"], r["prob1"], r["labels"]):
        subj_p.setdefault(s, []).append(p)
        subj_y[s] = y
subs = sorted(subj_p)
p = np.array([np.mean(subj_p[s]) for s in subs])
y = np.array([subj_y[s] for s in subs])
print(f"=== subject-level clinical metrics ({len(subs)} subjects, {int(y.sum())} positive) ===")

auc = roc_auc_score(y, p)
# subject bootstrap CI for AUROC
aucs = []
idx = np.arange(len(y))
for _ in range(2000):
    b = RNG.choice(idx, len(idx), replace=True)
    if len(np.unique(y[b])) == 2:
        aucs.append(roc_auc_score(y[b], p[b]))
lo, hi = np.percentile(aucs, [2.5, 97.5])
print(f"AUROC: {auc:.3f}  (95% CI {lo:.3f}-{hi:.3f})")

fpr, tpr, thr = roc_curve(y, p)
j = np.argmax(tpr - fpr)                      # Youden's J
def sens_spec(t):
    pred = (p >= t).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    acc = (tp + tn) / len(y)
    return sens, spec, acc
for name, t in [("Youden-optimal", thr[j]), ("0.5 threshold", 0.5)]:
    se, sp, ac = sens_spec(t)
    print(f"{name:16s} (thr={t:.3f}): sensitivity {se:.3f}  specificity {sp:.3f}  accuracy {ac:.3f}")
