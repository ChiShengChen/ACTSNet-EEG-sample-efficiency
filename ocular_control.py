"""Ocular-contamination control on the Mumtaz MDD cohort (CPU only).

Question: could group-discriminative non-neural signal (blinks / eye movements, which load
on frontal low frequencies) explain the MDD result? Two facts bound the risk before any
analysis: the pipeline z-scores every (channel, sub-band) row within each window, so
absolute power cannot reach the classifier; and every model in the study saw the same
input. What can still leak is waveform shape. This script quantifies, on the *raw*
eyes-closed recordings, how separable the groups are on the classic ocular signatures:

  1. frontal (Fp1, Fp2, F7, F8) delta (0.5-4 Hz) and theta (4-8 Hz) log-power, per subject
  2. the same at a posterior reference (P3, Pz, P4, O1, O2)
  3. ocular-band variance: variance of the 0.5-4 Hz band-passed Fp1/Fp2 signal
  4. blink rate proxy: count of Fp1/Fp2 peaks > 4 SD in the 0.5-4 Hz band, per minute

Each feature is scored as a single-feature subject-level classifier (AUROC over the 58
subjects; Mann-Whitney U). If frontal delta separated the groups as well as the models do
(subject-level AUROC 0.95), an ocular confound would be plausible; if it is near 0.5, it
is not. Output: results/ocular_control.json and a printed table.
"""
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, find_peaks
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from build_mumtaz import iter_mumtaz          # noqa: E402
from montages import build_index_map          # noqa: E402

FS = 256
FRONTAL = ["FP1", "FP2", "F7", "F8"]
POSTERIOR = ["P3", "PZ", "P4", "O1", "O2"]
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0)}


def resample_to(x, src_fs, dst_fs):
    if abs(src_fs - dst_fs) < 1e-6:
        return x
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(round(src_fs)), int(round(dst_fs)))
    return resample_poly(x, int(round(dst_fs)) // g, int(round(src_fs)) // g, axis=-1)


def bandpass(x, lo, hi, fs=FS):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


rows = []
for rec in iter_mumtaz(modality="EC"):
    idx, names = build_index_map(rec["ch_names"], "10_20_19")
    x = resample_to(rec["eeg"][idx], rec["src_fs"], FS)
    x = bandpass(x, 0.5, 45.0)                      # same broadband limits as the pipeline
    names = [n.upper() for n in names]
    f, pxx = welch(x, fs=FS, nperseg=FS * 4, axis=-1)
    def logpow(chs, band):
        ci = [names.index(c) for c in chs if c in names]
        lo, hi = BANDS[band]
        m = (f >= lo) & (f < hi)
        return float(np.log10(pxx[ci][:, m].mean()))
    feat = {"subject": rec["subject"], "label": rec["label"]}
    for band in BANDS:
        feat[f"frontal_{band}"] = logpow(FRONTAL, band)
        feat[f"posterior_{band}"] = logpow(POSTERIOR, band)
    fp = [names.index(c) for c in ("FP1", "FP2") if c in names]
    ocular = bandpass(x[fp], 0.5, 4.0)
    feat["ocular_var"] = float(np.log10(ocular.var()))
    thr = 4 * ocular.std()
    peaks = sum(len(find_peaks(np.abs(ch), height=thr, distance=FS // 4)[0]) for ch in ocular)
    feat["blink_rate_per_min"] = float(peaks / (ocular.shape[-1] / FS / 60))
    rows.append(feat)

y = np.array([r["label"] for r in rows])
out = {"n_subjects": len(rows), "n_mdd": int(y.sum()), "features": {}}
print(f"Mumtaz EC, {len(rows)} subjects ({int(y.sum())} MDD). Single-feature subject-level separability:")
print(f"{'feature':24s}{'MDD mean':>10}{'HC mean':>10}{'AUROC':>8}{'MW p':>9}")
for k in [k for k in rows[0] if k not in ("subject", "label")]:
    v = np.array([r[k] for r in rows])
    auc = roc_auc_score(y, v); auc = max(auc, 1 - auc)          # direction-free
    p = mannwhitneyu(v[y == 1], v[y == 0]).pvalue
    out["features"][k] = {"mdd_mean": float(v[y == 1].mean()), "hc_mean": float(v[y == 0].mean()),
                          "auroc_direction_free": float(auc), "mannwhitney_p": float(p)}
    print(f"{k:24s}{v[y == 1].mean():>10.3f}{v[y == 0].mean():>10.3f}{auc:>8.3f}{p:>9.3f}")
out["reference"] = {"model_subject_level_auroc": 0.954,
                    "note": "pipeline z-scores each (channel, sub-band) row per window, so absolute power never reaches the classifier"}
os.makedirs("results", exist_ok=True)
json.dump({"rows": rows, **out}, open("results/ocular_control.json", "w"), indent=2)
print("-> results/ocular_control.json")
