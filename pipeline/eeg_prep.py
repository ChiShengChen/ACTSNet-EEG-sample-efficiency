"""Shared EEG preprocessing core for the ACTSNet papers.

Emits the tensor contract from 00_shared_infrastructure.md:
    data.npy     (N, C, S, T)  float32   channels x sub-bands x time
    labels.npy   (N,)          int64
    subjects.npy (N,)          str        for LOSO grouping (no subject leakage)
    meta.json                  provenance + shapes

ACTSNet v2 consumes (N, C, S, T) directly; ACTSNet v1 wants (N, C*S, T) — use
`flatten_cst()` (or load with `--flatten`) for the v1 view of byte-identical signal.

Per-dataset adapters (build_tuab.py, build_seediv.py) supply an iterator of
(eeg, src_fs, raw_ch_names, label, subject) records; everything else is shared.
"""

import json
import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from math import gcd

from montages import build_index_map

# 5 EEG sub-bands (Hz) — matches ACTSNet v2 README / run_eegfm_benchmark SUBBANDS.
SUBBANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


# ── signal ops (operate on (C, T) arrays, zero-phase) ───────────────────────
def resample_to(x, src_fs, dst_fs):
    if int(src_fs) == int(dst_fs):
        return x
    g = gcd(int(round(src_fs)), int(round(dst_fs)))
    up, down = int(round(dst_fs)) // g, int(round(src_fs)) // g
    return resample_poly(x, up, down, axis=-1)


def notch_lines(x, fs, freqs=(50.0, 60.0), q=30.0):
    nyq = fs / 2.0
    for f0 in freqs:
        if f0 < nyq:
            b, a = iirnotch(f0 / nyq, q)
            x = filtfilt(b, a, x, axis=-1)
    return x


def _bandpass(x, fs, lo, hi, order=4):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    # padlen guard for short signals
    padlen = 3 * max(len(a), len(b))
    if x.shape[-1] <= padlen:
        return None
    return filtfilt(b, a, x, axis=-1)


def subband_decompose(x, fs):
    """(C, T) -> (C, S, T): apply the 5 sub-band bandpass filters (zero-phase)."""
    bands = []
    for _, lo, hi in SUBBANDS:
        y = _bandpass(x, fs, lo, hi)
        if y is None:
            return None
        bands.append(y)
    return np.stack(bands, axis=1)  # (C, S, T)


def zscore_per_window(seg, eps=1e-6, mode="perrow"):
    """(C, S, T) normalization.
    mode='perrow'  : z-score each (channel, sub-band) row over time (v1; DESTROYS relative
                     inter-band energy).
    mode='global'  : one mean/std over the whole (C,S,T) window (v2; PRESERVES the relative
                     channel x sub-band energy that SubBandFusion/FreqLens rely on).
    mode='none'    : raw band-passed signal (let the model's RevIN normalize).
    """
    if mode == "none":
        return seg
    if mode == "global":
        return (seg - seg.mean()) / (seg.std() + eps)
    mu = seg.mean(axis=-1, keepdims=True)
    sd = seg.std(axis=-1, keepdims=True)
    return (seg - mu) / (sd + eps)


def segment_windows(cst, win_T, stride_T, max_windows=None):
    """(C, S, T_full) -> (n, C, S, win_T) non-/overlapping windows along time."""
    C, S, T = cst.shape
    starts = list(range(0, T - win_T + 1, stride_T))
    if max_windows is not None:
        starts = starts[:max_windows]
    return np.stack([cst[:, :, s:s + win_T] for s in starts], axis=0) if starts \
        else np.empty((0, C, S, win_T), dtype=cst.dtype)


def flatten_cst(data):
    """(N, C, S, T) -> (N, C*S, T) view for ACTSNet v1."""
    n, c, s, t = data.shape
    return data.reshape(n, c * s, t)


# ── driver ──────────────────────────────────────────────────────────────────
def run_pipeline(records, *, montage, dst_fs=256, win_sec=10.0, stride_sec=None,
                 windows_per_recording=None, max_total_windows=None,
                 out_dir, dataset_name, dtype="float32", log_every=50, norm="perrow"):
    """Consume an iterator of records and write the (N,C,S,T) cache.

    Each record: dict(eeg=(C_raw,T_raw) float, src_fs, ch_names=[...],
                       label=int, subject=str, extra=dict(optional)).
    Returns the meta dict.
    """
    stride_sec = stride_sec if stride_sec is not None else win_sec
    win_T = int(round(win_sec * dst_fs))
    stride_T = int(round(stride_sec * dst_fs))

    data_chunks, labels, subjects, extras = [], [], [], []
    n_seen = n_used = n_skipped = total_windows = 0
    skip_reasons = {}

    for rec in records:
        n_seen += 1
        eeg, src_fs, ch_names = rec["eeg"], rec["src_fs"], rec["ch_names"]
        try:
            idx, canon = build_index_map(ch_names, montage)
        except KeyError as e:
            n_skipped += 1
            skip_reasons["montage"] = skip_reasons.get("montage", 0) + 1
            continue
        x = np.asarray(eeg, dtype=np.float64)[idx]          # (C, T_raw)
        x = resample_to(x, src_fs, dst_fs)                   # (C, T)
        x = notch_lines(x, dst_fs)
        bb = _bandpass(x, dst_fs, 0.5, 45.0)                 # broadband clean
        if bb is None:
            n_skipped += 1
            skip_reasons["too_short"] = skip_reasons.get("too_short", 0) + 1
            continue
        cst = subband_decompose(bb, dst_fs)                  # (C, S, T)
        if cst is None:
            n_skipped += 1
            skip_reasons["too_short"] = skip_reasons.get("too_short", 0) + 1
            continue
        wins = segment_windows(cst, win_T, stride_T, windows_per_recording)
        if wins.shape[0] == 0:
            n_skipped += 1
            skip_reasons["no_window"] = skip_reasons.get("no_window", 0) + 1
            continue
        wins = np.stack([zscore_per_window(w, mode=norm) for w in wins]).astype(dtype)

        if max_total_windows is not None and total_windows + wins.shape[0] > max_total_windows:
            wins = wins[: max_total_windows - total_windows]
        data_chunks.append(wins)
        labels.extend([rec["label"]] * wins.shape[0])
        subjects.extend([str(rec["subject"])] * wins.shape[0])
        extras.extend([rec.get("extra", {})] * wins.shape[0])
        total_windows += wins.shape[0]
        n_used += 1
        if n_used % log_every == 0:
            print(f"  [{dataset_name}] {n_used} recordings used, {total_windows} windows")
        if max_total_windows is not None and total_windows >= max_total_windows:
            print(f"  [{dataset_name}] reached max_total_windows={max_total_windows}, stopping")
            break

    if not data_chunks:
        raise RuntimeError(f"No windows produced for {dataset_name}. skip_reasons={skip_reasons}")

    data = np.concatenate(data_chunks, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects)
    C, S, T = data.shape[1:]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "data.npy"), data)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "subjects.npy"), subjects)

    uniq, counts = np.unique(labels, return_counts=True)
    meta = {
        "dataset": dataset_name,
        "shape_NCST": list(data.shape),
        "C": C, "S": S, "T": T,
        "channels": MONTAGE_NAMES(montage),
        "subbands": [b[0] for b in SUBBANDS],
        "dst_fs": dst_fs, "win_sec": win_sec, "stride_sec": stride_sec,
        "n_recordings_seen": n_seen, "n_recordings_used": n_used,
        "n_recordings_skipped": n_skipped, "skip_reasons": skip_reasons,
        "n_windows": int(data.shape[0]),
        "n_subjects": int(len(np.unique(subjects))),
        "class_counts": {int(k): int(v) for k, v in zip(uniq, counts)},
        "dtype": dtype,
        "v1_view": "data.reshape(N, C*S, T) -> (N, %d, %d)" % (C * S, T),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{dataset_name}] DONE -> {out_dir}")
    print(f"  data {data.shape} {dtype} | subjects {meta['n_subjects']} | classes {meta['class_counts']}")
    return meta


def MONTAGE_NAMES(montage):
    from montages import MONTAGES
    return MONTAGES[montage] if isinstance(montage, str) else list(montage)
