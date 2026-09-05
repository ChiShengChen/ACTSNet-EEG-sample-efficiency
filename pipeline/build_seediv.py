"""SEED-IV adapter -> (N, C, S, T) cache.

Layout: eeg_raw_data/{1,2,3}/<subjId>_<date>.mat ; each .mat has 24 trial arrays
keyed `<initials>_eegN` (N=1..24), shape (62, T_trial), downsampled to 200 Hz.
Labels are per-session (from ReadMe.txt), shared across subjects within a session.
Subject id = filename prefix number; session = parent dir. LOSO groups by subject.
Emotions: 0 neutral, 1 sad, 2 fear, 3 happy.
"""

import argparse
import glob
import os
import re
import numpy as np
import scipy.io as sio

from eeg_prep import run_pipeline

# Corpus root: SEED-IV, the eeg_raw_data directory.
# Set the SEEDIV_ROOT environment variable to point at your local copy.
SEEDIV_ROOT = os.environ.get("SEEDIV_ROOT", "")
SRC_FS = 200  # SEED-IV eeg_raw_data is downsampled to 200 Hz

# 62-channel order from channel_62_pos.locs (used as ch_names for montage mapping)
SEEDIV_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3",
    "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2",
    "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1",
    "O1", "Oz", "O2", "CB2",
]

SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


def iter_seediv(root=SEEDIV_ROOT, sessions=(1, 2, 3), max_recordings=None):
    n = 0
    for sess in sessions:
        labels = SESSION_LABELS[sess]
        for f in sorted(glob.glob(os.path.join(root, str(sess), "*.mat"))):
            subject = os.path.basename(f).split("_")[0]   # e.g. "10"
            m = sio.loadmat(f)
            keys = [k for k in m if not k.startswith("__")]
            # order trials by their numeric suffix eegN
            def trial_no(k):
                mm = re.search(r"_eeg(\d+)$", k)
                return int(mm.group(1)) if mm else 0
            keys = sorted(keys, key=trial_no)
            for k in keys:
                t = trial_no(k)                # 1..24
                if t < 1 or t > 24:
                    continue
                eeg = np.asarray(m[k], dtype=np.float64)   # (62, T_trial)
                if eeg.shape[0] != len(SEEDIV_CHANNELS):
                    print(f"  skip {os.path.basename(f)}:{k} unexpected n_ch {eeg.shape[0]}")
                    continue
                yield {
                    "eeg": eeg,
                    "src_fs": SRC_FS,
                    "ch_names": SEEDIV_CHANNELS,
                    "label": int(labels[t - 1]),
                    "subject": subject,  # group all 3 sessions per subject for LOSO
                    "extra": {"session": sess, "trial": t},
                }
            n += 1
            if max_recordings is not None and n >= max_recordings:
                return


def main():
    ap = argparse.ArgumentParser(description="Build SEED-IV (N,C,S,T) cache")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         os.pardir, "prep_cache", "seed_iv"))
    ap.add_argument("--montage", default="10_20_19")
    ap.add_argument("--dst_fs", type=int, default=256)
    ap.add_argument("--win_sec", type=float, default=4.0,
                    help="SEED-IV convention is 4 s windows")
    ap.add_argument("--stride_sec", type=float, default=None)
    ap.add_argument("--windows_per_recording", type=int, default=8)
    ap.add_argument("--max_total_windows", type=int, default=20000)
    ap.add_argument("--max_recordings", type=int, default=None)
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--norm", default="perrow", choices=["perrow", "global", "none"])
    args = ap.parse_args()

    records = iter_seediv(sessions=tuple(args.sessions),
                          max_recordings=args.max_recordings)
    run_pipeline(records, montage=args.montage, dst_fs=args.dst_fs,
                 win_sec=args.win_sec, stride_sec=args.stride_sec,
                 windows_per_recording=args.windows_per_recording,
                 max_total_windows=args.max_total_windows,
                 out_dir=args.out_dir, dataset_name="seed_iv", norm=args.norm)


if __name__ == "__main__":
    main()
