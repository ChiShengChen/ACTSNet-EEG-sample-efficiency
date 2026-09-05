"""TUAB (TUH Abnormal) adapter -> (N, C, S, T) cache.

Layout:
  .../tuh_eeg_abnormal/v3.0.1/edf/{train,eval}/{abnormal,normal}/01_tcp_ar/<subj>_sNNN_tNNN.edf
Label: abnormal=1, normal=0 (from path).  Subject: filename prefix (e.g. aaaaabdo).
Official train/eval split kept in `extra` for reference; LOSO regroups by subject.
"""

import argparse
import glob
import os
import mne

from eeg_prep import run_pipeline

# Corpus root: TUAB v3.0.1 (requires a TUH data use agreement).
# Set the TUAB_ROOT environment variable to point at your local copy.
TUAB_ROOT = os.environ.get("TUAB_ROOT", "")


def iter_tuab(root=TUAB_ROOT, splits=("train", "eval"), max_recordings=None,
              crop_sec=None):
    files = []
    for split in splits:
        for cls in ("abnormal", "normal"):
            files += sorted(glob.glob(os.path.join(root, split, cls, "**", "*.edf"),
                                      recursive=True))
    if max_recordings is not None:
        # interleave classes so a cap still yields both labels
        files = sorted(files, key=lambda p: os.path.basename(p))[:max_recordings]
    for f in files:
        parts = f.split(os.sep)
        label = 1 if "abnormal" in parts else 0
        split = "train" if "/train/" in f else "eval"
        subject = os.path.basename(f).split("_")[0]
        try:
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        except Exception as e:
            print(f"  skip (read error) {os.path.basename(f)}: {e}")
            continue
        if crop_sec is not None and raw.n_times / raw.info["sfreq"] > crop_sec:
            raw.crop(tmax=crop_sec)
        yield {
            "eeg": raw.get_data(),               # (C, T) in Volts
            "src_fs": raw.info["sfreq"],
            "ch_names": raw.ch_names,
            "label": label,
            "subject": subject,
            "extra": {"split": split, "file": os.path.basename(f)},
        }


def main():
    ap = argparse.ArgumentParser(description="Build TUAB (N,C,S,T) cache")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         os.pardir, "prep_cache", "tuab"))
    ap.add_argument("--montage", default="10_20_19")
    ap.add_argument("--dst_fs", type=int, default=256)
    ap.add_argument("--win_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=None)
    ap.add_argument("--windows_per_recording", type=int, default=5)
    ap.add_argument("--max_total_windows", type=int, default=20000)
    ap.add_argument("--max_recordings", type=int, default=None)
    ap.add_argument("--crop_sec", type=float, default=120.0,
                    help="read only first crop_sec of each recording (TUAB is ~20min)")
    ap.add_argument("--splits", nargs="+", default=["train", "eval"])
    args = ap.parse_args()

    records = iter_tuab(splits=tuple(args.splits),
                        max_recordings=args.max_recordings, crop_sec=args.crop_sec)
    run_pipeline(records, montage=args.montage, dst_fs=args.dst_fs,
                 win_sec=args.win_sec, stride_sec=args.stride_sec,
                 windows_per_recording=args.windows_per_recording,
                 max_total_windows=args.max_total_windows,
                 out_dir=args.out_dir, dataset_name="tuab")


if __name__ == "__main__":
    main()
