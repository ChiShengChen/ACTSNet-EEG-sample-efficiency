"""Cavanagh MDD adapter (OpenNeuro ds003478) -> (N, C, S, T) cache.

Second public depression cohort (resting-state EEG), for reproducing the low-data advantage
beyond Mumtaz. BIDS / EEGLAB .set files: sub-XXX/eeg/sub-XXX_task-Rest_run-*_eeg.set.
Labels from participants.tsv Beck Depression Inventory (BDI):
  HC  (label 0): BDI < hc_max (default 7)
  MDD (label 1): BDI > mdd_min (default 13)
  subthreshold (hc_max<=BDI<=mdd_min): excluded.
Subject id = participant_id (sub-XXX). Resting -> 10 s windows, like the Mumtaz clinical setting.
"""

import argparse
import csv
import glob
import os
import mne

from eeg_prep import run_pipeline

# Corpus root: Cavanagh et al. resting-state MDD.
# Set the CAV_ROOT environment variable to point at your local copy.
CAV_ROOT = os.environ.get("CAV_ROOT", "")


def load_labels(root, hc_max, mdd_min):
    lab = {}
    with open(os.path.join(root, "participants.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pid = row["participant_id"]
            try:
                bdi = float(row.get("BDI", "nan"))
            except ValueError:
                continue
            if bdi < hc_max:
                lab[pid] = 0
            elif bdi > mdd_min:
                lab[pid] = 1
            # else subthreshold -> skip
    return lab


def iter_cavanagh(root=CAV_ROOT, hc_max=7, mdd_min=13, crop_sec=None, max_recordings=None):
    labels = load_labels(root, hc_max, mdd_min)
    files = sorted(glob.glob(os.path.join(root, "sub-*", "eeg", "*task-Rest*_eeg.set")))
    n = 0
    for f in files:
        pid = os.path.basename(f).split("_")[0]        # sub-XXX
        if pid not in labels:
            continue                                    # subthreshold / no BDI
        try:
            raw = mne.io.read_raw_eeglab(f, preload=True, verbose="ERROR")
        except Exception as e:
            print(f"  skip (read error) {os.path.basename(f)}: {e}"); continue
        if crop_sec is not None and raw.n_times / raw.info["sfreq"] > crop_sec:
            raw.crop(tmax=crop_sec)
        yield {
            "eeg": raw.get_data(), "src_fs": raw.info["sfreq"], "ch_names": raw.ch_names,
            "label": labels[pid], "subject": pid,
            "extra": {"file": os.path.basename(f)},
        }
        n += 1
        if max_recordings is not None and n >= max_recordings:
            return


def main():
    ap = argparse.ArgumentParser(description="Build Cavanagh MDD (N,C,S,T) cache")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         os.pardir, "prep_cache", "cavanagh"))
    ap.add_argument("--montage", default="10_20_19")
    ap.add_argument("--dst_fs", type=int, default=256)
    ap.add_argument("--win_sec", type=float, default=10.0)
    ap.add_argument("--windows_per_recording", type=int, default=20)
    ap.add_argument("--max_total_windows", type=int, default=20000)
    ap.add_argument("--hc_max", type=float, default=7)
    ap.add_argument("--mdd_min", type=float, default=13)
    ap.add_argument("--crop_sec", type=float, default=None)
    ap.add_argument("--max_recordings", type=int, default=None)
    args = ap.parse_args()

    records = iter_cavanagh(hc_max=args.hc_max, mdd_min=args.mdd_min,
                            crop_sec=args.crop_sec, max_recordings=args.max_recordings)
    run_pipeline(records, montage=args.montage, dst_fs=args.dst_fs, win_sec=args.win_sec,
                 windows_per_recording=args.windows_per_recording,
                 max_total_windows=args.max_total_windows,
                 out_dir=args.out_dir, dataset_name="cavanagh")


if __name__ == "__main__":
    main()
