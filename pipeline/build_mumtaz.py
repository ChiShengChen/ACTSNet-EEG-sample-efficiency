"""Mumtaz 2016 MDD adapter -> (N, C, S, T) cache.

Dataset: "MDD Patients and Healthy Controls EEG Data" (Mumtaz 2016, figshare 4244171).
33 MDD + 30 healthy, 19-ch 10-20 (linked-ear ref), 256 Hz, 5-min eyes-closed (EC) /
eyes-open (EO) / TASK sessions. This is ACTSNet v1's *native* clinical task (MDD).

Files: "<group> S<n> <modality>.edf", group in {H, MDD}, modality in {EC, EO, TASK}.
Label: H=0 (healthy), MDD=1.  Subject id = "<group>_S<n>" (H and MDD numbering are
disjoint people). Default modality EC (eyes-closed resting — standard for MDD EEG).
"""

import argparse
import glob
import os
import mne

from eeg_prep import run_pipeline

# Corpus root: Mumtaz 2016 MDD (figshare 4244171).
# Set the MUMTAZ_ROOT environment variable to point at your local copy.
MUMTAZ_ROOT = os.environ.get("MUMTAZ_ROOT", "")


def iter_mumtaz(root=MUMTAZ_ROOT, modality="EC", crop_sec=None, max_recordings=None):
    files = sorted(glob.glob(os.path.join(root, "**", f"* {modality}.edf"), recursive=True))
    if max_recordings is not None:
        files = files[:max_recordings]
    for f in files:
        base = os.path.basename(f)[:-4]            # strip .edf
        parts = base.split()                        # ["H"|"MDD", "S<n>", modality]
        if len(parts) < 3:
            print(f"  skip (unparsed name) {base}"); continue
        group = parts[0].upper()
        if group not in ("H", "MDD"):
            print(f"  skip (unknown group) {base}"); continue
        label = 1 if group == "MDD" else 0
        subject = f"{group}_{parts[1]}"
        try:
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        except Exception as e:
            print(f"  skip (read error) {base}: {e}"); continue
        if crop_sec is not None and raw.n_times / raw.info["sfreq"] > crop_sec:
            raw.crop(tmax=crop_sec)
        yield {
            "eeg": raw.get_data(), "src_fs": raw.info["sfreq"], "ch_names": raw.ch_names,
            "label": label, "subject": subject,
            "extra": {"group": group, "modality": modality, "file": base},
        }


def main():
    ap = argparse.ArgumentParser(description="Build Mumtaz MDD (N,C,S,T) cache")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         os.pardir, "prep_cache", "mumtaz"))
    ap.add_argument("--montage", default="10_20_19")
    ap.add_argument("--modality", default="EC", choices=["EC", "EO", "TASK"])
    ap.add_argument("--dst_fs", type=int, default=256)
    ap.add_argument("--win_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=None)
    ap.add_argument("--windows_per_recording", type=int, default=20)
    ap.add_argument("--max_total_windows", type=int, default=20000)
    ap.add_argument("--crop_sec", type=float, default=None)
    ap.add_argument("--max_recordings", type=int, default=None)
    args = ap.parse_args()

    records = iter_mumtaz(modality=args.modality, crop_sec=args.crop_sec,
                          max_recordings=args.max_recordings)
    run_pipeline(records, montage=args.montage, dst_fs=args.dst_fs,
                 win_sec=args.win_sec, stride_sec=args.stride_sec,
                 windows_per_recording=args.windows_per_recording,
                 max_total_windows=args.max_total_windows,
                 out_dir=args.out_dir, dataset_name="mumtaz")


if __name__ == "__main__":
    main()
