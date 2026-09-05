# ACTSNet preprocessing pipeline — TUAB & SEED-IV

Turns raw EEG into the `(N, C, S, T)` contract from
[../00_shared_infrastructure.md](../00_shared_infrastructure.md). Same signal feeds
both models: **v2** takes `(N,C,S,T)` directly; **v1** takes the flattened
`(N, C*S, T)` view (`eeg_prep.flatten_cst`).

## Env
```bash
conda activate pytorch291        # torch 2.9.1+cu128, mne 1.11, scipy 1.17, sklearn 1.7
```

## Files
| File | Role |
|---|---|
| `eeg_prep.py` | shared core: resample→notch→0.5–45 bandpass→5 sub-bands→window→z-score→save |
| `montages.py` | canonical 10-20 names; `10_20_19` (19 ch) and `frontal7` (7 ch) |
| `build_tuab.py` | TUAB adapter (EDF, abnormal/normal) |
| `build_seediv.py` | SEED-IV adapter (.mat, 4 emotions) |

## Output (per dataset, in `../prep_cache/<ds>/`)
`data.npy (N,C,S,T) f32` · `labels.npy (N,)` · `subjects.npy (N,)` · `meta.json`

`S=5` sub-bands = δ/θ/α/β/γ. `T = win_sec × 256`. With `--montage frontal7`,
`C·S = 7×5 = 35` — **identical to the original ACTSNet thesis input**.

## Smoke test (validated)
```bash
python build_tuab.py   --max_recordings 8 --crop_sec 60 --windows_per_recording 3 --out_dir /tmp/tuab_tiny
python build_seediv.py --max_recordings 2 --windows_per_recording 4 --out_dir /tmp/seediv_tiny
# -> tuab (24,19,5,2560), seed_iv (192,19,5,1024); both load + forward through ACTSNet v1.
```

## Full runs (19-ch, journal cache)
```bash
# TUAB: ~2993 recordings. crop_sec limits read length; caps bound RAM/disk.
python build_tuab.py --out_dir ../prep_cache/tuab \
    --montage 10_20_19 --win_sec 10 --windows_per_recording 5 \
    --crop_sec 120 --max_total_windows 20000

# SEED-IV: 15 subj × 3 sessions × 24 trials, 4 s windows.
python build_seediv.py --out_dir ../prep_cache/seed_iv \
    --montage 10_20_19 --win_sec 4 --windows_per_recording 8 \
    --max_total_windows 20000
```
Add `--montage frontal7` to build the Paper-1 frontal ablation cache (`prep_cache/tuab_frontal7`, etc).

## Key parameters
| Flag | Meaning | Default |
|---|---|---|
| `--montage` | `10_20_19` or `frontal7` | `10_20_19` |
| `--dst_fs` | resample target | 256 |
| `--win_sec` | window length (T = win_sec·fs) | TUAB 10 / SEED-IV 4 |
| `--windows_per_recording` | cap windows per recording | 5 / 8 |
| `--max_total_windows` | global cap (RAM/disk guard) | 20000 |
| `--crop_sec` (TUAB) | read only first N s per EDF | 120 |
| `--max_recordings` | cap recordings (debug) | none |

## Notes / current limitations
- **ICA not yet applied.** Core does notch + broadband + sub-band only. ICA artifact
  removal (infra doc §3 step 5) is a planned add-on; bandpass + per-window z-score is the
  current denoiser. Flag when reporting.
- **Memory:** one window ≈ `C·S·T·4` bytes (19×5×2560×4 ≈ 0.93 MB). 20k windows ≈ 19 GB
  on disk/RAM — raise/lower `--max_total_windows` to taste; for the full corpus, switch the
  driver to a memmap two-pass (TODO in `eeg_prep.run_pipeline`).
- **TUAB fs varies** (250/256); resampled uniformly to `dst_fs`. Non-EEG channels
  (EKG/PHOTIC/IBI/…) auto-dropped by `montages.normalize_name`.
- **LOSO:** group by `subjects.npy`. TUAB subject = EDF filename prefix; SEED-IV subject =
  numeric id (all 3 sessions of a subject share the id → no cross-session leakage within a fold).
