#!/usr/bin/env bash
# H4 frontal-7 montage ablation: 7 frontal electrodes (FP1 FP2 F7 F3 Fz F4 F8) -> C=7,
# C*S=35 = the original 2021 thesis input. Build frontal-7 caches, then v1 LOSO on each.
# Compare to the full 19-ch results (SEED-IV 0.349, TUAB 0.774). Idempotent.
#
# Durable launch:
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_h4_frontal7.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/h4_frontal7.log
echo "=== H4 frontal-7 START $(date) ===" >> "$LOG"

# 1) Build frontal-7 caches (skip if present)
if [ ! -f prep_cache/seed_iv_frontal7/meta.json ]; then
  echo ">>> build seed_iv frontal7 $(date)" >> "$LOG"
  ( cd pipeline && "$PY" build_seediv.py --montage frontal7 --win_sec 4 \
      --windows_per_recording 8 --max_total_windows 20000 \
      --out_dir ../prep_cache/seed_iv_frontal7 ) >> "$LOG" 2>&1
fi
if [ ! -f prep_cache/tuab_frontal7/meta.json ]; then
  echo ">>> build tuab frontal7 $(date)" >> "$LOG"
  ( cd pipeline && "$PY" build_tuab.py --montage frontal7 --win_sec 10 --crop_sec 120 \
      --windows_per_recording 5 --max_total_windows 20000 \
      --out_dir ../prep_cache/tuab_frontal7 ) >> "$LOG" 2>&1
fi

# 2) v1 LOSO on frontal-7 (same protocol/seeds as the full-montage runs)
if [ ! -f results/h4/seed_iv_frontal7/summary.json ]; then
  echo ">>> v1 LOSO seed_iv frontal7 $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/seed_iv_frontal7 \
    --epochs 100 --seeds 42 123 456 --batch_size 64 \
    --output_dir results/h4/seed_iv_frontal7 >> "$LOG" 2>&1
fi
if [ ! -f results/h4/tuab_frontal7/summary.json ]; then
  echo ">>> v1 LOSO tuab frontal7 $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/tuab_frontal7 \
    --epochs 100 --eval_every 5 --seeds 42 123 456 --batch_size 64 --class_weight \
    --output_dir results/h4/tuab_frontal7 >> "$LOG" 2>&1
fi
echo "=== H4 frontal-7 DONE $(date) ===" >> "$LOG"
