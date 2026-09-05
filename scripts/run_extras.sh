#!/usr/bin/env bash
# Additional baselines (TapNet, transformer), clinical metrics, and the hyperparameter sweep.
# Ordered by value/speed. Idempotent, detached.
#   setsid bash scripts/run_extras.sh >/dev/null 2>&1 < /dev/null & disown
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/extras.log; mkdir -p results/sweep
echo "=== EXTRAS START $(date) ===" >> "$LOG"
skip() { [ -f "$1/summary.json" ]; }
run() { echo ">>> $1 $(date)" >> "$LOG"; shift; CUDA_VISIBLE_DEVICES=0 "$PY" -u "$@" >> "$LOG" 2>&1; }

# --- #5 clinical metrics: ACTSNet on MDD with dumped probs (subject-level sens/spec/AUROC) ---
skip results/mumtaz_v1_preds || run "MDD dump (clinical)" run_loso.py --cache_dir prep_cache/mumtaz \
  --epochs 100 --eval_every 5 --seeds 42 123 456 --batch_size 64 --class_weight --dump_preds \
  --output_dir results/mumtaz_v1_preds

# --- #4 TapNet standalone (= LSTM branch + prototypical head) on TUAB (have SEED-IV/MDD from ablations) ---
skip results/tuab_tapnet || run "TapNet TUAB" run_loso.py --cache_dir prep_cache/tuab \
  --epochs 100 --eval_every 5 --seeds 42 123 456 --batch_size 64 --class_weight \
  --branch lstm --head proto --output_dir results/tuab_tapnet

# --- #4 transformer baseline, full data on the 3 main datasets ---
run "transformer SEED-IV" true; skip results/seed_iv_transformer || CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  run_loso_baseline.py --cache_dir prep_cache/seed_iv --model transformer --epochs 100 \
  --seeds 42 123 456 --batch_size 256 --output_dir results/seed_iv_transformer >> "$LOG" 2>&1
skip results/mumtaz_transformer || CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py \
  --cache_dir prep_cache/mumtaz --model transformer --epochs 100 --seeds 42 123 456 \
  --batch_size 256 --class_weight --output_dir results/mumtaz_transformer >> "$LOG" 2>&1
skip results/tuab_transformer || CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py \
  --cache_dir prep_cache/tuab --model transformer --epochs 100 --seeds 42 123 456 \
  --batch_size 256 --class_weight --output_dir results/tuab_transformer >> "$LOG" 2>&1

# --- #7 hyperparameter sensitivity (SEED-IV, 1 seed, vary one hparam at a time) ---
sweep() { local tag=$1; shift; skip results/sweep/$tag && return
  echo ">>> sweep $tag $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/seed_iv --epochs 100 \
    --seeds 42 "$@" --output_dir results/sweep/$tag >> "$LOG" 2>&1; }
sweep default
sweep pdim64  --prototype_dim 64
sweep pdim256 --prototype_dim 256
sweep grp2    --n_groups 2
sweep grp5    --n_groups 5
sweep lr5e4   --lr 0.0005
sweep lr2e3   --lr 0.002
echo "=== EXTRAS DONE $(date) ===" >> "$LOG"
