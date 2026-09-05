#!/usr/bin/env bash
# Idempotent baseline runner: runs only the (dataset, model) combos whose
# results/<ds>_<model>/summary.json is still missing. Safe to re-run after a
# disconnect — already-finished baselines are skipped.
#
# Durable launch (survives Claude session disconnect):
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid nohup bash run_remaining_baselines.sh >/dev/null 2>&1 < /dev/null &
#   echo "detached PID: $!"
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
BLOG=results/baselines_resume.log
echo "=== BASELINE RESUME (setsid) START $(date) ===" >> "$BLOG"

run_one() {  # args: dataset model class_weight_flag
  local ds=$1 model=$2 cw=$3
  local out="results/${ds}_${model}"
  if [ -f "$out/summary.json" ]; then
    echo ">>> SKIP $ds $model (already done)" >> "$BLOG"; return
  fi
  echo ">>> RUN $ds $model $(date)" >> "$BLOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py --cache_dir "prep_cache/$ds" \
    --model "$model" --epochs 100 --seeds 42 123 456 --batch_size 256 $cw \
    --output_dir "$out" >> "$BLOG" 2>&1
}

# EEGNet baselines at batch 256 (tiny model -> GPU headroom).
# v1 stays batch 64; baselines use their own batch (noted in the paper).
# ShallowConv DROPPED (2026-06-27): architecturally memory/compute-bound on 95-ch
# long windows (~38 min/fold), ~20h+ for 3 seeds; EEGNet is the standard primary
# baseline. Re-add `run_one <ds> shallowconv` with a lighter config if ever needed.
run_one seed_iv eegnet      ""
run_one tuab    eegnet      --class_weight
echo "=== BASELINE RESUME DONE $(date) ===" >> "$BLOG"
