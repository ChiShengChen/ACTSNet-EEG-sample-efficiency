#!/usr/bin/env bash
# Second depression cohort (Cavanagh ds003478): build cache, then reproduce the low-data
# inductive-bias result — v1 vs EEGNet vs BigCNN(capacity-matched) at fracs {0.10,0.25,0.50,1.0}.
# GroupKFold(10), class_weight (74 HC / 45 MDD). Idempotent, detached.
#   setsid bash scripts/run_cavanagh.sh >/dev/null 2>&1 < /dev/null & disown
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/cavanagh.log; mkdir -p results/h1
echo "=== CAVANAGH START $(date) ===" >> "$LOG"

if [ ! -f prep_cache/cavanagh/meta.json ]; then
  echo ">>> build cavanagh cache $(date)" >> "$LOG"
  ( cd pipeline && "$PY" build_cavanagh.py --win_sec 10 --windows_per_recording 20 \
      --max_total_windows 20000 --out_dir ../prep_cache/cavanagh ) >> "$LOG" 2>&1
fi

v1() {  local out=$2; [ -f "$out/summary.json" ] && return
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/cavanagh --epochs 100 \
    --eval_every 5 --seeds 42 123 456 --batch_size 64 --class_weight $1 --output_dir "$out" >> "$LOG" 2>&1; }
bl() {  local m=$1 out=$3; [ -f "$out/summary.json" ] && return
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py --cache_dir prep_cache/cavanagh --model $m \
    --epochs 100 --seeds 42 123 456 --batch_size 256 --class_weight $2 --output_dir "$out" >> "$LOG" 2>&1; }

echo ">>> ACTSNet $(date)" >> "$LOG"
v1 "" results/cavanagh_v1
for f in "0.10 f010" "0.25 f025" "0.50 f050"; do set -- $f; v1 "--train_frac $1" results/h1/cavanagh_v1_$2; done
for M in eegnet bigcnn; do
  echo ">>> $M $(date)" >> "$LOG"
  bl $M "" results/cavanagh_$M
  for f in "0.10 f010" "0.25 f025" "0.50 f050"; do set -- $f; bl $M "--train_frac $1" results/h1/cavanagh_${M}_$2; done
done
echo "=== CAVANAGH DONE $(date) ===" >> "$LOG"
