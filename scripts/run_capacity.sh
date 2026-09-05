#!/usr/bin/env bash
# Crux control: capacity-matched plain CNN (BigCNN, ~5.4e5 params ≈ ACTSNet) low-data curves on
# SEED-IV + MDD. If a same-size CNN with no metric-learning bias gets no low-data benefit, the
# advantage is inductive bias, not capacity. Idempotent.
#   setsid bash scripts/run_capacity.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/capacity.log
mkdir -p results/h1
echo "=== CAPACITY CONTROL START $(date) ===" >> "$LOG"
run() { local out=$3; [ -f "$out/summary.json" ] && { echo ">>> SKIP $out" >> "$LOG"; return; }
  echo ">>> $out $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py --cache_dir "prep_cache/$1" --model bigcnn \
    --epochs 100 --seeds 42 123 456 --batch_size 256 $2 --output_dir "$out" >> "$LOG" 2>&1; }
runf() { local out=$4; [ -f "$out/summary.json" ] && { echo ">>> SKIP $out" >> "$LOG"; return; }
  echo ">>> $out (frac $3) $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py --cache_dir "prep_cache/$1" --model bigcnn \
    --epochs 100 --seeds 42 123 456 --batch_size 256 $2 --train_frac "$3" --output_dir "$out" >> "$LOG" 2>&1; }

# SEED-IV (no class_weight)
run    seed_iv "" results/seed_iv_bigcnn
for f in "0.10 f010" "0.25 f025" "0.50 f050"; do set -- $f; runf seed_iv "" "$1" results/h1/seed_iv_bigcnn_$2; done
# MDD (class_weight)
run    mumtaz --class_weight results/mumtaz_bigcnn
for f in "0.10 f010" "0.25 f025" "0.50 f050"; do set -- $f; runf mumtaz --class_weight "$1" results/h1/mumtaz_bigcnn_$2; done
echo "=== CAPACITY CONTROL DONE $(date) ===" >> "$LOG"
