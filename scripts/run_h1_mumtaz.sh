#!/usr/bin/env bash
# H1 low-data curve on Mumtaz MDD (v1's native task): v1 vs EEGNet at frac {0.1,0.25,0.5}.
# frac=1.0 reuses results/mumtaz_v1 and results/mumtaz_eegnet. Settings match those 100% points
# (v1: eval_every=5 + class_weight; EEGNet: batch 256 + class_weight). Idempotent.
#
# Durable launch:
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_h1_mumtaz.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/h1_mumtaz.log
mkdir -p results/h1
echo "=== H1 Mumtaz curve START $(date) ===" >> "$LOG"

v1_point() {  # frac tag
  local frac=$1 out="results/h1/mumtaz_v1_$2"
  [ -f "$out/summary.json" ] && { echo ">>> SKIP v1 $2" >> "$LOG"; return; }
  echo ">>> v1 frac=$frac $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/mumtaz \
    --epochs 100 --eval_every 5 --seeds 42 123 456 --batch_size 64 --class_weight \
    --train_frac "$frac" --output_dir "$out" >> "$LOG" 2>&1
}
eg_point() {  # frac tag
  local frac=$1 out="results/h1/mumtaz_eegnet_$2"
  [ -f "$out/summary.json" ] && { echo ">>> SKIP eegnet $2" >> "$LOG"; return; }
  echo ">>> eegnet frac=$frac $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py --cache_dir prep_cache/mumtaz \
    --model eegnet --epochs 100 --seeds 42 123 456 --batch_size 256 --class_weight \
    --train_frac "$frac" --output_dir "$out" >> "$LOG" 2>&1
}

for f in "0.10 f010" "0.25 f025" "0.50 f050"; do
  set -- $f
  v1_point "$1" "$2"
  eg_point "$1" "$2"
done
echo "=== H1 Mumtaz curve DONE $(date) ===" >> "$LOG"
