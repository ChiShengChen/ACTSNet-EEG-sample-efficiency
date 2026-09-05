#!/usr/bin/env bash
# H5 test-time robustness (SEED-IV + MDD, v1 vs EEGNet, noise/channel-dropout sweep, 1 seed).
# Waits for run_bonus.sh to finish first to avoid GPU contention. Idempotent.
#   setsid bash scripts/run_h5.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/h5.log
mkdir -p results/h5
echo "=== H5 START (waiting for bonus chain) $(date) ===" >> "$LOG"
while pgrep -f "run_bonus.sh" >/dev/null || pgrep -f "bin/python.*run_loso" >/dev/null; do sleep 120; done
echo ">>> GPU free, running H5 $(date)" >> "$LOG"

run() {  # cache model outdir [--class_weight]
  local out=$3
  [ -f "$out/summary.json" ] && { echo ">>> SKIP $out" >> "$LOG"; return; }
  echo ">>> H5 $1 $2 $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_h5.py --cache_dir "$1" --model "$2" \
    --seeds 42 --output_dir "$out" ${4:-} >> "$LOG" 2>&1
}
run prep_cache/seed_iv v1     results/h5/seed_iv_v1
run prep_cache/seed_iv eegnet results/h5/seed_iv_eegnet
run prep_cache/mumtaz  v1     results/h5/mumtaz_v1     --class_weight
run prep_cache/mumtaz  eegnet results/h5/mumtaz_eegnet --class_weight
echo "=== H5 DONE $(date) ===" >> "$LOG"
