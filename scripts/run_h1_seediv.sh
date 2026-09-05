#!/usr/bin/env bash
# H1 low-data learning curve on SEED-IV: v1 vs EEGNet at train_frac {0.1,0.25,0.5}.
# (frac=1.0 reuses existing results/seed_iv and results/seed_iv_eegnet.)
# Idempotent: skips any point whose summary.json already exists.
#
# Durable launch:
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_h1_seediv.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/h1_seediv.log
mkdir -p results/h1
echo "=== H1 SEED-IV curve START $(date) ===" >> "$LOG"

# v1 uses eval_every=1 to match the existing 100% SEED-IV point.
v1_point() {   # frac tag
  local frac=$1 tag=$2 out="results/h1/seed_iv_v1_$2"
  [ -f "$out/summary.json" ] && { echo ">>> SKIP v1 $tag" >> "$LOG"; return; }
  echo ">>> v1 frac=$frac $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/seed_iv \
    --epochs 100 --seeds 42 123 456 --batch_size 64 --train_frac "$frac" \
    --output_dir "$out" >> "$LOG" 2>&1
}
eg_point() {   # frac tag
  local frac=$1 tag=$2 out="results/h1/seed_iv_eegnet_$2"
  [ -f "$out/summary.json" ] && { echo ">>> SKIP eegnet $tag" >> "$LOG"; return; }
  echo ">>> eegnet frac=$frac $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py --cache_dir prep_cache/seed_iv \
    --model eegnet --epochs 100 --seeds 42 123 456 --batch_size 256 --train_frac "$frac" \
    --output_dir "$out" >> "$LOG" 2>&1
}

for f in "0.10 f010" "0.25 f025" "0.50 f050"; do
  set -- $f
  v1_point "$1" "$2"
  eg_point "$1" "$2"
done
echo "=== H1 SEED-IV curve DONE $(date) ===" >> "$LOG"
