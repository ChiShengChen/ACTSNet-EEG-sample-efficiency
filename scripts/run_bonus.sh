#!/usr/bin/env bash
# Paper 1 bonus experiments, ordered fast->slow so useful results land early. Idempotent.
#   1. MDD ablations (H2/H3 on the clinical task)      — fast
#   2. ShallowConvNet 2nd baseline on SEED-IV + MDD    — tractable
#   3. TUAB learning curve (v1 + EEGNet, frac 0.1/0.25/0.5) — moderate
#   4. ShallowConvNet on TUAB                          — slow (last)
# Durable launch:
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_bonus.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/bonus.log
mkdir -p results/h1
echo "=== BONUS START $(date) ===" >> "$LOG"
run() { echo ">>> $1 $(date)" >> "$LOG"; shift; CUDA_VISIBLE_DEVICES=0 "$PY" -u "$@" >> "$LOG" 2>&1; }
skip() { [ -f "$1/summary.json" ]; }

# 1. MDD ablations (match MDD v1 settings: eval_every 5, class_weight)
skip results/ablation/mumtaz_lstm_proto || run "MDD H2 lstm+proto" run_loso.py \
  --cache_dir prep_cache/mumtaz --epochs 100 --eval_every 5 --seeds 42 123 456 \
  --batch_size 64 --class_weight --branch lstm --head proto \
  --output_dir results/ablation/mumtaz_lstm_proto
skip results/ablation/mumtaz_ac_softmax || run "MDD H3 ac+softmax" run_loso.py \
  --cache_dir prep_cache/mumtaz --epochs 100 --eval_every 5 --seeds 42 123 456 \
  --batch_size 64 --class_weight --branch ac --head softmax \
  --output_dir results/ablation/mumtaz_ac_softmax

# 2. ShallowConvNet 2nd baseline (batch 64 to avoid the 95-ch activation blow-up)
skip results/seed_iv_shallowconv || run "ShallowConv SEED-IV" run_loso_baseline.py \
  --cache_dir prep_cache/seed_iv --model shallowconv --epochs 100 --seeds 42 123 456 \
  --batch_size 64 --output_dir results/seed_iv_shallowconv
skip results/mumtaz_shallowconv || run "ShallowConv MDD" run_loso_baseline.py \
  --cache_dir prep_cache/mumtaz --model shallowconv --epochs 100 --seeds 42 123 456 \
  --batch_size 64 --class_weight --output_dir results/mumtaz_shallowconv

# 3. TUAB learning curve (v1 eval_every5+cw batch64 ; EEGNet batch256+cw)
for spec in "0.10 f010" "0.25 f025" "0.50 f050"; do
  set -- $spec; frac=$1; tag=$2
  skip results/h1/tuab_v1_$tag || run "TUAB v1 frac=$frac" run_loso.py \
    --cache_dir prep_cache/tuab --epochs 100 --eval_every 5 --seeds 42 123 456 \
    --batch_size 64 --class_weight --train_frac $frac --output_dir results/h1/tuab_v1_$tag
  skip results/h1/tuab_eegnet_$tag || run "TUAB eegnet frac=$frac" run_loso_baseline.py \
    --cache_dir prep_cache/tuab --model eegnet --epochs 100 --seeds 42 123 456 \
    --batch_size 256 --class_weight --train_frac $frac --output_dir results/h1/tuab_eegnet_$tag
done

# 4. ShallowConvNet on TUAB (slow — last)
skip results/tuab_shallowconv || run "ShallowConv TUAB" run_loso_baseline.py \
  --cache_dir prep_cache/tuab --model shallowconv --epochs 100 --seeds 42 123 456 \
  --batch_size 64 --class_weight --output_dir results/tuab_shallowconv
echo "=== BONUS DONE $(date) ===" >> "$LOG"
