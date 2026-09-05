#!/usr/bin/env bash
# H2/H3 ablations on SEED-IV (v1's clearest-advantage dataset).
#   baseline ac+proto = results/seed_iv (0.349, already done)
#   H2  lstm+proto  : AC branch -> LSTM (TapNet original) — does AC beat LSTM?
#   H3  ac+softmax  : prototypical head -> softmax-linear — is the metric head the win?
# Same LOSO protocol/seeds, eval_every=1 (consistent with the seed_iv baseline). Idempotent.
#
# Durable launch:
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_h2h3_ablation.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/h2h3_ablation.log
mkdir -p results/ablation
echo "=== H2/H3 ablation START $(date) ===" >> "$LOG"

run() {  # branch head tag
  local branch=$1 head=$2 out="results/ablation/seed_iv_$3"
  [ -f "$out/summary.json" ] && { echo ">>> SKIP $3" >> "$LOG"; return; }
  echo ">>> $3 (branch=$branch head=$head) $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/seed_iv \
    --epochs 100 --seeds 42 123 456 --batch_size 64 \
    --branch "$branch" --head "$head" --output_dir "$out" >> "$LOG" 2>&1
}

run lstm proto   lstm_proto    # H2
run ac   softmax ac_softmax    # H3
echo "=== H2/H3 ablation DONE $(date) ===" >> "$LOG"
