#!/usr/bin/env bash
# Mumtaz H1 曲線的 v1 重跑,與 run_h1_mumtaz_recheck.sh 的 baseline 用同一個 batch size(32),
# 讓整條曲線在對等的訓練預算下可比。
#
# 背景:原始曲線 baseline 用 --batch_size 256、v1 用 64,而 run_loso.py:212 與
# run_loso_baseline.py:84 的 train loader 都是 drop_last=True。Mumtaz 每受試僅 ~20 窗,
# f0.10 的 inner-train 只有 80 窗 → v1 得到 1 批/epoch(有訓練),baseline 得到 0 批
# (完全沒訓練)。bs=32 讓兩邊在最小的 fraction 都有 >=2 批。
#
# 會先等 run_h1_mumtaz_recheck.sh 跑完再開始(避免 GPU 雙占)。
#
# Durable launch:
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_h1_mumtaz_recheck_v1.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/h1_mumtaz_recheck.log
mkdir -p results/h1_recheck

# 等 baseline 佇列結束
while pgrep -f "run_h1_mumtaz_recheck.sh" > /dev/null; do sleep 60; done
echo "=== Mumtaz H1 recheck (v1) START $(date) ===" >> "$LOG"

v1_point() {  # frac tag
  local frac=$1 out="results/h1_recheck/mumtaz_v1_$2"
  [ -f "$out/summary.json" ] && { echo ">>> SKIP v1 $2" >> "$LOG"; return; }
  echo ">>> v1 frac=$frac bs=32 $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/mumtaz \
    --epochs 100 --eval_every 5 --seeds 42 123 456 --batch_size 32 --class_weight \
    --train_frac "$frac" --output_dir "$out" >> "$LOG" 2>&1
}
for f in "0.10 f010" "0.25 f025" "0.50 f050" "1.00 f100"; do
  set -- $f; v1_point "$1" "$2"
done
echo "=== Mumtaz H1 recheck (v1) DONE $(date) ===" >> "$LOG"
