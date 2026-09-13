#!/usr/bin/env bash
# 重跑 Mumtaz H1 曲線的 baseline 低比例點,修正 drop_last 造成 0 batch 的 bug。
# 原因:原 run_h1_mumtaz.sh 用 --batch_size 256,但 Mumtaz 每受試僅 ~20 窗,
# f0.10 的 inner-train 只有 80 窗、f0.25 只有 200 窗 < 256,配合 loader 的
# drop_last=True → 每 epoch 0 個 batch → 模型停在隨機初始化,從未訓練。
# 這裡改用 --batch_size 32(確保低比例下仍有 >=2 個 batch),輸出到獨立目錄以保留原始產物。
#
# Durable launch:
#   cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash run_h1_mumtaz_recheck.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/h1_mumtaz_recheck.log
mkdir -p results/h1_recheck
echo "=== Mumtaz H1 recheck START $(date) ===" >> "$LOG"

bl() {  # model frac tag
  local m=$1 frac=$2 out="results/h1_recheck/mumtaz_$1_$3"
  [ -f "$out/summary.json" ] && { echo ">>> SKIP $m $3" >> "$LOG"; return; }
  echo ">>> $m frac=$frac bs=32 $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py --cache_dir prep_cache/mumtaz \
    --model "$m" --epochs 100 --seeds 42 123 456 --batch_size 32 --class_weight \
    --train_frac "$frac" --output_dir "$out" >> "$LOG" 2>&1
}
# 全比例重跑,讓整條曲線在同一個 batch size 下可比
for f in "0.10 f010" "0.25 f025" "0.50 f050" "1.00 f100"; do
  set -- $f
  bl eegnet "$1" "$2"
  bl bigcnn "$1" "$2"
done
echo "=== Mumtaz H1 recheck DONE $(date) ===" >> "$LOG"
