#!/usr/bin/env bash
# 一次把方案 B + 方案 C 的所有未完成 job 丟進 gq 佇列(gq submit),取代逐一阻塞提交。
#
# 為什麼:gq 是 FIFO,逐一提交等於每個 job 都重新排到隊尾;在 30+ session 共用時
# 每次排隊動輒數小時,40 個 job 在期限內跑不完。批次提交後排程器有空就派我們的 job。
#
# 順序(我方內部優先序,不用 -p 排擠他人):
#   1. 方案 C 交叉實驗 BigCNN+proto(決定 Discussion 走向)
#   2. Cavanagh baseline、小資料集 TapNet 曲線
#   3. 方案 C 其餘消融
#   4. TUAB(最大宗、最不影響結論)
#   5. bench_compute(-x,整張卡)最後
# frontal-7 需先建 cache(cpu-only);cache 完成後再執行 `bash submit_all.sh frontal7`。
#
# 用法:bash submit_all.sh            # 提交全部(已有 summary.json 的自動略過)
#       bash submit_all.sh frontal7   # 只提交 frontal-7 的 4 個 job(cache 建好後)
#       DRY=1 bash submit_all.sh      # 只列出會提交什麼
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
DRY=${DRY:-0}
LEDGER=results/gq_submitted.tsv
touch "$LEDGER"

gen() { DRY=1 bash "$1" | grep '^GQ_ARGS=\|^MASTER_LOG=' ; }

# 兩個 driver 的 job 清單(格式:GQ_ARGS="..." bash runjob.sh <out> <cmd...>)
ALL=$( { gen run_planB.sh; gen run_planC.sh; } )

order() {   # 依內部優先序重排
  local L="$1"
  grep 'bigcnnproto'                       <<<"$L"
  grep 'planB/cavanagh_'                   <<<"$L" | grep -v frontal7
  grep 'planB/seed_iv_tapnet\|planB/mumtaz_tapnet\|planB/cavanagh_tapnet' <<<"$L"
  grep 'planC/' <<<"$L" | grep -v 'bigcnnproto'
  grep 'planB/tuab_'                       <<<"$L"
}
if [ "${1:-}" = "frontal7" ]; then
  # run_planB.sh 的 DRY 模式在 Part 3 之前就退出,frontal-7 的 job 要在這裡自行組出
  PY=${PY:-python}
  LIST=""
  for ds in mumtaz cavanagh; do
    [ -f "prep_cache/${ds}_frontal7/data.npy" ] || { echo "缺 cache:${ds}_frontal7"; continue; }
    if [ "$ds" = mumtaz ]; then bs=32; else bs=64; fi
    LIST+="GQ_ARGS=\"-g 6G -c 2 -m 8\" bash runjob.sh results/planB/${ds}_frontal7_v1 $PY -u run_loso.py --cache_dir prep_cache/${ds}_frontal7 --epochs 100 --eval_every 5 --seeds 42 123 456 --batch_size $bs --class_weight --output_dir results/planB/${ds}_frontal7_v1
"
    LIST+="GQ_ARGS=\"-g 4G -c 2 -m 8\" bash runjob.sh results/planB/${ds}_frontal7_eegnet $PY -u run_loso_baseline.py --cache_dir prep_cache/${ds}_frontal7 --model eegnet --epochs 100 --seeds 42 123 456 --batch_size $bs --class_weight --output_dir results/planB/${ds}_frontal7_eegnet
"
  done
else
  LIST=$(order "$ALL" | awk '!seen[$0]++')     # 去重:order() 的條件會互相重疊
fi

n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  gqargs=$(sed -n 's/.*GQ_ARGS="\([^"]*\)".*/\1/p' <<<"$line")
  rest=${line#*bash runjob.sh }
  out=${rest%% *}
  cmd=${rest#* }
  [ -f "$out/summary.json" ] && continue
  # TapNet(LSTM)的 VRAM 宣告要等探針 #169 量出峰值(見 resubmit_tapnet_after_probe.sh);量到前不送
  if grep -q tapnet <<<"$out" && ! grep -q "探針峰值" results/tapnet_resubmit.log 2>/dev/null; then
    [ "$DRY" = 1 ] && echo "(等探針)跳過 $(basename "$out")"; continue; fi
  grep -q "	$out	" "$LEDGER" 2>/dev/null && { echo "已提交過,略過:$out"; continue; }
  name=$(basename "$out")
  if [ "$DRY" = 1 ]; then echo "[$gqargs] $name"; n=$((n+1)); continue; fi
  mkdir -p "$out"
  # shellcheck disable=SC2086
  id=$(gq submit $gqargs -n "$name" -- bash -c "$cmd > $out/run.log 2>&1" 2>&1 | grep -oE '#?[0-9]+' | head -1)
  printf '%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$out" "${id:-?}" "$gqargs" >> "$LEDGER"
  echo "submitted ${id:-?}  $name  [$gqargs]"
  n=$((n+1))
done <<<"$LIST"
echo "共 $n 個 job$( [ "$DRY" = 1 ] && echo '(dry run)' )"
