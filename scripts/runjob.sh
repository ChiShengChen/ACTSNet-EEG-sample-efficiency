#!/usr/bin/env bash
# 單一 job 執行器:idempotent(已有 summary.json 就跳過),各自寫 log,結果附到主 log。
#
# 機器規則(2026-09-05 起):所有吃 GPU 的工作一律經 gq 排程器提交,不得直接跑。
# 這裡用 `gq run`(阻塞、等顯存空出才啟動、stdout/exit code 與直接執行相同)。
# 資源由呼叫端以環境變數 GQ_ARGS 誠實宣告,例如:
#   GQ_ARGS="-g 4G -c 2 -m 8"   小型 EEG 模型、seed_iv/cavanagh/mumtaz 的 cache(<5 GB RAM)
#   GQ_ARGS="-g 6G -c 2 -m 16"  TUAB(cache 13.6 GB 常駐 RAM)
#   GQ_ARGS="-x -c 2 -m 16"     計時 benchmark,需要整張卡才量得準
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
MASTER=${MASTER_LOG:-results/planB.log}
out=$1; shift
if [ -f "$out/summary.json" ]; then echo "SKIP  $out" >> "$MASTER"; exit 0; fi
mkdir -p "$out"
name=$(basename "$out")
echo ">>>   $out  QUEUED $(date +%H:%M:%S)  [gq ${GQ_ARGS:-"-g 4G"}]" >> "$MASTER"
t0=$(date +%s)
# shellcheck disable=SC2086
if gq run ${GQ_ARGS:-"-g 4G"} -n "$name" -- "$@" > "$out/run.log" 2>&1; then
  echo "OK    $out  $(( ($(date +%s)-t0)/60 )) min (含排隊)" >> "$MASTER"
else
  echo "FAIL  $out  (見 $out/run.log)" >> "$MASTER"
  grep -m2 -E "\[gate\]|Error|Traceback" "$out/run.log" >> "$MASTER" 2>/dev/null
fi
