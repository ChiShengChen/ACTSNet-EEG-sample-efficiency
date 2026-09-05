#!/usr/bin/env bash
# 單一 job 執行器:idempotent(已有 summary.json 就跳過),各自寫 log,結果附到主 log。
set -u
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
MASTER=results/planB.log
out=$1; shift
if [ -f "$out/summary.json" ]; then echo "SKIP  $out" >> "$MASTER"; exit 0; fi
mkdir -p "$out"
echo ">>>   $out  START $(date +%H:%M:%S)" >> "$MASTER"
t0=$(date +%s)
if "$@" > "$out/run.log" 2>&1; then
  echo "OK    $out  $(( ($(date +%s)-t0)/60 )) min" >> "$MASTER"
else
  echo "FAIL  $out  (見 $out/run.log)" >> "$MASTER"
  grep -m2 -E "\[gate\]|Error|Traceback" "$out/run.log" >> "$MASTER" 2>/dev/null
fi
