#!/usr/bin/env bash
# Mumtaz MDD end-to-end: wait for download -> unzip -> build cache (EC resting) ->
# v1 LOSO + EEGNet baseline. Idempotent. MDD vs healthy = v1's native clinical task.
#
# Durable launch (after the download was started detached):
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_mumtaz.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
set -u
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
ROOT=${MUMTAZ_DIR:?請設為 Mumtaz 2016 MDD 語料的所在目錄}
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=results/mumtaz.log
echo "=== MUMTAZ pipeline START $(date) ===" >> "$LOG"

# 1) wait for the detached download to finish
while [ ! -f "$ROOT/dl.status" ]; do sleep 30; done
echo ">>> download done: $(cat $ROOT/dl.status)" >> "$LOG"

# 2) unzip (skip if already extracted)
if ! ls "$ROOT"/*.edf >/dev/null 2>&1; then
  echo ">>> unzip $(date)" >> "$LOG"
  ( cd "$ROOT" && unzip -o -q mumtaz.zip ) >> "$LOG" 2>&1
fi
echo ">>> edf count: $(ls "$ROOT"/*.edf 2>/dev/null | wc -l)" >> "$LOG"

# 3) build cache (EC = eyes-closed resting)
if [ ! -f prep_cache/mumtaz/meta.json ]; then
  echo ">>> build mumtaz cache (EC) $(date)" >> "$LOG"
  ( cd pipeline && "$PY" build_mumtaz.py --modality EC --win_sec 10 \
      --windows_per_recording 20 --max_total_windows 20000 \
      --out_dir ../prep_cache/mumtaz ) >> "$LOG" 2>&1
fi

# 4) v1 LOSO  (MDD ~63 subjects -> GroupKFold(10); class-weighted)
if [ ! -f results/mumtaz_v1/summary.json ]; then
  echo ">>> v1 LOSO mumtaz $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso.py --cache_dir prep_cache/mumtaz \
    --epochs 100 --eval_every 5 --seeds 42 123 456 --batch_size 64 --class_weight \
    --output_dir results/mumtaz_v1 >> "$LOG" 2>&1
fi

# 5) EEGNet baseline
if [ ! -f results/mumtaz_eegnet/summary.json ]; then
  echo ">>> EEGNet LOSO mumtaz $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run_loso_baseline.py --cache_dir prep_cache/mumtaz \
    --model eegnet --epochs 100 --seeds 42 123 456 --batch_size 256 --class_weight \
    --output_dir results/mumtaz_eegnet >> "$LOG" 2>&1
fi
echo "=== MUMTAZ pipeline DONE $(date) ===" >> "$LOG"
