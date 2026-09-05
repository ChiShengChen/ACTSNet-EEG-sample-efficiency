#!/usr/bin/env bash
# ============================================================================
# Matched-training-budget reruns, plus the baseline coverage the paper was missing.
#
# 背景:原始實驗中 v1 一律 --batch_size 64、所有 baseline 一律 256,epochs 都是 100,
# 而兩邊的 train loader 都是 drop_last=True。結果 baseline 每 epoch 的梯度更新只有 v1
# 的約 1/4;在 Mumtaz 低比例更直接變成 0 個 batch(模型從未訓練,停在隨機初始化)。
# 詳見 revision/FINDINGS_reanalysis.md。
#
# 本佇列把所有模型對齊到 v1 已在使用的 batch 64(Mumtaz 每受試僅 ~20 窗,改用 32),
# 因此 v1 既有結果可沿用,只需重跑 baseline。ShallowConv 依方案 B 不重跑(光是 TUAB
# 就需 ~111 GPU 小時),維持原設定並在 Limitations 揭露。
#
# Output goes to results/planB/ and never overwrites the original results/, which are
# kept as the record of what the first version of this work actually ran.
# 每個 job 都經過 run_loso.py 的 trainability_gate:0 batch/epoch 直接中止,不再靜默跑空。
#
# Durable launch:
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_planB.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
#
# 平行度預設 1。實測(2026-09-05):8 個 job 各限 2 核併跑需 321s,單獨跑 41s×8=328s
# → 加速比 1.0,併行毫無幫助。瓶頸是單一 job 就已飽和的資料搬運/記憶體頻寬,不是 CPU
# 核數(每 job 2 執行緒即滿速)也不是 GPU(使用率僅 7%)。併行反而讓多個 job 同時從
# Elements 外接碟冷載入數 GB 的 cache 而互搶 I/O。
#   進度 tail -f results/planB.log   |   中止 pkill -f run_planB.sh; pkill -f run_loso
#   DRY=1 bash run_planB.sh  → 只印 job 清單不執行
# ============================================================================
set -u
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
PAR=${PAR:-1}
# 每 job 2 執行緒即滿速(實測 1核 77s / 2核 41s / 24核 42s),多給純屬浪費
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
DRY=${DRY:-0}
JOBS=results/planB_jobs.txt
mkdir -p results/planB
: > "$JOBS"

FRACS=(1.00 0.10 0.25 0.50)
TAGS=(f100 f010 f025 f050)

add() { echo "bash runjob.sh $*" >> "$JOBS"; }

# 每個資料集的 batch size 與 class_weight(沿用原始設定;SEED-IV 類別平衡故不加權)
bs_of()  { case $1 in mumtaz) echo 32;; *) echo 64;; esac; }
cw_of()  { case $1 in seed_iv) echo "";; *) echo "--class_weight";; esac; }

# ---------------------------------------------------------------- Part 1
# baseline 在對等 batch 下重跑(不含 ShallowConv)。
# Mumtaz 的 eegnet/bigcnn 已在 results/h1_recheck/ 以 bs32 跑完,故排除。
for ds in seed_iv mumtaz cavanagh tuab; do
  bs=$(bs_of $ds); cw=$(cw_of $ds)
  for m in eegnet bigcnn transformer; do  # transformer 最後,可隨時中止
    [ "$ds" = mumtaz ] && [ "$m" != transformer ] && continue
    for i in 0 1 2 3; do
      out=results/planB/${ds}_${m}_${TAGS[$i]}
      add "$out" $PY -u run_loso_baseline.py --cache_dir prep_cache/$ds --model $m \
          --epochs 100 --seeds 42 123 456 --batch_size $bs $cw --train_frac ${FRACS[$i]} \
          --output_dir "$out"
    done
  done
done

# ---------------------------------------------------------------- Part 2
# TapNet (= the LSTM branch + prototypical head) was compared only at full data;
# these fill in its learning curves.
for ds in seed_iv mumtaz cavanagh tuab; do
  bs=$(bs_of $ds); cw=$(cw_of $ds)
  for i in 0 1 2 3; do
    out=results/planB/${ds}_tapnet_${TAGS[$i]}
    add "$out" $PY -u run_loso.py --cache_dir prep_cache/$ds --epochs 100 --eval_every 5 \
        --seeds 42 123 456 --batch_size $bs $cw --branch lstm --head proto \
        --train_frac ${FRACS[$i]} --output_dir "$out"
  done
done

if [ "$DRY" = 1 ]; then echo "--- $(wc -l < "$JOBS") jobs ---"; cat "$JOBS"; exit 0; fi

echo "=== PLAN B START $(date) — $(wc -l < "$JOBS") jobs, 平行度 $PAR ===" >> results/planB.log
xargs -P "$PAR" -a "$JOBS" -d '\n' -I@ bash -c '@'
echo "=== PLAN B PART1+2 DONE $(date) ===" >> results/planB.log

# ---------------------------------------------------------------- Part 3
# Extend the 7-channel frontal montage to the Mumtaz and Cavanagh cohorts, where a
# low-density montage is the clinically motivated configuration. Needs its own cache.
for ds in mumtaz cavanagh; do
  if [ ! -f "prep_cache/${ds}_frontal7/data.npy" ]; then
    echo ">>> 建 ${ds}_frontal7 cache $(date)" >> results/planB.log
    # 參數與原始全 montage cache 完全一致(見 run_mumtaz.sh / run_cavanagh.sh),
    # 只改 --montage,否則窗切法不同就無法比較。逐一建,勿併行(會截斷 data.npy)。
    ( cd pipeline && "$PY" build_${ds}.py --montage frontal7 --win_sec 10 \
        --windows_per_recording 20 --max_total_windows 20000 \
        --out_dir ../prep_cache/${ds}_frontal7 ) >> results/planB.log 2>&1
  fi
done
: > "$JOBS"
for ds in mumtaz cavanagh; do
  [ -f "prep_cache/${ds}_frontal7/data.npy" ] || continue
  bs=$(bs_of $ds); cw=$(cw_of $ds)
  out=results/planB/${ds}_frontal7_v1
  add "$out" $PY -u run_loso.py --cache_dir prep_cache/${ds}_frontal7 --epochs 100 \
      --eval_every 5 --seeds 42 123 456 --batch_size $bs $cw --output_dir "$out"
  out=results/planB/${ds}_frontal7_eegnet
  add "$out" $PY -u run_loso_baseline.py --cache_dir prep_cache/${ds}_frontal7 --model eegnet \
      --epochs 100 --seeds 42 123 456 --batch_size $bs $cw --output_dir "$out"
done
[ -s "$JOBS" ] && xargs -P "$PAR" -a "$JOBS" -d '\n' -I@ bash -c '@'
echo "=== PLAN B ALL DONE $(date) ===" >> results/planB.log
