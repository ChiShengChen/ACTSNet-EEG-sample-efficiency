#!/usr/bin/env bash
# ============================================================================
# 方案 C — 審稿人要求、需要改 code 才能跑的四個實驗
#
#  1. Reviewer AN #2  交叉對照:BigCNN encoder + prototypical head(crossover.py)
#     補齊「架構 × 分類頭」的 2x2,才能把低資料量優勢單獨歸給架構而非損失函數。
#     另三格已存在:v1+proto(主模型)、v1+softmax(H3 消融)、BigCNN+softmax(baseline)。
#  2. 編輯 #8        multi-scale 分支消融(--multiscale off)
#  3. 編輯 #3        batch 內 support/query 不相交的 episodic 訓練(--episodic)
#  4. Reviewer L #4  計算成本 benchmark(bench_compute.py;需 GPU 淨空才準)
#
# 會先等 run_planB.sh 跑完。產物寫到 results/planC/。
#
# Durable launch:
#   cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash run_planC.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
#   DRY=1 bash run_planC.sh   → 只印 job 清單不執行
# ============================================================================
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
JOBS=results/planC_jobs.txt
mkdir -p results/planC
: > "$JOBS"
add() { echo "MASTER_LOG=results/planC.log GQ_ARGS=\"$GQ\" bash runjob.sh $*" >> "$JOBS"; }

FRACS=(1.00 0.10 0.25 0.50); TAGS=(f100 f010 f025 f050)
bs_of() { case $1 in mumtaz) echo 32;; *) echo 64;; esac; }
gq_v1() { case $1 in tuab) echo "-g 8G -c 2 -m 16";; *) echo "-g 6G -c 2 -m 8";; esac; }
cw_of() { case $1 in seed_iv) echo "";; *) echo "--class_weight";; esac; }

# 1) AN-2 交叉:BigCNN encoder + proto 頭,整條學習曲線(優勢宣稱所在的兩個資料集)
for ds in seed_iv mumtaz; do
  bs=$(bs_of $ds); cw=$(cw_of $ds); GQ=$(gq_v1 $ds)
  for i in 0 1 2 3; do
    out=results/planC/${ds}_bigcnnproto_${TAGS[$i]}
    add "$out" $PY -u run_loso.py --cache_dir prep_cache/$ds --epochs 100 --eval_every 5 \
        --seeds 42 123 456 --batch_size $bs $cw --encoder bigcnn \
        --train_frac ${FRACS[$i]} --output_dir "$out"
  done
done
# 2x2 的第四格:v1 + softmax 在 Mumtaz 只有全資料量,補上曲線以對齊
GQ=$(gq_v1 mumtaz)
for i in 1 2 3; do
  out=results/planC/mumtaz_ac_softmax_${TAGS[$i]}
  add "$out" $PY -u run_loso.py --cache_dir prep_cache/mumtaz --epochs 100 --eval_every 5 \
      --seeds 42 123 456 --batch_size 32 --class_weight --head softmax \
      --train_frac ${FRACS[$i]} --output_dir "$out"
done

# 2) 編輯 #8:multi-scale 分支消融(與既有 H2/H3 消融一致,全資料量)
for ds in seed_iv mumtaz; do
  GQ=$(gq_v1 $ds)
  out=results/planC/${ds}_no_multiscale
  add "$out" $PY -u run_loso.py --cache_dir prep_cache/$ds --epochs 100 --eval_every 5 \
      --seeds 42 123 456 --batch_size $(bs_of $ds) $(cw_of $ds) --multiscale off \
      --output_dir "$out"
done

# 3) 編輯 #3:support/query 不相交的 episodic 訓練
for ds in seed_iv mumtaz; do
  GQ=$(gq_v1 $ds)
  out=results/planC/${ds}_episodic
  add "$out" $PY -u run_loso.py --cache_dir prep_cache/$ds --epochs 100 --eval_every 5 \
      --seeds 42 123 456 --batch_size $(bs_of $ds) $(cw_of $ds) --episodic \
      --output_dir "$out"
done

if [ "${DRY:-0}" = 1 ]; then echo "--- $(wc -l < "$JOBS") jobs ---"; cat "$JOBS"; exit 0; fi
while pgrep -f "run_plan[B]\.sh" > /dev/null; do sleep 300; done
echo "=== PLAN C START $(date) — $(wc -l < "$JOBS") jobs ===" >> results/planC.log
xargs -a "$JOBS" -d '\n' -I@ bash -c '@'

# 4) Reviewer L #4:計算成本。放最後,確保量測時 GPU 已淨空。
for ds in seed_iv mumtaz; do
  [ -f results/planC/compute_cost_${ds}.json ] && continue
  # 計時需要整張卡(-x),否則其他 session 的負載會污染延遲量測
  gq run -x -c 2 -m 16 -n bench_$ds -- "$PY" bench_compute.py --cache_dir prep_cache/$ds \
      --out results/planC/compute_cost_${ds}.json >> results/planC.log 2>&1
done
echo "=== PLAN C DONE $(date) ===" >> results/planC.log
