#!/usr/bin/env bash
# ============================================================================
# Controls that required new model code rather than only new run configurations.
#
#  1. Encoder x head crossover: BigCNN encoder + prototypical head (crossover.py).
#     補齊「架構 × 分類頭」的 2x2,才能把低資料量優勢單獨歸給架構而非損失函數。
#     另三格已存在:v1+proto(主模型)、v1+softmax(H3 消融)、BigCNN+softmax(baseline)。
#  2. Multi-scale branch ablation (--multiscale off).
#  3. Episodic training with disjoint support/query within each batch (--episodic).
#  4. Computational-cost benchmark (bench_compute.py; needs an otherwise idle GPU).
#
# 會先等 run_planB.sh 跑完。產物寫到 results/planC/。
#
# Durable launch:
#   # 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
#   setsid bash scripts/run_planC.sh >/dev/null 2>&1 < /dev/null & disown; echo ok
# ============================================================================
set -u
# 腳本位於 scripts/,專案根目錄是上一層
cd "$(dirname "$(readlink -f "$0")")/.."
PY=${PY:-python}      # 覆寫範例:PY=/path/to/env/bin/python bash scripts/xxx.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
JOBS=results/planC_jobs.txt
mkdir -p results/planC
: > "$JOBS"
add() { echo "bash runjob.sh $*" >> "$JOBS"; }

FRACS=(1.00 0.10 0.25 0.50); TAGS=(f100 f010 f025 f050)
bs_of() { case $1 in mumtaz) echo 32;; *) echo 64;; esac; }
cw_of() { case $1 in seed_iv) echo "";; *) echo "--class_weight";; esac; }

# 1) AN-2 交叉:BigCNN encoder + proto 頭,整條學習曲線(優勢宣稱所在的兩個資料集)
for ds in seed_iv mumtaz; do
  bs=$(bs_of $ds); cw=$(cw_of $ds)
  for i in 0 1 2 3; do
    out=results/planC/${ds}_bigcnnproto_${TAGS[$i]}
    add "$out" $PY -u run_loso.py --cache_dir prep_cache/$ds --epochs 100 --eval_every 5 \
        --seeds 42 123 456 --batch_size $bs $cw --encoder bigcnn \
        --train_frac ${FRACS[$i]} --output_dir "$out"
  done
done
# 2x2 的第四格:v1 + softmax 在 Mumtaz 只有全資料量,補上曲線以對齊
for i in 1 2 3; do
  out=results/planC/mumtaz_ac_softmax_${TAGS[$i]}
  add "$out" $PY -u run_loso.py --cache_dir prep_cache/mumtaz --epochs 100 --eval_every 5 \
      --seeds 42 123 456 --batch_size 32 --class_weight --head softmax \
      --train_frac ${FRACS[$i]} --output_dir "$out"
done

# 2) Multi-scale branch ablation, at full data to match the other ablations.
for ds in seed_iv mumtaz; do
  out=results/planC/${ds}_no_multiscale
  add "$out" $PY -u run_loso.py --cache_dir prep_cache/$ds --epochs 100 --eval_every 5 \
      --seeds 42 123 456 --batch_size $(bs_of $ds) $(cw_of $ds) --multiscale off \
      --output_dir "$out"
done

# 3) Episodic training with disjoint support/query sets.
for ds in seed_iv mumtaz; do
  out=results/planC/${ds}_episodic
  add "$out" $PY -u run_loso.py --cache_dir prep_cache/$ds --epochs 100 --eval_every 5 \
      --seeds 42 123 456 --batch_size $(bs_of $ds) $(cw_of $ds) --episodic \
      --output_dir "$out"
done

while pgrep -f "run_planB.sh" > /dev/null; do sleep 300; done
echo "=== PLAN C START $(date) — $(wc -l < "$JOBS") jobs ===" >> results/planC.log
xargs -a "$JOBS" -d '\n' -I@ bash -c '@'

# 4) Computational cost. Last, so the GPU is idle when the timings are taken.
for ds in seed_iv mumtaz; do
  [ -f results/planC/compute_cost_${ds}.json ] && continue
  "$PY" bench_compute.py --cache_dir prep_cache/$ds \
      --out results/planC/compute_cost_${ds}.json >> results/planC.log 2>&1
done
echo "=== PLAN C DONE $(date) ===" >> results/planC.log
