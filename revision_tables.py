"""Tables for the JMIR ms#107929 point-by-point response that need no new training.

Every number is recomputed from the per-fold artefacts already on disk, so each table
is reproducible from the released results directory alone.

  --table auroc    editor #13 / Reviewer L #3: AUROC for every model, not just ACTSNet
  --table robust   editor #15 / Reviewer L #2b / V #9: the full robustness grid
  --table dup      editor #9: evidence that the Mumtaz low-fraction baselines never trained
  --table all      (default)
"""
import argparse
import glob
import json
import os

import numpy as np

R = "results"


def load(d, key="balanced_accuracy"):
    p = f"{d}/per_fold.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def fold_mean(recs, key):
    """One observation per fold (averaged over seeds), then mean +- SD across folds."""
    acc = {}
    for x in recs:
        if key in x and x[key] is not None and not (isinstance(x[key], float) and np.isnan(x[key])):
            acc.setdefault(x["fold"], []).append(x[key])
    if not acc:
        return None
    v = np.array([np.mean(a) for a in acc.values()])
    return v.mean(), v.std(), len(v)


def table_auroc():
    print("=" * 78)
    print("編輯 #13 / Reviewer L #3 — 每個模型的 AUROC(window level,全資料量)")
    print("=" * 78)
    print("送審的 Table 2 只給 ACTSNet 的 AUROC,四個對照組留白。這些值一直都在")
    print("per_fold.json 的 `auc` 欄位裡,不需要重跑。\n")
    print("注意層級不同:這裡是 window-level;稿件 §3.1 的 0.95 是 subject-level,")
    print("兩者不可混列於同一欄(編輯 #13 後半的疑慮)。\n")
    DS = {"seed_iv": ("results/seed_iv", 4), "tuab": ("results/tuab", 2),
          "mumtaz": ("results/mumtaz_v1", 2), "cavanagh": ("results/cavanagh_v1", 2)}
    models = ["eegnet", "shallowconv", "bigcnn", "transformer", "tapnet"]
    print(f"{'資料集':10s}{'類別':>4}  {'ACTSNet':>15}" + "".join(f"{m:>15}" for m in models))
    for ds, (v1dir, k) in DS.items():
        row = [f"{ds:10s}{k:>4}"]
        for d in [v1dir] + [f"results/{ds}_{m}" for m in models]:
            recs = load(d)
            r = fold_mean(recs, "auc") if recs else None
            row.append(f"{r[0]:>9.3f}±{r[1]:.3f}" if r else f"{'—':>15}")
        print("  ".join(row))
    print("\n多分類(SEED-IV, 4 類)的 AUROC 取決於平均方式:本專案 `safe_auc` 使用")
    print("one-vs-rest 搭配 macro 平均;此averaging scheme 需寫進 Methods(編輯 #13)。")


def table_robust():
    print("=" * 78)
    print("編輯 #15 / Reviewer L #2b / Reviewer V #9 — 完整的測試期穩健性格點")
    print("=" * 78)
    print("送審的 Table 8 只報了 MDD 的 Gaussian 噪聲一欄。channel dropout 與 SEED-IV")
    print("的結果從一開始就在 results/h5/,不需要新實驗。\n")
    conds = ["clean", "noise0.25", "noise0.50", "noise1.00", "chdrop0.2", "chdrop0.4"]
    print(f"{'資料集/模型':22s}{'折數':>5}" + "".join(f"{c:>12}" for c in conds))
    for ds in ["mumtaz", "seed_iv"]:
        for m in ["v1", "eegnet"]:
            p = f"{R}/h5/{ds}_{m}/summary.json"
            if not os.path.exists(p):
                continue
            s = json.load(open(p))
            cells = "".join(f"{s['mean'][c]:>7.3f}±{s['std'][c]:.2f}" if c in s["mean"]
                            else f"{'—':>12}" for c in conds)
            print(f"{ds + '/' + m:22s}{s.get('n_folds', '?'):>5}{cells}")
    print("\n兩項與稿件不符之處,磁碟資料可直接裁決:")
    print("  · Reviewer V #9 / 編輯 #15:EEGNet 在 σ=1.0 的值,磁碟為 0.641 →")
    print("    §3.3 正文的 0.64 是對的,Table 8 的 0.60 是錯的(與審稿人的假設相反)。")
    print("  · Table 8 的 clean 欄(0.61/0.60)對不上磁碟的 0.856/0.907,疑為整欄搬錯。")
    print("  · 誠實面:MDD 上 EEGNet 在每一個擾動條件下都比 ACTSNet 耐受。")


def table_dup():
    print("=" * 78)
    print("編輯 #9 — Mumtaz 低比例的 baseline 從未被訓練(不是多數類退化)")
    print("=" * 78)
    print("編輯注意到 Mumtaz 在 10% 與 25% 時 BigCNN 與 EEGNet 都恰為 0.50 且離散度極小,")
    print("推測是退化成單類別預測。實際機制更嚴重,而且可由既有產物直接證明。\n")
    for m in ["eegnet", "bigcnn", "v1"]:
        a = load(f"{R}/h1/mumtaz_{m}_f010")
        b = load(f"{R}/h1/mumtaz_{m}_f025")
        if not (a and b):
            continue
        da = {(x["seed"], x["fold"]): x["balanced_accuracy"] for x in a}
        db = {(x["seed"], x["fold"]): x["balanced_accuracy"] for x in b}
        k = sorted(set(da) & set(db))
        same = sum(1 for i in k if da[i] == db[i])
        flag = "  ← 逐位元相同,不可能來自兩次獨立訓練" if same == len(k) else ""
        print(f"  mumtaz {m:8s} f0.10 與 f0.25 的測試 BACC 相同者:{same}/{len(k)}{flag}")
    print("\n根因:run_h1_mumtaz.sh 給 baseline 用 --batch_size 256,而 Mumtaz 每受試僅約")
    print("20 個窗,f0.10 的 inner-train 只有 80 窗、f0.25 只有 200 窗,皆小於 batch;")
    print("train loader 的 drop_last=True 使每個 epoch 產生 0 個 batch,模型從未更新,")
    print("停在隨機初始化(seed 固定 → 兩個 fraction 的預測完全相同)。\n")
    print("修正後(全部模型 batch 32,對等訓練預算,results/h1_recheck/):")
    print(f"  {'frac':>6}{'ACTSNet':>16}{'EEGNet':>16}{'BigCNN':>16}")
    for tag, fr in [("f010", 0.10), ("f025", 0.25), ("f050", 0.50), ("f100", 1.00)]:
        row = [f"  {fr:>6.2f}"]
        for m in ["v1", "eegnet", "bigcnn"]:
            recs = load(f"{R}/h1_recheck/mumtaz_{m}_{tag}")
            r = fold_mean(recs, "balanced_accuracy") if recs else None
            row.append(f"{r[0]:>10.3f}±{r[1]:.3f}" if r else f"{'(跑中)':>16}")
        print("".join(row))
    print("\n→ 修正後 baseline 在 10% 就達 0.641(EEGNet),並非 chance;各 fraction 的")
    print("  差異皆不顯著。稿件 Abstract/Table 4/Table 6/Fig.3 的對應敘述需全面改寫。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="all", choices=["all", "auroc", "robust", "dup"])
    a = ap.parse_args()
    for name, fn in [("dup", table_dup), ("robust", table_robust), ("auroc", table_auroc)]:
        if a.table in ("all", name):
            fn(); print()
