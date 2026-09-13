"""Leave-One-Subject-Out (LOSO) driver for ACTSNet v1 on a prep_cache dataset.

Reads a cache built by pipeline/ (data.npy (N,C,S,T), labels.npy, subjects.npy),
flattens to the v1 view (N, C*S, T), and runs cross-subject CV:

  - n_subjects <= --loso_max  -> LeaveOneGroupOut (true LOSO)
  - otherwise                 -> GroupKFold(--n_folds)   (grouped by subject)

Protocol (no test leakage):
  per fold, hold out test subject(s); carve a group-disjoint inner-validation set
  from the remaining train subjects for best-model selection (by balanced accuracy);
  finally evaluate the selected model on the test fold using ALL train subjects as
  the prototypical support set. Repeats over --seeds; reports mean ± std of
  balanced accuracy / Cohen's kappa / weighted-F1 / accuracy / AUC over (seed × fold).

Usage:
  conda activate pytorch291
  python run_loso.py --cache_dir prep_cache/seed_iv --epochs 100 --seeds 42 123 456
  python run_loso.py --cache_dir prep_cache/tuab   --epochs 100 --class_weight
  # quick smoke:
  python run_loso.py --cache_dir <tiny> --epochs 2 --seeds 42 --max_folds 2
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import (
    LeaveOneGroupOut, GroupKFold, GroupShuffleSplit,
)
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, f1_score, accuracy_score,
)

# Resolve sibling repos regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "pipeline"))
# The ACTSNet architecture lives in a separate repository:
#   https://github.com/ChiShengChen/ACTSNetv1
# Set ACTSNET_V1 to a local checkout, or clone it next to this repository.
sys.path.insert(0, os.environ.get("ACTSNET_V1",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ACTSNetv1")))

from eeg_prep import flatten_cst                      # noqa: E402
from actsnet.config import ACTSNetConfig              # noqa: E402
from actsnet.model import ACTSNet                     # noqa: E402
from actsnet.train import set_seed, safe_auc          # noqa: E402


# ── data ────────────────────────────────────────────────────────────────────
def load_cache(cache_dir):
    # mmap:cache 以檔案映射載入,不複製成匿名記憶體(TUAB 的 data.npy 有 13.6 GB;
    # 全量載入時 RSS 超過 gq 宣告並耗盡 SWAP,job 被排程器暫停)。頁面由 OS 快取,
    # 同機多個 job 可共享,數值與非 mmap 完全相同。
    data = np.load(os.path.join(cache_dir, "data.npy"), mmap_mode="r")   # (N,C,S,T)
    labels = np.load(os.path.join(cache_dir, "labels.npy"))      # (N,)
    subjects = np.load(os.path.join(cache_dir, "subjects.npy"))  # (N,)
    # copy=False:cache 已是 float32,flatten_cst 又只是 reshape view,所以原本的
    # astype 是一份多餘的整份複製。TUAB 的 data.npy 有 13.6 GB,省掉這份複製可讓
    # 峰值記憶體從 ~27 GB 降到 ~13.6 GB,方案 B 才能安全地 3 個 job 併跑。
    flat = flatten_cst(data).astype(np.float32, copy=False)       # (N, C*S, T)
    return flat, labels.astype(np.int64), subjects


class _LazySubset(torch.utils.data.Dataset):
    """Index the (memory-mapped) cache per item instead of copying X[idx] up front.

    TensorDataset(torch.from_numpy(X[idx])) materialised the whole training subset in
    anonymous RAM (up to 10.5 GB for TUAB at 100%), which together with the cache
    exceeded the declared RAM, exhausted swap, and got the job paused by the scheduler.
    Values, dtypes and batch composition are identical; only the memory footprint changes.
    """
    def __init__(self, X, y, idx):
        self.X, self.y, self.idx = X, y, np.asarray(idx)
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return torch.from_numpy(np.array(self.X[j])), torch.as_tensor(int(self.y[j]))   # np.array = 可寫複製,避免 memmap 唯讀警告


def make_loader(X, y, idx, batch_size, shuffle, drop_last=False):
    return DataLoader(_LazySubset(X, y, idx), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last)


# ── train / eval ────────────────────────────────────────────────────────────
def _disjoint_support_query(y, generator):
    """Class-stratified split of a batch into disjoint support / query halves.

    Editor comment 3: the default training loop passes the mini-batch as its own
    support set, so every query sample contributes to the prototype it is scored
    against, which partially trivialises the metric objective relative to standard
    episodic training. This splits each class's samples in half instead. A class with
    a single sample in the batch goes entirely to support, since a prototype cannot be
    formed without it (it then contributes no query term).
    """
    sup, qry = [], []
    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=True)[0]
        # 在 CPU 上抽 permutation(generator 是 CPU 的)再搬到 idx 所在裝置
        idx = idx[torch.randperm(len(idx), generator=generator).to(idx.device)]
        k = max(1, len(idx) // 2)
        sup.append(idx[:k])
        if len(idx) > k:
            qry.append(idx[k:])
    if not qry:
        return None, None
    return torch.cat(sup), torch.cat(qry)


def train_epoch(model, loader, opt, device, class_weight=None, episodic=False,
                generator=None):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if episodic:
            s, q = _disjoint_support_query(y, generator)
            if s is None:                      # 整個 batch 只有單類別,跳過
                continue
            log_probs = model(x[q], support_x=x[s], support_labels=y[s])
            loss = F.nll_loss(log_probs, y[q], weight=class_weight)
        else:
            log_probs = model(x, support_x=x, support_labels=y)
            loss = F.nll_loss(log_probs, y, weight=class_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def _metrics(labs, preds, probs, return_preds=False):
    m = {
        "balanced_accuracy": balanced_accuracy_score(labs, preds),
        "cohen_kappa": cohen_kappa_score(labs, preds),
        "weighted_f1": f1_score(labs, preds, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labs, preds),
        "auc": safe_auc(labs, probs),
    }
    if return_preds:
        m["preds"] = [int(v) for v in preds]
        m["labels"] = [int(v) for v in labs]
        if probs.shape[1] == 2:                       # positive-class prob (binary clinical)
            m["prob1"] = [float(v) for v in probs[:, 1]]
    return m


@torch.no_grad()
def eval_full(model, support_loader, query_loader, device, max_support=5000, return_preds=False):
    """Support from `support_loader` (train), query = `query_loader` (val/test).

    Softmax-head ablation has no support set — evaluate by plain forward.
    """
    model.eval()
    if getattr(model, "head", "proto") == "softmax":
        preds, labs, probs = [], [], []
        for x, y in query_loader:
            lp = model(x.to(device))
            preds.extend(lp.argmax(1).cpu().numpy())
            labs.extend(y.numpy())
            probs.append(torch.exp(lp).cpu().numpy())
        labs = np.asarray(labs); preds = np.asarray(preds)
        probs = np.concatenate(probs, 0)
        return _metrics(labs, preds, probs, return_preds)
    sx, sy, n = [], [], 0
    for x, y in support_loader:
        sx.append(x); sy.append(y); n += x.size(0)
        if n >= max_support:
            break
    sx = torch.cat(sx)[:max_support].to(device)
    sy = torch.cat(sy)[:max_support].to(device)
    # 支撐集分批編碼。批大小只影響記憶體不影響數值(eval 模式下各樣本獨立);
    # 用 64 而非 256:cuDNN LSTM 的工作區與 batch×T 成正比,256×2560 時曾達 6.9 GiB 而 OOM。
    semb = []
    for i in range(0, len(sx), 64):
        semb.append(model.encode(sx[i:i + 64]))
    semb = torch.cat(semb, 0)

    preds, labs, probs = [], [], []
    for x, y in query_loader:
        lp = model.proto(model.encode(x.to(device)), semb, sy)
        preds.extend(lp.argmax(1).cpu().numpy())
        labs.extend(y.numpy())
        probs.append(torch.exp(lp).cpu().numpy())
    labs = np.asarray(labs); preds = np.asarray(preds)
    probs = np.concatenate(probs, 0)
    return _metrics(labs, preds, probs, return_preds)


def class_weight_from(y_idx, n_classes, device):
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def balanced_support_idx(idx, y, max_support, seed):
    """Class-stratified random support subsample of `idx` (<= max_support).

    Guarantees every class is represented in the prototypical support set. Without
    this, a cache block-ordered by class (e.g. TUAB = all-abnormal then all-normal)
    makes the first-N support single-class -> the missing class's prototype collapses
    -> systematically inverted test predictions (negative kappa, AUC < 0.5).
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y[idx])
    per = max(1, max_support // len(classes))
    chosen = []
    for c in classes:
        ci = idx[y[idx] == c]
        if len(ci) > per:
            ci = rng.choice(ci, per, replace=False)
        chosen.append(ci)
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def subsample_train_subjects(train_idx, groups, frac, seed):
    """Keep a random `frac` of the TRAINING subjects (group-level) for H1 low-data
    curves. Test fold is untouched. Deterministic per seed; always keeps >=1 subject.
    """
    if frac >= 1.0:
        return train_idx
    g = groups[train_idx]
    uniq = np.unique(g)
    k = max(1, int(round(len(uniq) * frac)))
    # The draw is seeded by the run seed AND the identity of the training fold. The
    # original implementation used the seed alone, so within a seed nearly the same
    # subject positions were drawn in every fold (at 10% each seed used ~2 distinct
    # SEED-IV subjects across 15 folds, and one seed drew controls only on Mumtaz).
    # All results reported before 2026-09-13 used the seed-only draw; see Methods.
    fold_tag = int(np.uint32(hash(tuple(uniq.tolist())) & 0xFFFFFFFF))
    rng = np.random.RandomState((seed * 1000 + 7 + fold_tag) % (2 ** 32))
    keep = set(rng.choice(uniq, k, replace=False).tolist())
    return train_idx[np.isin(g, list(keep))]


def inner_split(train_idx, y, groups, frac, seed):
    """Group-disjoint inner validation split of the training indices.

    Falls back to a plain random split when <2 train groups (degenerate / smoke).
    Returns (inner_train_idx, inner_val_idx) as absolute indices into the dataset.
    """
    g = groups[train_idx]
    if len(np.unique(g)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        a, b = next(gss.split(train_idx, y[train_idx], g))
        return train_idx[a], train_idx[b]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_idx))
    n_val = max(1, int(len(train_idx) * frac))
    return train_idx[perm[n_val:]], train_idx[perm[:n_val]]



def trainability_gate(inner_tr, groups, batch_size, seed, tag):
    """Fail loudly when the training loader would yield too few batches.

    The train loader uses drop_last=True, so `len(inner_tr) // batch_size` batches
    per epoch. When that is 0 the epoch loop silently does nothing and the model is
    evaluated at its random initialisation -- this produced the invalid Mumtaz
    low-data baseline curve (see revision/FINDINGS_reanalysis.md). Always log the
    count; abort on 0.
    """
    nb = len(inner_tr) // batch_size
    n_sub = len(set(groups[inner_tr].tolist()))
    print(f"[gate] {tag} seed={seed} inner_train={len(inner_tr)} windows / {n_sub} subjects"
          f" | batch={batch_size} -> {nb} batches/epoch", flush=True)
    if nb == 0:
        raise RuntimeError(
            f"[gate] {tag}: 0 batches/epoch ({len(inner_tr)} windows < batch {batch_size} "
            f"with drop_last=True) -- the model would never train. Lower --batch_size.")
    if nb < 2:
        print(f"[gate] WARNING {tag}: only {nb} batch/epoch -- training budget is very thin "
              f"and may not be comparable across models.", flush=True)
    return nb


# ── one fold ────────────────────────────────────────────────────────────────
def run_fold(X, y, groups, train_idx, test_idx, n_classes, args, device, seed):
    if getattr(args, "train_frac", 1.0) < 1.0:
        train_idx = subsample_train_subjects(train_idx, groups, args.train_frac, seed)
    inner_tr, inner_val = inner_split(train_idx, y, groups, args.inner_val_frac, seed)

    cfg = ACTSNetConfig(
        n_channels=X.shape[1], n_classes=n_classes,
        n_groups=args.n_groups, prototype_dim=args.prototype_dim,
        latent_dim_u=args.latent_dim_u, dropout=args.dropout,
        branch=args.branch, head=args.head,
        seed=seed, device=str(device),
    )
    cfg.n_times = X.shape[2]
    # 消融/交叉對照的唯一分派點(見 crossover.py):只有模型換掉,折切分、seed、
    # 支撐集建構、訓練迴圈、選模準則全部相同。
    from crossover import build_model
    model = build_model(cfg, getattr(args, "encoder", "actsnet"),
                        getattr(args, "multiscale", "on")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    cw = class_weight_from(y[inner_tr], n_classes, device) if args.class_weight else None

    trainability_gate(inner_tr, groups, args.batch_size, seed,
                      f"{getattr(args,'encoder','actsnet')}/ms={getattr(args,'multiscale','on')}/"
                      f"{args.branch}/{args.head}{'/episodic' if getattr(args,'episodic',False) else ''} "
                      f"frac={getattr(args,'train_frac',1.0)}")
    tr_loader = make_loader(X, y, inner_tr, args.batch_size, True, drop_last=True)
    itr_support = make_loader(X, y, balanced_support_idx(inner_tr, y, 5000, seed),
                              args.batch_size, False)
    ival_loader = make_loader(X, y, inner_val, args.batch_size, False)

    best_balacc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, tr_loader, opt, device, cw,
                    episodic=getattr(args, 'episodic', False),
                    generator=torch.Generator(device='cpu').manual_seed(seed + epoch))
        # inner-val model selection only every eval_every epochs (+ always last);
        # the per-epoch support re-encoding is the cost bottleneck on long windows.
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            vm = eval_full(model, itr_support, ival_loader, device)
            if vm["balanced_accuracy"] > best_balacc:
                best_balacc = vm["balanced_accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test: support = class-balanced sample of ALL train subjects, query = test.
    full_support = make_loader(X, y, balanced_support_idx(train_idx, y, 5000, seed),
                               args.batch_size, False)
    test_loader = make_loader(X, y, test_idx, args.batch_size, False)
    tm = eval_full(model, full_support, test_loader, device,
                   return_preds=getattr(args, "dump_preds", False))
    if getattr(args, "dump_preds", False):
        tm["subjects"] = [str(s) for s in groups[test_idx]]   # aligned with preds (shuffle=False)
    tm["inner_val_balacc"] = best_balacc
    if torch.cuda.is_available():
        # 誠實宣告 gq 資源用:每折印出本行程的 VRAM 峰值
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        tm["peak_vram_gb"] = round(peak, 2)
        print(f"[vram] peak {peak:.2f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
    return tm


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ACTSNet v1 LOSO driver")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--loso_max", type=int, default=20,
                    help="<= this many subjects -> true LOSO; else GroupKFold")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--inner_val_frac", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=1,
                    help="inner-val model-selection eval every N epochs (speedup; "
                         "final test eval is unchanged)")
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="H1 low-data curve: keep this fraction of TRAIN subjects "
                         "(group-level, per-seed). Test fold unchanged.")
    ap.add_argument("--dump_preds", action="store_true",
                    help="store per-fold test preds+labels in per_fold.json (for confusion matrix)")
    ap.add_argument("--class_weight", action="store_true",
                    help="class-balanced NLL (recommended for TUAB/imbalanced)")
    ap.add_argument("--max_folds", type=int, default=None, help="cap folds (smoke)")
    # model hparams
    ap.add_argument("--encoder", default="actsnet", choices=["actsnet", "bigcnn"],
                    help="AN-2 crossover: ACTSNet encoder (default) or the capacity-matched "
                         "BigCNN trunk, both feeding the same prototypical head")
    ap.add_argument("--multiscale", default="on", choices=["on", "off"],
                    help="editor #8 ablation: drop the TapNet-inherited multi-scale branch")
    ap.add_argument("--episodic", action="store_true",
                    help="editor #3: split each training batch into disjoint support/query "
                         "halves instead of using the batch as its own support set")
    ap.add_argument("--branch", default="ac", choices=["ac", "lstm"],
                    help="H2 ablation: AC (default) or LSTM (TapNet original) branch")
    ap.add_argument("--head", default="proto", choices=["proto", "softmax"],
                    help="H3 ablation: prototypical (default) or softmax-linear head")
    ap.add_argument("--n_groups", type=int, default=3)
    ap.add_argument("--prototype_dim", type=int, default=128)
    ap.add_argument("--latent_dim_u", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X, y, groups = load_cache(args.cache_dir)
    n_classes = int(len(np.unique(y)))
    n_subjects = int(len(np.unique(groups)))
    ds_name = os.path.basename(os.path.normpath(args.cache_dir))
    out_dir = Path(args.output_dir or os.path.join(_HERE, "results", ds_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_subjects <= args.loso_max:
        splitter, proto = LeaveOneGroupOut(), f"LOSO ({n_subjects} folds)"
    else:
        splitter, proto = GroupKFold(n_splits=args.n_folds), f"GroupKFold({args.n_folds})"

    print(f"[{ds_name}] X={X.shape} classes={n_classes} subjects={n_subjects} | {proto} "
          f"| device={device} | seeds={args.seeds} | class_weight={args.class_weight}")

    folds = list(splitter.split(X, y, groups))
    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    per_fold = []
    t0 = time.time()
    for seed in args.seeds:
        set_seed(seed)
        for fi, (tr, te) in enumerate(folds):
            tm = run_fold(X, y, groups, tr, te, n_classes, args, device, seed)
            test_subj = sorted(set(groups[te].tolist()))
            tm.update({"seed": seed, "fold": fi, "n_test": int(len(te)),
                       "test_subjects": test_subj[:10]})
            per_fold.append(tm)
            print(f"  seed {seed} fold {fi:2d} | BalAcc {tm['balanced_accuracy']:.4f} "
                  f"kappa {tm['cohen_kappa']:.4f} wF1 {tm['weighted_f1']:.4f} "
                  f"AUC {tm['auc']:.4f} (innerVal {tm['inner_val_balacc']:.3f})")

    def agg(key):
        v = np.array([r[key] for r in per_fold], dtype=np.float64)
        return float(v.mean()), float(v.std())

    summary = {
        "dataset": ds_name, "protocol": proto, "n_classes": n_classes,
        "n_subjects": n_subjects, "x_shape": list(X.shape),
        "seeds": args.seeds, "epochs": args.epochs, "class_weight": args.class_weight,
        "train_frac": args.train_frac, "encoder": args.encoder,
        "multiscale": args.multiscale, "episodic": args.episodic,
        "branch": args.branch, "head": args.head,
        "n_results": len(per_fold), "elapsed_sec": round(time.time() - t0, 1),
        "metrics": {k: {"mean": agg(k)[0], "std": agg(k)[1]}
                    for k in ["balanced_accuracy", "cohen_kappa",
                              "weighted_f1", "accuracy", "auc"]},
    }
    with open(out_dir / "per_fold.json", "w") as f:
        json.dump(per_fold, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    m = summary["metrics"]
    print(f"\n[{ds_name}] DONE ({summary['elapsed_sec']}s) over {len(per_fold)} (seed×fold)")
    print(f"  BalAcc {m['balanced_accuracy']['mean']:.4f} ± {m['balanced_accuracy']['std']:.4f}")
    print(f"  Kappa  {m['cohen_kappa']['mean']:.4f} ± {m['cohen_kappa']['std']:.4f}")
    print(f"  wF1    {m['weighted_f1']['mean']:.4f} ± {m['weighted_f1']['std']:.4f}")
    print(f"  AUC    {m['auc']['mean']:.4f} ± {m['auc']['std']:.4f}")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
