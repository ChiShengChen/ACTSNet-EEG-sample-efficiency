"""LOSO driver for the EEGNet / ShallowConvNet baselines.

Mirrors run_loso.py exactly (same cross-subject folds, inner-val model selection,
metrics) but trains a standard softmax CNN with CrossEntropy instead of the
prototypical head — so any v1-vs-baseline gap is the model, not the protocol/data.

Input: the SAME (N, C*S, T) tensor v1 ingests (default). `--collapse_bands` instead
sums the 5 sub-bands -> (N, C, T) for a "standard EEGNet on broadband" variant.

Usage:
  conda activate pytorch291
  python run_loso_baseline.py --cache_dir prep_cache/seed_iv --model eegnet --epochs 100 --seeds 42 123 456
  python run_loso_baseline.py --cache_dir prep_cache/tuab    --model shallowconv --class_weight
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
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, f1_score, accuracy_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "pipeline"))
# The ACTSNet architecture lives in a separate repository:
#   https://github.com/ChiShengChen/ACTSNetv1
# Set ACTSNET_V1 to a local checkout, or clone it next to this repository.
sys.path.insert(0, os.environ.get("ACTSNET_V1",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ACTSNetv1")))

# reuse the exact data + split + helper machinery from the v1 driver
from run_loso import (  # noqa: E402
    trainability_gate,
    load_cache, make_loader, inner_split, class_weight_from, subsample_train_subjects,
)
from actsnet.train import set_seed, safe_auc                                  # noqa: E402
from baselines import BASELINES                                               # noqa: E402


def train_epoch_clf(model, loader, opt, device, class_weight=None):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y, weight=class_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def eval_clf(model, loader, device):
    model.eval()
    preds, labs, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1)
        preds.extend(logits.argmax(1).cpu().numpy())
        labs.extend(y.numpy())
        probs.append(p.cpu().numpy())
    labs = np.asarray(labs); preds = np.asarray(preds)
    probs = np.concatenate(probs, 0)
    return {
        "balanced_accuracy": balanced_accuracy_score(labs, preds),
        "cohen_kappa": cohen_kappa_score(labs, preds),
        "weighted_f1": f1_score(labs, preds, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labs, preds),
        "auc": safe_auc(labs, probs),
    }


def run_fold(X, y, groups, tr, te, n_classes, args, device, seed):
    if getattr(args, "train_frac", 1.0) < 1.0:
        tr = subsample_train_subjects(tr, groups, args.train_frac, seed)
    inner_tr, inner_val = inner_split(tr, y, groups, args.inner_val_frac, seed)
    Model = BASELINES[args.model]
    model = Model(n_chans=X.shape[1], n_classes=n_classes, n_times=X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cw = class_weight_from(y[inner_tr], n_classes, device) if args.class_weight else None

    trainability_gate(inner_tr, groups, args.batch_size, seed,
                      f"{args.model} frac={getattr(args,'train_frac',1.0)}")
    tr_loader = make_loader(X, y, inner_tr, args.batch_size, True, drop_last=True)
    val_loader = make_loader(X, y, inner_val, args.batch_size, False)

    best, best_state = -1.0, None
    for _ in range(args.epochs):
        train_epoch_clf(model, tr_loader, opt, device, cw)
        vm = eval_clf(model, val_loader, device)
        if vm["balanced_accuracy"] > best:
            best = vm["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    tm = eval_clf(model, make_loader(X, y, te, args.batch_size, False), device)
    tm["inner_val_balacc"] = best
    return tm


def main():
    ap = argparse.ArgumentParser(description="EEGNet/ShallowConvNet LOSO baseline")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--model", required=True, choices=list(BASELINES))
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--loso_max", type=int, default=20)
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--inner_val_frac", type=float, default=0.2)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="H1 low-data curve: keep this fraction of TRAIN subjects")
    ap.add_argument("--collapse_bands", action="store_true",
                    help="sum the 5 sub-bands -> (N,C,T) broadband instead of (N,C*S,T)")
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X, y, groups = load_cache(args.cache_dir)            # (N, C*S, T)
    if args.collapse_bands:
        meta = json.load(open(os.path.join(args.cache_dir, "meta.json")))
        C, S = meta["C"], meta["S"]
        X = X.reshape(X.shape[0], C, S, X.shape[2]).sum(axis=2)   # (N, C, T)
    n_classes = int(len(np.unique(y)))
    n_subjects = int(len(np.unique(groups)))
    ds_name = os.path.basename(os.path.normpath(args.cache_dir))
    tag = f"{ds_name}_{args.model}" + ("_bb" if args.collapse_bands else "")
    out_dir = Path(args.output_dir or os.path.join(_HERE, "results", tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_subjects <= args.loso_max:
        splitter, proto = LeaveOneGroupOut(), f"LOSO ({n_subjects} folds)"
    else:
        splitter, proto = GroupKFold(n_splits=args.n_folds), f"GroupKFold({args.n_folds})"

    print(f"[{tag}] X={X.shape} classes={n_classes} subjects={n_subjects} | {proto} "
          f"| device={device} | seeds={args.seeds} | cw={args.class_weight}")

    folds = list(splitter.split(X, y, groups))
    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    per_fold, t0 = [], time.time()
    for seed in args.seeds:
        set_seed(seed)
        for fi, (tr, te) in enumerate(folds):
            tm = run_fold(X, y, groups, tr, te, n_classes, args, device, seed)
            tm.update({"seed": seed, "fold": fi, "n_test": int(len(te))})
            per_fold.append(tm)
            print(f"  seed {seed} fold {fi:2d} | BalAcc {tm['balanced_accuracy']:.4f} "
                  f"kappa {tm['cohen_kappa']:.4f} wF1 {tm['weighted_f1']:.4f} "
                  f"AUC {tm['auc']:.4f} (innerVal {tm['inner_val_balacc']:.3f})")

    def agg(k):
        v = np.array([r[k] for r in per_fold], dtype=np.float64)
        return float(v.mean()), float(v.std())

    summary = {
        "dataset": ds_name, "model": args.model, "collapse_bands": args.collapse_bands,
        "protocol": proto, "n_classes": n_classes, "n_subjects": n_subjects,
        "x_shape": list(X.shape), "seeds": args.seeds, "epochs": args.epochs,
        "class_weight": args.class_weight, "train_frac": args.train_frac,
        "n_results": len(per_fold),
        "elapsed_sec": round(time.time() - t0, 1),
        "metrics": {k: {"mean": agg(k)[0], "std": agg(k)[1]}
                    for k in ["balanced_accuracy", "cohen_kappa",
                              "weighted_f1", "accuracy", "auc"]},
    }
    json.dump(per_fold, open(out_dir / "per_fold.json", "w"), indent=2)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)

    m = summary["metrics"]
    print(f"\n[{tag}] DONE ({summary['elapsed_sec']}s) over {len(per_fold)} (seed×fold)")
    print(f"  BalAcc {m['balanced_accuracy']['mean']:.4f} ± {m['balanced_accuracy']['std']:.4f} | "
          f"Kappa {m['cohen_kappa']['mean']:.4f} | wF1 {m['weighted_f1']['mean']:.4f} | "
          f"AUC {m['auc']['mean']:.4f}")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
