"""H5 test-time robustness: train clean (LOSO), then evaluate the selected model on the
test fold under a sweep of test-time perturbations (Gaussian noise / channel dropout).
Compares ACTSNet v1 vs EEGNet degradation. Data is z-scored per (channel,sub-band), so
noise sigma is in normalized-signal units. 1 seed by default (supplementary robustness).

Usage:
  python run_h5.py --cache_dir prep_cache/seed_iv --model v1     --output_dir results/h5/seed_iv_v1
  python run_h5.py --cache_dir prep_cache/mumtaz  --model eegnet --class_weight --output_dir results/h5/mumtaz_eegnet
"""
import argparse, json, os, sys
import numpy as np, torch
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import balanced_accuracy_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "pipeline"))
# The ACTSNet architecture lives in a separate repository:
#   https://github.com/ChiShengChen/ACTSNetv1
# Set ACTSNET_V1 to a local checkout, or clone it next to this repository.
sys.path.insert(0, os.environ.get("ACTSNET_V1",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ACTSNetv1")))

from run_loso import (load_cache, make_loader, inner_split, class_weight_from,   # noqa: E402
                      balanced_support_idx, train_epoch, eval_full)
from run_loso_baseline import train_epoch_clf, eval_clf                          # noqa: E402
from actsnet.config import ACTSNetConfig                                         # noqa: E402
from actsnet.model import ACTSNet                                               # noqa: E402
from actsnet.train import set_seed                                              # noqa: E402
from baselines import BASELINES                                                 # noqa: E402

# perturbation levels applied to the query tensor (N, C*S, T)
PERTURBS = ["clean", "noise0.25", "noise0.50", "noise1.00", "chdrop0.2", "chdrop0.4"]

def perturb(x, kind, gen):
    if kind == "clean":
        return x
    if kind.startswith("noise"):
        s = float(kind[5:])
        return x + s * torch.randn(x.shape, generator=gen, device=x.device)
    if kind.startswith("chdrop"):
        p = float(kind[6:]); C = x.shape[1]
        mask = (torch.rand(x.shape[0], C, 1, generator=gen, device=x.device) > p).float()
        return x * mask
    return x


@torch.no_grad()
def eval_perturbed(model, X, y, tr_idx, te_idx, device, kind, gen, is_v1, max_support=5000):
    model.eval()
    xt = torch.from_numpy(X[te_idx]).to(device); yt = y[te_idx]
    if is_v1:
        si = balanced_support_idx(tr_idx, y, max_support, 0)
        sx = torch.from_numpy(X[si]).to(device); sy = torch.from_numpy(y[si]).to(device)
        semb = torch.cat([model.encode(sx[i:i+256]) for i in range(0, len(sx), 256)], 0)
    preds = []
    for i in range(0, len(xt), 256):
        xb = perturb(xt[i:i+256], kind, gen)
        lp = model.proto(model.encode(xb), semb, sy) if is_v1 else model(xb)
        preds.extend(lp.argmax(1).cpu().numpy())
    return balanced_accuracy_score(yt, np.array(preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--model", required=True, choices=["v1", "eegnet"])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--loso_max", type=int, default=20)
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X, y, groups = load_cache(args.cache_dir)
    ncls = int(len(np.unique(y))); nsubj = int(len(np.unique(groups)))
    is_v1 = args.model == "v1"
    splitter = LeaveOneGroupOut() if nsubj <= args.loso_max else GroupKFold(args.n_folds)
    folds = list(splitter.split(X, y, groups))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        for fi, (tr, te) in enumerate(folds):
            itr, ival = inner_split(tr, y, groups, 0.2, seed)
            cw = class_weight_from(y[itr], ncls, device) if args.class_weight else None
            if is_v1:
                cfg = ACTSNetConfig(n_channels=X.shape[1], n_classes=ncls, seed=seed, device=str(device))
                model = ACTSNet(cfg).to(device)
            else:
                model = BASELINES["eegnet"](n_chans=X.shape[1], n_classes=ncls, n_times=X.shape[2]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                   weight_decay=0.0 if is_v1 else 1e-4)
            trl = make_loader(X, y, itr, args.batch_size, True, drop_last=True)
            isup = make_loader(X, y, balanced_support_idx(itr, y, 5000, seed), args.batch_size, False)
            ivl = make_loader(X, y, ival, args.batch_size, False)
            best, best_state = -1.0, None
            for ep in range(1, args.epochs + 1):
                (train_epoch if is_v1 else train_epoch_clf)(model, trl, opt, device, cw)
                if ep % args.eval_every == 0 or ep == args.epochs:
                    vm = (eval_full(model, isup, ivl, device) if is_v1 else eval_clf(model, ivl, device))
                    if vm["balanced_accuracy"] > best:
                        best = vm["balanced_accuracy"]
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if best_state: model.load_state_dict(best_state)
            rec = {"seed": seed, "fold": fi}
            for kind in PERTURBS:
                rec[kind] = eval_perturbed(model, X, y, tr, te, device, kind, gen, is_v1)
            rows.append(rec)
            print(f"  seed {seed} fold {fi:2d} | " + " ".join(f"{k}={rec[k]:.3f}" for k in PERTURBS), flush=True)

    summary = {"dataset": os.path.basename(args.cache_dir), "model": args.model,
               "n_folds": len(rows), "perturbations": PERTURBS,
               "mean": {k: float(np.mean([r[k] for r in rows])) for k in PERTURBS},
               "std": {k: float(np.std([r[k] for r in rows])) for k in PERTURBS}}
    json.dump(rows, open(os.path.join(args.output_dir, "per_fold.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)
    print("\n" + args.model + " robustness (mean BACC):")
    for k in PERTURBS:
        print(f"  {k:<10} {summary['mean'][k]:.4f}")


if __name__ == "__main__":
    main()
