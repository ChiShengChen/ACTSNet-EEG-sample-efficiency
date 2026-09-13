"""Computational-cost comparison for the JMIR revision (Reviewer L, comment 4).

The manuscript reports accuracy only. Reviewer L asks for, on the same hardware
(one RTX 3090): training time per fold, single-window inference latency, and memory
footprint for ACTSNet, EEGNet, ShallowConvNet and BigCNN -- separating the encoder
cost from the prototypical head's support-set overhead, since ACTSNet additionally
carries a support set of up to 5000 embeddings at inference time.

Run it with an otherwise idle GPU; concurrent jobs invalidate the timings.

    python bench_compute.py --cache_dir prep_cache/seed_iv --out results/compute_cost.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# The ACTSNet architecture lives in a separate repository:
#   https://github.com/ChiShengChen/ACTSNetv1
# Set ACTSNET_V1 to a local checkout, or clone it next to this repository.
sys.path.insert(0, os.environ.get("ACTSNET_V1",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ACTSNetv1")))
from actsnet.config import ACTSNetConfig                      # noqa: E402
from baselines import BASELINES                               # noqa: E402
from crossover import build_model                             # noqa: E402
from run_loso import load_cache, make_loader                  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_median(fn, warmup=10, trials=50):
    """Median wall time of `fn`, in milliseconds, after warm-up."""
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def build(name, C, T, n_classes, device):
    """Return (model, is_proto). Proto models take (x, support_x, support_labels)."""
    if name in ("actsnet", "actsnet_bigcnn"):
        cfg = ACTSNetConfig(n_channels=C, n_classes=n_classes, seed=42, device=str(device))
        cfg.n_times = T
        enc = "bigcnn" if name == "actsnet_bigcnn" else "actsnet"
        return build_model(cfg, enc, "on").to(device), True
    return BASELINES[name](n_chans=C, n_classes=n_classes, n_times=T).to(device), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="prep_cache/seed_iv")
    ap.add_argument("--models", nargs="+",
                    default=["actsnet", "eegnet", "shallowconv", "bigcnn", "transformer"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--support_size", type=int, default=5000)
    ap.add_argument("--epoch_batches", type=int, default=30,
                    help="batches timed to estimate per-epoch training cost")
    ap.add_argument("--out", default="results/compute_cost.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X, y, _ = load_cache(args.cache_dir)
    N, C, T = X.shape
    n_classes = int(y.max()) + 1
    print(f"cache {args.cache_dir}: {N} windows, C={C}, T={T}, {n_classes} classes\n")

    idx = np.arange(min(N, args.batch_size * args.epoch_batches))
    loader = make_loader(X, y, idx, args.batch_size, True, drop_last=True)
    sup_idx = np.arange(min(N, args.support_size))
    sup_x = torch.from_numpy(X[sup_idx]).to(device)
    sup_y = torch.from_numpy(y[sup_idx]).to(device)
    one = torch.from_numpy(X[:1]).to(device)                  # single window

    rows = []
    for name in args.models:
        model, is_proto = build(name, C, T, n_classes, device)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # --- training: time a fixed number of steps, scale to a full epoch ---
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _sync(); t0 = time.perf_counter(); nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb, xb, yb) if is_proto else model(xb)
            loss = (torch.nn.functional.nll_loss(out, yb) if is_proto
                    else torch.nn.functional.cross_entropy(out, yb))
            opt.zero_grad(); loss.backward(); opt.step(); nb += 1
        _sync()
        ms_per_step = (time.perf_counter() - t0) * 1000 / max(nb, 1)
        train_mem = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0

        # --- inference ---
        model.eval()
        with torch.no_grad():
            if is_proto:
                # encoder-only cost, kept separate from the support-set overhead.
                # The support set is encoded in chunks of 64, exactly as run_loso.eval_full
                # does at inference; encoding 5000 windows in one call needs ~5 GiB and is
                # neither what deployment does nor safe on a shared GPU.
                def encode_support():
                    return torch.cat([model.encode(sup_x[i:i + 64]) for i in range(0, len(sup_x), 64)])
                enc_ms = time_median(lambda: model.encode(one))
                sup_ms = time_median(encode_support, warmup=2, trials=5)   # one-off per support set
                semb = encode_support()
                # per-query cost once prototypes exist: encode one window + distances to K prototypes
                full_ms = time_median(lambda: model.proto(model.encode(one), semb, sup_y))
            else:
                enc_ms = time_median(lambda: model(one))
                sup_ms = 0.0
                full_ms = enc_ms
        infer_mem = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0

        rows.append(dict(model=name, params=n_par, ms_per_train_step=round(ms_per_step, 2),
                         train_peak_mem_mb=round(train_mem, 1),
                         encoder_latency_ms=round(enc_ms, 3),
                         support_encode_ms=round(sup_ms, 1),
                         full_inference_ms=round(full_ms, 3),
                         peak_mem_mb=round(infer_mem, 1)))
        print(f"{name:14s} params {n_par:>9,} | train {ms_per_step:7.1f} ms/step "
              f"| encoder {enc_ms:6.3f} ms | support({len(sup_idx)}) {sup_ms:8.1f} ms "
              f"| peak {infer_mem:7.1f} MB")
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta = dict(cache=args.cache_dir, N=int(N), C=int(C), T=int(T),
                batch_size=args.batch_size, support_size=int(len(sup_idx)),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                rows=rows)
    json.dump(meta, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
