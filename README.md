# ACTSNet — analysis code for the small-sample clinical/affective EEG study

This repository contains **the code that produced every number reported in the
manuscript**, together with the per-fold results those numbers were computed from.

> **Relationship to `ChiShengChen/ACTSNetv1`.** That repository holds the ACTSNet
> *architecture* and a legacy training/evaluation pair. It does **not** contain the
> preprocessing pipeline, the cross-validation protocol, the baselines, or any of the
> analyses reported in the manuscript — those live here. Two scripts in that repository
> must not be used to reproduce the paper; see [Deprecated code paths](#deprecated-code-paths).

---

## What is here

| Path | Role |
|---|---|
| `pipeline/` | Raw EDF/`.mat` → `(N, C, S, T)` cache. `eeg_prep.py` (core), `montages.py` (channel selection incl. the 7-channel frontal montage), `build_{tuab,seediv,mumtaz,cavanagh}.py` (per-corpus adapters) |
| `run_loso.py` | ACTSNet driver: subject-grouped CV, inner-validation model selection, train-as-support evaluation, ablation switches |
| `run_loso_baseline.py` | Same protocol, same folds, same seeds, for the softmax baselines |
| `baselines.py` | EEGNet, ShallowConvNet, BigCNN (capacity-matched control), EEGTransformer |
| `crossover.py` | Encoder × head factorial: BigCNN trunk + prototypical head, and the multi-scale-branch ablation |
| `stats_paper1.py` | Paired Wilcoxon + rank-biserial + bootstrap CI, Holm–Bonferroni over the comparison family |
| `analyze_h1.py` | Learning curves and AULC |
| `revision_tables.py` | AUROC, robustness and diagnostic tables recomputed from stored per-fold results |
| `bench_compute.py` | Training time, single-window inference latency, memory; encoder cost separated from support-set overhead |
| `scripts/run_*.sh` | The exact commands used, one script per experiment family |
| `results/` | `per_fold.json` and `summary.json` for every run reported in the paper |

## Quick start

The analyses that only re-read the stored per-fold results need nothing but Python:

```bash
python stats_paper1.py               # all significance tests, Holm-corrected
python stats_paper1.py --unit seedfold   # the originally submitted (inflated-n) analysis
python analyze_h1.py mumtaz          # learning curve + AULC, with the arithmetic shown
python revision_tables.py            # AUROC / robustness / diagnostic tables
```

Re-training needs the ACTSNet architecture and the preprocessed caches:

```bash
git clone https://github.com/ChiShengChen/ACTSNetv1   # next to this repository
export ACTSNET_V1=../ACTSNetv1                        # or wherever you put it

export SEEDIV_ROOT=/path/to/SEED_IV/eeg_raw_data      # corpus roots, per dataset
python pipeline/build_seediv.py                       # -> prep_cache/seed_iv/

PY=python bash scripts/run_h1_seediv.sh               # or call run_loso.py directly
```

Preprocessed caches are **not** redistributed: they are 32 GB, and TUAB/TUEV are covered
by a Temple University data use agreement. Rebuild them with `pipeline/build_*.py`; the
exact arguments used for the paper are in the `scripts/run_*.sh` files. Corpus roots and
the interpreter are read from the environment (`SEEDIV_ROOT`, `TUAB_ROOT`, `MUMTAZ_ROOT`,
`CAV_ROOT`, `ACTSNET_V1`, `PY`), so no local path is baked into the code.

## Which script produced which table

| Manuscript item | Command | Results |
|---|---|---|
| Table 2, main benchmark | `run_loso.py` / `run_loso_baseline.py` at `--train_frac 1.0` | `results/<ds>{,_<model>}/` |
| Tables 3–6, Fig. 3–4, learning curves | `scripts/run_h1_*.sh`, `scripts/run_capacity.sh`, `scripts/run_cavanagh.sh` | `results/h1/` |
| Table 6, capacity-matched control | `scripts/run_capacity.sh` (`--model bigcnn`) | `results/h1/*_bigcnn_*` |
| Table 7, ablations | `run_loso.py --branch lstm` / `--head softmax` | `results/ablation/` |
| Frontal-7 montage | `scripts/run_h4_frontal7.sh` | `results/h4/` |
| Table 8, test-time robustness | `scripts/run_h5.sh` → `run_h5.py` | `results/h5/` |
| Confusion matrix | `build_confmat.py` with `--dump_preds` | `results/*_preds/` |
| All significance tests | `stats_paper1.py` | stdout |
| AULC | `analyze_h1.py <dataset>` | stdout |

## Protocol guarantees

These are the properties a reader should be able to check directly in the code:

- **No test-set model selection.** Checkpoints are selected on a group-disjoint inner
  validation split (`run_loso.py`, `best_balacc` updated only from `ival_loader`); the
  test fold is touched once, after selection, for the final evaluation.
- **No label leakage at evaluation.** Test-time prototypes are built from *training*
  embeddings only (`full_support` is drawn from `train_idx`), never from the query set.
- **Class-stratified support set.** `balanced_support_idx()` samples up to 5000 training
  embeddings with every class guaranteed present, which matters for class-block-ordered
  caches where a naive first-N sample would be single-class.
- **Subject-disjoint splits throughout.** Folds and the inner validation split are formed
  over subject groups, never over windows.
- **Trained from scratch.** No pretrained encoder or checkpoint is loaded anywhere in
  `run_loso.py` / `run_loso_baseline.py`; no result in the manuscript uses pretraining.
- **Trainability gate.** Every fold logs `inner_train windows → batches/epoch` and aborts
  if a configuration would yield zero batches. See [Corrections](#corrections).

## Deprecated code paths

Two scripts in `ChiShengChen/ACTSNetv1` implement neither the protocol described in the
manuscript nor the one used here, and **must not** be used to reproduce it:

- `actsnet/evaluate.py` — `evaluate_model()` passes the evaluation set as its own support
  set together with its ground-truth labels, so prototypes are built from the labels of
  the samples being classified.
- `actsnet/train.py` — partitions data with `random_split` over *windows*, placing
  segments from the same subject on both sides of the split.

Neither produced any reported result. They are retained there only as the original
single-file demo and are being marked deprecated in that repository.

`run_eegfm_benchmark.py` in that repository uses the EEG-FM-Bench splits and caches, not
the preprocessing described in the manuscript, and its README numbers (SEED-IV 0.3008,
TUAB 0.7457) are therefore **not** the manuscript's numbers (0.35, 0.77) and should not be
compared with them. The pipeline in this repository is the authoritative one for the paper.

## Corrections

While preparing the revision we found and fixed a defect that invalidated part of the
originally submitted low-data results:

`run_h1_mumtaz.sh` passed `--batch_size 256` to the baselines, but the Mumtaz cache holds
only ~20 windows per subject; at 10% and 25% of training subjects the inner-training set
(80 and 200 windows) was smaller than one batch, and with `drop_last=True` each epoch
yielded **zero** batches. Those baselines were therefore evaluated at their random
initialisation and never trained, which is why their scores were identical across two
different training fractions. Corrected runs (all models at a matched batch size) are in
`results/h1_recheck/`; `revision_tables.py --table dup` reproduces both the evidence and
the corrected curve. A trainability gate now aborts any run that would repeat this.

## Provenance

ACTSNet was originally proposed in the first author's 2021 National Taiwan University
master's thesis on MDD EEG classification for TMS treatment-response prediction
(Chen, Chi-Sheng; DOI [10.6342/NTU202101201](https://doi.org/10.6342/NTU202101201);
<https://tdr.lib.ntu.edu.tw/handle/123456789/82206>). The thesis introduced the
architecture and evaluated it on a private clinical cohort.

What is new in the present study, relative to the thesis: public-corpus validation under
a leakage-free subject-grouped protocol; the sample-efficiency question and the learning-
curve analysis; the size-matched capacity control and the encoder × head factorial; the
component ablations with correction for multiple comparisons; the reduced-montage and
test-time robustness analyses; and the preprocessing pipeline and evaluation code in this
repository. No result from the thesis is reused.

## Environment

Python 3.11, PyTorch 2.9.1+cu128, MNE 1.11, scikit-learn 1.7, SciPy 1.17; single RTX 3090.
Seeds 42/123/456 are fixed per run and recorded in every `summary.json`.
