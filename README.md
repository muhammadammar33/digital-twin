# Cardiac Digital Twin — Stream 1 backbone

Phase 1 of a reproduction of Narigina et al., *A Neuro-Fuzzy Digital Twin for Interpretable
Cardiac Disease Recognition* (Appl. Sci. 2026): the accurate ECG classifier backbone
(xresnet1d101 + cross-lead attention → 512-d embedding → 5 superclass logits) on PTB-XL.
Target: test-fold macro-AUC ≈ 0.928 (Strodthoff et al. baseline).

> Status: milestones 1–3 done (data prep, Dataset/normalisation, model). Next: training (4),
> evaluation + baseline reproduction (5). Scope: **Stream 1 only** (no interpretation layer yet).

Goal: reproduce the baseline first (reference line), then **beat** it (macro-F1 0.754 / macro-AUC
0.928) with raw-waveform-only improvements, each one a config toggle recorded as its own row in
[`benchmarks.md`](benchmarks.md).

## Setup (any machine: Linux, macOS, Windows)

Everything except the dataset and generated caches is in this repo. Starting from a fresh clone:

```bash
git clone https://github.com/muhammadammar33/digital-twin.git
```

```bash
cd digital-twin
```

```bash
python -m venv .venv
```

Activate it (`source .venv/bin/activate` on Linux/macOS, `.venv\Scripts\activate` on Windows), then
install PyTorch for your hardware first (pick the command at https://pytorch.org/get-started/locally/,
e.g. a CUDA build if you have an NVIDIA GPU), then the rest:

```bash
pip install -r requirements.txt
```

Download PTB-XL v1.0.3 (≈1.8 GB zip → 3.2 GB, public, CC BY 4.0) into `data/ptb-xl/`, checksum-verified:

```bash
python scripts/download_data.py
```

Already have PTB-XL somewhere? Skip the download and pass its folder instead:
`--set data.root=/path/to/ptb-xl` on any script (or edit `data.root` in `configs/default.yaml`).

Tested with Python 3.13, torch 2.15, numpy 2.5, pandas 3.0, scikit-learn 1.9, wfdb 4.3
(Python ≥ 3.10 required).

### What is (and isn't) in git

| Path | In git? | How to recreate |
|---|---|---|
| code, configs, README, requirements | yes | n/a |
| `data/ptb-xl/` (raw dataset, 3.2 GB) | no | `python scripts/download_data.py` |
| `artifacts/cache/` (label table + signal cache, 1–5 GB) | no | `python scripts/prepare_data.py` |
| `runs/` (checkpoints, logs) | no | produced by training |

Label table and splits are deterministic, so the cache rebuilt on your machine is identical to the one here.

## 1. Prepare data (labels, splits, signal cache, sanity checks)

```bash
python scripts/prepare_data.py
```

For 500 Hz (builds a separate ~5 GB memory-mapped cache):

```bash
python scripts/prepare_data.py --set data.sampling_rate=500
```

Expected result (100 Hz): 21,388 records kept (411 without a superclass dropped);
train/val/test = 17,084 / 2,146 / 2,158; 0 patients shared between splits; `ALL CHECKS PASSED`.

**Label protocol** (same as Strodthoff's `ecg_ptbxl_benchmarking`): each SCP code with
`diagnostic == 1` in `scp_statements.csv` contributes its `diagnostic_class`. Likelihood values
are ignored. Records with no superclass are dropped (`data.drop_unlabeled: true`).
Splits: `strat_fold` 1–8 train, 9 validation, 10 test. The folds are already split by patient,
and the script checks that no patient appears in more than one split.

## 2. Check the Dataset / normalisation

```bash
python scripts/check_data.py
```

Computes per-lead mean/std on the **train folds only** (cached in `artifacts/cache/norm_stats_<fs>hz.json`)
and applies them unchanged to val/test as `(x - mean) / (std + eps)`. Then checks one batch per split
(`x [64, 12, 1000]`, `y [64, 5]`), the flat-lead record, the 2.5 s crop fallback, the windowing helper,
seed reproducibility and the HYP oversampler.

## 3. Check the model

```bash
python scripts/check_model.py
```

B0 is the **pure xresnet1d101** (port of Strodthoff's `xresnet1d.py`: bottleneck blocks (3, 4, 23, 3),
widths (64, 128, 256, 512) ×4 expansion → 2048 channels, kernel 5) → average pooling → 512-d
projection (Linear-BN-ReLU) → dropout + Linear → 5 logits. 34.07 M parameters (33.02 M backbone).

`model(x)` returns `StreamOutput(logits [B,5], embedding [B,512])`. The embedding is the only path to
the logits (`model.head(model.encoder(x))`). `model.freeze_encoder()` freezes the whole encoder,
BatchNorm statistics included, for a later layer trained on the embedding.

Attention is a toggle (`model.attention.type`): `none` (B0), `lead` (cross-lead multi-head attention
over the 12 lead tokens at each time step, before the backbone), `temporal` (self-attention over the
backbone's time steps). Both are residual with a zero-initialised gate, so they start exactly equal to
B0. To add a variant, register a module in `src/models/attention.py`.

## Accuracy roadmap: where each lever plugs in

Defaults are the baseline. Every lever is off by default and is enabled from the config.

| Lever | Config key | Code | Status |
|---|---|---|---|
| Cross-lead / temporal attention | `model.attention.type` | `src/models/attention.py` | done (M3); first benchmark row vs B0 |
| 2.5 s random crops (Strodthoff fallback) | `data.input.mode: random_crop` | `src/data/windows.py` | done (M2) |
| Overlapping-window aggregation at eval | `eval.window_agg` | `src/data/windows.py` → predictor | helper done; wired in M5 |
| Train-time augmentation / TTA | `data.augment`, `eval.tta` | `src/data/augment.py` registry | seam only (empty) |
| HYP oversampling | `data.sampler.type: hyp_oversample` | `src/data/dataset.py` | done (M2) |
| Asymmetric + focal loss | `train.loss` | `src/train/loss.py` | M4 |
| EMA / SWA | `train.ema`, `train.swa` | `src/train/` | M4 |
| Multi-seed / multi-architecture ensembles | `ensemble` (list of run dirs); `model.backbone.name` (xresnet1d18…152) | `scripts/evaluate.py` averages saved probabilities | after M5 |
| Self-supervised pretraining (masked / contrastive) | `model.init_from` (backbone checkpoint) | loader done (`load_encoder_weights`); pretrain script after M5 | seam ready |
| 500 Hz input | `data.sampling_rate: 500` | everywhere | supported |

**Interpretation-layer boundary (kept clean, not built):** the model returns `(logits [B,5],
embedding [B,512])`. A later stage can attach to the 512-d embedding with the backbone frozen, so
the backbone is never retrained.

## Repo layout

```
configs/        YAML configs (all experiment settings)
src/data/       label mapping, splits, signal cache, Dataset, normalisation, crops/windows, augmentation registry
src/models/     xresnet1d101 backbone, cross-lead attention, head
src/train/      training loop, losses, optimiser/schedule, EMA/SWA
src/eval/       metrics, threshold search, confusion matrices
scripts/        download_data.py, prepare_data.py, check_data.py, check_model.py, train.py, evaluate.py
benchmarks.md   running results table (one change per row)
```
