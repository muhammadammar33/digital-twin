# Cardiac Digital Twin — Stream 1 backbone

Phase 1 of a reproduction of Narigina et al., *A Neuro-Fuzzy Digital Twin for Interpretable
Cardiac Disease Recognition* (Appl. Sci. 2026): the accurate ECG classifier backbone
(xresnet1d101 + cross-lead attention → 512-d embedding → 5 superclass logits) on PTB-XL.
Target: test-fold macro-AUC ≈ 0.928 (Strodthoff et al. baseline).

> Status: milestones 1–4 done (data prep, Dataset/normalisation, model, training loop). Next:
> full evaluation + B0 baseline reproduction (5). Scope: **Stream 1 only** (no interpretation layer yet).

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

## 4. Train

Single-batch overfit test (≈2 min on CPU; loss should approach 0):

```bash
python scripts/train.py --overfit-batch
```

Full B0 baseline run (50 epochs, AdamW peak LR 2e-3, wd 1e-4, 2-epoch warm-up + cosine, BCE):

```bash
python scripts/train.py --run-name b0
```

Continue an interrupted run from its last completed epoch (exact: same weights, optimizer,
schedule, EMA/SWA, early-stopping state and RNG as an uninterrupted run):

```bash
python scripts/train.py --run-name b0 --resume
```

Variants are config overrides, e.g. the first improvement row:

```bash
python scripts/train.py --run-name a1_lead --set model.attention.type=lead
```

Each epoch logs train/val loss, val macro-AUC, val macro-F1 (per-class thresholds tuned on val),
macro-F1 at 0.5 and per-class AUC to the console, `runs/<name>/train.log` and `runs/<name>/log.csv`.
`last.pt` is rewritten after every epoch (atomically, so a kill mid-save cannot corrupt it).
`best.pt` holds the epoch with the best **threshold-optimised val macro-F1**, together with its val
thresholds. The same metric drives early stopping (`train.early_stopping.patience`, default 15).
EMA (`train.ema.enabled`) and SWA (`train.swa.enabled`, writes `swa.pt`) are off in B0.

On CUDA, fp16 mixed precision is used automatically (`train.amp: auto`). On CPU the B0 run takes
about 12 min per epoch, so use a GPU for the full run.

### Running B0 on Kaggle

1. Notebook settings: GPU on. Internet on as well, if you want the exact PhysioNet v1.0.3 download.
2. The repo is private, so cloning needs a token. Create a GitHub fine-grained token with read-only
   *Contents* access to this repo only, and save it in Kaggle under *Add-ons → Secrets* as `GITHUB_TOKEN`.
   Then, in a Python cell (Kaggle already has torch with CUDA, so only wfdb is installed):

```python
from kaggle_secrets import UserSecretsClient; import subprocess, os
tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
subprocess.run(["git", "clone", "-q", "-b", "claude/serene-maxwell-2hehyp",
                f"https://x-access-token:{tok}@github.com/muhammadammar33/digital-twin.git"], check=True)
os.chdir("digital-twin")
# drop the token from .git/config: /kaggle/working is saved as notebook output
subprocess.run(["git", "remote", "set-url", "origin",
                "https://github.com/muhammadammar33/digital-twin.git"], check=True)
subprocess.run(["pip", "install", "-q", "wfdb"], check=True)
```

   Run the shell commands below from `/kaggle/working/digital-twin`, with a `!` prefix in a notebook cell.

3. Point at the data. With internet on, run `python scripts/download_data.py` and leave out `--set`.
   With a Kaggle PTB-XL dataset attached, find its folder with `find /kaggle/input -name ptbxl_database.csv`
   and check that `prepare_data.py` prints the expected counts (21,388 records; 17,084 / 2,146 / 2,158).
   Some Kaggle uploads are older PTB-XL versions.

One-line B0 run (prepare the cache, then train):

```bash
python scripts/prepare_data.py --set data.root=/kaggle/input/<dataset>/<folder-with-ptbxl_database.csv> && python scripts/train.py --run-name b0
```

Resume in a new session: repeat step 2, copy the old run folder (from the previous version's output
attached as input) into `runs/`, rebuild the cache, and continue:

```bash
mkdir -p runs && cp -r /kaggle/input/<previous-output>/digital-twin/runs/b0 runs/ && python scripts/prepare_data.py --set data.root=/kaggle/input/<dataset>/<folder> && python scripts/train.py --run-name b0 --resume
```

Use "Save Version → Save & Run All" so the run keeps going after you close the tab and `runs/b0`
is kept as notebook output.

## Accuracy roadmap: where each lever plugs in

Defaults are the baseline. Every lever is off by default and is enabled from the config.

| Lever | Config key | Code | Status |
|---|---|---|---|
| Cross-lead / temporal attention | `model.attention.type` | `src/models/attention.py` | done (M3); first benchmark row vs B0 |
| 2.5 s random crops (Strodthoff fallback) | `data.input.mode: random_crop` | `src/data/windows.py` | done (M2) |
| Overlapping-window aggregation at eval | `eval.window_agg` | `src/eval/predict.py` | done (M4); auto-on with crops |
| Train-time augmentation / TTA | `data.augment`, `eval.tta` | `src/data/augment.py` registry | seam only (empty) |
| HYP oversampling | `data.sampler.type: hyp_oversample` | `src/data/dataset.py` | done (M2) |
| Asymmetric + focal loss | `train.loss.type` | `src/train/loss.py` registry | seam (bce only so far) |
| EMA / SWA | `train.ema.enabled`, `train.swa.enabled` | `src/train/averaging.py` | done (M4) |
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
