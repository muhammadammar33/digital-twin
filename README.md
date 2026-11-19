# Cardiac Digital Twin — Stream 1 backbone

Phase 1 of a reproduction of Narigina et al., *A Neuro-Fuzzy Digital Twin for Interpretable
Cardiac Disease Recognition* (Appl. Sci. 2026): the accurate ECG classifier backbone
(xresnet1d101 + cross-lead attention → 512-d embedding → 5 superclass logits) on PTB-XL.
Target: test-fold macro-AUC ≈ 0.928 (Strodthoff et al. baseline).

> Status: milestone 1 (data preparation + sanity checks). Later milestones add the Dataset,
> model, training loop and evaluation.

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

## Repo layout

```
configs/        YAML configs (all experiment settings)
src/data/       label mapping, splits, signal cache, Dataset + normalisation
src/models/     xresnet1d101 backbone, cross-lead attention, head
src/train/      training loop, losses, optimiser/schedule, EMA/SWA
src/eval/       metrics, threshold search, confusion matrices
scripts/        prepare_data.py, train.py, evaluate.py
```
