# Cardiac Digital Twin — Stream 1 backbone

Phase 1 of a reproduction of Narigina et al., *A Neuro-Fuzzy Digital Twin for Interpretable
Cardiac Disease Recognition* (Appl. Sci. 2026): the accurate ECG classifier backbone
(xresnet1d101 + cross-lead attention → 512-d embedding → 5 superclass logits) on PTB-XL.
Target: test-fold macro-AUC ≈ 0.928 (Strodthoff et al. baseline).

> Status: milestone 1 (data preparation + sanity checks). Later milestones add the Dataset,
> model, training loop and evaluation.

## Setup

```bash
pip install -r requirements.txt          # for CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Get PTB-XL v1.0.3 from PhysioNet (≈1.8 GB zip, public):

```bash
curl -L -o ptb-xl-1.0.3.zip https://physionet.org/content/ptb-xl/get-zip/1.0.3/ && unzip ptb-xl-1.0.3.zip
```

The dataset path is never hard-coded; pass it with `--set data.root=...` (or edit `configs/default.yaml`).

## 1. Prepare data (labels, splits, signal cache, sanity checks)

```bash
python scripts/prepare_data.py --config configs/default.yaml --set data.root=/path/to/ptb-xl-...-1.0.3
```

For 500 Hz, add `data.sampling_rate=500` (builds a separate ~5 GB memory-mapped cache).

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
