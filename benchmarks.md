# Stream 1 benchmarks: PTB-XL superdiagnostic, test fold 10

Protocol, identical for every row (otherwise the row is not comparable):
- Train folds 1–8, validation fold 9, test fold 10; 21,388 records (unlabelled dropped).
- Per-class decision thresholds grid-searched on **validation**, then applied unchanged to test.
- Model selection / early stopping on threshold-optimised **validation** macro-F1.
- **One change per row** relative to the row named in *vs*, so each gain is attributable.
- Report the mean ± std over seeds when a row is multi-seed; say so in the notes.

Reference lines (literature numbers, not run here). Each run is compared with ITS OWN reference:
- **ref-B0** is what **B0** must reproduce (`eval.targets.B0`).
- **ref-A1** is what **A1** (B0 + cross-lead attention) is compared with (`eval.targets.A1`).
- **ref-twin** is the full neuro-fuzzy twin from a later phase. It is listed for context only and is
  **not** a Stream 1 target.

Rows are written by `scripts/evaluate.py --row <id>`. Metric cells are `value [95% bootstrap CI]` on test fold 10.

| # | config (file + overrides) | vs | macro-F1 | macro-AUC | HYP-F1 | notes |
|---|---|---|---|---|---|---|
| ref-B0 | Strodthoff et al.: pure xresnet1d101 (attention off) | | 0.74 | 0.928 | | reference for B0 |
| ref-A1 | Narigina et al. 2026 Stream-1 predictor: xresnet1d101 + cross-lead attention | | 0.754 | | | reference for A1 |
| ref-twin | Narigina et al. 2026 full twin (with fuzzy layer; later phase) | | 0.751 [0.740–0.767] | | | context only, NOT a Stream 1 target |
| B0 | `configs/default.yaml`: pure xresnet1d101, 10 s, 100 Hz, BCE, avg-pool → 512 → 5 | | | | | baseline reproduction (Milestone 5); 34.07 M params |
| A1 | B0 + `model.attention.type=lead` (cross-lead MHA on input, zero-gated) | B0 | | | | first improvement row; +4,962 params |
| A1b | B0 + `model.attention.type=temporal` (self-attention over T/32 tokens, zero-gated) | B0 | | | | alternative reading of the paper's attention; +1.31 M params |
