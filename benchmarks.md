# Stream 1 benchmarks: PTB-XL superdiagnostic, test fold 10

Protocol, identical for every row (otherwise the row is not comparable):
- Train folds 1–8, validation fold 9, test fold 10; 21,388 records (unlabelled dropped).
- Per-class decision thresholds grid-searched on **validation**, then applied unchanged to test.
- Model selection / early stopping on threshold-optimised **validation** macro-F1.
- **One change per row** relative to the row named in *vs*, so each gain is attributable.
- Report the mean ± std over seeds when a row is multi-seed; say so in the notes.

Reference targets (Narigina et al. 2026): macro-F1 **0.754**, macro-AUC **0.928**.

| # | config (file + overrides) | vs | macro-F1 | macro-AUC | HYP-F1 | notes |
|---|---|---|---|---|---|---|
| ref | Narigina et al. 2026, Stream 1 (reported) | | 0.754 | 0.928 | | paper numbers, not run here |
| B0 | `configs/default.yaml` (10 s, 100 Hz, BCE) | | | | | baseline reproduction (Milestone 5) |
