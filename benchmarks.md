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
| B0 | `configs/default.yaml`: pure xresnet1d101, 10 s, 100 Hz, BCE, avg-pool → 512 → 5 | | 0.734 [0.718–0.750] | 0.921 [0.914–0.929] | 0.588 [0.538–0.640] | baseline reproduction (Milestone 5); 34.07M params; run `b0` best.pt ep 6; test N=2158; 95% bootstrap CI B=1000; val thresholds frozen; 2026-11-23. Both ref-B0 targets inside the CI (AUC 0.928, F1 0.74): **reproduced**. Best epoch 6, then overfitting (val loss 0.28→0.44, train loss →0.13) |
| A1 | B0 + `model.attention.type=lead` (cross-lead MHA on input, zero-gated) | B0 | 0.748 [0.733–0.763]; Δ +0.014 (unpaired) | pending | pending | first improvement row; +4,962 params · **keeper**: all 5 classes improved vs B0; ref-A1 (0.754) inside CI, so it reproduces Narigina's Stream-1 predictor. Best val macro-F1 0.7605 (B0 0.7539), peak epoch 6, ~2x slower/epoch. Trained before commit 7634b10, so not bit-reproducible on GPU (memory-efficient attention backward). AUC / HYP-F1 / paired Δ pending: rerun `evaluate.py --run-name a1_lead --row A1 --vs B0 --vs-run b0` |
| A2 | `configs/a2_aug.yaml`: A1 + augmentation (time shift, baseline wander, Gaussian noise, amplitude scale, lead dropout, mixup) + early-stop patience 25 | B0, A1 | 0.751 [0.736–0.766]; ΔB0 +0.016 (P=0.009); ΔA1 +0.002 (P=0.35, ns) | 0.929 [0.921–0.936]; ΔB0 +0.008 (P=0.001); ΔA1 +0.006 (P=0.002) | 0.600 | **keeper**; paired bootstrap Δ (P = P(Δ≤0), B=1000). Significant vs B0 on F1 and AUC; vs A1 only AUC is significant. Matches the ref-twin numbers (0.751 / 0.929) at Stream-1 level. HYP-F1 unmoved (0.600). Δ CIs and HYP-F1 CI not transcribed (see the run's report.md). Patience 25 is bundled; an A1 + patience 25 control would isolate it |
| S1 | A1 + `data.sampler.type=hyp_oversample` (factor 2) | A1, B0 | | | | user-run: HYP oversampling **hurt**; numbers not recorded here. Not carried forward |
| A3 | `configs/a3_asl.yaml`: A2 + asymmetric loss (γ-=4, γ+=0, clip 0.05) | A2, B0 | | | | aimed at HYP-F1; one change vs A2 |
| E1 | 5-seed ensemble of A2 (seeds 42, 1337, 2024, 7, 314), probability-averaged; thresholds tuned on the ensemble's validation probabilities | B0, A1, A2 | | | | `scripts/run_seeds.py` → `scripts/ensemble.py` |
| A1b | B0 + `model.attention.type=temporal` (self-attention over T/32 tokens, zero-gated) | B0 | | | | alternative reading of the paper's attention; +1.31 M params |
