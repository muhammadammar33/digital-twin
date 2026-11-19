"""Non-parametric bootstrap 95% CIs on test-fold metrics.

Test records are resampled with replacement (B replicates, fixed seed). Decision thresholds are
the FROZEN validation thresholds in every replicate: they are never re-tuned, so the CI reflects
test-sampling variability of this fixed classifier (not threshold-selection variability).
Percentile intervals.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thresholds, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    y = y.astype(bool)
    pred = p >= np.asarray(thresholds, dtype=float)[None, :]
    rng = np.random.default_rng(seed)
    N, K = y.shape
    f1s = np.empty((n_boot, K))
    aucs = np.full((n_boot, K), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        yb, pb, prb = y[idx], pred[idx], p[idx]
        tp = (pb & yb).sum(0)
        fp = (pb & ~yb).sum(0)
        fn = (~pb & yb).sum(0)
        f1s[b] = _f1_from_counts(tp, fp, fn)
        for k in range(K):
            if 0 < yb[:, k].sum() < N:
                aucs[b, k] = roc_auc_score(yb[:, k], prb[:, k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def ci(a):
        return [float(np.nanpercentile(a, lo)), float(np.nanpercentile(a, hi))]

    return {
        "n_boot": n_boot, "seed": seed, "alpha": alpha,
        "macro_f1": ci(f1s.mean(1)),
        "macro_auc": ci(np.nanmean(aucs, 1)),
        "per_class_f1": [ci(f1s[:, k]) for k in range(K)],
        "per_class_auc": [ci(aucs[:, k]) for k in range(K)],
    }
