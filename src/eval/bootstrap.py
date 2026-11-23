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


def paired_delta(y: np.ndarray, p_new: np.ndarray, thr_new, p_ref: np.ndarray, thr_ref,
                 hyp_idx: int, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired bootstrap of (new - ref) on the SAME test records, each model at its own frozen
    validation thresholds. Returns point delta, percentile CI and the fraction of replicates
    with delta <= 0 (a one-sided bootstrap p-value for 'new is better')."""
    y = y.astype(bool)
    pred_n = p_new >= np.asarray(thr_new, dtype=float)[None, :]
    pred_r = p_ref >= np.asarray(thr_ref, dtype=float)[None, :]
    N, K = y.shape

    def stats(idx):
        yb = y[idx]
        out = {}
        for tag, pred, prob in (("new", pred_n, p_new), ("ref", pred_r, p_ref)):
            pb, prb = pred[idx], prob[idx]
            f1 = _f1_from_counts((pb & yb).sum(0), (pb & ~yb).sum(0), (~pb & yb).sum(0))
            aucs = [roc_auc_score(yb[:, k], prb[:, k]) for k in range(K) if 0 < yb[:, k].sum() < len(idx)]
            out[tag] = (f1.mean(), float(np.mean(aucs)), f1[hyp_idx])
        return np.array(out["new"]) - np.array(out["ref"])

    point = stats(np.arange(N))
    rng = np.random.default_rng(seed)
    reps = np.stack([stats(rng.integers(0, N, N)) for _ in range(n_boot)])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    res = {"n_boot": n_boot, "seed": seed}
    for j, name in enumerate(("macro_f1", "macro_auc", "hyp_f1")):
        res[name] = {"delta": float(point[j]),
                     "ci": [float(np.percentile(reps[:, j], lo)), float(np.percentile(reps[:, j], hi))],
                     "p_delta_le_0": float((reps[:, j] <= 0).mean())}
    return res
