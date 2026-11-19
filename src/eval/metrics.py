"""Multilabel metrics: macro-AUC, per-class decision thresholds, threshold-optimised macro-F1.

Protocol (the paper's fix): per-class thresholds are grid-searched on the VALIDATION fold to
maximise each class's F1; the same thresholds are then applied unchanged to the test fold.
The threshold-optimised validation macro-F1 is also the model-selection / early-stopping metric.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def per_class_auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.full(y.shape[1], np.nan)
    for k in range(y.shape[1]):
        if 0 < y[:, k].sum() < len(y):
            out[k] = roc_auc_score(y[:, k], p[:, k])
    return out


def _f1_counts(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """y [N], pred [N, G] bool -> F1 per grid point [G] (F1 = 0 when tp+fp+fn = 0 is impossible here)."""
    yb = y.astype(bool)[:, None]
    tp = (pred & yb).sum(0)
    fp = (pred & ~yb).sum(0)
    fn = (~pred & yb).sum(0)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def find_thresholds(y: np.ndarray, p: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> np.ndarray:
    """Per-class threshold maximising F1 on (y, p). Ties -> the threshold closest to 0.5."""
    thr = np.empty(y.shape[1])
    for k in range(y.shape[1]):
        f1 = _f1_counts(y[:, k], p[:, k][:, None] >= grid[None, :])
        best = np.flatnonzero(f1 >= f1.max() - 1e-12)
        thr[k] = grid[best[np.argmin(np.abs(grid[best] - 0.5))]]
    return thr


def per_class_f1(y: np.ndarray, p: np.ndarray, thr) -> np.ndarray:
    thr = np.broadcast_to(np.asarray(thr, dtype=float), (y.shape[1],))
    return np.array([_f1_counts(y[:, k], (p[:, k] >= thr[k])[:, None])[0] for k in range(y.shape[1])])


def summarize(y: np.ndarray, p: np.ndarray, classes: list[str], thresholds=None) -> dict:
    """thresholds=None -> search them on (y, p) (validation); else apply the given ones (test)."""
    tuned = thresholds is None
    thr = find_thresholds(y, p) if tuned else np.asarray(thresholds, dtype=float)
    auc = per_class_auc(y, p)
    f1 = per_class_f1(y, p, thr)
    f1_05 = per_class_f1(y, p, 0.5)
    out = {
        "macro_auc": float(np.nanmean(auc)),
        "macro_f1": float(f1.mean()),
        "macro_f1_at_0.5": float(f1_05.mean()),
        "thresholds": thr.tolist(),
        "thresholds_tuned_here": tuned,
    }
    for k, c in enumerate(classes):
        out[f"auc_{c}"] = float(auc[k])
        out[f"f1_{c}"] = float(f1[k])
    return out
