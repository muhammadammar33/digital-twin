"""Per-lead z-score normalisation with statistics from the TRAIN split only (no leakage).

Two modes (config data.normalization.type):
  per_lead_zscore    (default) mean/std per lead over all train records & timesteps;
                     the same 12 means/stds are applied to train, val and test.
  per_record_zscore  each record standardised by its own per-lead mean/std (an ablation
                     option; uses no dataset statistics at all, so it cannot leak either).
Both divide by (std + eps) so a flat lead can never cause a divide-by-zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_train_stats(X: np.ndarray, train_rows: np.ndarray, chunk: int = 1024) -> dict:
    """Streaming float64 per-lead mean/std over X[train_rows] ([N, 12, T] memmap)."""
    train_rows = np.sort(np.asarray(train_rows))
    n_leads = X.shape[1]
    s = np.zeros(n_leads)
    ss = np.zeros(n_leads)
    count = 0
    for i in range(0, len(train_rows), chunk):
        x = np.asarray(X[train_rows[i:i + chunk]], dtype=np.float64)
        s += x.sum(axis=(0, 2))
        ss += (x ** 2).sum(axis=(0, 2))
        count += x.shape[0] * x.shape[2]
    mean = s / count
    std = np.sqrt(np.maximum(ss / count - mean ** 2, 0.0))
    return {"mean": mean.tolist(), "std": std.tolist(), "n_records": int(len(train_rows))}


def load_or_compute_stats(path: Path, X: np.ndarray, train_rows: np.ndarray,
                          train_folds: list[int]) -> dict:
    """Cached in the cache dir; recomputed if the train folds or train set size changed."""
    if path.exists():
        st = json.loads(path.read_text())
        if st.get("train_folds") == list(train_folds) and st.get("n_records") == len(train_rows):
            return st
    st = compute_train_stats(X, train_rows)
    st["train_folds"] = list(train_folds)
    path.write_text(json.dumps(st, indent=2))
    return st


class Normalizer:
    def __init__(self, kind: str, eps: float, mean=None, std=None):
        if kind not in ("per_lead_zscore", "per_record_zscore", "none"):
            raise ValueError(f"unknown normalization type {kind!r}")
        self.kind, self.eps = kind, float(eps)
        if kind == "per_lead_zscore":
            self.mean = np.asarray(mean, dtype=np.float32)[:, None]   # [12, 1]
            self.std = np.asarray(std, dtype=np.float32)[:, None]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "per_lead_zscore":
            return (x - self.mean) / (self.std + self.eps)
        if self.kind == "per_record_zscore":
            return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + self.eps)
        return x
