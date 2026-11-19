"""Per-class (one-vs-rest) confusion matrices for the multilabel task at fixed thresholds."""
from __future__ import annotations

import numpy as np


def per_class_confusion(y: np.ndarray, p: np.ndarray, thresholds, classes: list[str]) -> list[dict]:
    y = y.astype(bool)
    pred = p >= np.asarray(thresholds, dtype=float)[None, :]
    out = []
    for k, c in enumerate(classes):
        tp = int((pred[:, k] & y[:, k]).sum())
        fp = int((pred[:, k] & ~y[:, k]).sum())
        fn = int((~pred[:, k] & y[:, k]).sum())
        tn = int((~pred[:, k] & ~y[:, k]).sum())
        out.append({
            "class": c, "threshold": float(thresholds[k]), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1), "support": tp + fn,
        })
    return out


def format_confusion(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        lines.append(
            f"{r['class']:<5} thr={r['threshold']:.2f}   "
            f"pred-  pred+\n"
            f"      true- {r['tn']:>6} {r['fp']:>6}    sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}\n"
            f"      true+ {r['fn']:>6} {r['tp']:>6}    prec {r['precision']:.3f}  support {r['support']}")
    return "\n".join(lines)
