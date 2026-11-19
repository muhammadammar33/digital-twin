"""Loss registry (config train.loss.type). B0 uses plain BCEWithLogits.

Seam: asymmetric multilabel loss + focal term and class weighting are added here later as
new registry entries; each is a separate benchmarks.md row against B0.
"""
from __future__ import annotations

import torch
from torch import nn


def build_loss(cfg) -> nn.Module:
    lc = cfg.train.loss
    if lc.type == "bce":
        pw = lc.get("pos_weight")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32) if pw else None)
    raise KeyError(f"unknown loss {lc.type!r}; implemented: bce")
