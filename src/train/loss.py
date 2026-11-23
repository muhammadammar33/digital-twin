"""Loss registry (config `train.loss.type`). B0 / A1 / A2 use plain BCEWithLogits.

  bce    BCEWithLogits, optional per-class pos_weight (list of 5).
  asl    Asymmetric loss (Ridnik et al., ICCV 2021) for multilabel imbalance:
             L+ = (1-p)^gamma_pos * log(p)
             L- = p_m^gamma_neg * log(1 - p_m),   p_m = max(p - clip, 0)
         gamma_neg > gamma_pos down-weights the many easy negatives (most records are not HYP);
         clip (probability margin m) discards very easy negatives entirely.
         Defaults gamma_pos=0, gamma_neg=4, clip=0.05 follow the paper's recommendation.
  focal  Focal loss (Lin et al. 2017), symmetric: (1-p_t)^gamma * BCE, optional alpha balancing
         (alpha weights positives, 1-alpha negatives; null = no balancing).

All losses: optional `class_weights` (list of 5) multiplies each class's term (e.g. up-weight HYP),
mean reduction over batch x classes (same scale convention as BCE), soft targets allowed (mixup).
With gamma_pos = gamma_neg = 0 and clip = 0, `asl` equals `bce`; focal with gamma = 0 and alpha null
equals `bce` (checked in scripts/check_loss.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _Weighted(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        w = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", w, persistent=False)

    def reduce(self, per_elem: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            per_elem = per_elem * self.class_weights
        return per_elem.mean()


class AsymmetricLoss(_Weighted):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05,
                 eps: float = 1e-8, class_weights=None):
        super().__init__(class_weights)
        self.gamma_pos, self.gamma_neg, self.clip, self.eps = float(gamma_pos), float(gamma_neg), float(clip), eps

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        p = torch.sigmoid(logits)
        # positive term: log p via logsigmoid (stable)
        log_p = F.logsigmoid(logits)
        loss_pos = y * log_p
        if self.gamma_pos > 0:
            loss_pos = loss_pos * (1 - p).pow(self.gamma_pos)
        # negative term with probability shifting
        if self.clip > 0:
            p_m = (p - self.clip).clamp(min=0)
            log_1m = torch.log((1 - p_m).clamp(min=self.eps))
        else:
            p_m = p
            log_1m = F.logsigmoid(-logits)
        loss_neg = (1 - y) * log_1m
        if self.gamma_neg > 0:
            loss_neg = loss_neg * p_m.pow(self.gamma_neg)
        return self.reduce(-(loss_pos + loss_neg))


class FocalLoss(_Weighted):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, class_weights=None):
        super().__init__(class_weights)
        self.gamma, self.alpha = float(gamma), alpha

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        p_t = y * p + (1 - y) * (1 - p)
        loss = bce * (1 - p_t).pow(self.gamma) if self.gamma > 0 else bce
        if self.alpha is not None:
            loss = loss * (y * self.alpha + (1 - y) * (1 - self.alpha))
        return self.reduce(loss)


def build_loss(cfg) -> nn.Module:
    lc = cfg.train.loss
    cw = lc.get("class_weights")
    if lc.type == "bce":
        pw = lc.get("pos_weight")
        if cw:
            raise ValueError("bce: use pos_weight (or switch to asl/focal for class_weights)")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32) if pw else None)
    if lc.type == "asl":
        a = lc.get("asl") or {}
        return AsymmetricLoss(a.get("gamma_pos", 0.0), a.get("gamma_neg", 4.0), a.get("clip", 0.05),
                              class_weights=cw)
    if lc.type == "focal":
        f = lc.get("focal") or {}
        return FocalLoss(f.get("gamma", 2.0), f.get("alpha"), class_weights=cw)
    raise KeyError(f"unknown loss {lc.type!r}; implemented: bce, asl, focal")
