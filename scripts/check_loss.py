#!/usr/bin/env python
"""Checks for the loss registry: ASL/focal reduce to BCE in their limits, behave as intended,
handle soft (mixup) targets, and give finite gradients.

    python scripts/check_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.loss import AsymmetricLoss, FocalLoss, build_loss  # noqa: E402
from src.utils.config import load_config  # noqa: E402

FAIL = []


def check(cond, msg):
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAIL.append(msg)


torch.manual_seed(0)
logits = torch.randn(256, 5) * 3
y = (torch.rand(256, 5) < 0.2).float()
bce = F.binary_cross_entropy_with_logits(logits, y)

check(torch.allclose(AsymmetricLoss(0, 0, 0)(logits, y), bce, atol=1e-6), "ASL(γ+=0, γ-=0, clip=0) == BCE")
check(torch.allclose(FocalLoss(0, None)(logits, y), bce, atol=1e-6), "Focal(γ=0, α=None) == BCE")

# ASL: an easy negative (confidently negative) contributes ~0; a hard positive keeps full BCE weight
easy_neg = AsymmetricLoss(0, 4, 0.05)(torch.tensor([[-4.0]]), torch.tensor([[0.0]]))
check(easy_neg.item() == 0.0, f"ASL: easy negative (p=0.018 < clip) contributes 0 (got {easy_neg.item():.2e})")
hard_pos_asl = AsymmetricLoss(0, 4, 0.05)(torch.tensor([[-2.0]]), torch.tensor([[1.0]]))
hard_pos_bce = F.binary_cross_entropy_with_logits(torch.tensor([[-2.0]]), torch.tensor([[1.0]]))
check(torch.allclose(hard_pos_asl, hard_pos_bce), "ASL(γ+=0): positives keep the full BCE loss")
neg_asl = AsymmetricLoss(0, 4, 0.05)(logits, torch.zeros_like(y))
neg_bce = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(y))
check(neg_asl < 0.5 * neg_bce, f"ASL down-weights negatives (neg loss {neg_asl:.3f} vs BCE {neg_bce:.3f})")

# class weights scale only that class
w = [1, 1, 1, 1, 3]
base = AsymmetricLoss()(logits, y)
lw = AsymmetricLoss(class_weights=w)(logits, y)
check(lw > base, "class_weights [1,1,1,1,3] increase the loss (HYP up-weighted)")

# soft targets (mixup) and gradients
ys = torch.rand(256, 5)
for name, fn in (("asl", AsymmetricLoss()), ("focal", FocalLoss(2.0, 0.25))):
    lg = logits.clone().requires_grad_(True)
    out = fn(lg, ys)
    out.backward()
    check(torch.isfinite(out) and torch.isfinite(lg.grad).all(), f"{name}: finite loss and gradients with soft targets")

# registry from configs
for t in ("bce", "asl", "focal"):
    cfg = load_config("configs/default.yaml", [f"train.loss.type={t}"])
    check(build_loss(cfg)(logits, y).item() > 0, f"build_loss(type={t}) works from the config")
cfg = load_config("configs/a3_asl.yaml")
check(isinstance(build_loss(cfg), AsymmetricLoss) and cfg.model.attention.type == "lead"
      and len(cfg.data.augment) == 5 and cfg.data.sampler.type == "none",
      "configs/a3_asl.yaml = A2 (lead attention + augmentation, no oversampling) + ASL")

print("\nALL CHECKS PASSED" if not FAIL else "\nFAILED: " + "; ".join(FAIL))
sys.exit(1 if FAIL else 0)
