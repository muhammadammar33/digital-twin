"""AdamW + linear warm-up + cosine decay (stepped per batch)."""
from __future__ import annotations

import math

import torch
from torch import nn


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """No weight decay on biases, BatchNorm/LayerNorm weights, attention gates and lead embeddings."""
    oc = cfg.train.optim
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith("lead_emb") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(oc.weight_decay)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(oc.lr), betas=tuple(oc.betas), eps=float(oc.eps))


def warmup_cosine(total_steps: int, warmup_steps: int, final_lr_frac: float):
    """LR multiplier: 0 -> 1 linearly over warm-up, then cosine 1 -> final_lr_frac."""
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return final_lr_frac + (1 - final_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


def build_scheduler(opt: torch.optim.Optimizer, cfg, steps_per_epoch: int):
    sc = cfg.train.sched
    total = int(cfg.train.epochs) * steps_per_epoch
    warm = int(round(float(sc.warmup_epochs) * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, warm, float(sc.final_lr_frac)))
