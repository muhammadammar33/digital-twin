"""Cropping and overlapping-window helpers (the 2.5 s crop fallback and test-time window aggregation).

Training with input.mode=random_crop takes one random crop per sample per epoch (Strodthoff).
At eval time the full 10 s record is split into overlapping windows, each window is scored,
and the window logits are reduced per class (max as in Strodthoff, or mean).
"""
from __future__ import annotations

import numpy as np
import torch


def random_crop(x: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    T = x.shape[-1]
    if length >= T:
        return x
    start = int(rng.integers(0, T - length + 1))
    return x[..., start:start + length]


def window_starts(T: int, length: int, stride: int) -> list[int]:
    """Start indices covering [0, T); the last window is aligned to the end so no tail is lost."""
    if length >= T:
        return [0]
    starts = list(range(0, T - length + 1, stride))
    if starts[-1] != T - length:
        starts.append(T - length)
    return starts


def sliding_windows(x: torch.Tensor, length: int, stride: int) -> torch.Tensor:
    """[B, C, T] -> [B, W, C, length]."""
    starts = window_starts(x.shape[-1], length, stride)
    return torch.stack([x[..., s:s + length] for s in starts], dim=1)


def aggregate_windows(logits: torch.Tensor, reduce: str = "max") -> torch.Tensor:
    """[B, W, K] -> [B, K]."""
    if reduce == "max":
        return logits.max(dim=1).values
    if reduce == "mean":
        return logits.mean(dim=1)
    raise ValueError(f"unknown window reduce {reduce!r}")
