"""Train-time ECG augmentation (config `data.augment`, `data.mixup`). Off in B0/A1.

Per-record transforms run inside the Dataset AFTER per-lead z-scoring, on the training split only,
so amplitudes are in normalised units (multiples of a lead's train-set std). Each entry is
    {name: <transform>, p: <probability per record>, ...kwargs}
and is applied independently with probability p, in list order:

  time_shift       circular shift by U(-max_shift_s, +max_shift_s) seconds (mode roll) or a shift
                   with zero fill (mode pad)
  baseline_wander  adds a sum of `n_components` low-frequency sinusoids (freq_hz range) per lead,
                   amplitude U(0, max_amplitude), random phase: simulates respiration/motion drift
  gaussian_noise   adds N(0, sigma^2) with sigma ~ U(sigma_min, sigma_max)
  amplitude_scale  multiplies by U(min_scale, max_scale), per lead (per_lead: true) or globally
  lead_dropout     zeroes each lead with probability p_lead (at most max_leads, never all 12)

Mixup is batch-level (it mixes records AND their multilabel targets), so it lives in the training
loop (`mixup_batch`), not here: x = lam*x_i + (1-lam)*x_j, y = lam*y_i + (1-lam)*y_j, lam ~ Beta(a, a).

Defaults below are common ECG-augmentation choices, not values taken from the paper.
Adding a transform: decorate a factory with @register("name"); it receives fs plus its kwargs and
returns f(x [12, T] float32, rng) -> same shape. Every new transform is its own benchmarks.md row.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

Transform = Callable[[np.ndarray, np.random.Generator], np.ndarray]
REGISTRY: dict[str, Callable[..., Transform]] = {}


def register(name: str):
    def deco(factory):
        REGISTRY[name] = factory
        return factory
    return deco


@register("time_shift")
def time_shift(fs: int, max_shift_s: float = 1.0, mode: str = "roll") -> Transform:
    max_shift = int(round(max_shift_s * fs))

    def f(x, rng):
        s = int(rng.integers(-max_shift, max_shift + 1))
        if s == 0:
            return x
        if mode == "roll":
            return np.roll(x, s, axis=-1)
        out = np.zeros_like(x)
        if s > 0:
            out[:, s:] = x[:, :-s]
        else:
            out[:, :s] = x[:, -s:]
        return out
    return f


@register("baseline_wander")
def baseline_wander(fs: int, max_amplitude: float = 0.3, freq_hz=(0.05, 0.5),
                    n_components: int = 2) -> Transform:
    def f(x, rng):
        C, T = x.shape
        t = np.arange(T, dtype=np.float32) / fs
        wander = np.zeros((C, T), dtype=np.float32)
        for _ in range(n_components):
            freq = rng.uniform(freq_hz[0], freq_hz[1], size=(C, 1))
            amp = rng.uniform(0, max_amplitude, size=(C, 1))
            phase = rng.uniform(0, 2 * np.pi, size=(C, 1))
            wander += (amp * np.sin(2 * np.pi * freq * t[None, :] + phase)).astype(np.float32)
        return x + wander
    return f


@register("gaussian_noise")
def gaussian_noise(fs: int, sigma_min: float = 0.01, sigma_max: float = 0.05) -> Transform:
    def f(x, rng):
        sigma = rng.uniform(sigma_min, sigma_max)
        return x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return f


@register("amplitude_scale")
def amplitude_scale(fs: int, min_scale: float = 0.8, max_scale: float = 1.2, per_lead: bool = True) -> Transform:
    def f(x, rng):
        size = (x.shape[0], 1) if per_lead else (1, 1)
        return x * rng.uniform(min_scale, max_scale, size=size).astype(np.float32)
    return f


@register("lead_dropout")
def lead_dropout(fs: int, p_lead: float = 0.1, max_leads: int = 3) -> Transform:
    def f(x, rng):
        drop = np.flatnonzero(rng.random(x.shape[0]) < p_lead)
        if len(drop) == 0:
            return x
        if len(drop) > max_leads:
            drop = rng.choice(drop, size=max_leads, replace=False)
        drop = drop[: x.shape[0] - 1]                     # never drop every lead
        x = x.copy()
        x[drop] = 0.0
        return x
    return f


class Compose:
    def __init__(self, specs: list[dict] | None, fs: int = 100):
        self.ops: list[tuple[str, float, Transform]] = []
        for spec in specs or []:
            spec = dict(spec)
            name, p = spec.pop("name"), float(spec.pop("p", 1.0))
            if spec.pop("enabled", True) is False:
                continue
            if name not in REGISTRY:
                raise KeyError(f"augmentation {name!r} not registered; available: {sorted(REGISTRY)}")
            self.ops.append((name, p, REGISTRY[name](fs=fs, **spec)))

    def __bool__(self) -> bool:
        return bool(self.ops)

    def names(self) -> list[str]:
        return [n for n, _, _ in self.ops]

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for _, p, op in self.ops:
            if rng.random() < p:
                x = op(x, rng)
        return x.astype(np.float32, copy=False)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, p: float,
                rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-level mixup with probability p. Targets become soft labels (fine for BCE)."""
    if alpha <= 0 or rng.random() >= p:
        return x, y
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)                         # keep the original record dominant
    perm = torch.as_tensor(rng.permutation(x.shape[0]), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
