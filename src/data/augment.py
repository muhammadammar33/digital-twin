"""Train-time signal augmentation registry (seam for accuracy experiments; empty by default).

Config:  data.augment: [{name: <registered name>, p: 0.5, ...kwargs}]
Each transform takes (x [12, T] float32 np.ndarray, rng) and returns an array of the same shape.
The same registry is meant to be reused for test-time augmentation (eval.tta) later.
Every new transform is a separate benchmarks.md row.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Transform = Callable[[np.ndarray, np.random.Generator], np.ndarray]
REGISTRY: dict[str, Callable[..., Transform]] = {}


def register(name: str):
    def deco(factory):
        REGISTRY[name] = factory
        return factory
    return deco


class Compose:
    def __init__(self, specs: list[dict] | None):
        self.ops: list[tuple[float, Transform]] = []
        for spec in specs or []:
            spec = dict(spec)
            name, p = spec.pop("name"), float(spec.pop("p", 1.0))
            if name not in REGISTRY:
                raise KeyError(f"augmentation {name!r} not registered; available: {sorted(REGISTRY)}")
            self.ops.append((p, REGISTRY[name](**spec)))

    def __bool__(self) -> bool:
        return bool(self.ops)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for p, op in self.ops:
            if rng.random() < p:
                x = op(x, rng)
        return x
