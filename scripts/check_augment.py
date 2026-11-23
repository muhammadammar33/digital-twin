#!/usr/bin/env python
"""Checks for the augmentation suite (configs/a2_aug.yaml): shapes, determinism, effect sizes,
train-only application, mixup soft labels.

    python scripts/check_augment.py [--config configs/a2_aug.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.augment import REGISTRY, Compose, mixup_batch  # noqa: E402
from src.data.dataset import build_dataloaders  # noqa: E402
from src.utils.config import load_config, seed_everything  # noqa: E402

FAIL: list[str] = []


def check(cond, msg):
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAIL.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/a2_aug.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    fs = int(cfg.data.sampling_rate)
    T = 10 * fs
    x = np.random.default_rng(0).standard_normal((12, T)).astype(np.float32)

    print("Per-record transforms (p=1, defaults from the config)")
    specs = {s["name"]: {k: v for k, v in s.items() if k not in ("name", "p")} for s in cfg.data.augment}
    for name in REGISTRY:
        f = REGISTRY[name](fs=fs, **specs.get(name, {}))
        a, b = f(x.copy(), np.random.default_rng(1)), f(x.copy(), np.random.default_rng(1))
        d = a - x
        print(f"  {name:<16} mean|Δ| {np.abs(d).mean():.4f}  max|Δ| {np.abs(d).max():.3f}")
        check(a.shape == x.shape and a.dtype == np.float32, f"{name}: shape/dtype kept")
        check(np.array_equal(a, b), f"{name}: deterministic for a fixed rng")

    rng = np.random.default_rng(3)
    sh = REGISTRY["time_shift"](fs=fs, max_shift_s=1.0)(x, rng)
    shifts = [s for s in range(-fs, fs + 1) if np.array_equal(np.roll(x, s, axis=-1), sh)]
    check(len(shifts) == 1, f"time_shift is a pure circular shift ({shifts[0] if shifts else '?'} samples)")
    sc = REGISTRY["amplitude_scale"](fs=fs, min_scale=0.8, max_scale=1.2)(x, rng) / x
    check(0.8 <= sc.min() and sc.max() <= 1.2, "amplitude_scale within [0.8, 1.2]")
    bw = REGISTRY["baseline_wander"](fs=fs, max_amplitude=0.3, freq_hz=(0.05, 0.5))(np.zeros_like(x), rng)
    power = (np.abs(np.fft.rfft(bw * np.hanning(T), axis=-1)) ** 2).sum(0)   # Hann window: limits leakage
    freqs = np.fft.rfftfreq(T, 1 / fs)
    frac = power[freqs > 1.0].sum() / power.sum()
    check(frac < 0.01, f"baseline_wander power is below 1 Hz ({100 * frac:.3f}% above 1 Hz)")
    drops = [int((REGISTRY["lead_dropout"](fs=fs, p_lead=0.5, max_leads=3)(x, np.random.default_rng(s))
                  == 0).all(axis=1).sum()) for s in range(200)]
    check(max(drops) <= 3 and min(drops) >= 0, f"lead_dropout drops 0..3 leads (seen {min(drops)}..{max(drops)})")
    noise = REGISTRY["gaussian_noise"](fs=fs, sigma_min=0.05, sigma_max=0.05)(np.zeros_like(x), rng)
    check(abs(noise.std() - 0.05) < 0.005, f"gaussian_noise sigma 0.05 (measured {noise.std():.4f})")
    check(np.array_equal(Compose([{"name": "gaussian_noise", "p": 0.0}], fs)(x, rng), x), "p=0 is a no-op")
    check(not Compose([{"name": "time_shift", "enabled": False}], fs), "enabled: false removes a transform")

    print("Mixup")
    xb, yb = torch.randn(8, 12, T), (torch.rand(8, 5) > 0.5).float()
    xm, ym = mixup_batch(xb, yb, alpha=0.2, p=1.0, rng=np.random.default_rng(0))
    check(xm.shape == xb.shape and ym.shape == yb.shape, "mixup keeps shapes")
    check(bool(((ym > 0) & (ym < 1)).any()) and float(ym.min()) >= 0 and float(ym.max()) <= 1,
          "mixup produces soft labels in [0, 1]")
    x0, y0 = mixup_batch(xb, yb, alpha=0.2, p=0.0, rng=np.random.default_rng(0))
    check(x0 is xb and y0 is yb, "mixup p=0 is a no-op")

    print("Dataset integration")
    out = build_dataloaders(cfg)
    tr, va = out["datasets"]["train"], out["datasets"]["val"]
    check(tr.augment is not None and tr.augment.names() == [s["name"] for s in cfg.data.augment],
          f"train dataset augments: {tr.augment.names() if tr.augment else None}")
    check(va.augment is None and out["datasets"]["test"].augment is None, "val/test are never augmented")
    tr.set_epoch(1); a = tr[0]["x"]
    tr.set_epoch(1); b = tr[0]["x"]
    tr.set_epoch(2); c = tr[0]["x"]
    check(torch.equal(a, b) and not torch.equal(a, c), "train augmentation reproducible per epoch, varies across epochs")
    check(torch.equal(va[0]["x"], va[0]["x"]), "val item is deterministic")

    print("\nALL CHECKS PASSED" if not FAIL else "\nFAILED: " + "; ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
