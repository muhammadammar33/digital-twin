#!/usr/bin/env python
"""Milestone-2 checks: Dataset/DataLoader, train-fold-only normalisation, one-batch shapes.

    python scripts/check_data.py [--config configs/default.yaml] [--set key=value ...]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import build_dataloaders, cache_paths  # noqa: E402
from src.data.normalization import compute_train_stats  # noqa: E402
from src.data.windows import sliding_windows  # noqa: E402
from src.utils.config import load_config, seed_everything  # noqa: E402

FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAIL.append(msg)


def hr(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.seed)
    fs = int(cfg.data.sampling_rate)
    T = fs * 10

    out = build_dataloaders(cfg)
    rows, st = out["rows"], out["norm_stats"]
    X = np.load(cache_paths(cfg)["signals"], mmap_mode="r")

    hr("Normalisation statistics (train folds only)")
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    print(f"computed on {st['n_records']} records from folds {st['train_folds']}")
    print("lead   " + "".join(f"{l:>8}" for l in leads))
    print("mean   " + "".join(f"{m:8.4f}" for m in st["mean"]))
    print("std    " + "".join(f"{s:8.4f}" for s in st["std"]))
    ref = compute_train_stats(X, rows["train"])
    check(np.allclose(ref["mean"], st["mean"]) and np.allclose(ref["std"], st["std"]),
          "cached stats == fresh recomputation over train rows")
    check(st["n_records"] == len(rows["train"]), "stats use exactly the train split row count")
    every = compute_train_stats(X, np.arange(len(X)))
    check(not np.allclose(every["std"], st["std"], rtol=0, atol=1e-7),
          "stats differ from all-split stats (val/test not included)")
    check(min(st["std"]) > 1e-3, f"no near-zero lead std (min {min(st['std']):.4f})")

    hr("One batch per split (mode=full)")
    for split, dl in out["loaders"].items():
        b = next(iter(dl))
        x, y = b["x"], b["y"]
        print(f"{split:<5} x {tuple(x.shape)} {x.dtype}  y {tuple(y.shape)} {y.dtype}  "
              f"ecg_id[:3]={b['ecg_id'][:3].tolist()}  x mean {x.mean():+.3f} std {x.std():.3f}")
        check(x.shape == (cfg.train.batch_size, 12, T) and y.shape == (cfg.train.batch_size, 5),
              f"{split}: shapes [B,12,{T}] / [B,5]")
        check(bool(torch.isfinite(x).all()), f"{split}: all finite")
        check(set(y.unique().tolist()) <= {0.0, 1.0}, f"{split}: binary labels")
    print(f"batches/epoch: train {len(out['loaders']['train'])}, val {len(out['loaders']['val'])}, "
          f"test {len(out['loaders']['test'])}")

    tr = out["datasets"]["train"]
    sub = np.stack([tr[i]["x"].numpy() for i in range(0, len(tr), 17)])  # ~1000 train records
    m, s = sub.mean(axis=(0, 2)), sub.std(axis=(0, 2))
    print(f"normalised train subset: per-lead mean in [{m.min():+.3f}, {m.max():+.3f}], "
          f"std in [{s.min():.3f}, {s.max():.3f}]")
    check(np.abs(m).max() < 0.1 and np.abs(s - 1).max() < 0.15, "train data ~zero-mean / unit-std per lead")

    hr("Flat-lead record")
    flat = [i for i in range(len(X)) if (np.asarray(X[i]).std(axis=1) < 1e-6).any()]
    for i in flat:
        split = next(k for k, r in rows.items() if i in set(r))
        pos = int(np.flatnonzero(rows[split] == i)[0])
        item = out["datasets"][split][pos]
        lead = int(np.flatnonzero(np.asarray(X[i]).std(axis=1) < 1e-6)[0])
        print(f"ecg_id {item['ecg_id']} ({split}), flat lead {leads[lead]}: "
              f"normalised value {item['x'][lead, 0]:+.4f}")
        check(bool(torch.isfinite(item["x"]).all()), "flat-lead record normalises to finite values")
    for kind in ("per_record_zscore",):
        c2 = copy.deepcopy(cfg)
        c2.data.normalization.type = kind
        o2 = build_dataloaders(c2)
        if flat:
            split = next(k for k, r in rows.items() if flat[0] in set(r))
            pos = int(np.flatnonzero(rows[split] == flat[0])[0])
            check(bool(torch.isfinite(o2["datasets"][split][pos]["x"]).all()),
                  f"{kind}: flat lead -> finite (eps guard)")

    hr("Fallback: 2.5 s random crops + eval windows")
    c3 = copy.deepcopy(cfg)
    c3.data.input.mode = "random_crop"
    o3 = build_dataloaders(c3)
    b = next(iter(o3["loaders"]["train"]))
    L = int(c3.data.input.crop_seconds * fs)
    print(f"train crop batch x {tuple(b['x'].shape)}")
    check(b["x"].shape[-1] == L, f"train crops have length {L}")
    vb = next(iter(o3["loaders"]["val"]))
    check(vb["x"].shape[-1] == T, "val stays full-length (windowed in predictor)")
    w = sliding_windows(vb["x"], L, int(cfg.eval.window_agg.stride_seconds * fs))
    print(f"val windows {tuple(w.shape)}  ([B, W, 12, {L}])")
    check(w.shape[1] == 7 and w.shape[-1] == L, "7 overlapping windows of 2.5 s @ 1.25 s stride")

    hr("Reproducibility")
    b1 = next(iter(build_dataloaders(c3)["loaders"]["train"]))
    b2 = next(iter(build_dataloaders(c3)["loaders"]["train"]))
    check(torch.equal(b1["x"], b2["x"]) and torch.equal(b1["ecg_id"], b2["ecg_id"]),
          "same seed -> identical shuffled + cropped first batch")

    hr("Sampler seam: hyp_oversample")
    c4 = copy.deepcopy(cfg)
    c4.data.sampler.type = "hyp_oversample"
    o4 = build_dataloaders(c4)
    idx = np.array(list(iter(o4["loaders"]["train"].sampler)))
    hyp = o4["datasets"]["train"].labels[:, 4]
    print(f"HYP share: natural {hyp.mean():.3f} -> sampled {hyp[idx].mean():.3f} "
          f"(factor {c4.data.sampler.factor})")
    check(hyp[idx].mean() > hyp.mean() * 1.5, "oversampling raises HYP share")

    hr("Result")
    if FAIL:
        sys.exit("FAILED: " + "; ".join(FAIL))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
