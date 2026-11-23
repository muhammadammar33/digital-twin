"""PyTorch Dataset / DataLoaders over the cached PTB-XL signal array."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.augment import Compose
from src.data.normalization import Normalizer, load_or_compute_stats
from src.data.windows import random_crop
from src.utils.config import Config, resolve_path


def cache_paths(cfg: Config) -> dict[str, Path]:
    cache = resolve_path(cfg.data.cache_dir)
    fs = int(cfg.data.sampling_rate)
    return {
        "labels": cache / f"labels_{cfg.data.task}.csv",
        "signals": cache / f"signals_{fs}hz.npy",
        "ecg_ids": cache / f"signals_{fs}hz.ecg_ids.npy",
        "stats": cache / f"norm_stats_{fs}hz.json",
    }


class PTBXLDataset(Dataset):
    """Returns dict(x=[12, T] float32, y=[K] float32, ecg_id=int).

    train=True enables random cropping (data.input.mode=random_crop) and augmentation;
    eval datasets always return the full 10 s record (window aggregation happens in the predictor).
    The memmap is opened lazily inside each worker: pickling an open memmap to spawned
    workers (Windows/macOS) would copy the whole array.
    """

    def __init__(self, signals_path: Path, rows: np.ndarray, labels: np.ndarray, ecg_ids: np.ndarray,
                 normalizer: Normalizer, train: bool = False, crop_len: int | None = None,
                 augment: Compose | None = None):
        self.signals_path = signals_path
        self.rows = np.asarray(rows)
        self.labels = labels.astype(np.float32)
        self.ecg_ids = ecg_ids
        self.normalizer = normalizer
        self.train = train
        self.crop_len = crop_len if train else None
        self.augment = augment if train else None
        self._X = None
        self._rng = None

    def __len__(self) -> int:
        return len(self.rows)

    def _lazy_init(self) -> None:
        if self._X is None:
            self._X = np.load(self.signals_path, mmap_mode="r")
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            # seeded from the DataLoader's seeded generator -> reproducible crops/augmentations
            self._rng = np.random.default_rng(info.seed if info else torch.initial_seed())

    def __getitem__(self, i: int) -> dict:
        self._lazy_init()
        x = np.array(self._X[self.rows[i]], dtype=np.float32)       # copy out of the memmap
        x = self.normalizer(x)
        if self.crop_len:
            x = random_crop(x, self.crop_len, self._rng)
        if self.augment:
            x = self.augment(x, self._rng)
        return {"x": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                "y": torch.from_numpy(self.labels[i]),
                "ecg_id": int(self.ecg_ids[i])}


def _make_sampler(cfg: Config, y_train: np.ndarray, classes: list[str],
                  generator: torch.Generator):
    """Class-balancing seam. none | hyp_oversample (records with HYP drawn `factor` x as often)."""
    s = cfg.data.sampler
    if s.type == "none":
        return None
    if s.type == "hyp_oversample":
        hyp = y_train[:, classes.index("HYP")]
        w = np.where(hyp > 0, float(s.factor), 1.0)
        return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w),
                                     replacement=True, generator=generator)
    raise ValueError(f"unknown sampler {s.type!r}")


def build_datasets(cfg: Config) -> dict:
    d = cfg.data
    p = cache_paths(cfg)
    for k in ("labels", "signals", "ecg_ids"):
        if not p[k].exists():
            raise FileNotFoundError(f"{p[k]} missing: run scripts/prepare_data.py "
                                    f"(with data.sampling_rate={d.sampling_rate}) first")
    table = pd.read_csv(p["labels"], index_col="ecg_id")
    ids = np.load(p["ecg_ids"])
    if not np.array_equal(ids, table.index.values):
        raise RuntimeError("signal cache rows do not match the label table; re-run prepare_data.py --rebuild")
    classes = list(d.classes)
    y = table[classes].values.astype(np.float32)
    rows = {s: np.flatnonzero(table["split"].values == s) for s in ("train", "val", "test")}

    X = np.load(p["signals"], mmap_mode="r")
    nrm = d.normalization
    if nrm.type == "per_lead_zscore":
        st = load_or_compute_stats(p["stats"], X, rows["train"], list(d.train_folds))
        normalizer = Normalizer(nrm.type, nrm.eps, st["mean"], st["std"])
    else:
        st = None
        normalizer = Normalizer(nrm.type, nrm.eps)
    del X

    fs = int(d.sampling_rate)
    crop_len = int(round(d.input.crop_seconds * fs)) if d.input.mode == "random_crop" else None
    if d.input.mode not in ("full", "random_crop"):
        raise ValueError(f"unknown data.input.mode {d.input.mode!r}")
    augment = Compose(d.get("augment"))

    ds = {s: PTBXLDataset(p["signals"], r, y[r], table.index.values[r], normalizer,
                          train=(s == "train"), crop_len=crop_len, augment=augment)
          for s, r in rows.items()}
    return {"datasets": ds, "classes": classes, "norm_stats": st, "table": table, "rows": rows}


def build_dataloaders(cfg: Config, batch_size: int | None = None) -> dict:
    out = build_datasets(cfg)
    ds = out["datasets"]
    bs = int(batch_size or cfg.train.batch_size)
    nw = int(cfg.data.num_workers)
    g = torch.Generator().manual_seed(int(cfg.seed))
    sampler = _make_sampler(cfg, ds["train"].labels, out["classes"], g)
    common = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=False)
    out["loaders"] = {
        "train": DataLoader(ds["train"], batch_size=bs, shuffle=sampler is None, sampler=sampler,
                            drop_last=True, generator=g, **common),
        "val": DataLoader(ds["val"], batch_size=bs, shuffle=False, **common),
        "test": DataLoader(ds["test"], batch_size=bs, shuffle=False, **common),
    }
    return out
