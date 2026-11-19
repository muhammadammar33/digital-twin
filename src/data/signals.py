"""Raw waveform loading (WFDB) and a single .npy cache per sampling rate.

The cache is an [N, 12, T] float32 array in the row order of the labels table, so the Dataset
can memory-map it (500 Hz is ~5 GB) instead of opening 21k WFDB files every epoch.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

N_LEADS = 12
DURATION_S = 10


def filename_column(sampling_rate: int) -> str:
    if sampling_rate == 100:
        return "filename_lr"
    if sampling_rate == 500:
        return "filename_hr"
    raise ValueError(f"PTB-XL ships 100 Hz and 500 Hz only, got {sampling_rate}")


def load_record(root: Path, rel_path: str) -> np.ndarray:
    import wfdb
    sig, _ = wfdb.rdsamp(str(root / rel_path))          # [T, 12], physical units (mV)
    return sig.T.astype(np.float32)                      # -> [12, T]


def build_signal_cache(root: Path, df: pd.DataFrame, sampling_rate: int, out: Path,
                       workers: int = 8) -> np.ndarray:
    """Read every record listed in df (in df order) into a memmapped [N, 12, T] array."""
    T = sampling_rate * DURATION_S
    paths = df[filename_column(sampling_rate)].tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(df), N_LEADS, T))

    def work(i):
        x = load_record(root, paths[i])
        if x.shape != (N_LEADS, T):
            raise ValueError(f"{paths[i]}: shape {x.shape}, expected {(N_LEADS, T)}")
        arr[i] = x

    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(work, range(len(paths))), total=len(paths), desc=f"cache {sampling_rate} Hz"))
    arr.flush()
    del arr
    tmp.rename(out)
    return np.load(out, mmap_mode="r")
