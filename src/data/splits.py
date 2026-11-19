"""Patient-wise splits from PTB-XL's `strat_fold` column (1-8 train, 9 val, 10 test)."""
from __future__ import annotations

import pandas as pd


def assign_splits(df: pd.DataFrame, train_folds, val_folds, test_folds) -> pd.Series:
    folds = {"train": set(train_folds), "val": set(val_folds), "test": set(test_folds)}
    all_f = [f for s in folds.values() for f in s]
    if len(all_f) != len(set(all_f)):
        raise ValueError(f"Folds overlap between splits: {folds}")
    split = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, fs in folds.items():
        split[df["strat_fold"].isin(fs)] = name
    if split.isna().any():
        raise ValueError(f"{split.isna().sum()} records fall in no split (strat_fold not covered)")
    return split


def check_patient_leakage(df: pd.DataFrame, split: pd.Series) -> dict[tuple[str, str], int]:
    """Number of patients shared between each pair of splits (must all be 0)."""
    pats = {s: set(df.loc[split == s, "patient_id"]) for s in ("train", "val", "test")}
    return {(a, b): len(pats[a] & pats[b])
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
