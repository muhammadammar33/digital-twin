"""PTB-XL label mapping: scp_codes -> 5 diagnostic superclasses (multilabel).

Mirrors Strodthoff et al. (ecg_ptbxl_benchmarking, utils.compute_label_aggregations):
every SCP code that is a *diagnostic* statement (scp_statements.diagnostic == 1) contributes
its `diagnostic_class`, independent of the likelihood value attached to the code.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_database(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)  # stored as a dict literal string
    return df


def load_diagnostic_map(root: Path) -> dict[str, str]:
    """code -> diagnostic superclass, for diagnostic statements only."""
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]
    return scp["diagnostic_class"].dropna().to_dict()


def aggregate_superclasses(scp_codes: dict, code_map: dict[str, str]) -> list[str]:
    return sorted({code_map[c] for c in scp_codes if c in code_map})


def build_label_matrix(df: pd.DataFrame, code_map: dict[str, str],
                       classes: list[str] = SUPERCLASSES) -> tuple[pd.Series, np.ndarray]:
    """Returns (list-of-superclasses per record, binary matrix [N, len(classes)])."""
    supers = df["scp_codes"].apply(lambda d: aggregate_superclasses(d, code_map))
    unknown = set().union(*supers) - set(classes)
    if unknown:
        raise ValueError(f"scp_statements produced unexpected superclasses: {unknown}")
    idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(df), len(classes)), dtype=np.float32)
    for row, labs in enumerate(supers):
        for lab in labs:
            y[row, idx[lab]] = 1.0
    return supers, y
