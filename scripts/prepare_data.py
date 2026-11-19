#!/usr/bin/env python
"""Build the PTB-XL label/split table and signal cache, then run sanity checks.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml --set data.root=/path/to/ptb-xl
    python scripts/prepare_data.py ... --set data.sampling_rate=500   # build the 500 Hz cache

Outputs (in data.cache_dir):
    labels_<task>.csv           one row per kept record: ecg_id, patient_id, strat_fold, split, 5 label columns
    signals_<fs>hz.npy          [N, 12, T] float32, rows aligned with labels_<task>.csv
    signals_<fs>hz.ecg_ids.npy  ecg_id per row (alignment check used by the Dataset)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import build_label_matrix, load_database, load_diagnostic_map  # noqa: E402
from src.data.signals import build_signal_cache, filename_column  # noqa: E402
from src.data.splits import assign_splits, check_patient_leakage  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. data.root=/x")
    ap.add_argument("--skip-signals", action="store_true", help="only build the label table")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the signal cache even if present")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg.data
    if not d.root:
        sys.exit("data.root is not set. Pass --set data.root=/path/to/ptb-xl or edit the config.")
    root = resolve_path(d.root)
    cache = resolve_path(d.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    classes = list(d.classes)
    fs = int(d.sampling_rate)
    failures: list[str] = []

    # ------------------------------------------------------------------ labels
    hr(f"PTB-XL at {root}")
    df = load_database(root)
    code_map = load_diagnostic_map(root)
    supers, y = build_label_matrix(df, code_map, classes)
    n_all = len(df)
    n_unlab = int((y.sum(1) == 0).sum())
    print(f"records in ptbxl_database.csv : {n_all}")
    print(f"patients                      : {df.patient_id.nunique()}")
    print(f"diagnostic SCP codes mapped   : {len(code_map)}  "
          f"({sorted(set(code_map.values()))})")
    print(f"records with no superclass    : {n_unlab}  "
          f"-> {'dropped' if d.drop_unlabeled else 'kept as all-zero rows'}")

    if d.drop_unlabeled:
        keep = y.sum(1) > 0
        df, y, supers = df[keep], y[keep], supers[keep]

    split = assign_splits(df, d.train_folds, d.val_folds, d.test_folds)

    table = df[["patient_id", "strat_fold", "filename_lr", "filename_hr", "age", "sex"]].copy()
    table["split"] = split
    for i, c in enumerate(classes):
        table[c] = y[:, i].astype(np.int8)
    out_csv = cache / f"labels_{d.task}.csv"
    table.to_csv(out_csv)
    print(f"wrote {out_csv}  ({len(table)} rows)")

    # ------------------------------------------------------------------ class counts
    hr("Class counts per split (positives, % of split)")
    hdr = f"{'split':<7}{'N':>7}" + "".join(f"{c:>14}" for c in classes) + f"{'no-label':>10}"
    print(hdr)
    for s in ("train", "val", "test", "all"):
        m = np.ones(len(table), bool) if s == "all" else (table.split == s).values
        ys = y[m]
        cells = "".join(f"{int(ys[:, i].sum()):>6} ({100 * ys[:, i].mean():4.1f}%)" for i in range(len(classes)))
        print(f"{s:<7}{m.sum():>7}{cells}{int((ys.sum(1) == 0).sum()):>10}")
    print(f"\nfold sizes: {table.strat_fold.value_counts().sort_index().to_dict()}")

    card = y.sum(1).astype(int)
    print("labels per record (multilabel cardinality): "
          + ", ".join(f"{k}: {v}" for k, v in sorted(zip(*np.unique(card, return_counts=True)))))
    combos = supers.apply(lambda l: "+".join(l) if l else "<none>").value_counts().head(8)
    print("most frequent label combinations:")
    for k, v in combos.items():
        print(f"    {k:<20}{v:>6}")

    # ------------------------------------------------------------------ leakage
    hr("Patient leakage across splits (must be 0)")
    for (a, b), n in check_patient_leakage(table, split).items():
        print(f"{a:>5} ∩ {b:<5}: {n} shared patients")
        if n:
            failures.append(f"{n} patients shared between {a} and {b}")
    for s in ("train", "val", "test"):
        print(f"{s:<5} patients: {table.loc[table.split == s, 'patient_id'].nunique()}")
    if any((table.split == s).sum() == 0 for s in ("train", "val", "test")):
        failures.append("an empty split")
    for i, c in enumerate(classes):
        for s in ("train", "val", "test"):
            if y[(table.split == s).values, i].sum() == 0:
                failures.append(f"class {c} has no positives in {s} (AUC undefined)")

    # ------------------------------------------------------------------ signals
    if not args.skip_signals:
        hr(f"Signal cache @ {fs} Hz")
        sig_path = cache / f"signals_{fs}hz.npy"
        ids_path = cache / f"signals_{fs}hz.ecg_ids.npy"
        ids = table.index.values
        if sig_path.exists() and ids_path.exists() and not args.rebuild \
                and np.array_equal(np.load(ids_path), ids):
            X = np.load(sig_path, mmap_mode="r")
            print(f"reusing {sig_path}")
        else:
            missing = [p for p in table[filename_column(fs)] if not (root / f"{p}.hea").exists()]
            if missing:
                sys.exit(f"{len(missing)} WFDB headers missing under {root}, e.g. {missing[:3]}")
            X = build_signal_cache(root, table, fs, sig_path)
            np.save(ids_path, ids)
        print(f"array shape : {X.shape}  dtype {X.dtype}  ({X.nbytes / 1e9:.2f} GB)")
        expected = (len(table), 12, fs * 10)
        if X.shape != expected:
            failures.append(f"signal array shape {X.shape} != {expected}")
        # finite check + amplitude summary, streamed in chunks to stay memory-light at 500 Hz
        n_nonfinite, flat = 0, 0
        lead_abs_max = np.zeros(12)
        for s0 in range(0, len(X), 2048):
            chunk = np.asarray(X[s0:s0 + 2048])
            n_nonfinite += int((~np.isfinite(chunk)).sum())
            flat += int((chunk.std(axis=2) < 1e-6).any(axis=1).sum())
            lead_abs_max = np.maximum(lead_abs_max, np.abs(chunk).max(axis=(0, 2)))
        print(f"non-finite values            : {n_nonfinite}")
        print(f"records with a flat lead     : {flat}")
        print(f"per-lead max |amplitude| (mV): {np.round(lead_abs_max, 1).tolist()}")
        if n_nonfinite:
            failures.append(f"{n_nonfinite} non-finite samples in signals")
        ex = X[0]
        print(f"example ecg_id={ids[0]}: shape {ex.shape}, lead I mean={ex[0].mean():.3f} "
              f"std={ex[0].std():.3f} mV")

    hr("Sanity-check result")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
