#!/usr/bin/env python
"""Evaluate an ensemble of already-evaluated runs by averaging their saved predictions.

    python scripts/ensemble.py --name a2_ens5 --members a2_aug a2_s1337 a2_s2024 a2_s7 a2_s314 \\
        --row E1 --vs B0 A1 A2 --vs-run b0 a1_lead a2_aug

Each member must have runs/<member>/eval_best/predictions.npz (written by scripts/evaluate.py).
Protocol (same as a single run):
  1. Member val and test probabilities are averaged (probabilities by default; --average logit
     averages log-odds instead).
  2. Per-class thresholds are tuned on the ENSEMBLE's validation probabilities and frozen to
     runs/<name>/eval_best/thresholds.json before any test metric is computed.
  3. Test metrics, bootstrap CIs, paired Δ vs --vs-run references, report, benchmarks row:
     the shared code in src/eval/report.py, identical to evaluate.py.
The ensemble's own predictions.npz is written too, so it can be a --vs-run reference later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import find_thresholds  # noqa: E402
from src.eval.report import freeze_thresholds, finalize  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output run folder for the ensemble, e.g. a2_ens5")
    ap.add_argument("--members", nargs="+", required=True, help="run names whose eval_best/ predictions to average")
    ap.add_argument("--average", choices=["prob", "logit"], default="prob")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="allow members whose configs differ in more than the seed (multi-architecture ensembles)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--config", default="configs/default.yaml", help="reporting settings (targets, bootstrap)")
    ap.add_argument("--row", default=None)
    ap.add_argument("--vs", nargs="+", default=["B0"])
    ap.add_argument("--vs-run", nargs="*", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--benchmarks", default="benchmarks.md")
    ap.add_argument("--plumbing-check", action="store_true")
    args = ap.parse_args()

    runs_dir = resolve_path(args.runs_dir)
    if len(set(args.members)) != len(args.members):
        sys.exit("duplicate --members")
    missing = [m for m in args.members if not (runs_dir / m / "eval_best" / "predictions.npz").exists()]
    if missing:
        sys.exit(f"members without eval_best/predictions.npz (run scripts/evaluate.py on them first): {missing}")

    # ---------------------------------------------------------------- load + align members
    ref = np.load(runs_dir / args.members[0] / "eval_best" / "predictions.npz", allow_pickle=False)
    classes = [str(c) for c in ref["classes"]]
    val_ids, test_ids = ref["val_ids"], ref["test_ids"]
    val_y, test_y = ref["val_labels"], ref["test_labels"]
    vals, tests, seeds, cfgs = [], [], [], []
    for m in args.members:
        r = np.load(runs_dir / m / "eval_best" / "predictions.npz", allow_pickle=False)
        if [str(c) for c in r["classes"]] != classes:
            sys.exit(f"{m}: class order differs")
        out = []
        for split, ids, y in (("val", val_ids, val_y), ("test", test_ids, test_y)):
            order = {int(i): j for j, i in enumerate(r[f"{split}_ids"])}
            if set(order) != set(int(i) for i in ids):
                sys.exit(f"{m}: {split} records differ from {args.members[0]}")
            idx = np.array([order[int(i)] for i in ids])
            if not np.array_equal(r[f"{split}_labels"][idx], y):
                sys.exit(f"{m}: {split} labels differ from {args.members[0]}")
            out.append(r[f"{split}_probs"][idx])
        vals.append(out[0])
        tests.append(out[1])
        cfg_path = runs_dir / m / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        cfgs.append(_flat(cfg))
        seeds.append(cfg.get("seed"))

    ignore = ("seed", "data.root", "data.cache_dir", "data.num_workers", "train.device", "train.log_every")
    diffs = sorted({k for c in cfgs[1:] for k in set(c) | set(cfgs[0])
                    if k not in ignore and c.get(k) != cfgs[0].get(k)})
    if diffs and not args.allow_mixed:
        sys.exit(f"member configs differ beyond the seed in {diffs}; pass --allow-mixed for a multi-architecture ensemble")
    if len(set(seeds)) != len(seeds):
        print(f"WARNING: duplicate seeds among members: {seeds}")

    def combine(ps):
        ps = np.stack(ps)
        if args.average == "prob":
            return ps.mean(0)
        eps = 1e-6
        p = np.clip(ps, eps, 1 - eps)
        return 1 / (1 + np.exp(-np.log(p / (1 - p)).mean(0)))

    val = {"probs": combine(vals), "labels": val_y, "ecg_ids": val_ids}
    test = {"probs": combine(tests), "labels": test_y, "ecg_ids": test_ids}

    # ---------------------------------------------------------------- thresholds: ensemble val, frozen
    out_dir = runs_dir / args.name / "eval_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "members.json").write_text(json.dumps(
        {"members": args.members, "seeds": seeds, "average": args.average, "config_diffs": diffs}, indent=2))
    thr = find_thresholds(val["labels"], val["probs"])
    frozen = freeze_thresholds(out_dir, classes, thr, "tuned on the ensemble's averaged validation probabilities")

    rep_cfg = load_config(args.config)
    target_key = args.target or (args.row if args.row in rep_cfg.eval.targets else "B0")
    if target_key not in rep_cfg.eval.targets:
        sys.exit(f"--target {target_key!r} not in eval.targets ({list(rep_cfg.eval.targets)})")
    banner = ("PLUMBING CHECK (harness validation on short smoke runs, NOT a result)"
              if args.plumbing_check else "TEST-FOLD EVALUATION: ENSEMBLE")
    member_txt = ", ".join(f"`{m}` (seed {s})" for m, s in zip(args.members, seeds))
    finalize(
        out_dir=out_dir,
        header=[f"# {banner}", "",
                f"ensemble `{args.name}`: {len(args.members)} members, {args.average}-averaged: {member_txt}"
                + (f" · member configs differ in {diffs}" if diffs else " · members differ only in seed")],
        val=val, test=test, classes=classes, frozen=frozen,
        threshold_note="source: grid-searched on the ensemble's averaged VALIDATION probabilities (member "
                       "thresholds are not reused); written to thresholds.json before test metrics",
        rep_cfg=rep_cfg, target_key=target_key, row=args.row, vs=args.vs, vs_runs=args.vs_run,
        runs_dir=runs_dir, benchmarks=resolve_path(args.benchmarks), plumbing=args.plumbing_check,
        row_notes=f"ensemble `{args.name}` = {args.average}-average of {len(args.members)} runs "
                  f"({', '.join(args.members)}; seeds {seeds})" + (f"; {args.notes}" if args.notes else ""),
        config_cell=f"{args.average}-average of {len(args.members)} seeds of `{args.members[0]}`'s config"
                    + (f" (mixed: {diffs})" if diffs else ""),
        meta={"banner": banner, "ensemble": args.name, "members": args.members, "seeds": seeds,
              "average": args.average, "threshold_source": "ensemble validation"},
    )


if __name__ == "__main__":
    main()
