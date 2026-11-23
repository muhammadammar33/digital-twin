"""Shared test-fold reporting for single runs (scripts/evaluate.py) and ensembles (scripts/ensemble.py).

Given validation/test probabilities and FROZEN validation thresholds, this computes test metrics,
bootstrap CIs, paired Δ vs reference runs, writes report.md / metrics.json / predictions.npz /
thresholds.json, and (optionally) the benchmarks.md row. Keeping one code path guarantees that an
ensemble row is computed exactly like a single-run row.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from src.eval.benchmarks import read_row_values, write_row
from src.eval.bootstrap import bootstrap_ci, paired_delta
from src.eval.confusion import format_confusion, per_class_confusion
from src.eval.metrics import summarize
from src.utils.config import REPO_ROOT, git_commit


def fmt_ci(v: float, ci) -> str:
    return f"{v:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]"


def freeze_thresholds(out_dir: Path, classes: list[str], thresholds, source: str) -> np.ndarray:
    """Write the validation thresholds to disk BEFORE any test metric is computed."""
    thr = np.asarray(thresholds, dtype=float)
    (out_dir / "thresholds.json").write_text(json.dumps(
        {"classes": classes, "thresholds": thr.tolist(), "source": source, "fold": "validation",
         "frozen_before_test": True}, indent=2))
    return thr


def load_reference(npz: Path, test_ids: np.ndarray, test_labels: np.ndarray):
    """Reference run's test probs/thresholds, reordered to this run's test record order."""
    r = np.load(npz, allow_pickle=False)
    order = {int(i): j for j, i in enumerate(r["test_ids"])}
    if set(order) != set(int(i) for i in test_ids):
        sys.exit(f"{npz}: test records differ from this run's test fold; cannot pair")
    idx = np.array([order[int(i)] for i in test_ids])
    if not np.array_equal(r["test_labels"][idx], test_labels):
        sys.exit(f"{npz}: test labels differ; cannot pair")
    return r["test_probs"][idx], r["thresholds"]


def finalize(*, out_dir: Path, header: list[str], val: dict, test: dict, classes: list[str],
             frozen: np.ndarray, threshold_note: str, rep_cfg, target_key: str, row: str | None,
             vs: list[str], vs_runs: list[str] | None, runs_dir: Path, benchmarks: Path,
             plumbing: bool, row_notes: str, config_cell: str, meta: dict) -> dict:
    targets = rep_cfg.eval.targets[target_key]
    bs = rep_cfg.eval.bootstrap
    t0 = time.time()

    # frozen-threshold guard: the file written before test metrics must still hold these values
    on_disk = np.asarray(json.loads((out_dir / "thresholds.json").read_text())["thresholds"])
    assert np.array_equal(on_disk, frozen), "thresholds changed after freezing"
    val_m = summarize(val["labels"], val["probs"], classes, thresholds=frozen)
    test_m = summarize(test["labels"], test["probs"], classes, thresholds=frozen)
    assert test_m["thresholds_tuned_here"] is False, "thresholds must never be tuned on test"
    conf = per_class_confusion(test["labels"], test["probs"], frozen, classes)
    ci = bootstrap_ci(test["labels"], test["probs"], frozen, n_boot=int(bs.n_boot), seed=int(bs.seed))
    hyp = classes.index("HYP")

    # ---------------------------------------------------------------- Δ vs reference rows
    vs_runs = list(vs_runs or [])
    if len(vs_runs) > len(vs):
        sys.exit(f"--vs-run has {len(vs_runs)} entries but --vs has {len(vs)}; pair them by position")
    vs_runs += [None] * (len(vs) - len(vs_runs))
    cmps = []
    if row and row != "B0":
        for vs_id, vs_run in zip(vs, vs_runs):
            if vs_id == row:
                continue
            ref_npz = Path(runs_dir) / vs_run / "eval_best" / "predictions.npz" if vs_run else None
            if ref_npz is not None and ref_npz.exists():
                p_ref, thr_ref = load_reference(ref_npz, test["ecg_ids"], test["labels"])
                cmps.append({"kind": "paired", "vs": vs_id, "vs_run": vs_run,
                             **paired_delta(test["labels"], test["probs"], frozen, p_ref, thr_ref, hyp,
                                            n_boot=int(bs.n_boot), seed=int(bs.seed))})
                continue
            if vs_run:
                print(f"WARNING: {ref_npz} not found; falling back to an UNPAIRED delta vs {vs_id} from benchmarks.md")
            ref_vals = read_row_values(benchmarks, vs_id)
            if ref_vals:
                cur = {"macro_f1": test_m["macro_f1"], "macro_auc": test_m["macro_auc"],
                       "hyp_f1": test_m[f"f1_{classes[hyp]}"]}
                cmps.append({"kind": "unpaired", "vs": vs_id,
                             **{k: {"delta": cur[k] - v} for k, v in ref_vals.items() if v is not None}})
            else:
                print(f"WARNING: no '{vs_id}' row with values in {benchmarks}; no delta vs {vs_id}")

    # ---------------------------------------------------------------- report
    def target_row(name, value, run_ci, tgt, tgt_ci=None):
        if tgt is None:
            return f"| {name} | {fmt_ci(value, run_ci)} | n/a | | |"
        tgt = float(tgt)
        t = f"{tgt:.3f}" + (f" [{tgt_ci[0]:.3f}–{tgt_ci[1]:.3f}]" if tgt_ci else "")
        inside = "yes" if run_ci[0] <= tgt <= run_ci[1] else "no"
        return f"| {name} | {fmt_ci(value, run_ci)} | {t} | {value - tgt:+.3f} | {inside} |"

    lines = header + [
        f"test fold N = {len(test['labels'])} · val fold N = {len(val['labels'])} · code {git_commit()}",
        "",
        "## Thresholds (validation fold, frozen before test)",
        "",
        threshold_note,
        "",
        "| class | " + " | ".join(classes) + " |",
        "|---|" + "---|" * len(classes),
        "| threshold | " + " | ".join(f"{t:.2f}" for t in frozen) + " |",
        "",
        f"## Headline (test fold, 95% bootstrap CI, B = {ci['n_boot']})",
        "",
        f"target `{target_key}`: {targets.label}",
        "",
        "| metric | this run | target | Δ | target inside CI? |",
        "|---|---|---|---|---|",
        target_row("macro-AUC", test_m["macro_auc"], ci["macro_auc"], targets.get("macro_auc")),
        target_row("macro-F1 (val thresholds)", test_m["macro_f1"], ci["macro_f1"],
                   targets.get("macro_f1"), targets.get("macro_f1_ci")),
        f"| macro-F1 @ 0.5 (reference only) | {test_m['macro_f1_at_0.5']:.3f} | | | |",
        "",
        "## Per class (test fold)",
        "",
        "| class | AUC [95% CI] | F1 [95% CI] | sens | spec | prec | support |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, c in enumerate(classes):
        r = conf[k]
        lines.append(f"| {c} | {fmt_ci(test_m[f'auc_{c}'], ci['per_class_auc'][k])} | "
                     f"{fmt_ci(test_m[f'f1_{c}'], ci['per_class_f1'][k])} | {r['sensitivity']:.3f} | "
                     f"{r['specificity']:.3f} | {r['precision']:.3f} | {r['support']} |")
    lines += ["", "## Confusion matrices (one-vs-rest, test fold, frozen thresholds)", "", "```",
              format_confusion(conf), "```", "",
              f"validation (frozen thresholds): macro-AUC {val_m['macro_auc']:.4f}, macro-F1 {val_m['macro_f1']:.4f}", ""]
    for cmp in cmps:
        lines += [f"## Δ vs {cmp['vs']} ({cmp['kind']}"
                  + (f" bootstrap on the same {len(test['labels'])} test records, run `{cmp['vs_run']}`)"
                     if cmp["kind"] == "paired" else ", from benchmarks.md point values; no CI)"), "",
                  "| metric | Δ (this − ref) | 95% CI | P(Δ ≤ 0) |", "|---|---|---|---|"]
        for k, name in (("macro_f1", "macro-F1"), ("macro_auc", "macro-AUC"), ("hyp_f1", "HYP-F1")):
            if k in cmp:
                c = cmp[k]
                lines.append(f"| {name} | {c['delta']:+.3f} | "
                             + (f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['p_delta_le_0']:.3f} |"
                                if "ci" in c else "n/a | n/a |"))
        lines.append("")
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report)
    (out_dir / "metrics.json").write_text(json.dumps({
        **meta, "thresholds": frozen.tolist(), "test": test_m, "val": val_m, "bootstrap": ci,
        "confusion": conf, "target_key": target_key, "target": targets.to_dict(), "delta_vs": cmps,
        "code": git_commit(),
    }, indent=2))
    # saved predictions: what ensembles average and what --vs-run pairs against
    np.savez_compressed(out_dir / "predictions.npz", val_probs=val["probs"], val_labels=val["labels"],
                        val_ids=val["ecg_ids"], test_probs=test["probs"], test_labels=test["labels"],
                        test_ids=test["ecg_ids"], thresholds=frozen, classes=np.array(classes))
    print(report)

    # ---------------------------------------------------------------- benchmarks.md row
    if row:
        bench = benchmarks
        if plumbing:
            scratch = out_dir / "benchmarks_plumbing_copy.md"
            scratch.write_text(Path(benchmarks).read_text())
            bench = scratch
        notes = (f"{row_notes}; test N={len(test['labels'])}; 95% bootstrap CI (B={ci['n_boot']}); "
                 f"val thresholds frozen; code {git_commit()}; {time.strftime('%Y-%m-%d')}")

        def cell(value: str, key: str) -> str:
            for cmp in cmps:
                if key in cmp:
                    c = cmp[key]
                    value += (f"; Δ{cmp['vs']} {c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
                              if "ci" in c else f"; Δ{cmp['vs']} {c['delta']:+.3f} (unpaired)")
            return value
        for cmp in cmps:
            notes += (f"; Δ vs {cmp['vs']}: paired bootstrap vs run `{cmp['vs_run']}`"
                      + (f" (P(Δ≤0) F1 {cmp['macro_f1']['p_delta_le_0']:.3f}, AUC {cmp['macro_auc']['p_delta_le_0']:.3f})"
                         if "ci" in cmp.get("macro_f1", {}) else "")
                      if cmp["kind"] == "paired" else f"; Δ vs {cmp['vs']}: unpaired point difference")
        line = write_row(bench, row,
                         macro_f1=cell(fmt_ci(test_m["macro_f1"], ci["macro_f1"]), "macro_f1"),
                         macro_auc=cell(fmt_ci(test_m["macro_auc"], ci["macro_auc"]), "macro_auc"),
                         hyp_f1=cell(fmt_ci(test_m[f"f1_{classes[hyp]}"], ci["per_class_f1"][hyp]), "hyp_f1"),
                         notes=notes, config=config_cell, vs=", ".join(vs) if row != "B0" else "")
        shown = bench.relative_to(REPO_ROOT) if Path(bench).is_relative_to(REPO_ROOT) else bench
        print(f"\nbenchmark row written to {shown}:\n{line}")
    print(f"(reporting took {time.time() - t0:.0f}s)")
    return {"test": test_m, "val": val_m, "ci": ci, "delta_vs": cmps}
