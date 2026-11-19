#!/usr/bin/env python
"""Final test-fold evaluation of a trained run.

    python scripts/evaluate.py --run-name b0 --row B0
    python scripts/evaluate.py --run-name a1_lead --row A1
    python scripts/evaluate.py --run-name smoke_b0 --plumbing-check     # harness check, not a result

Protocol
  1. Thresholds: the per-class thresholds chosen on the VALIDATION fold at model selection
     (stored in best.pt / swa.pt) are FROZEN and written to thresholds.json before any test
     prediction is made. They are re-derived on validation only as a consistency check.
     (last.pt stores none: then they are tuned on validation here, never on test.)
  2. Test fold is predicted once; metrics use the frozen thresholds; nothing is tuned on test.
  3. Bootstrap 95% CIs (test records resampled, thresholds kept fixed).
  4. Report (console + eval_<ckpt>/report.md + metrics.json + predictions.npz) and the
     benchmarks.md row (--row) are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import build_dataloaders  # noqa: E402
from src.eval.benchmarks import write_row  # noqa: E402
from src.eval.bootstrap import bootstrap_ci  # noqa: E402
from src.eval.confusion import format_confusion, per_class_confusion  # noqa: E402
from src.eval.metrics import find_thresholds, summarize  # noqa: E402
from src.eval.predict import predict, window_agg_enabled  # noqa: E402
from src.models.model import build_model  # noqa: E402
from src.train.loop import pick_device  # noqa: E402
from src.utils.config import REPO_ROOT, Config, load_config, resolve_path, seed_everything  # noqa: E402

MACHINE_KEYS = ("data.root", "data.cache_dir", "data.num_workers", "train.device")


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(run_cfg: dict, default_cfg: dict) -> list[str]:
    a, b = _flat(run_cfg), _flat(default_cfg)
    skip = ("train.runs_dir", "train.log_every", "data.num_workers", "train.device", "eval.")
    return [f"{k}={a[k]}" for k in sorted(a)
            if not k.startswith(skip) and k in b and a[k] != b[k]]


def fmt_ci(v: float, ci) -> str:
    return f"{v:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--checkpoint", default="best", choices=["best", "swa", "last"])
    ap.add_argument("--config", default="configs/default.yaml",
                    help="source of reporting settings (eval.targets, eval.bootstrap) and of the "
                         "reference config for the benchmark 'config' cell")
    ap.add_argument("--set", nargs="*", default=[], help=f"machine overrides only: {MACHINE_KEYS}")
    ap.add_argument("--row", default=None, help="benchmarks.md row id to write, e.g. B0 or A1")
    ap.add_argument("--vs", default="B0", help="'vs' cell for a new benchmark row")
    ap.add_argument("--target", default=None,
                    help="eval.targets key to compare against (default: --row if it has a target, else B0)")
    ap.add_argument("--notes", default="", help="extra text for the notes cell")
    ap.add_argument("--benchmarks", default="benchmarks.md")
    ap.add_argument("--plumbing-check", action="store_true",
                    help="label everything as a harness check; write the row to a scratch copy only")
    args = ap.parse_args()

    run_dir = resolve_path(args.runs_dir) / args.run_name
    ckpt_path = run_dir / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        sys.exit(f"{ckpt_path} not found")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config.wrap(ck["config"])
    for ov in args.set:
        k = ov.split("=", 1)[0]
        if k not in MACHINE_KEYS:
            sys.exit(f"--set {k}: only machine-specific keys may be overridden at evaluation ({MACHINE_KEYS})")
    if args.set:
        cfg = load_config_from_dict(cfg, args.set)
    rep_cfg = load_config(args.config)
    target_key = args.target or (args.row if args.row in rep_cfg.eval.targets else "B0")
    if target_key not in rep_cfg.eval.targets:
        sys.exit(f"--target {target_key!r} not in eval.targets ({list(rep_cfg.eval.targets)})")
    targets = rep_cfg.eval.targets[target_key]
    bs_cfg = rep_cfg.eval.bootstrap
    classes = list(ck.get("classes") or cfg.data.classes)

    out_dir = run_dir / f"eval_{args.checkpoint}"
    out_dir.mkdir(exist_ok=True)
    banner = ("PLUMBING CHECK (harness validation on a short smoke run, NOT a result)"
              if args.plumbing_check else "TEST-FOLD EVALUATION")

    seed_everything(cfg.seed)
    device = pick_device(cfg.train.get("device", "auto"))
    data = build_dataloaders(cfg)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    amp = torch.float16 if device.type == "cuda" and cfg.train.get("amp", "auto") in ("auto", True) else None

    # ---------------------------------------------------------------- 1. validation: frozen thresholds
    t0 = time.time()
    val = predict(model, data["loaders"]["val"], device, cfg, amp_dtype=amp)
    val_thr_now = find_thresholds(val["labels"], val["probs"])
    if ck.get("thresholds") is not None:
        frozen = np.asarray(ck["thresholds"], dtype=float)
        source = f"{args.checkpoint}.pt (selected on val at epoch {ck.get('epoch', '?')})"
    else:
        frozen = val_thr_now
        source = "tuned on validation now (checkpoint stores none)"
    consistent = bool(np.allclose(frozen, val_thr_now))
    thr_file = out_dir / "thresholds.json"
    thr_file.write_text(json.dumps({"classes": classes, "thresholds": frozen.tolist(), "source": source,
                                    "fold": "validation", "frozen_before_test": True}, indent=2))
    val_m = summarize(val["labels"], val["probs"], classes, thresholds=frozen)

    # ---------------------------------------------------------------- 2. test: predicted once, frozen thresholds
    test = predict(model, data["loaders"]["test"], device, cfg, amp_dtype=amp)
    frozen_on_disk = np.asarray(json.loads(thr_file.read_text())["thresholds"])
    assert np.array_equal(frozen_on_disk, frozen), "thresholds changed after freezing"
    test_m = summarize(test["labels"], test["probs"], classes, thresholds=frozen)
    assert test_m["thresholds_tuned_here"] is False, "thresholds must never be tuned on test"
    conf = per_class_confusion(test["labels"], test["probs"], frozen, classes)

    # ---------------------------------------------------------------- 3. bootstrap CIs
    ci = bootstrap_ci(test["labels"], test["probs"], frozen, n_boot=int(bs_cfg.n_boot), seed=int(bs_cfg.seed))
    secs = time.time() - t0

    # ---------------------------------------------------------------- 4. report
    hyp = classes.index("HYP")

    def target_row(name: str, value: float, run_ci, tgt, tgt_ci=None) -> str:
        if tgt is None:
            return f"| {name} | {fmt_ci(value, run_ci)} | n/a | | |"
        tgt = float(tgt)
        t = f"{tgt:.3f}" + (f" [{tgt_ci[0]:.3f}–{tgt_ci[1]:.3f}]" if tgt_ci else "")
        inside = "yes" if run_ci[0] <= tgt <= run_ci[1] else "no"
        return f"| {name} | {fmt_ci(value, run_ci)} | {t} | {value - tgt:+.3f} | {inside} |"
    lines = [
        f"# {banner}",
        "",
        f"run `{args.run_name}` · checkpoint `{args.checkpoint}.pt` (epoch {ck.get('epoch', '?')}, "
        f"weights: {ck.get('weights', 'raw')}) · {cfg.data.sampling_rate} Hz · input {cfg.data.input.mode}"
        f"{' + window aggregation' if window_agg_enabled(cfg) else ''} · attention {cfg.model.attention.type}",
        f"test fold N = {len(test['labels'])} · val fold N = {len(val['labels'])} · {secs:.0f}s on {device}",
        "",
        "## Thresholds (validation fold, frozen before test)",
        "",
        f"source: {source}; re-derived on val now: {'identical' if consistent else 'DIFFERENT (see metrics.json)'}",
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
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report)
    (out_dir / "metrics.json").write_text(json.dumps({
        "banner": banner, "run": args.run_name, "checkpoint": args.checkpoint, "epoch": ck.get("epoch"),
        "thresholds": frozen.tolist(), "threshold_source": source,
        "val_thresholds_rederived": val_thr_now.tolist(), "thresholds_consistent": consistent,
        "test": test_m, "val": val_m, "bootstrap": ci, "confusion": conf,
        "target_key": target_key, "target": targets.to_dict(),
    }, indent=2))
    # saved predictions: the seam for multi-seed / multi-architecture ensembling
    np.savez_compressed(out_dir / "predictions.npz", val_probs=val["probs"], val_labels=val["labels"],
                        val_ids=val["ecg_ids"], test_probs=test["probs"], test_labels=test["labels"],
                        test_ids=test["ecg_ids"], thresholds=frozen, classes=np.array(classes))
    print(report)

    # ---------------------------------------------------------------- 5. benchmarks.md row
    if args.row:
        bench = resolve_path(args.benchmarks)
        if args.plumbing_check:
            scratch = out_dir / "benchmarks_plumbing_copy.md"
            scratch.write_text(bench.read_text())
            bench = scratch
        diff = config_diff(cfg.to_dict(), load_config(args.config).to_dict())
        notes = (f"run `{args.run_name}` {args.checkpoint}.pt ep {ck.get('epoch', '?')}; test N={len(test['labels'])}; "
                 f"95% bootstrap CI (B={ci['n_boot']}); val thresholds frozen; {time.strftime('%Y-%m-%d')}"
                 + (f"; {args.notes}" if args.notes else ""))
        row = write_row(bench, args.row,
                        macro_f1=fmt_ci(test_m["macro_f1"], ci["macro_f1"]),
                        macro_auc=fmt_ci(test_m["macro_auc"], ci["macro_auc"]),
                        hyp_f1=fmt_ci(test_m[f"f1_{classes[hyp]}"], ci["per_class_f1"][hyp]),
                        notes=notes,
                        config="`configs/default.yaml`" + (" " + " ".join(f"`{d}`" for d in diff) if diff else ""),
                        vs=args.vs if args.row != "B0" else "")
        print(f"\nbenchmark row written to {bench.relative_to(REPO_ROOT) if bench.is_relative_to(REPO_ROOT) else bench}:\n{row}")


def load_config_from_dict(cfg: Config, overrides: list[str]) -> Config:
    import yaml
    d = cfg.to_dict()
    for ov in overrides:
        k, v = ov.split("=", 1)
        node = d
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = yaml.safe_load(v)
    return Config.wrap(d)


if __name__ == "__main__":
    main()
