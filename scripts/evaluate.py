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
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import build_dataloaders  # noqa: E402
from src.eval.metrics import find_thresholds  # noqa: E402
from src.eval.predict import predict, window_agg_enabled  # noqa: E402
from src.eval.report import freeze_thresholds, finalize  # noqa: E402
from src.models.model import build_model  # noqa: E402
from src.train.loop import pick_device  # noqa: E402
from src.utils.config import Config, load_config, resolve_path, seed_everything  # noqa: E402

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
    def show(v):
        if isinstance(v, list) and v and all(isinstance(e, dict) and "name" in e for e in v):
            return "[" + ",".join(e["name"] for e in v if e.get("enabled", True) is not False) + "]"
        return v
    return [f"{k}={show(a[k])}" for k in sorted(a)
            if not k.startswith(skip) and k in b and a[k] != b[k]]


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
    ap.add_argument("--vs", nargs="+", default=["B0"],
                    help="reference row id(s) to report Δ against, e.g. --vs B0 A1")
    ap.add_argument("--vs-run", nargs="*", default=None,
                    help="run name(s) of the --vs rows, same order (e.g. --vs-run b0 a1_lead): each run's "
                         "eval_best/predictions.npz enables a PAIRED bootstrap Δ on the same test records; "
                         "a reference without a run falls back to an unpaired Δ from benchmarks.md")
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
        thr = np.asarray(ck["thresholds"], dtype=float)
        source = f"{args.checkpoint}.pt (selected on val at epoch {ck.get('epoch', '?')})"
    else:
        thr = val_thr_now
        source = "tuned on validation now (checkpoint stores none)"
    consistent = bool(np.allclose(thr, val_thr_now))
    frozen = freeze_thresholds(out_dir, classes, thr, source)     # on disk before any test prediction

    # ---------------------------------------------------------------- 2. test: predicted once
    test = predict(model, data["loaders"]["test"], device, cfg, amp_dtype=amp)
    secs = time.time() - t0

    # ---------------------------------------------------------------- 3-5. metrics, CIs, Δ, report, row
    diff = config_diff(cfg.to_dict(), load_config(args.config).to_dict())
    finalize(
        out_dir=out_dir,
        header=[f"# {banner}", "",
                f"run `{args.run_name}` · checkpoint `{args.checkpoint}.pt` (epoch {ck.get('epoch', '?')}, "
                f"weights: {ck.get('weights', 'raw')}) · {cfg.data.sampling_rate} Hz · input {cfg.data.input.mode}"
                f"{' + window aggregation' if window_agg_enabled(cfg) else ''} · attention {cfg.model.attention.type}"
                f" · loss {cfg.train.loss.type} · seed {cfg.seed} · predicted in {secs:.0f}s on {device}"],
        val=val, test=test, classes=classes, frozen=frozen,
        threshold_note=f"source: {source}; re-derived on val now: "
                       f"{'identical' if consistent else 'DIFFERENT (see metrics.json)'}",
        rep_cfg=rep_cfg, target_key=target_key, row=args.row, vs=args.vs, vs_runs=args.vs_run,
        runs_dir=resolve_path(args.runs_dir), benchmarks=resolve_path(args.benchmarks),
        plumbing=args.plumbing_check,
        row_notes=f"run `{args.run_name}` {args.checkpoint}.pt ep {ck.get('epoch', '?')}"
                  + (f"; {args.notes}" if args.notes else ""),
        config_cell="`configs/default.yaml`" + (" " + " ".join(f"`{d}`" for d in diff) if diff else ""),
        meta={"banner": banner, "run": args.run_name, "checkpoint": args.checkpoint, "epoch": ck.get("epoch"),
              "seed": cfg.seed, "threshold_source": source, "val_thresholds_rederived": val_thr_now.tolist(),
              "thresholds_consistent": consistent},
    )


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
