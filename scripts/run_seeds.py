#!/usr/bin/env python
"""Train + evaluate one config at several seeds, then evaluate their ensemble. Resumable across sessions.

    python scripts/run_seeds.py --config configs/a2_aug.yaml --prefix a2 \\
        --seeds 42 1337 2024 7 314 --reuse 42=a2_aug --time-limit-min 690 \\
        --ensemble-name a2_ens5 --row E1 --vs B0 A1 A2 --vs-run b0 a1_lead a2_aug

Just re-run the SAME command in the next session (with runs/ carried over): every step checks what
already exists on disk and skips or resumes it:
  - seed done        runs/<run>/eval_best/predictions.npz exists               -> skip
  - seed trained     last.pt exists and training is finished                   -> evaluate only
  - seed partial     last.pt exists, unfinished                                -> train.py --resume
  - seed not started                                                           -> train.py --set seed=<s>
The whole command respects --time-limit-min: training stops cleanly between epochs before the
budget runs out (exit code 3), and the runner stops; the ensemble is evaluated once every seed is done.
Members are named <prefix>_s<seed>, except seeds mapped to an existing run with --reuse SEED=RUN.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
T0 = time.time()


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=REPO)


def remaining_min(limit: float | None) -> float | None:
    return None if limit is None else limit - (time.time() - T0) / 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefix", required=True, help="member run names: <prefix>_s<seed>")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--reuse", nargs="*", default=[], help="SEED=RUN: use an existing run for that seed")
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides for NEW runs (and machine keys)")
    ap.add_argument("--time-limit-min", type=float, default=None, help="budget for this whole session")
    ap.add_argument("--min-epoch-budget-min", type=float, default=3.0,
                    help="do not start/resume a training session with less than this many minutes left")
    ap.add_argument("--ensemble-name", default=None)
    ap.add_argument("--average", choices=["prob", "logit"], default="prob")
    ap.add_argument("--row", default=None)
    ap.add_argument("--vs", nargs="*", default=None)
    ap.add_argument("--vs-run", nargs="*", default=None)
    ap.add_argument("--plumbing-check", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    reuse = {int(k): v for k, v in (r.split("=", 1) for r in args.reuse)}
    members = [reuse.get(s, f"{args.prefix}_s{s}") for s in args.seeds]
    runs = REPO / "runs"
    machine = [s for s in args.set if s.split("=", 1)[0] in
               ("data.root", "data.cache_dir", "data.num_workers", "train.device", "train.log_every")]

    status = {}
    for seed, name in zip(args.seeds, members):
        rd = runs / name
        if (rd / "eval_best" / "predictions.npz").exists():
            status[name] = "done (evaluated)"
            continue
        left = remaining_min(args.time_limit_min)
        if left is not None and left < args.min_epoch_budget_min:
            status[name] = "not reached (time budget)"
            break
        tl = ["--time-limit-min", f"{left:.1f}"] if left is not None else []
        if (rd / "last.pt").exists():
            code = run([py, "scripts/train.py", "--run-name", name, "--resume", *tl]
                       + (["--set", *machine] if machine else []))
        else:
            if seed in reuse:
                sys.exit(f"--reuse {seed}={name}: runs/{name} has no last.pt")
            code = run([py, "scripts/train.py", "--config", args.config, "--run-name", name, *tl,
                        "--set", f"seed={seed}", *args.set])
        if code == 3:
            status[name] = "partial (time budget; resume next session)"
            break
        if code != 0:
            sys.exit(f"training {name} failed with exit code {code}")
        code = run([py, "scripts/evaluate.py", "--run-name", name] + (["--plumbing-check"] if args.plumbing_check else []))
        if code != 0:
            sys.exit(f"evaluating {name} failed with exit code {code}")
        status[name] = "done (trained + evaluated this session)"

    print("\n================ seed status ================")
    for name in members:
        print(f"  {name:<14} {status.get(name, 'pending')}")
    done = all(str(status.get(m, "")).startswith("done") for m in members)
    if not done:
        print("\nNot all seeds are finished. Carry runs/ over and re-run the same command in the next session.")
        sys.exit(3)
    if args.ensemble_name:
        cmd = [py, "scripts/ensemble.py", "--name", args.ensemble_name, "--members", *members,
               "--average", args.average]
        if args.row:
            cmd += ["--row", args.row]
        if args.vs:
            cmd += ["--vs", *args.vs]
        if args.vs_run:
            cmd += ["--vs-run", *args.vs_run]
        if args.plumbing_check:
            cmd += ["--plumbing-check"]
        sys.exit(run(cmd))


if __name__ == "__main__":
    main()
