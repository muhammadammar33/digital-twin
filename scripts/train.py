#!/usr/bin/env python
"""Train the Stream 1 model.

    python scripts/train.py --run-name b0                          # B0 baseline (configs/default.yaml)
    python scripts/train.py --run-name b0 --resume                 # continue runs/b0 from last.pt
    python scripts/train.py --run-name a1 --set model.attention.type=lead
    python scripts/train.py --overfit-batch                        # single-batch overfit sanity test

On --resume the run's saved config.yaml is used; --set may still override machine-specific
keys (data.root, data.cache_dir, data.num_workers, train.device, train.log_every), so the same
run can continue on a different machine/session.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import build_dataloaders  # noqa: E402
from src.models.model import build_model, count_params  # noqa: E402
from src.train.loop import Trainer, pick_device  # noqa: E402
from src.utils.config import load_config, git_commit, resolve_path, sdpa_backends, seed_everything  # noqa: E402

RESUME_SAFE = ("data.root", "data.cache_dir", "data.num_workers", "train.device", "train.log_every")


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def overfit_batch(cfg, steps: int, batch_size: int) -> None:
    """Train on ONE fixed batch: a correct model/loss/optimizer drives the loss towards 0."""
    cfg.model.head_dropout = 0.0
    seed_everything(cfg.seed)
    out = build_dataloaders(cfg, batch_size=batch_size)
    b = next(iter(out["loaders"]["train"]))
    device = pick_device(cfg.train.get("device", "auto"))
    model = build_model(cfg).to(device).train()
    x, y = b["x"].to(device), b["y"].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    lossf = torch.nn.BCEWithLogitsLoss()
    print(f"overfitting one batch: x {tuple(x.shape)}, {int(y.sum())} positive labels, device {device}")
    t0 = time.time()
    for s in range(1, steps + 1):
        loss = lossf(model(x).logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s == 1 or s % 10 == 0:
            print(f"  step {s:>4}  loss {loss.item():.5f}  ({time.time() - t0:.0f}s)")
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x).logits)
    acc = ((p > 0.5).float() == y).float().mean().item()
    final = lossf(model(x).logits, y).item()
    print(f"final train-mode loss {loss.item():.5f}; eval-mode loss {final:.5f}; "
          f"label accuracy (eval mode, thr 0.5) {acc:.3f}")
    ok = loss.item() < 0.05 and acc == 1.0
    print("OVERFIT TEST " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


T_START = time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=3")
    ap.add_argument("--run-name", default=None, help="run directory name under train.runs_dir")
    ap.add_argument("--resume", action="store_true", help="continue runs/<run-name> from last.pt")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after N epochs this session")
    ap.add_argument("--time-limit-min", type=float, default=None,
                    help="wall-clock budget for this session; stops cleanly between epochs (exit code 3 = unfinished)")
    ap.add_argument("--overfit-batch", action="store_true", help="single-batch overfit test")
    ap.add_argument("--overfit-steps", type=int, default=150)
    ap.add_argument("--overfit-bs", type=int, default=16)
    args = ap.parse_args()

    if args.overfit_batch:
        overfit_batch(load_config(args.config, args.set), args.overfit_steps, args.overfit_bs)
        return

    base_cfg = load_config(args.config, args.set)
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = resolve_path(base_cfg.train.runs_dir) / run_name

    if args.resume:
        if not (run_dir / "last.pt").exists():
            sys.exit(f"--resume: {run_dir / 'last.pt'} not found")
        cfg = load_config(run_dir / "config.yaml",
                          [s for s in args.set if s.split("=", 1)[0] in RESUME_SAFE])
        ignored = [s for s in args.set if s.split("=", 1)[0] not in RESUME_SAFE]
        if ignored:
            print(f"--resume: ignoring overrides that would change the experiment: {ignored}")
    else:
        if (run_dir / "last.pt").exists():
            sys.exit(f"{run_dir} already has a checkpoint; use --resume or a different --run-name")
        cfg = base_cfg
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    log = Tee(run_dir / "train.log")
    seed_everything(cfg.seed, deterministic=bool(cfg.train.get("deterministic", True)))
    data = build_dataloaders(cfg)
    model = build_model(cfg)
    trainer = Trainer(cfg, model, data["loaders"], data["classes"], run_dir, log=log)
    log(f"code commit {git_commit()}")
    log(f"run {run_dir}  device {trainer.device}  amp {trainer.amp_dtype}  "
        f"params {count_params(model) / 1e6:.2f}M  steps/epoch {trainer.steps_per_epoch}  "
        f"attention {cfg.model.attention.type}  fs {cfg.data.sampling_rate} Hz  input {cfg.data.input.mode}  "
        f"deterministic {cfg.train.get('deterministic', True)} (attention kernels: {sdpa_backends()})")
    if args.resume:
        trainer.resume(run_dir / "last.pt")
    deadline = T_START + 60 * args.time_limit_min if args.time_limit_min else None
    res = trainer.fit(stop_after=args.stop_after, deadline=deadline)
    log(f"done: {res['epochs_done']} epochs, best val macro-F1 {res['best']['metric']:.4f} "
        f"at epoch {res['best']['epoch']}" + ("" if res["finished"] else " (unfinished: use --resume)"))
    sys.exit(0 if res["finished"] else 3)


if __name__ == "__main__":
    main()
