"""Training loop with per-epoch validation, checkpointing and exact resume.

Run directory (runs/<name>/):
    config.yaml   resolved config of the run
    last.pt       full state after the latest completed epoch (model, optimizer, scheduler,
                  AMP scaler, EMA/SWA, early-stopping state, RNG states) -> --resume continues here
    best.pt       weights of the best epoch by threshold-optimised VAL macro-F1, plus its val
                  thresholds and metrics (what evaluate.py loads)
    swa.pt        SWA weights (only if train.swa.enabled)
    log.csv       one row per epoch (train/val loss, val macro-AUC, val macro-F1, per-class AUC/F1, lr)

Model selection & early stopping use val macro-F1 with per-class thresholds grid-searched on val.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data.augment import mixup_batch
from src.eval.metrics import summarize
from src.eval.predict import predict
from src.train.averaging import build_ema, build_swa, swa_update_bn
from src.train.loss import build_loss
from src.train.optim import build_optimizer, build_scheduler


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename, so a kill mid-save never corrupts the last checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


class Trainer:
    def __init__(self, cfg, model, loaders: dict, classes: list[str], run_dir: Path, log=print):
        self.cfg, self.classes, self.run_dir, self.log = cfg, classes, run_dir, log
        tc = cfg.train
        self.device = pick_device(tc.get("device", "auto"))
        self.model = model.to(self.device)
        self.loaders = loaders
        self.limit_train = tc.get("limit_train_batches")
        self.limit_eval = tc.get("limit_eval_batches")
        self.steps_per_epoch = min(len(loaders["train"]), self.limit_train or math.inf)
        self.loss_fn = build_loss(cfg).to(self.device)
        self.opt = build_optimizer(self.model, cfg)
        self.sched = build_scheduler(self.opt, cfg, self.steps_per_epoch)
        amp = tc.get("amp", "auto")
        self.amp_dtype = torch.float16 if (amp is True or (amp == "auto" and self.device.type == "cuda")) else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype is torch.float16
                                           and self.device.type == "cuda")
        self.ema = build_ema(self.model, cfg)
        self.swa = build_swa(self.model, cfg)
        self.epoch = 0                       # number of completed epochs
        self.global_step = 0
        self.best = {"metric": -math.inf, "epoch": None}
        self.bad_epochs = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None,
            "swa": self.swa.state_dict() if self.swa else None,
            "epoch": self.epoch, "global_step": self.global_step, "best": self.best,
            "bad_epochs": self.bad_epochs, "history": self.history, "rng": _rng_state(),
            "config": self.cfg.to_dict(),
        }

    def load_state_dict(self, st: dict) -> None:
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        self.scaler.load_state_dict(st["scaler"])
        if self.ema and st.get("ema"):
            self.ema.load_state_dict(st["ema"])
        if self.swa and st.get("swa"):
            self.swa.load_state_dict(st["swa"])
        self.epoch, self.global_step = st["epoch"], st["global_step"]
        self.best, self.bad_epochs, self.history = st["best"], st["bad_epochs"], st["history"]
        _set_rng_state(st["rng"])

    def resume(self, ckpt_path: Path) -> None:
        st = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.load_state_dict(st)
        self.log(f"resumed from {ckpt_path}: {self.epoch} epochs done, best val macro-F1 "
                 f"{self.best['metric']:.4f} (epoch {self.best['epoch']})")

    # ------------------------------------------------------------------ epoch
    def _eval_model(self):
        return self.ema.module if (self.ema and self.cfg.train.ema.evaluate) else self.model

    def train_one_epoch(self) -> float:
        self.model.train()
        # reseed the shuffle/sampler/worker RNG from (seed, epoch): identical order after --resume
        self.loaders["train"].generator.manual_seed(int(self.cfg.seed) + self.epoch)
        self.loaders["train"].dataset.set_epoch(int(self.cfg.seed) * 100003 + self.epoch)
        mix = self.cfg.data.get("mixup") or {}
        mix_on = bool(mix.get("enabled", False))
        mix_rng = np.random.default_rng([int(self.cfg.seed), self.epoch, 7])   # resume-exact
        loss_sum, n = 0.0, 0
        clip = self.cfg.train.get("grad_clip")
        t0 = time.time()
        for i, b in enumerate(self.loaders["train"]):
            if self.limit_train and i >= self.limit_train:
                break
            x = b["x"].to(self.device, non_blocking=True)
            y = b["y"].to(self.device, non_blocking=True)
            if mix_on:
                x, y = mixup_batch(x, y, float(mix.get("alpha", 0.2)), float(mix.get("p", 0.5)), mix_rng)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                logits = self.model(x).logits
            loss = self.loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {self.epoch + 1}, step {i}")
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if clip:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip))
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.global_step += 1
            if self.ema:
                self.ema.update_parameters(self.model)
            loss_sum += loss.item() * len(y)
            n += len(y)
            every = int(self.cfg.train.get("log_every", 50))
            if every and (i + 1) % every == 0:
                self.log(f"  ep {self.epoch + 1} step {i + 1}/{self.steps_per_epoch} "
                         f"loss {loss_sum / n:.4f} lr {self.sched.get_last_lr()[0]:.2e} "
                         f"({(time.time() - t0) / (i + 1):.2f} s/step)")
        return loss_sum / max(n, 1)

    def evaluate(self, split: str, model=None) -> dict:
        loader = self.loaders[split]
        if self.limit_eval:
            loader = _Limited(loader, self.limit_eval)
        res = predict(model or self._eval_model(), loader, self.device, self.cfg,
                      amp_dtype=self.amp_dtype, loss_fn=self.loss_fn)
        m = summarize(res["labels"], res["probs"], self.classes)
        m["loss"] = res["loss"]
        return m

    # ------------------------------------------------------------------ fit
    def fit(self, stop_after: int | None = None) -> dict:
        tc = self.cfg.train
        epochs = int(tc.epochs)
        patience = tc.early_stopping.get("patience")
        swa_start = int(tc.swa.start_epoch) if self.swa else None
        ran = 0
        while self.epoch < epochs:
            if patience is not None and self.bad_epochs >= int(patience):
                self.log(f"early stopping: no val macro-F1 improvement for {self.bad_epochs} epochs")
                break
            if stop_after is not None and ran >= stop_after:
                self.log(f"stopping after {ran} epoch(s) this session (--stop-after); resume with --resume")
                break
            t0 = time.time()
            lr = self.sched.get_last_lr()[0]
            train_loss = self.train_one_epoch()
            self.epoch += 1
            ran += 1
            if self.swa and self.epoch >= swa_start:
                self.swa.update_parameters(self.model)
            val = self.evaluate("val")
            improved = val["macro_f1"] > self.best["metric"]
            row = {"epoch": self.epoch, "lr_start": lr, "train_loss": train_loss, "val_loss": val["loss"],
                   "val_macro_auc": val["macro_auc"], "val_macro_f1": val["macro_f1"],
                   "val_macro_f1_at_0.5": val["macro_f1_at_0.5"],
                   **{f"val_auc_{c}": val[f"auc_{c}"] for c in self.classes},
                   **{f"val_f1_{c}": val[f"f1_{c}"] for c in self.classes},
                   "seconds": time.time() - t0, "best": improved}
            self.history.append(row)
            self.log(f"epoch {self.epoch:>3}/{epochs}  train {train_loss:.4f}  val {val['loss']:.4f}  "
                     f"AUC {val['macro_auc']:.4f}  F1(thr) {val['macro_f1']:.4f}  "
                     f"F1@0.5 {val['macro_f1_at_0.5']:.4f}  "
                     + " ".join(f"{c}:{val[f'auc_{c}']:.3f}" for c in self.classes)
                     + f"  {row['seconds']:.0f}s" + ("  *best*" if improved else ""))
            if improved:
                self.best = {"metric": val["macro_f1"], "epoch": self.epoch}
                self.bad_epochs = 0
                atomic_save({"model": self._eval_model().state_dict(), "epoch": self.epoch,
                             "val_metrics": val, "thresholds": val["thresholds"],
                             "weights": "ema" if self._eval_model() is not self.model else "raw",
                             "config": self.cfg.to_dict(), "classes": self.classes},
                            self.run_dir / "best.pt")
            else:
                self.bad_epochs += 1
            self._write_log()
            atomic_save(self.state_dict(), self.run_dir / "last.pt")

        finished = self.epoch >= epochs or (patience is not None and self.bad_epochs >= int(patience))
        if finished and self.swa and self.swa.n_averaged.item() > 0 and not (self.run_dir / "swa.pt").exists():
            self.log(f"SWA: averaged {self.swa.n_averaged.item()} epochs; recomputing BN statistics")
            swa_update_bn(self.swa.module, self.loaders["train"], self.device)
            val = self.evaluate("val", model=self.swa.module)
            self.log(f"SWA val: AUC {val['macro_auc']:.4f}  F1(thr) {val['macro_f1']:.4f}")
            atomic_save({"model": self.swa.module.state_dict(), "val_metrics": val,
                         "thresholds": val["thresholds"], "weights": "swa",
                         "config": self.cfg.to_dict(), "classes": self.classes}, self.run_dir / "swa.pt")
        return {"finished": finished, "best": self.best, "epochs_done": self.epoch}

    def _write_log(self) -> None:
        with open(self.run_dir / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            w.writeheader()
            w.writerows(self.history)
        (self.run_dir / "best.json").write_text(json.dumps(self.best, indent=2))


class _Limited:
    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        for i, b in enumerate(self.loader):
            if i >= self.n:
                break
            yield b

    def __len__(self):
        return min(self.n, len(self.loader))
