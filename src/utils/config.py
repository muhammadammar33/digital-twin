"""YAML config loading with dotted-key CLI overrides, plus global seeding."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """dict with attribute access (cfg.data.root) that stays YAML-serialisable."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Config:
    """Load a YAML config. A config may set `base: other.yaml` to inherit from another file.

    overrides: ["data.root=/x", "train.epochs=3"]; values are parsed as YAML (so 500 -> int, true -> bool).
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base")).to_dict()
        cfg = _deep_update(base, cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, yaml.safe_load(v))
    return Config.wrap(copy.deepcopy(cfg))


def resolve_path(p: str | os.PathLike) -> Path:
    """Relative paths in the config are relative to the repo root, not the CWD."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO_ROOT / p


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch. deterministic=True also forces deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Fused scaled-dot-product attention kernels (flash / memory-efficient / cuDNN) have
        # non-deterministic backward passes on CUDA. Force the plain "math" kernel so runs with
        # attention (A1/A1b) are bit-reproducible too. No effect on CPU or on attention-free B0.
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
            if hasattr(torch.backends.cuda, name):
                getattr(torch.backends.cuda, name)(False)
        torch.backends.cuda.enable_math_sdp(True)


def sdpa_backends() -> str:
    """Which CUDA attention kernels are enabled (for the run log)."""
    import torch
    names = {"flash": "flash_sdp_enabled", "mem_efficient": "mem_efficient_sdp_enabled",
             "cudnn": "cudnn_sdp_enabled", "math": "math_sdp_enabled"}
    on = [k for k, f in names.items() if hasattr(torch.backends.cuda, f) and getattr(torch.backends.cuda, f)()]
    return ",".join(on) or "none"


def git_commit() -> str:
    """Short commit hash of the code being run (+ '-dirty' if there are local edits), or 'unknown'."""
    import subprocess
    try:
        h = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (h + ("-dirty" if dirty else "")) if h else "unknown"
    except Exception:
        return "unknown"
