#!/usr/bin/env python
"""Milestone-3 checks: architecture, parameter count, forward shapes, attention toggles,
and the frozen-encoder / embedding boundary.

    python scripts/check_model.py [--set key=value ...]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import build_dataloaders  # noqa: E402
from src.models.backbone import ResBlock  # noqa: E402
from src.models.model import build_model, count_params  # noqa: E402
from src.utils.config import load_config, seed_everything  # noqa: E402

FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAIL.append(msg)


def hr(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # ------------------------------------------------------------------ B0 architecture
    hr(f"B0: pure {cfg.model.backbone.name} (attention: {cfg.model.attention.type})")
    model = build_model(cfg)
    bb = model.encoder.backbone
    blocks = [sum(isinstance(m, ResBlock) for m in stage.modules()) for stage in bb.stages]
    out_ch = [stage[-1].convs[-1][0].out_channels for stage in bb.stages]
    convs = sum(isinstance(m, torch.nn.Conv1d) for m in bb.modules())
    print(f"bottleneck blocks per stage : {tuple(blocks)}")
    print(f"stage widths (inner)        : {tuple(cfg.model.backbone.widths)}  "
          f"-> output channels {tuple(out_ch)}")
    print(f"conv layers in backbone     : {convs}  (3 stem + 3x{sum(blocks)} block convs "
          f"+ {convs - 3 - 3 * sum(blocks)} shortcut projections)")
    check(tuple(blocks) == (3, 4, 23, 3), "blocks (3, 4, 23, 3)")
    check(tuple(out_ch) == (256, 512, 1024, 2048), "bottleneck expansion 4 -> 2048 channels")
    check(model.encoder.pre_attn is None and model.encoder.post_attn is None, "no attention in B0")

    parts = {"backbone (xresnet1d101)": bb, "projection 2048->512": model.encoder.projection,
             "head 512->5": model.head}
    for k, m in parts.items():
        print(f"params  {k:<26}{count_params(m):>12,}")
    total = count_params(model)
    print(f"params  {'TOTAL':<26}{total:>12,}  ({total / 1e6:.2f} M)")

    # ------------------------------------------------------------------ forward on real data
    hr("Forward pass on a real batch")
    loader = build_dataloaders(cfg)["loaders"]["train"]
    batch = next(iter(loader))
    x, y = batch["x"], batch["y"]
    model.eval()
    with torch.no_grad():
        fmap = model.encoder.features(x)
        out = model(x)
    logits, emb = out                                # NamedTuple unpacks like a tuple
    print(f"input      {tuple(x.shape)}")
    print(f"feature map{tuple(fmap.shape)}   (backbone output, T/32)")
    print(f"logits     {tuple(logits.shape)}   embedding {tuple(emb.shape)}")
    print(f"embedding stats: min {emb.min():.3f}  mean {emb.mean():.3f}  "
          f"frac>0 {(emb > 0).float().mean():.2f}")
    B = x.shape[0]
    check(logits.shape == (B, 5), "logits [B, 5]")
    check(emb.shape == (B, 512), "embedding [B, 512]")
    check(out.logits is logits and out.embedding is emb, "named outputs (out.logits / out.embedding)")
    check(bool(torch.isfinite(logits).all() and torch.isfinite(emb).all()), "finite outputs")
    with torch.no_grad():
        emb2 = model.encoder(x)
        logits2 = model.head(emb2)
    check(torch.allclose(emb, emb2) and torch.allclose(logits, logits2),
          "model(x) == head(encoder(x)): the embedding is the only path to the logits")

    for shape, what in [((2, 12, 5000), "500 Hz full"), ((2, 12, 250), "2.5 s crop @100 Hz")]:
        with torch.no_grad():
            o = model(torch.randn(*shape))
        check(o.logits.shape == (2, 5) and o.embedding.shape == (2, 512), f"{what} {shape} -> [2,5]/[2,512]")

    # ------------------------------------------------------------------ backward + timing
    hr("Backward pass + CPU timing")
    model.train()
    t0 = time.time()
    out = model(x)
    loss = F.binary_cross_entropy_with_logits(out.logits, y)
    loss.backward()
    dt = time.time() - t0
    grads_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"BCE loss at init {loss.item():.4f}  (ln2 = 0.693 for uninformed logits)")
    print(f"fwd+bwd, batch {B}: {dt:.2f} s on {torch.get_num_threads()} CPU threads "
          f"-> ~{dt * len(loader) / 60:.0f} min / epoch on this machine")
    check(grads_ok, "finite gradients everywhere")
    model.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ attention toggles
    hr("Attention toggles (zero-gated: identical to B0 at init)")
    base_sd = model.state_dict()
    model.eval()
    with torch.no_grad():
        ref = model(x[:8])
    for kind in ("lead", "temporal"):
        c = copy.deepcopy(cfg)
        c.model.attention.type = kind
        m = build_model(c)
        attn = m.encoder.pre_attn or m.encoder.post_attn
        missing, unexpected = m.load_state_dict(base_sd, strict=False)
        check(not unexpected and all(k.split(".")[1] in ("pre_attn", "post_attn") for k in missing),
              f"{kind}: B0 weights load; only attention params are new")
        m.eval()
        with torch.no_grad():
            o = m(x[:8])
        print(f"{kind:<9} placement={attn.placement:<4} params {count_params(attn):>10,}  "
              f"max |Δlogits| vs B0 at init = {(o.logits - ref.logits).abs().max():.2e}")
        check(torch.allclose(o.logits, ref.logits, atol=1e-5), f"{kind}: equals B0 at init (gate=0)")
        m.train()
        m(x[:8]).logits.sum().backward()
        check(attn.gate.grad is not None and attn.gate.grad.abs().item() > 0,
              f"{kind}: gate receives gradient (can learn to open)")

    # ------------------------------------------------------------------ frozen boundary
    hr("Frozen-encoder boundary")
    model.load_state_dict(base_sd)
    model.freeze_encoder()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"trainable after freeze: {trainable}  ({count_params(model, trainable_only=True):,} params)")
    check(all(n.startswith("head.") for n in trainable), "only the head is trainable")
    enc_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    check(not model.encoder.training and model.head.training, "train() keeps frozen encoder in eval mode")
    for _ in range(3):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(x).logits, y).backward()
        opt.step()
    same = all(torch.equal(enc_before[k], v) for k, v in model.encoder.state_dict().items())
    check(same, "encoder weights AND BatchNorm running stats unchanged after 3 head-only steps")
    with torch.no_grad():
        e1, e2 = model.embed(x[:4]), model.embed(x[:4])
    check(torch.equal(e1, e2), "frozen embedding is deterministic")
    model.freeze_encoder(False)
    check(count_params(model, trainable_only=True) == total, "unfreeze restores all params")

    hr("Result")
    if FAIL:
        sys.exit("FAILED: " + "; ".join(FAIL))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
