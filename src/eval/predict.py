"""Inference over a DataLoader -> (probabilities, labels, embeddings?), with optional
overlapping-window aggregation (eval.window_agg) for models trained on crops.

Seam for test-time augmentation: `eval.tta` will wrap `_forward` with transformed copies of x.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.windows import aggregate_windows, sliding_windows


def window_agg_enabled(cfg) -> bool:
    w = cfg.eval.window_agg.enabled
    if w == "auto":
        return cfg.data.input.mode == "random_crop"
    return bool(w)


@torch.no_grad()
def predict(model, loader, device, cfg, amp_dtype=None, return_embeddings: bool = False,
            loss_fn=None) -> dict:
    model.eval()
    fs = int(cfg.data.sampling_rate)
    use_win = window_agg_enabled(cfg)
    win_len = int(round(cfg.eval.window_agg.window_seconds * fs))
    stride = int(round(cfg.eval.window_agg.stride_seconds * fs))
    probs, labels, embs, ids = [], [], [], []
    loss_sum, n = 0.0, 0
    for b in loader:
        x = b["x"].to(device, non_blocking=True)
        y = b["y"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            if use_win:
                w = sliding_windows(x, win_len, stride)                # [B, W, C, L]
                B, W = w.shape[:2]
                out = model(w.reshape(B * W, *w.shape[2:]))
                logits = aggregate_windows(out.logits.float().reshape(B, W, -1),
                                           cfg.eval.window_agg.reduce)
                emb = out.embedding.float().reshape(B, W, -1).mean(1)
            else:
                out = model(x)
                logits, emb = out.logits.float(), out.embedding.float()
        if loss_fn is not None:
            loss_sum += loss_fn(logits, y).item() * len(y)
            n += len(y)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
        ids.append(b["ecg_id"])
        if return_embeddings:
            embs.append(emb.cpu())
    res = {"probs": torch.cat(probs).numpy(), "labels": torch.cat(labels).numpy(),
           "ecg_ids": torch.cat(ids).numpy()}
    if loss_fn is not None:
        res["loss"] = loss_sum / max(n, 1)
    if return_embeddings:
        res["embeddings"] = torch.cat(embs).numpy()
    return res
