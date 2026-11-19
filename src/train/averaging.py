"""EMA and SWA weight averaging (config train.ema / train.swa), both off in B0.

EMA: exponential moving average of weights AND BatchNorm buffers, updated every step.
     If train.ema.evaluate is true, validation/model selection use the EMA weights.
SWA: equal-weight average of the weights at the end of each epoch from swa.start_epoch on.
     BatchNorm statistics are recomputed on the training set at the end (update_bn), and the
     SWA model is evaluated and saved separately (swa.pt) so it can be its own benchmark row.
"""
from __future__ import annotations

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def build_ema(model, cfg):
    ec = cfg.train.ema
    if not ec.enabled:
        return None
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(ec.decay)), use_buffers=True)


def build_swa(model, cfg):
    if not cfg.train.swa.enabled:
        return None
    return AveragedModel(model, use_buffers=False)


@torch.no_grad()
def swa_update_bn(swa_model, loader, device) -> None:
    """Recompute BN running stats for the SWA weights (dict batches, so not torch's update_bn)."""
    bns = [m for m in swa_model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    momenta = {}
    for m in bns:
        m.reset_running_stats()
        momenta[m] = m.momentum
        m.momentum = None                 # cumulative average over the epoch
    was_training = swa_model.training
    swa_model.train()
    for b in loader:
        swa_model(b["x"].to(device))
    for m in bns:
        m.momentum = momenta[m]
    swa_model.train(was_training)
