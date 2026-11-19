"""Optional attention modules for Stream 1 (OFF in the B0 baseline: model.attention.type: none).

The paper under-specifies its "cross-lead attention", so variants are pluggable:

  lead      Cross-lead attention on the raw input, BEFORE the backbone. The xresnet mixes all
            12 leads in its first conv, so the lead axis only exists at the input. Each lead is
            embedded by a shared 1-D conv (+ a learned lead-identity embedding); at every time
            step a multi-head attention runs over the 12 lead tokens; the result is projected
            back to one channel per lead and added residually.
  temporal  Self-attention over the backbone's temporal feature tokens ([B, 2048, L], L = T/32),
            AFTER the backbone and before pooling.

Both are residual with a learnable scalar gate initialised to 0 (ReZero): at initialisation
the module is the identity, so an attention model starts as exactly the B0 network and any
benchmark gain is attributable to what the attention learns.

Add a variant: write an nn.Module with forward(x) -> same shape, register it in ATTENTION,
and give it a `placement` ("pre" = on the [B, 12, T] input, "post" = on backbone features).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LeadAttention(nn.Module):
    placement = "pre"

    def __init__(self, n_leads: int = 12, d_model: int = 32, heads: int = 4, kernel_size: int = 7,
                 dropout: float = 0.0, **_):
        super().__init__()
        self.embed = nn.Conv1d(1, d_model, kernel_size, padding=kernel_size // 2)
        self.lead_emb = nn.Parameter(torch.zeros(n_leads, d_model))
        nn.init.normal_(self.lead_emb, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(d_model, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [B, C, T]
        B, C, T = x.shape
        h = self.embed(x.reshape(B * C, 1, T))                       # [B*C, d, T]
        h = h.reshape(B, C, -1, T).permute(0, 3, 1, 2)               # [B, T, C, d]
        h = h + self.lead_emb                                        # lead identity
        tok = self.norm(h.reshape(B * T, C, -1))                     # one 12-token sequence per time step
        a, _ = self.attn(tok, tok, tok, need_weights=False)          # [B*T, C, d]
        delta = self.out(a).reshape(B, T, C).transpose(1, 2)         # [B, C, T]
        return x + self.gate * delta


def _sinusoidal(L: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(L, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TemporalAttention(nn.Module):
    placement = "post"

    def __init__(self, in_channels: int = 2048, d_model: int = 256, heads: int = 8,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.down = nn.Linear(in_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.up = nn.Linear(d_model, in_channels)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:          # f [B, C, L]
        tok = f.transpose(1, 2)                                      # [B, L, C]
        h = self.down(tok)
        h = self.norm(h + _sinusoidal(h.shape[1], h.shape[2], h.device))
        a, _ = self.attn(h, h, h, need_weights=False)
        return f + self.gate * self.up(a).transpose(1, 2)


ATTENTION = {"lead": LeadAttention, "temporal": TemporalAttention}


def build_attention(cfg: dict, n_leads: int, feat_channels: int) -> nn.Module | None:
    kind = cfg.get("type", "none")
    if kind in (None, "none"):
        return None
    if kind not in ATTENTION:
        raise KeyError(f"unknown attention {kind!r}; available: none, {', '.join(ATTENTION)}")
    kwargs = dict(cfg.get(kind) or {})
    cls = ATTENTION[kind]
    if cls.placement == "pre":
        return cls(n_leads=n_leads, **kwargs)
    return cls(in_channels=feat_channels, **kwargs)
