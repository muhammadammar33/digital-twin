"""Stream 1 model: [optional pre-attention] -> xresnet1d -> [optional post-attention]
-> adaptive pooling -> 512-d projection (= THE EMBEDDING) -> linear head -> 5 logits.

Interpretation-layer boundary
-----------------------------
    model.encoder(x)  -> embedding [B, 512]      (everything up to and including the projection)
    model.head(emb)   -> logits    [B, 5]        (dropout + one Linear; nothing else)
    model(x)          -> StreamOutput(logits, embedding)   (a NamedTuple; unpacks as a tuple)

`model.freeze_encoder()` stops gradients into the encoder and pins it (incl. BatchNorm) in eval
mode, so a later layer can be trained on the embedding without the backbone ever changing.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from src.models.attention import build_attention
from src.models.backbone import build_backbone


class StreamOutput(NamedTuple):
    logits: torch.Tensor      # [B, n_classes], pre-sigmoid
    embedding: torch.Tensor   # [B, embed_dim]


class AdaptivePool(nn.Module):
    """avg (paper spec) | max | concat (avg ‖ max, as in the fastai head)."""

    def __init__(self, kind: str = "avg"):
        super().__init__()
        if kind not in ("avg", "max", "concat"):
            raise ValueError(f"unknown pooling {kind!r}")
        self.kind = kind

    def factor(self) -> int:
        return 2 if self.kind == "concat" else 1

    def forward(self, f: torch.Tensor) -> torch.Tensor:          # [B, C, L] -> [B, C*factor]
        if self.kind == "avg":
            return f.mean(dim=-1)
        if self.kind == "max":
            return f.amax(dim=-1)
        return torch.cat([f.mean(dim=-1), f.amax(dim=-1)], dim=1)


class Stream1Encoder(nn.Module):
    def __init__(self, backbone: nn.Module, pre_attn: nn.Module | None, post_attn: nn.Module | None,
                 pooling: str, embed_dim: int, proj_dropout: float = 0.0):
        super().__init__()
        self.pre_attn = pre_attn
        self.backbone = backbone
        self.post_attn = post_attn
        self.pool = AdaptivePool(pooling)
        in_f = backbone.out_channels * self.pool.factor()
        # projection: Linear -> BN -> ReLU; the post-ReLU 512-d vector is the embedding
        self.projection = nn.Sequential(
            nn.Dropout(proj_dropout) if proj_dropout else nn.Identity(),
            nn.Linear(in_f, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.projection[1].weight)
        self.embed_dim = embed_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 12, T] -> backbone feature map [B, 2048, T/32] (after any attention)."""
        if self.pre_attn is not None:
            x = self.pre_attn(x)
        f = self.backbone(x)
        if self.post_attn is not None:
            f = self.post_attn(f)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.features(x)))


class Stream1Model(nn.Module):
    def __init__(self, encoder: Stream1Encoder, n_classes: int, head_dropout: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(encoder.embed_dim, n_classes))
        self._encoder_frozen = False

    def forward(self, x: torch.Tensor) -> StreamOutput:
        emb = self.encoder(x)
        return StreamOutput(self.head(emb), emb)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only embedding (no grad); call model.eval() first for deterministic BN/dropout."""
        return self.encoder(x)

    def freeze_encoder(self, frozen: bool = True) -> "Stream1Model":
        self._encoder_frozen = frozen
        for p in self.encoder.parameters():
            p.requires_grad_(not frozen)
        self.train(self.training)
        return self

    def train(self, mode: bool = True) -> "Stream1Model":
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()          # keep BN running stats & dropout fixed when frozen
        return self


def build_model(cfg) -> Stream1Model:
    m = cfg.model
    n_leads = 12
    backbone = build_backbone(m.backbone.name, input_channels=n_leads,
                              kernel_size=m.backbone.kernel_size,
                              kernel_size_stem=m.backbone.kernel_size_stem,
                              widths=m.backbone.widths)
    attn = build_attention(m.attention, n_leads, backbone.out_channels)
    pre = attn if attn is not None and attn.placement == "pre" else None
    post = attn if attn is not None and attn.placement == "post" else None
    encoder = Stream1Encoder(backbone, pre, post, m.pooling, m.embed_dim, m.get("proj_dropout", 0.0))
    model = Stream1Model(encoder, len(cfg.data.classes), m.head_dropout)
    if m.get("init_from"):
        load_encoder_weights(model, m.init_from)
    return model


def load_encoder_weights(model: Stream1Model, path: str, prefix: str = "backbone.") -> None:
    """Seam for self-supervised pretraining: load backbone weights saved by a pretraining stage.

    Accepts a checkpoint with a 'backbone' state_dict, or a full model/encoder state_dict
    (keys are matched after stripping 'encoder.' / 'backbone.' prefixes).
    """
    from src.utils.config import resolve_path
    ckpt = torch.load(resolve_path(path), map_location="cpu", weights_only=True)
    sd = ckpt.get("backbone", ckpt.get("model", ckpt))
    clean = {}
    for k, v in sd.items():
        k = k.removeprefix("encoder.")
        if k.startswith(prefix):
            k = k[len(prefix):]
        clean[k] = v
    missing, unexpected = model.encoder.backbone.load_state_dict(clean, strict=False)
    if missing:
        raise RuntimeError(f"init_from {path}: backbone keys missing: {missing[:5]}...")
    if unexpected:
        print(f"init_from {path}: ignored {len(unexpected)} non-backbone keys")


def count_params(module: nn.Module | None, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
