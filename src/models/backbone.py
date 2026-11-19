"""xresnet1d: 1-D port of fastai's XResNet ("bag of tricks" ResNet), following Strodthoff et al.
(ecg_ptbxl_benchmarking/code/models/xresnet1d.py), without the fastai dependency.

Architecture (xresnet1d101): stem of three conv-BN-ReLU (12->32->32->64, k=5, first stride 2),
max-pool /2, then 4 stages of bottleneck ResBlocks (expansion 4) with
blocks (3, 4, 23, 3) and widths (64, 128, 256, 512) -> 2048 output channels.
Downsampling stages use stride 2 in the middle conv and AvgPool on the shortcut ("ResNet-D").
Last BN of each block is zero-initialised so every block starts as identity.

Only the feature extractor is here (output [B, 2048, L]); pooling/projection/head live in
src/models/model.py so the 512-d embedding boundary is defined in one place.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv(ni: int, nf: int, ks: int = 3, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=False)


def conv_layer(ni: int, nf: int, ks: int = 3, stride: int = 1, zero_bn: bool = False,
               act: bool = True) -> nn.Sequential:
    bn = nn.BatchNorm1d(nf)
    nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
    layers = [_conv(ni, nf, ks, stride), bn]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, expansion: int, ni: int, nh: int, stride: int = 1, kernel_size: int = 5):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        if expansion == 1:
            layers = [conv_layer(ni, nh, kernel_size, stride=stride),
                      conv_layer(nh, nf, kernel_size, zero_bn=True, act=False)]
        else:  # bottleneck: 1x1 reduce -> kxk (strided) -> 1x1 expand
            layers = [conv_layer(ni, nh, 1),
                      conv_layer(nh, nh, kernel_size, stride=stride),
                      conv_layer(nh, nf, 1, zero_bn=True, act=False)]
        self.convs = nn.Sequential(*layers)
        self.idconv = nn.Identity() if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool1d(2, ceil_mode=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.idconv(self.pool(x)))


class XResNet1d(nn.Module):
    def __init__(self, expansion: int, layers: list[int], input_channels: int = 12,
                 widths: tuple[int, ...] = (64, 128, 256, 512), kernel_size: int = 5,
                 kernel_size_stem: int = 5):
        super().__init__()
        stem_sizes = [input_channels, 32, 32, 64]
        self.stem = nn.Sequential(
            *[conv_layer(stem_sizes[i], stem_sizes[i + 1], ks=kernel_size_stem, stride=2 if i == 0 else 1)
              for i in range(3)],
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        block_szs = [64 // expansion, *widths]
        self.stages = nn.Sequential(*[
            nn.Sequential(*[ResBlock(expansion, block_szs[i] if j == 0 else block_szs[i + 1],
                                     block_szs[i + 1], stride=(1 if i == 0 else 2) if j == 0 else 1,
                                     kernel_size=kernel_size)
                            for j in range(n)])
            for i, n in enumerate(layers)
        ])
        self.out_channels = widths[-1] * expansion
        self.apply(_init_cnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))          # [B, out_channels, T / 32]


def _init_cnn(m: nn.Module) -> None:
    """kaiming-normal conv/linear weights, zero biases (fastai init_cnn). BN set in conv_layer."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# name -> (expansion, blocks per stage). Registry = seam for multi-architecture ensembles.
XRESNET_SPECS = {
    "xresnet1d18": (1, [2, 2, 2, 2]),
    "xresnet1d34": (1, [3, 4, 6, 3]),
    "xresnet1d50": (4, [3, 4, 6, 3]),
    "xresnet1d101": (4, [3, 4, 23, 3]),
    "xresnet1d152": (4, [3, 8, 36, 3]),
}


def build_backbone(name: str, input_channels: int = 12, kernel_size: int = 5,
                   kernel_size_stem: int = 5, widths=(64, 128, 256, 512)) -> XResNet1d:
    if name not in XRESNET_SPECS:
        raise KeyError(f"unknown backbone {name!r}; available: {sorted(XRESNET_SPECS)}")
    expansion, layers = XRESNET_SPECS[name]
    return XResNet1d(expansion, layers, input_channels, tuple(widths), kernel_size, kernel_size_stem)
