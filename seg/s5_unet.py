"""Stage 5 (Day 5): a U-Net, written without looking at a reference.

    input (B, in_ch, H, W)
      enc0: DoubleConv(in_ch, base)            -> skip 0     (H, W)
      down: MaxPool2d(2) then DoubleConv(base, 2*base)       (H/2, W/2)
      ...   `depth` times, doubling channels each time (the last one is the bottleneck)
      up:   ConvTranspose2d(c, c//2, kernel_size=2, stride=2), concat with the
            matching skip along channels (dim=1), then DoubleConv(c, c//2)
      head: Conv2d(base, n_classes, kernel_size=1)
    output logits (B, n_classes, H, W)   (H and W divisible by 2**depth)
"""
import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv2d 3x3 padding=1, no bias -> BatchNorm2d -> ReLU) twice. Keeps H and W."""

    def __init__(self, in_ch, out_ch):
        raise NotImplementedError("TODO: __init__")

    def forward(self, x):
        raise NotImplementedError("TODO: forward")


class UNet(nn.Module):
    """See the module docstring. Use nn.ModuleList for the per-level blocks
    (a plain Python list would hide the parameters from the optimiser)."""

    def __init__(self, in_channels=3, n_classes=3, base=16, depth=3):
        raise NotImplementedError("TODO: __init__")

    def forward(self, x):
        raise NotImplementedError("TODO: forward")
