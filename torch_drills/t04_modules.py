"""PyTorch drill 04: writing nn.Modules (target: 15 min).

Assigning an nn.Module or nn.Parameter to an attribute in __init__ REGISTERS it:
it then shows up in .parameters() and .state_dict(), and moves with .to(). A plain
Python list of layers is NOT registered (use nn.ModuleList or nn.Sequential), and
neither is a plain tensor attribute. State that isn't trained, like running
statistics, goes in self.register_buffer(name, tensor). Call super().__init__() first.
"""
import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected network. sizes = [d_in, h_1, ..., d_out].

    One nn.Linear between each pair of consecutive sizes. After every Linear EXCEPT the
    last one: ReLU, then dropout with probability `dropout`.
    forward(x (N, d_in)) -> logits (N, d_out).
    """

    def __init__(self, sizes, dropout=0.0):
        raise NotImplementedError("TODO: MLP.__init__")

    def forward(self, x):
        raise NotImplementedError("TODO: MLP.forward")


class LayerNorm(nn.Module):
    """Layer normalisation over the last axis, matching nn.LayerNorm(d, eps=eps).

    Two parameters of shape (d,): `weight` (starts at ones) and `bias` (starts at zeros).
    y = (x - mean) / sqrt(var + eps) * weight + bias, with the mean and the BIASED variance
    taken over the last axis. Don't use nn.LayerNorm or F.layer_norm.
    """

    def __init__(self, d, eps=1e-5):
        raise NotImplementedError("TODO: LayerNorm.__init__")

    def forward(self, x):
        raise NotImplementedError("TODO: LayerNorm.forward")


class RunningCenter(nn.Module):
    """Centres features and keeps a running mean, the way BatchNorm keeps its statistics.

    A buffer named `running_mean`, shape (d,), starting at zeros.
    forward(x (N, d)):
      training mode: running_mean <- (1 - momentum) * running_mean + momentum * (batch mean),
                     updated without tracking gradients; return x - (batch mean).
      eval mode:     return x - running_mean, leaving the buffer unchanged.
    """

    def __init__(self, d, momentum=0.1):
        raise NotImplementedError("TODO: RunningCenter.__init__")

    def forward(self, x):
        raise NotImplementedError("TODO: RunningCenter.forward")
