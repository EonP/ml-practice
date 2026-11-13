"""Drill 13: attention (target: 35 min). NumPy, then PyTorch."""
import math

import numpy as np
import torch
from torch import nn


def causal_mask(L):
    """Boolean (L, L) mask, True where attention is ALLOWED (query i sees keys j <= i)."""
    raise NotImplementedError("TODO: causal_mask")


def scaled_dot_product_attention(q, k, v, mask=None):
    """NumPy attention.

    q: (..., Lq, d), k: (..., Lk, d), v: (..., Lk, dv).
    mask: optional bool array broadcastable to (..., Lq, Lk), True = allowed.
    Returns (out (..., Lq, dv), weights (..., Lq, Lk)).
    Use a numerically stable softmax and work for any number of leading batch dims.
    """
    raise NotImplementedError("TODO: scaled_dot_product_attention")


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention in PyTorch, written by hand.

    __init__(d_model, n_heads): create four nn.Linear(d_model, d_model) layers
    named q_proj, k_proj, v_proj, out_proj (with bias). The tests rely on these names.
    forward(x, causal=False): x (B, L, d_model) -> (B, L, d_model).
    Split into heads with view + transpose to (B, h, L, d_head); do not call
    F.scaled_dot_product_attention or nn.MultiheadAttention.
    """

    def __init__(self, d_model, n_heads):
        raise NotImplementedError("TODO: __init__")

    def forward(self, x, causal=False):
        raise NotImplementedError("TODO: forward")
