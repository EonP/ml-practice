"""Drill 10: 2D convolution and max-pooling (target: 40 min). NumPy only.

Layout is PyTorch's: x (N, C, H, W), weights (F, C, KH, KW).
Like PyTorch, "convolution" here means cross-correlation (no kernel flip).
Output size: H_out = (H + 2 * padding - KH) // stride + 1.
"""
import numpy as np


def conv2d_naive(x, w, b, stride=1, padding=0):
    """Loops are fine here: loop over output positions (and filters if you like)
    and take a dot product with the receptive field. Returns (N, F, H_out, W_out)."""
    raise NotImplementedError("TODO: conv2d_naive")


def conv2d_fast(x, w, b, stride=1, padding=0):
    """Same result with NO Python loops over N, H or W.

    Hint: np.lib.stride_tricks.sliding_window_view gives every receptive field
    as a view; then a single np.einsum (or reshape + matmul, i.e. im2col)."""
    raise NotImplementedError("TODO: conv2d_fast")


def maxpool2d(x, k=2, stride=2):
    """Max-pooling with a k x k window. x (N, C, H, W) -> (N, C, H_out, W_out)."""
    raise NotImplementedError("TODO: maxpool2d")
