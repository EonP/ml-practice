"""NumPy drill 14: mixing slices with index arrays (target: 12 min). No loops.

Images are batched (N, C, H, W). When the index arrays (and ints) in an index
sit next to each other, their broadcast shape takes the place of those axes.
When a slice separates them, the broadcast shape moves to the FRONT of the
result. Try x[0, :, [0, 1]] on a (2, 3, 4) array and look at the shape.
"""
import numpy as np


def select_channels(x, ch):
    """x (N, C, H, W), ch (K,) ints -> (N, K, H, W) with out[:, k] = x[:, ch[k]]."""
    raise NotImplementedError("TODO: select_channels")


def channel_per_sample(x, ch):
    """x (N, C, H, W), ch (N,) ints -> (N, H, W) with out[n] = x[n, ch[n]]."""
    raise NotImplementedError("TODO: channel_per_sample")


def pixel_per_sample(x, ys, xs):
    """x (N, C, H, W), ys and xs (N,) ints -> (N, C) with out[n] = x[n, :, ys[n], xs[n]]."""
    raise NotImplementedError("TODO: pixel_per_sample")


def columns_of_sample(x, n, cols):
    """x (N, H, W), n an int, cols (K,) ints -> (H, K) with out[h, k] = x[n, h, cols[k]]."""
    raise NotImplementedError("TODO: columns_of_sample")
