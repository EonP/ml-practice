"""NumPy drill 07: boolean masks (target: 8 min). No loops.

x[mask], where mask is a boolean array with x's shape, returns a 1D COPY of
the selected entries in row-major order. On the left of `=` it writes into x.
Combine conditions with &, | and ~, and put parentheses around each
comparison: `and`, `or`, `not` and chained `lo <= x < hi` raise on arrays.
"""
import numpy as np


def positives(x):
    """All strictly positive entries of x (any shape), as a 1D array in row-major order."""
    raise NotImplementedError("TODO: positives")


def count_in_range(x, lo, hi):
    """Number of entries v with lo <= v < hi, as a Python int."""
    raise NotImplementedError("TODO: count_in_range")


def clip_outliers(x, k):
    """A copy of x in which every entry more than k standard deviations from the mean
    is replaced by the mean. The mean and std are taken over all of x. Do not modify x.
    """
    raise NotImplementedError("TODO: clip_outliers")


def masked_mean_loss(loss, target, ignore_index):
    """Mean of a per-pixel loss over the pixels whose target is not ignore_index.

    loss: (N, H, W) float, target: (N, H, W) int. Returns a Python float, and
    0.0 when every pixel is ignored (no NaN, no warnings).
    """
    raise NotImplementedError("TODO: masked_mean_loss")
