"""NumPy drill 10: None and broadcasting (target: 8 min). No loops.

Broadcasting lines shapes up from the RIGHT, and size-1 axes stretch to match.
x[:, None] (np.newaxis is the same object) inserts a size-1 axis, which lets
you choose which axes line up.
"""
import numpy as np


def outer(a, b):
    """a (N,), b (M,) -> (N, M) with out[i, j] = a[i] * b[j]. No np.outer or einsum."""
    raise NotImplementedError("TODO: outer")


def scale_rows(x, s):
    """x (N, D), s (N,) -> x with row i multiplied by s[i]."""
    raise NotImplementedError("TODO: scale_rows")


def add_per_channel(x, bias):
    """x (N, C, H, W), bias (C,) -> x with bias[c] added to every value in channel c."""
    raise NotImplementedError("TODO: add_per_channel")


def causal_mask(n):
    """(n, n) bool, True where column j <= row i (position i may attend to j). No np.tril / np.triu."""
    raise NotImplementedError("TODO: causal_mask")


def pairwise_diff(x):
    """x (N, D) -> (N, N, D) with out[i, j] = x[i] - x[j]."""
    raise NotImplementedError("TODO: pairwise_diff")
