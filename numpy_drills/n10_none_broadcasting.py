"""NumPy drill 10: None and broadcasting (target: 8 min). No loops.

Broadcasting lines shapes up from the RIGHT, and size-1 axes stretch to match.
x[:, None] (np.newaxis is the same object) inserts a size-1 axis, which lets
you choose which axes line up.
"""
import numpy as np


def outer(a, b):
    """a (N,), b (M,) -> (N, M) with out[i, j] = a[i] * b[j]. No np.outer or einsum."""
    return a[:, None] * b[None] # (N, 1), (1, M) -> (N , M) , Note that a[:, None] * b also works and is shoter


def scale_rows(x, s):
    """x (N, D), s (N,) -> x with row i multiplied by s[i]."""
    return x * s[:, None]


def add_per_channel(x, bias):
    """x (N, C, H, W), bias (C,) -> x with bias[c] added to every value in channel c."""
    return x + bias[None, :, None, None]


def causal_mask(n):
    """(n, n) bool, True where column j <= row i (position i may attend to j). No np.tril / np.triu."""
    cols = np.arange(n) # (N,)
    rows = np.arange(n) # (N,)
    return cols <= rows[:, None] # cols (N,) i.e. (1, N) and rows is (N, 1)


def pairwise_diff(x):
    """x (N, D) -> (N, N, D) with out[i, j] = x[i] - x[j]."""
    return x[:, None] - x # (N, 1, D) and (1, N, D)