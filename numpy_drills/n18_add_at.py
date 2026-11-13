"""NumPy drill 18: scatter-add with np.add.at (target: 10 min). No loops.

`out[idx] += v` looks like a scatter-add, but it goes wrong as soon as idx has
repeated entries (try a tiny example and see which additions get lost).
np.add.at(out, idx, v) is the unbuffered version: it applies every occurrence.
"""
import numpy as np


def bincount(x, K):
    """x (N,) ints in [0, K) -> (K,) int counts. No np.bincount."""
    raise NotImplementedError("TODO: bincount")


def scatter_add_rows(src, idx, n):
    """src (N, D), idx (N,) ints in [0, n) -> (n, D) where out[j] is the sum of the rows src[i] with idx[i] == j."""
    raise NotImplementedError("TODO: scatter_add_rows")


def segment_mean(values, seg, n):
    """values (N, D), seg (N,) ints in [0, n) -> (n, D): mean of the rows in each segment.

    Empty segments give 0 (and no warnings).
    """
    raise NotImplementedError("TODO: segment_mean")
