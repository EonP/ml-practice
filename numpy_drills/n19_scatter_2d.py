"""NumPy drill 19: scattering in 2D (target: 12 min). No loops.

np.add.at and its siblings (np.maximum.at, np.minimum.at, ...) accept the same
indices as x[...], so (rows, cols) pairs work, and so does a multi-dimensional
index array that selects whole rows.
"""
import numpy as np


def splat(points, weights, H, W):
    """points (K, 2) ints as (row, col), weights (K,) -> (H, W) float image.

    Each point adds its weight to its pixel, and repeated points accumulate.
    """
    raise NotImplementedError("TODO: splat")


def embedding_grad(grad_out, ids, V):
    """Backward pass of an embedding lookup out = table[ids].

    grad_out (N, T, D) is dLoss/dout, and ids (N, T) are ints in [0, V).
    Return dLoss/dtable, shape (V, D).
    """
    raise NotImplementedError("TODO: embedding_grad")


def scatter_max(src, idx, n):
    """src (N,) floats, idx (N,) ints in [0, n) -> (n,) where out[j] is the max of src[i] over i with idx[i] == j.

    Slots that nothing lands in are -inf.
    """
    raise NotImplementedError("TODO: scatter_max")
