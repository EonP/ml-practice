"""NumPy drill 11: indexing with integer arrays (target: 8 min). No loops.

x[idx], where idx is an integer array, is a GATHER: the result has idx's shape
(followed by x's remaining axes) and is always a copy. On the left of `=`,
x[idx] = v is a SCATTER.
"""
import numpy as np


def shuffle_together(x, y, rng):
    """Shuffle x (N, ...) and y (N,) with the same random permutation, so row i of x still goes with y[i].

    rng is an np.random.Generator. Returns (x_shuffled, y_shuffled). Do not modify the inputs.
    """
    raise NotImplementedError("TODO: shuffle_together")


def one_hot(y, K):
    """y: int array of any shape with values in [0, K) -> float array of shape y.shape + (K,)."""
    raise NotImplementedError("TODO: one_hot")


def colorize(labels, palette):
    """labels (H, W) ints in [0, K), palette (K, 3) uint8 -> (H, W, 3) uint8 RGB image."""
    raise NotImplementedError("TODO: colorize")


def invert_permutation(p):
    """p is a permutation of 0..N-1. Return inv with inv[p[i]] == i, so p[inv] == arange(N).

    One scatter, no sorting.
    """
    raise NotImplementedError("TODO: invert_permutation")
