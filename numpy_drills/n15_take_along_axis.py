"""NumPy drill 15: np.take_along_axis (target: 10 min). No loops.

np.take_along_axis(x, idx, axis) gathers along a single axis. idx must have
the same NUMBER of dimensions as x (size 1 on the other axes is fine; it
broadcasts). It pairs naturally with argsort, argmax(..., keepdims=True) and
argpartition.
"""
import numpy as np


def sort_rows_by(keys, values):
    """keys, values (N, D) -> values with each row reordered so the matching row of keys is ascending."""
    raise NotImplementedError("TODO: sort_rows_by")


def value_at_argmax(a, b, axis):
    """The value of b wherever a is largest along `axis`.

    a and b have the same shape. The result has that shape with `axis` removed.
    """
    raise NotImplementedError("TODO: value_at_argmax")


def target_logits(logits, target):
    """Segmentation-style gather: logits (N, K, H, W), target (N, H, W) ints in [0, K).

    Returns (N, H, W) with out[n, h, w] = logits[n, target[n, h, w], h, w].
    """
    raise NotImplementedError("TODO: target_logits")
