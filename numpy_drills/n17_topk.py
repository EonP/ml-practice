"""NumPy drill 17: per-row top-k (target: 12 min). No loops.

np.argpartition(x, kth, axis) moves the kth-smallest entry to position kth,
with everything smaller before it, in O(D) rather than the O(D log D) of a
full sort. The first k slots come out in no particular order, so sort only
those k afterwards. For the largest entries, partition -x.
"""
import numpy as np


def topk(x, k):
    """x (N, D), 1 <= k <= D -> (values, indices), both (N, k).

    Each row is sorted by value, LARGEST first (like torch.topk).
    """
    raise NotImplementedError("TODO: topk")


def topk_mask(x, k):
    """(N, D) bool with exactly k True per row, at that row's k largest entries."""
    raise NotImplementedError("TODO: topk_mask")


def topk_accuracy(logits, y, k):
    """Fraction of rows whose true label y[i] is among the k highest logits, as a Python float.

    logits (N, K), y (N,) ints.
    """
    raise NotImplementedError("TODO: topk_accuracy")
