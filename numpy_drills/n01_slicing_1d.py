"""NumPy drill 01: 1D slicing (target: 5 min). Slices only: no loops, no np.flip.

x[start:stop:step], where negative indices count from the end. Each answer is
one line, but two of them have an edge case that bites. Every result should be
a view of x (slicing never copies).
"""
import numpy as np


def last_k(x, k):
    """The last k elements of the 1D array x, for 0 <= k <= len(x).

    last_k(np.arange(5), 2) -> [3, 4]. Make sure k = 0 gives an empty array.
    """
    return x[len(x) - k:]


def drop_ends(x, k):
    """Drop k elements from each end: (N,) -> (N - 2k,), for 0 <= 2k <= N.

    drop_ends(np.arange(6), 1) -> [1, 2, 3, 4]. Make sure k = 0 drops nothing.
    """
    return x[k : len(x) - k]


def every_kth(x, k, start=0):
    """x[start], x[start + k], x[start + 2k], ... for as long as they exist."""
    return x[start::k]

def odd_positions_reversed(x):
    """The elements at odd positions (1, 3, 5, ...), last one first.

    np.arange(7) -> [5, 3, 1] and np.arange(6) -> [5, 3, 1].
    x[::-2] is right for only one of those two lengths.
    """
    return x[1::2][::-1]
