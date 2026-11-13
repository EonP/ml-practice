"""NumPy drill 09: from masks to positions (target: 10 min). No loops.

np.nonzero(mask) returns one index array per axis, and np.argwhere(mask)
stacks them into a (K, ndim) array. argmax on a boolean array returns the
position of the FIRST True, but it also returns 0 when there is no True at all.
"""
import numpy as np


def coords_of(x, value):
    """(K, 2) int array of the (row, col) positions where the 2D array x equals value, in row-major order."""
    raise NotImplementedError("TODO: coords_of")


def first_true(mask):
    """mask (N, W) bool -> (N,) int: column of the first True in each row, or -1 if the row has none."""
    raise NotImplementedError("TODO: first_true")


def last_true(mask):
    """Like first_true, but the column of the LAST True in each row (-1 if none)."""
    raise NotImplementedError("TODO: last_true")


def bbox(mask):
    """Tight bounding box of the True pixels in a 2D mask: a tuple of Python ints (r0, c0, r1, c1).

    r1 and c1 are INCLUSIVE. Return None if the mask has no True pixel.
    """
    raise NotImplementedError("TODO: bbox")
