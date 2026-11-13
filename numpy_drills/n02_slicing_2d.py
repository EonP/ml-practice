"""NumPy drill 02: 2D slicing (target: 5 min). No loops.

x is (H, W). x[i] is row i and x[:, j] is column j. An integer index removes
its axis; a slice keeps it, even a slice of length 1. Use basic indexing only
(integers, slices, None), so that every result is a view of x.
"""
import numpy as np


def column(x, j, keepdims=False):
    """Column j of x: shape (H,), or (H, 1) when keepdims is True.

    j may be negative. Check j = -1 with keepdims=True.
    """
    raise NotImplementedError("TODO: column")


def interior(x, b=1):
    """Everything except a border of width b >= 1 on all four sides: (H - 2b, W - 2b)."""
    raise NotImplementedError("TODO: interior")


def corners(x):
    """The four corners as a (2, 2) view [[top-left, top-right], [bottom-left, bottom-right]].

    One slicing expression with steps, no index lists. Assume H, W >= 2.
    """
    raise NotImplementedError("TODO: corners")


def tile(x, i, j, size):
    """The size x size block in tile-row i, tile-column j.

    tile(x, 0, 0, s) is the top-left s x s block, tile(x, 0, 1, s) the block to
    its right. Assume the block lies inside x.
    """
    raise NotImplementedError("TODO: tile")
