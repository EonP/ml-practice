"""NumPy drill 12: paired index arrays, x[rows, cols] (target: 10 min). No loops.

x[rows, cols] pairs the index arrays up element by element (after broadcasting
them against each other): it picks x[rows[0], cols[0]], x[rows[1], cols[1]],
and so on. It does NOT pick a sub-grid; that's the next drill.
"""
import numpy as np


def pick_per_row(x, cols):
    """x (N, D), cols (N,) ints -> (N,) with out[i] = x[i, cols[i]]."""
    raise NotImplementedError("TODO: pick_per_row")


def set_per_row_(x, cols, value):
    """In place: x[i, cols[i]] = value for every row i. Return None."""
    raise NotImplementedError("TODO: set_per_row_")


def gather_pixels(img, pts):
    """img (H, W, C), pts (K, 2) ints as (row, col) -> (K, C): the pixel at each point."""
    raise NotImplementedError("TODO: gather_pixels")


def swap_rows_(x, i, j):
    """Swap rows i and j of x in place. Return None.

    The Python idiom x[i], x[j] = x[j], x[i] silently gets this wrong. Why?
    """
    raise NotImplementedError("TODO: swap_rows_")
