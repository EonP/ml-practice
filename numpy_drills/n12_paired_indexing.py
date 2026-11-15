"""NumPy drill 12: paired index arrays, x[rows, cols] (target: 10 min). No loops.

x[rows, cols] pairs the index arrays up element by element (after broadcasting
them against each other): it picks x[rows[0], cols[0]], x[rows[1], cols[1]],
and so on. It does NOT pick a sub-grid; that's the next drill.
"""
import numpy as np


def pick_per_row(x, cols):
    """x (N, D), cols (N,) ints -> (N,) with out[i] = x[i, cols[i]]."""
    return x[np.arange(x.shape[0]), cols]


def set_per_row_(x, cols, value):
    """In place: x[i, cols[i]] = value for every row i. Return None."""
    x[np.arange(x.shape[0]), cols] = value
    return


def gather_pixels(img, pts: np.ndarray):
    """img (H, W, C), pts (K, 2) ints as (row, col) -> (K, C): the pixel at each point."""
    return img[pts[:, 0], pts[:, 1]]


def swap_rows_(x, i, j):
    """Swap rows i and j of x in place. Return None.

    The Python idiom x[i], x[j] = x[j], x[i] silently gets this wrong. Why?
    """
    # Right side: integer-array gather, so a NEW 2-row array holding rows j, i.
    # Left side: scatter into rows i, j of x itself, which is the in-place write.
    # (x[i], x[j] = x[j], x[i] fails because x[j], x[i] are views: by the time
    # x[j] is written, the "row i" it reads from has already been overwritten.)
    x[[i, j]] = x[[j, i]]
    return
