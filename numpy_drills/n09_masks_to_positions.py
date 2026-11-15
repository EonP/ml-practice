"""NumPy drill 09: from masks to positions (target: 10 min). No loops.

np.nonzero(mask) returns one index array per axis, and np.argwhere(mask)
stacks them into a (K, ndim) array. argmax on a boolean array returns the
position of the FIRST True, but it also returns 0 when there is no True at all.
"""
import numpy as np


def coords_of(x, value):
    """(K, 2) int array of the (row, col) positions where the 2D array x equals value, in row-major order."""
    return np.argwhere(x == value)


def first_true(mask):
    """mask (N, W) bool -> (N,) int: column of the first True in each row, or -1 if the row has none."""
    # argmax and add -1 to the colums that are all false, either add 0
    return np.argmax(mask, axis=1) + np.max(np.where(mask, 0, -1), axis=1) 


def last_true(mask):
    """Like first_true, but the column of the LAST True in each row (-1 if none)."""
    _, W = mask.shape
    # np.arange(W) shape (W,) and mask is (N, W) -> broadcasted
    return np.where(mask, np.arange(W), -1).max(axis=1)


def bbox(mask):
    """Tight bounding box of the True pixels in a 2D mask: a tuple of Python ints (r0, c0, r1, c1).

    r1 and c1 are INCLUSIVE. Return None if the mask has no True pixel.
    """
    if not mask.any():
        return None
    first_true_col = first_true(mask) # (H,) first True column per row, -1 if none
    last_true_col = last_true(mask) # (H,) last True column per row, -1 if none
    row_has_true = first_true_col != -1 # (H,)
    H = row_has_true.shape[0]
    r0 = np.argmax(row_has_true) # first True in the 1D row mask
    r1 = H - 1 - np.argmax(row_has_true[::-1]) # last True: reverse, then convert back
    c0 = first_true_col[row_has_true].min() # drop the -1 rows, or they'd win the min
    c1 = last_true_col.max() # -1 rows can never win the max
    return int(r0), int(c0), int(r1), int(c1)
