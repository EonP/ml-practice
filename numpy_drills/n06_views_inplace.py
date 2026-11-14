"""NumPy drill 06: views, copies and in-place updates (target: 8 min). No loops.

A function that modifies x "in place" must change the caller's array and
return None. Inside a function, `x = x + 1` only rebinds the local name, while
`x += 1` and `x[...] = ...` write into the caller's memory. A trailing
underscore in the name (PyTorch style) marks an in-place function.
"""
import numpy as np


def crop_view(x, top, left, h, w):
    """The h x w window of the 2D array x whose top-left corner is (top, left), as a VIEW.

    Writing into the result must change x.
    """
    return x[top:top + h, left:left + w]


def crop_copy(x: np.ndarray, top, left, h, w):
    """The same window as an independent COPY: writing into the result must leave x alone."""
    return x[top:top + h, left:left + w].copy()


def zero_border_(x, b):
    """Set a border of width b >= 1 on all four sides of the 2D array x to 0, in place. Return None."""
    x[:b] = x[-b:] = 0          # top and bottom
    x[:, :b] = x[:, -b:] = 0    # left and right
    return


def normalize_rows_(x):
    """Divide each row of the float (N, D) array x by the row's sum, in place. Return None."""
    x /= x.sum(axis=1, keepdims=True)
    return
