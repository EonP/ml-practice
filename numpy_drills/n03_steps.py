"""NumPy drill 03: steps and slice assignment (target: 8 min). No loops, no np.flip / np.rot90.

Images are (H, W, C). A negative step walks backwards and still gives a view.
A slice can also go on the left of `=`, to write into part of an array.
"""
import numpy as np


def downsample(img, f):
    """Keep every f-th row and every f-th column, starting at 0: (ceil(H/f), ceil(W/f), C)."""
    raise NotImplementedError("TODO: downsample")


def flip_lr(img):
    """Mirror left-right, so the last column comes first. Must be a view."""
    raise NotImplementedError("TODO: flip_lr")


def rot180(img):
    """Rotate by 180 degrees. Must be a view."""
    raise NotImplementedError("TODO: rot180")


def interleave(a, b):
    """a[0], b[0], a[1], b[1], ... for two arrays with the same shape (N, ...) and dtype.

    Returns (2N, ...) with that dtype. Allocate the output once, then fill it
    with two strided slice assignments.
    """
    raise NotImplementedError("TODO: interleave")


def checkerboard(h, w):
    """(h, w) int array of 0s and 1s: 0 at the top-left, alternating along rows and columns.

    Start from zeros and use two strided slice assignments (no index arithmetic).
    """
    raise NotImplementedError("TODO: checkerboard")
