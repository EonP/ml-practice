"""NumPy drill 04: Ellipsis (target: 6 min). No loops.

Each function must work for ANY number of leading axes: (H, W, C),
(N, H, W, C), (T, N, H, W, C), ... Inside an index, `...` means "as many `:`
as needed". Every result should be a view.
"""
import numpy as np


def last_channel(x):
    """(..., C) -> (...): the last entry along the last axis."""
    return x[..., -1]


def bgr_to_rgb(x: np.ndarray):
    """(..., 3) -> (..., 3) with the order of the last axis reversed."""
    return x[..., ::-1]


def split_halves(x):
    """(..., 2D) -> a tuple of two (..., D) arrays: the first and second half of the last axis."""
    D = x.shape[-1] // 2
    return (x[..., :D], x[..., D:])


def center_crop(x, h, w):
    """The central h x w window of the LAST TWO axes: (..., H, W) -> (..., h, w).

    The top offset is (H - h) // 2 and the left offset is (W - w) // 2, so when
    the leftover is odd the extra row/column is cut from the bottom/right.
    """
    H, W = x.shape[-2:]
    top, left = (H - h) // 2, (W - w) // 2
    return x[..., top:top + h, left:left + w]
