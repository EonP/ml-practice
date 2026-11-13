"""NumPy drill 24: overlapping patches / im2col (target: 20 min). No loops.

Overlapping patches can't come out of a reshape. Use sliding_window_view (and
slice it with a step for the stride), or build broadcast index arrays.
"""
import numpy as np


def unfold(x, k, stride=1, padding=0):
    """im2col, matching torch.nn.functional.unfold: x (N, C, H, W) -> (N, C * k * k, L).

    L = H_out * W_out with H_out = (H + 2 * padding - k) // stride + 1 (same for
    W). Columns go in row-major order over the output positions, and each column
    is one patch flattened in (C, k, k) order. Padding is zeros.
    """
    raise NotImplementedError("TODO: unfold")


def crop_patches(img, centers, size):
    """img (H, W), centers (K, 2) ints as (row, col), odd size -> (K, size, size).

    Patch k is centred on centers[k]; any part that falls outside the image is 0.
    Hint: pad once, then a single fancy-indexing expression.
    """
    raise NotImplementedError("TODO: crop_patches")
