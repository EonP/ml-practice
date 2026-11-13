"""NumPy drill 21: resampling with index grids (target: 15 min). No loops.

Sampling an image at computed coordinates is fancy indexing: img[r, c] for a
list of points, img[r[:, None], c[None, :]] for a whole grid.
"""
import numpy as np


def nearest_resize(img, H2, W2):
    """Nearest-neighbour resize: (H, W, C) -> (H2, W2, C).

    Output row i reads input row (i * H) // H2 (integer arithmetic), and the
    same goes for columns.
    """
    raise NotImplementedError("TODO: nearest_resize")


def bilinear_sample(img, ys, xs):
    """img (H, W) float with H, W >= 2; ys, xs (K,) float -> (K,) bilinearly interpolated values.

    Pixel centres sit at integer coordinates, so sampling at (2.0, 3.0) gives
    img[2, 3]. Coordinates satisfy 0 <= ys <= H - 1 and 0 <= xs <= W - 1, and
    points may lie exactly on the last row or column.
    """
    raise NotImplementedError("TODO: bilinear_sample")
