"""NumPy drill 16: masks from broadcasting, and ragged data (target: 12 min). No loops.

Comparing an arange with a column vector builds a whole mask in one go.
Boolean-mask assignment, out[mask] = values, fills the True slots in
row-major order, which is exactly what padding ragged sequences needs.
"""
import numpy as np


def sequence_mask(lengths, T):
    """lengths (N,) ints -> (N, T) bool, True at [i, t] when t < lengths[i]."""
    raise NotImplementedError("TODO: sequence_mask")


def pad_ragged(flat, lengths, fill=0):
    """Pad N variable-length sequences into one (N, max(lengths)) array.

    flat holds the sequences back to back (len(flat) == lengths.sum()). Row i
    holds sequence i followed by `fill`. The result has flat's dtype.
    """
    raise NotImplementedError("TODO: pad_ragged")


def unpad(padded, lengths):
    """Inverse of pad_ragged: padded (N, T), lengths (N,) -> the unpadded rows concatenated into 1D."""
    raise NotImplementedError("TODO: unpad")


def radial_distance(H, W, cy, cx):
    """(H, W) float: the Euclidean distance of each pixel (row, col) from the point (cy, cx). No np.meshgrid."""
    raise NotImplementedError("TODO: radial_distance")
