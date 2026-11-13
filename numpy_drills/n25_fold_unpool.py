"""NumPy drill 25: putting patches back (target: 25 min). No loops.

The adjoint of a gather is a scatter-add, so overlapping patches SUM where
they overlap: fold() is to unfold() what np.add.at is to x[idx]. Max-pooling
with indices is a gather (argmax + take_along_axis), and unpooling is the
matching scatter.
"""
import numpy as np


def fold(cols, H, W, k, stride=1):
    """col2im, matching torch.nn.functional.fold: cols (N, C * k * k, L) -> (N, C, H, W).

    Same layout as drill 24's unfold, without padding. Entries that land on
    the same pixel are summed.
    """
    raise NotImplementedError("TODO: fold")


def maxpool_with_indices(x, k):
    """Non-overlapping k x k max-pooling that also returns where each max came from.

    Matches torch.nn.functional.max_pool2d(x, k, return_indices=True).
    x (N, C, H, W) -> (values, indices), both (N, C, H // k, W // k). indices
    are int64 positions in the flattened H * W plane of each (n, c). Rows and
    columns that don't fill a whole window are dropped.
    """
    raise NotImplementedError("TODO: maxpool_with_indices")


def max_unpool(values, indices, H, W):
    """The matching scatter, like torch.nn.functional.max_unpool2d.

    values and indices (N, C, h, w) -> (N, C, H, W): zeros everywhere except
    values[n, c, i, j] written at flat position indices[n, c, i, j] of plane (n, c).
    """
    raise NotImplementedError("TODO: max_unpool")
