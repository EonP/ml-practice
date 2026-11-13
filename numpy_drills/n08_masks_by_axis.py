"""NumPy drill 08: masks along one axis (target: 10 min). No loops.

A 1D boolean mask selects along a single axis: x[row_mask] keeps rows and
x[:, col_mask] keeps columns. To get one from a 2D condition, reduce it first
with .any(axis) or .all(axis).
"""
import numpy as np


def drop_nan_rows(x):
    """Remove every row of the (N, D) array x that contains at least one NaN."""
    raise NotImplementedError("TODO: drop_nan_rows")


def high_variance_columns(x, thresh):
    """Keep only the columns of x (N, D) whose variance is > thresh. Returns (N, D')."""
    raise NotImplementedError("TODO: high_variance_columns")


def fill_nan_with_column_mean(x):
    """A copy of x (N, D) with each NaN replaced by the mean of the non-NaN entries in its column.

    Columns that are entirely NaN become 0. Do not modify x, and don't trigger
    any warnings (np.nanmean warns on an all-NaN column, and so does 0 / 0).
    """
    raise NotImplementedError("TODO: fill_nan_with_column_mean")


def masked_row_mean(x, mask):
    """Mean of each row of x (N, D) over the entries where mask (N, D) is True.

    Returns (N,). Rows with no True entries give 0 (and no warnings).
    """
    raise NotImplementedError("TODO: masked_row_mean")
