"""NumPy drill 08: masks along one axis (target: 10 min). No loops.

A 1D boolean mask selects along a single axis: x[row_mask] keeps rows and
x[:, col_mask] keeps columns. To get one from a 2D condition, reduce it first
with .any(axis) or .all(axis).
"""
import numpy as np


def drop_nan_rows(x: np.ndarray):
    """Remove every row of the (N, D) array x that contains at least one NaN."""
    row_mask = ~np.isnan(x).any(axis=1) # (N,)
    return x[row_mask] # (N, D) and (N,)


def high_variance_columns(x: np.ndarray, thresh):
    """Keep only the columns of x (N, D) whose variance is > thresh. Returns (N, D')."""
    col_mask = x.var(axis=0) > thresh # (D,)
    return x[:, col_mask] # (N, D) and (D,)


def fill_nan_with_column_mean(x: np.ndarray):
    """A copy of x (N, D) with each NaN replaced by the mean of the non-NaN entries in its column.

    Columns that are entirely NaN become 0. Do not modify x, and don't trigger
    any warnings (np.nanmean warns on an all-NaN column, and so does 0 / 0).
    """
    non_nans = ~np.isnan(x) # (N, D)
    counts = np.maximum(non_nans.sum(axis=0), 1) # (D,) - the maximum is there in case counts are zero
    sums = np.nansum(x, axis=0) # (D,)
    means = sums / counts # (D,)
    return np.where(non_nans, x, means) # means gets broadcasted to (N, D)


def masked_row_mean(x: np.ndarray, mask: np.ndarray):
    """Mean of each row of x (N, D) over the entries where mask (N, D) is True.

    Returns (N,). Rows with no True entries give 0 (and no warnings).
    """
    counts = np.maximum(mask.sum(axis=1), 1)
    sums = np.sum(np.where(mask, x, 0), axis=1)
    return sums / counts

