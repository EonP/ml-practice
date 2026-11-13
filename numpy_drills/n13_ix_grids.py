"""NumPy drill 13: sub-grids with np.ix_ (target: 10 min). No loops.

To take the rows `rows` AND the columns `cols` (every combination), the two
index arrays must broadcast to a grid: x[np.ix_(rows, cols)], which is the
same as x[rows[:, None], cols[None, :]].
"""
import numpy as np


def submatrix(x, rows, cols):
    """x (N, D), rows (R,) and cols (S,) ints -> (R, S) with out[a, b] = x[rows[a], cols[b]]."""
    raise NotImplementedError("TODO: submatrix")


def set_submatrix_(x, rows, cols, block):
    """In place: write block (R, S) into the rows-by-cols grid of x, so x[rows[a], cols[b]] = block[a, b]. Return None."""
    raise NotImplementedError("TODO: set_submatrix_")


def subgraph(adj, nodes):
    """adj (N, N) adjacency matrix, nodes (K,) ints -> (K, K) adjacency among those nodes, in that order."""
    raise NotImplementedError("TODO: subgraph")


def masked_submatrix(x, row_mask, col_mask):
    """x (N, D), row_mask (N,) bool, col_mask (D,) bool -> the kept rows AND kept columns, (R, S)."""
    raise NotImplementedError("TODO: masked_submatrix")
