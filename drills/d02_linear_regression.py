"""Drill 02: linear / ridge regression (target: 20 min). NumPy only."""
import numpy as np


def fit_closed_form(X, y, l2=0.0):
    """Ridge regression with an (unregularised) bias term, closed form.

    X: (N, D), y: (N,), l2 >= 0. Minimises ||X w + b - y||^2 + l2 * ||w||^2.
    Returns (w of shape (D,), b as float).

    Hints: centre X and y so the bias drops out, then solve
    (Xc^T Xc + l2 I) w = Xc^T yc with np.linalg.solve (never np.linalg.inv).
    Recover b from the means.
    """
    raise NotImplementedError("TODO: fit_closed_form")


def fit_gd(X, y, lr=0.1, n_steps=1000):
    """Full-batch gradient descent on the MSE loss mean((X w + b - y)^2).

    Start from w = 0, b = 0. Record the loss *before* each update.
    Returns (w (D,), b float, losses: list of n_steps floats).
    """
    raise NotImplementedError("TODO: fit_gd")
