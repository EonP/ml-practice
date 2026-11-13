"""Drill 04: k-nearest neighbours (target: 15 min). NumPy only, no Python loops."""
import numpy as np


def pairwise_sq_dists(A, B):
    """Squared Euclidean distances between every row of A and every row of B.

    A: (N, D), B: (M, D) -> (N, M). No loops.
    Bonus: use the ||a||^2 - 2 a.b + ||b||^2 expansion so you never build an
    (N, M, D) array, and clip tiny negatives caused by rounding to 0.
    """
    raise NotImplementedError("TODO: pairwise_sq_dists")


def knn_predict(X_train, y_train, X_test, k=5):
    """Majority vote among the k nearest training points (Euclidean).

    y_train: (N,) non-negative ints. Returns (M,) ints. Ties go to the
    smallest label. You may loop over test points for the vote only.
    Hint: np.argpartition, np.bincount.
    """
    raise NotImplementedError("TODO: knn_predict")
