"""Drill 04: k-nearest neighbours (target: 15 min). NumPy only, no Python loops."""
import numpy as np


def pairwise_sq_dists(A, B):
    """Squared Euclidean distances between every row of A and every row of B.

    A: (N, D), B: (M, D) -> (N, M). No loops.
    Bonus: use the ||a||^2 - 2 a.b + ||b||^2 expansion so you never build an
    (N, M, D) array, and clip tiny negatives caused by rounding to 0.
    """

    """ VERSION 1 - no need for clipping as the (a - b)^2 is never negative
    diff = A[:, None] - B # N, M, 3
    return np.sum(diff ** 2, axis=-1)
    """

    """ VERSION 2
    """

    a_square = np.sum(A ** 2, axis=1) # (N,)
    b_square = np.sum(B ** 2, axis=1) # (M,)
    a_dot_b = A @ B.T # (N, M)
    res = a_square[:, None] - 2 * a_dot_b + b_square
    return np.clip(res, 0, None)


def knn_predict(X_train, y_train, X_test, k=5):
    """Majority vote among the k nearest training points (Euclidean).

    y_train: (N,) non-negative ints. Returns (M,) ints. Ties go to the
    smallest label. You may loop over test points for the vote only.
    Hint: np.argpartition, np.bincount.
    """
    # x_train (N, D), x_test (M, D)
    out = np.empty(X_test.shape[0], dtype=int)
    distances = pairwise_sq_dists(X_test, X_train) # (M, N)
    k_nearest_indices = np.argpartition(distances, kth=k-1, axis=1)[:, :k] # (M, k)
    knn = y_train[k_nearest_indices] # (M, k)
    for i, row in enumerate(knn):
        votes_per_label = np.bincount(row)
        winning_label = np.argmax(votes_per_label)
        out[i] = winning_label
    return out
