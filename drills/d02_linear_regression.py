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

    # We can solve for w on the centered data as we are sure the line passes through the origin
    # Mathematically the optimal bias in this case is (by taking derivative of loss wrt b) b = y_bar - x_bar^T w
    _, D = X.shape
    X_mean, y_mean = X.mean(axis=0), y.mean() # (D,) for X_mean and float for y_mean
    X_c, y_c = X - X_mean, y - y_mean 
    w_hat = np.linalg.solve(X_c.T @ X_c + l2 * np.eye(D), X_c.T @ y_c)
    b_hat = y_mean - np.dot(w_hat, X_mean)
    return w_hat, float(b_hat)


def fit_gd(X, y, lr=0.1, n_steps=1000):
    """Full-batch gradient descent on the MSE loss mean((X w + b - y)^2).

    Start from w = 0, b = 0. Record the loss *before* each update.
    Returns (w (D,), b float, losses: list of n_steps floats).
    """

    # loss = 1/N Sum((r^i)^2), r = (r_1, ..., r_N)
    # gradient wrt w gives 2 / N X^T r
    # gradient wrt b gives 2 / N Sum(r^i) = 2 * mean(r)
    N, D = X.shape
    w, b = np.zeros(D), 0.0
    losses = []
    for _ in range(n_steps):
        r = X @ w + b - y # (N,)
        losses.append(float(np.mean(r ** 2)))
        grad_w = 2 / N * (X.T @ r)
        grad_b = 2 * r.mean()
        w -= lr * grad_w
        b -= lr * grad_b

    return w, float(b), losses
