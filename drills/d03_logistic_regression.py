"""Drill 03: binary logistic regression (target: 25 min). NumPy only."""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid. Must not emit overflow warnings for |z| = 1000.

    Hint: for z < 0 use exp(z) / (1 + exp(z)). Avoid evaluating np.exp on
    large positive numbers at all (np.where evaluates BOTH branches).
    """
    raise NotImplementedError("TODO: sigmoid")


def bce_loss(w, b, X, y):
    """Mean binary cross-entropy, computed from the logits z = X w + b.

    X: (N, D), y: (N,) in {0, 1}. Returns a float.
    Hint: BCE = log(1 + e^z) - y z; np.logaddexp(0, z) computes log(1 + e^z) stably.
    """
    raise NotImplementedError("TODO: bce_loss")


def bce_grad(w, b, X, y):
    """Gradient of `bce_loss`. Returns (dw (D,), db float)."""
    raise NotImplementedError("TODO: bce_grad")


def fit(X, y, lr=0.5, n_steps=2000):
    """Gradient descent from zeros. Returns (w, b)."""
    raise NotImplementedError("TODO: fit")


def predict_proba(w, b, X):
    """P(y = 1 | x) for each row of X, shape (N,)."""
    raise NotImplementedError("TODO: predict_proba")
