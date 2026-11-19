"""Drill 03: binary logistic regression (target: 25 min). NumPy only."""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid. Must not emit overflow warnings for |z| = 1000.

    Hint: for z < 0 use exp(z) / (1 + exp(z)). Avoid evaluating np.exp on
    large positive numbers at all (np.where evaluates BOTH branches).
    """
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def bce_loss(w, b, X, y):
    """Mean binary cross-entropy, computed from the logits z = X w + b.

    X: (N, D), y: (N,) in {0, 1}. Returns a float.
    Hint: BCE = log(1 + e^z) ¨- y z; np.logaddexp(0, z) computes log(1 + e^z) stably.
    """

    # P(y_i|x_i) = p_i^y_i * (1 - p_i)^(1 - y_i)
    # minimize negative log likelihood -1/N sum(log(p(y_i | x_i)))
    
    z = X @ w + b
    return float(np.mean(np.logaddexp(0, z) - y * z))


def bce_grad(w, b, X, y):
    """Gradient of `bce_loss`. Returns (dw (D,), db float)."""
    N, _ = X.shape
    delta = sigmoid(X @ w + b) - y
    grad_w = X.T @ delta / N
    grad_b = np.sum(delta) / N
    return grad_w, float(grad_b)

def fit(X, y, lr=0.5, n_steps=2000):
    """Gradient descent from zeros. Returns (w, b)."""
    N, D = X.shape
    w, b = np.zeros(D), 0.0
    for _ in range(n_steps):
        grad_w, grad_b = bce_grad(w, b, X, y)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, b


def predict_proba(w, b, X):
    """P(y = 1 | x) for each row of X, shape (N,)."""
    return sigmoid(X @ w + b)

