"""Drill 01: softmax and cross-entropy (target: 15 min). NumPy only.

The building block of every classifier, and the classic place to show you
know about numerical stability.
"""
import numpy as np


def softmax(z: np.ndarray, axis=-1):
    """Numerically stable softmax along `axis`.

    z: array of any shape. Returns an array of the same shape whose values
    sum to 1 along `axis`. Must not overflow for z around 1000.
    Hint: softmax(z) == softmax(z - c) for any constant c.
    """
    e = np.exp(z - z.max(axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def log_softmax(z, axis=-1):
    """Numerically stable log-softmax along `axis`.

    Do NOT compute np.log(softmax(z)): it gives -inf for very negative logits.
    Hint: log-sum-exp trick.
    """
    # log(S) = z_i - m - log(sum(e^{z_j - m}))
    m = np.max(z, axis=axis, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))


def cross_entropy(logits, y):
    """Mean cross-entropy loss.

    logits: (N, K) float. y: (N,) int labels in [0, K).
    Returns a Python float. No loops.
    """
    # loss = -log(p_i)
    N = logits.shape[0]
    logp = log_softmax(logits, axis=-1) # shape (N, K)
    losses = -logp[np.arange(N), y]
    return float(np.mean(losses))

def cross_entropy_grad(logits, y):
    """Gradient of `cross_entropy` with respect to the logits, shape (N, K).

    Derive it on paper first: it is a famously simple expression.
    Do not forget the 1/N from the mean.
    """
    ## basically, gradient of loss(i) = softmax(z_i) - y_i_one_hot (vector size)
    ## so we stack them for N inputs

    N, K = logits.shape
    grad = (softmax(logits, axis=-1) - np.eye(K)[y]) / N # (N, K)
    return grad
