"""Drill 09: dropout and batch normalisation, forward and backward (target: 35 min).

NumPy only. Inputs are (N, D) (fully connected case).
"""
import numpy as np


def dropout(x, p, train, rng):
    """Inverted dropout with drop probability p.

    Train mode: zero each element with probability p and scale the survivors
    by 1 / (1 - p). Eval mode: identity. Returns (out, mask) where mask is the
    boolean keep-mask in train mode and None in eval mode.
    """
    raise NotImplementedError("TODO: dropout")


def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """Training-mode batch norm over the batch axis.

    x: (N, D), gamma/beta: (D,). Use the biased batch variance.
    Returns (out, cache).
    """
    raise NotImplementedError("TODO: batchnorm_forward")


def batchnorm_backward(dout, cache):
    """Backward pass of `batchnorm_forward`. Returns (dx, dgamma, dbeta).

    Hint: backprop through the graph step by step first (x -> mu -> var ->
    x_hat -> out). Then, as a bonus, simplify it to a one-line closed form for dx.
    """
    raise NotImplementedError("TODO: batchnorm_backward")


def update_running_stats(running_mean, running_var, x, momentum=0.1):
    """Return the new (running_mean, running_var) PyTorch-style:
    new = (1 - momentum) * old + momentum * batch_stat, where the running
    variance uses the UNBIASED batch variance (ddof=1)."""
    raise NotImplementedError("TODO: update_running_stats")


def batchnorm_eval(x, gamma, beta, running_mean, running_var, eps=1e-5):
    """Eval-mode batch norm: normalise with the running statistics."""
    raise NotImplementedError("TODO: batchnorm_eval")
