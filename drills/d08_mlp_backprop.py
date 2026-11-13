"""Drill 08: two-layer MLP with a hand-written backward pass (target: 40 min).

NumPy only. This is the most important drill: if you can write this cleanly
from a blank page, you understand backprop.

Architecture: X (N, D) -> Linear(D, H) -> ReLU -> Linear(H, K) -> logits (N, K)
Loss: mean softmax cross-entropy.
"""
import numpy as np


def init_params(D, H, K, seed=0):
    """Given: He-initialised weights. W1 (D, H), b1 (H,), W2 (H, K), b2 (K,)."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / D), (D, H)),
        "b1": np.zeros(H),
        "W2": rng.normal(0, np.sqrt(2.0 / H), (H, K)),
        "b2": np.zeros(K),
    }


def forward(params, X):
    """Forward pass. Returns (logits (N, K), cache) where cache holds
    whatever `backward` needs (inputs, pre-activations, activations...)."""
    raise NotImplementedError("TODO: forward")


def backward(params, cache, y):
    """Backward pass for the mean cross-entropy loss.

    Returns (loss float, grads) with grads a dict with the same keys and
    shapes as params. No autograd, no loops.
    """
    raise NotImplementedError("TODO: backward")


def train(X, y, H=64, lr=1.0, n_steps=2000, seed=0):
    """Full-batch gradient descent. Returns (params, losses list)."""
    raise NotImplementedError("TODO: train")
