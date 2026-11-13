"""Stage 3 (Day 3): PyTorch tensors, dtypes and the training loop."""
import numpy as np
import torch
from torch import nn


def image_to_tensor(img):
    """uint8 (H, W, 3) NumPy image -> float32 tensor (3, H, W) in [0, 1]."""
    raise NotImplementedError("TODO: image_to_tensor")


def mask_to_tensor(mask):
    """(H, W) integer NumPy mask -> int64 (long) tensor (H, W).
    CrossEntropyLoss needs long targets: this is a classic error message."""
    raise NotImplementedError("TODO: mask_to_tensor")


def count_params(model):
    """Number of trainable parameters."""
    raise NotImplementedError("TODO: count_params")


def train_softmax_regression(X, y, n_classes, epochs=300, lr=0.5):
    """Train nn.Linear(D, n_classes) with nn.CrossEntropyLoss and torch.optim.SGD,
    full batch, for `epochs` steps. X: float tensor (N, D), y: long tensor (N,).
    Returns (model, losses) with losses a list of Python floats (use .item()).
    Write the loop from memory: zero_grad, forward, loss, backward, step."""
    raise NotImplementedError("TODO: train_softmax_regression")
