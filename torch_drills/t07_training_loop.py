"""PyTorch drill 07: the training loop (target: 20 min).

Know these five lines cold, in this order:
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
Around them: model.train() before training; model.eval() AND torch.no_grad() for
evaluation. They do different things: eval() switches dropout and batchnorm to
inference behaviour, no_grad() stops building the graph. Log loss.item(), never the
tensor itself: holding on to `loss` keeps its whole graph alive.
"""
import copy

import torch


def train_one_epoch(model, loader, loss_fn, optimizer):
    """One pass over loader in training mode. loss_fn returns the MEAN loss of a batch.

    Returns the mean loss over all SAMPLES as a Python float. Careful: the last batch
    may be smaller than the others.
    """
    raise NotImplementedError("TODO: train_one_epoch")


def evaluate(model, loader, loss_fn):
    """(mean loss over samples, accuracy) as Python floats, in eval mode without building a graph.

    Accuracy compares model(x).argmax(1) with the labels.
    """
    raise NotImplementedError("TODO: evaluate")


def overfit_one_batch(model, x, y, loss_fn, steps=300, lr=1e-2):
    """Train on the single batch (x, y) with torch.optim.Adam(lr=lr) for `steps` steps.

    Returns the list of losses (Python floats, one per step). A working model can drive the
    loss on one small batch to almost zero; if it can't, something is broken. That makes
    this the first thing to try when a training run misbehaves.
    """
    raise NotImplementedError("TODO: overfit_one_batch")


def fit(model, train_loader, val_loader, loss_fn, optimizer, max_epochs, patience):
    """Train with early stopping.

    Each epoch: train_one_epoch, then the validation loss (computed as in evaluate). Stop when
    the validation loss hasn't improved for `patience` epochs in a row, or after max_epochs.
    Before returning, load the weights from the epoch with the LOWEST validation loss.
    Returns {"train_loss": [...], "val_loss": [...]}, one float per epoch that ran.
    Careful: what exactly does model.state_dict() give you?
    """
    raise NotImplementedError("TODO: fit")
