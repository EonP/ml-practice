"""PyTorch drill 06: Dataset and DataLoader (target: 15 min).

A map-style Dataset needs __len__ and __getitem__(i), which returns ONE sample.
The DataLoader picks indices (shuffled or not), calls __getitem__ for each, and
passes the list of samples to collate_fn to build a batch. The default collate_fn
stacks the samples, so samples of different sizes need a collate_fn of your own.
"""
import torch
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset):
    """Samples (x, y) from tensors X (N, ...) and y (N,).

    With a transform, __getitem__(i) returns (transform(X[i]), y[i]); the label is never transformed.
    """

    def __init__(self, X, y, transform=None):
        raise NotImplementedError("TODO: PairDataset.__init__")

    def __len__(self):
        raise NotImplementedError("TODO: PairDataset.__len__")

    def __getitem__(self, i):
        raise NotImplementedError("TODO: PairDataset.__getitem__")


def pad_collate(batch, pad_value=0.0):
    """collate_fn for variable-length sequences.

    batch: list of B samples (seq, label), seq float (L_i, D), label a Python int.
    Returns (padded (B, L_max, D) float, lengths (B,) long, labels (B,) long), in the
    order given, with every padding position set to pad_value.
    """
    raise NotImplementedError("TODO: pad_collate")


def make_loaders(X, y, batch_size, val_frac, seed):
    """Random train/validation split, and a DataLoader for each.

    The validation set gets int(len(X) * val_frac) samples, picked using a torch.Generator
    seeded with `seed`, so the same seed always gives the same split. The train loader
    reshuffles every epoch; the validation loader always uses the same order.
    Returns (train_loader, val_loader).
    """
    raise NotImplementedError("TODO: make_loaders")


def channel_mean_std(loader):
    """Per-channel mean and std of a whole image dataset, read one batch at a time.

    loader yields (images (B, C, H, W), labels), and the last batch may be smaller.
    Returns (mean (C,), std (C,)) as float64, where std is the population std (divide by
    the total number of values in the channel). Keep running sums: don't torch.cat all
    the batches, since a real dataset won't fit in memory.
    """
    raise NotImplementedError("TODO: channel_mean_std")
