"""Stage 6b-7 (Days 6-7): training, evaluation, sanity checks and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from seg.s1_images import IMAGENET_MEAN, IMAGENET_STD, denormalize, overlay
from seg.s6_losses_metrics import mean_iou


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """One pass over `loader` in train mode. Returns the mean loss (float),
    weighted by batch size."""
    raise NotImplementedError("TODO: train_one_epoch")


@torch.no_grad()
def evaluate(model, loader, loss_fn, n_classes, device):
    """Eval mode, no gradients. Returns {"loss": mean loss, "miou": mean IoU over
    the whole loader}. Compute mIoU on all predictions concatenated, not as an
    average of per-batch mIoUs."""
    raise NotImplementedError("TODO: evaluate")


def overfit_one_batch(model, x, y, loss_fn, steps=150, lr=1e-2):
    """The first sanity check for any new model: train on ONE batch with Adam
    until the loss collapses. Returns the list of losses (floats).
    If this does not go to ~0, the bug is in the model/loss/loop, not the data."""
    raise NotImplementedError("TODO: overfit_one_batch")


def plot_predictions(x, y, pred, n=4, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Figure with n rows and 3 columns: image | ground truth overlay | prediction overlay.
    x: (B, 3, H, W) normalised tensor, y and pred: (B, H, W) tensors.
    Use plt.subplots(n, 3, squeeze=False), denormalize, and the stage 1 `overlay`.
    Set the column titles on the first row. Returns the figure."""
    raise NotImplementedError("TODO: plot_predictions")


def plot_curves(history):
    """history: dict of lists, e.g. {"train_loss": [...], "val_loss": [...], "val_miou": [...]}.
    Two side-by-side axes: losses on the left (with legend), mIoU on the right.
    Label the x axes "epoch". Returns the figure."""
    raise NotImplementedError("TODO: plot_curves")
