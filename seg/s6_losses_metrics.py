"""Stage 6a (Day 6): losses and metrics in PyTorch.

logits: (B, K, H, W) float. target: (B, H, W) long. pred: (B, H, W) long.
"""
import torch
import torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1e-6):
    """Multi-class soft Dice loss: 1 - mean over classes of
    (2 * sum(p * g) + eps) / (sum(p) + sum(g) + eps), where p = softmax probs,
    g = one-hot target, and sums run over batch and pixels.
    Must be differentiable (no argmax). Hint: F.one_hot gives (B, H, W, K): permute it."""
    raise NotImplementedError("TODO: soft_dice_loss")


def ce_dice_loss(logits, target, dice_weight=1.0):
    """F.cross_entropy(logits, target) + dice_weight * soft_dice_loss(logits, target)."""
    raise NotImplementedError("TODO: ce_dice_loss")


@torch.no_grad()
def mean_iou(pred, target, K):
    """Mean IoU over classes present in pred or target (skip classes with an
    empty union). Returns a Python float. Vectorised: no loop over pixels."""
    raise NotImplementedError("TODO: mean_iou")
