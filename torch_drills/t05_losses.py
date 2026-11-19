"""PyTorch drill 05: loss functions (target: 15 min).

F.cross_entropy takes raw LOGITS (never apply softmax first). Shapes: logits (N, K)
with targets (N,), or logits (N, K, d1, d2, ...) with targets (N, d1, d2, ...): the
class axis is always axis 1, and targets are long class indices. Writing the losses
by hand is the classic way to show you understand the numerics.
"""
import torch
import torch.nn.functional as F


def cross_entropy(logits, y, ignore_index=-100):
    """Mean cross-entropy as a 0-dim tensor, matching F.cross_entropy(logits, y, ignore_index=...).

    logits (N, K), y (N,) long. Rows whose label is ignore_index don't count, and the mean
    is over the rows that do. Must be stable for logits around 1000. Use log_softmax or
    logsumexp, not F.cross_entropy or F.nll_loss.
    """
    raise NotImplementedError("TODO: cross_entropy")


def seg_cross_entropy(logits, target):
    """Pixel-wise mean cross-entropy when the model outputs CHANNELS-LAST logits.

    logits (N, H, W, K), target (N, H, W) long. You MAY use F.cross_entropy here:
    the work is putting the class axis where it expects it.
    """
    raise NotImplementedError("TODO: seg_cross_entropy")


def bce_with_logits(z, y, pos_weight=None):
    """Mean binary cross-entropy from logits, matching F.binary_cross_entropy_with_logits.

    z, y: same shape, y float in [0, 1]. pos_weight: None, or a tensor broadcastable to z
    that multiplies the positive term:
        loss = -[pos_weight * y * log s(z) + (1 - y) * log(1 - s(z))],   s = sigmoid.
    Must stay finite, with finite gradients, for |z| = 1000: never compute log(sigmoid(z)).
    Hint: log s(z) = -softplus(-z) and log(1 - s(z)) = -softplus(z).
    """
    raise NotImplementedError("TODO: bce_with_logits")


def label_smoothing_ce(logits, y, eps):
    """Cross-entropy against a smoothed target, matching F.cross_entropy(logits, y, label_smoothing=eps).

    Each row's target puts 1 - eps on the true class, plus eps / K on EVERY class
    (the true class included). logits (N, K), y (N,) long. Don't call F.cross_entropy.
    """
    raise NotImplementedError("TODO: label_smoothing_ce")
