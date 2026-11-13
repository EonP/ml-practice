"""Drill 11: segmentation metrics (target: 20 min). NumPy only, no loops over pixels.

pred and target are integer label maps of the same shape, e.g. (H, W) or (B, H, W).
"""
import numpy as np


def confusion_matrix(pred, target, K, ignore_index=None):
    """(K, K) matrix, rows = true class, columns = predicted class.

    Pixels whose target equals ignore_index are skipped.
    Hint: one np.bincount over K * target + pred.
    """
    raise NotImplementedError("TODO: confusion_matrix")


def iou_per_class(pred, target, K, ignore_index=None):
    """IoU = TP / (TP + FP + FN) per class, from the confusion matrix.
    Classes absent from both pred and target get NaN. Returns (K,) floats."""
    raise NotImplementedError("TODO: iou_per_class")


def mean_iou(pred, target, K, ignore_index=None):
    """Mean over classes, ignoring NaN entries."""
    raise NotImplementedError("TODO: mean_iou")


def dice_per_class(pred, target, K):
    """Dice = 2 TP / (2 TP + FP + FN) per class; NaN where the class is absent
    from both. Returns (K,) floats."""
    raise NotImplementedError("TODO: dice_per_class")
