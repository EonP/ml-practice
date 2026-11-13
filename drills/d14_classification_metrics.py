"""Drill 14: classification metrics (target: 25 min). NumPy only."""
import numpy as np


def precision_recall_f1(y_true, y_pred):
    """Binary precision, recall and F1 for labels in {0, 1}.
    Any 0/0 is defined as 0.0. Returns a tuple of three floats."""
    raise NotImplementedError("TODO: precision_recall_f1")


def roc_auc(y_true, scores):
    """ROC-AUC = P(score of a random positive > score of a random negative),
    counting ties as 1/2. Must be O(N log N), so no (pos x neg) matrix.

    Hint: rank all scores (average ranks for ties), then the Mann-Whitney U
    statistic: U = sum(ranks of positives) - n_pos (n_pos + 1) / 2.
    """
    raise NotImplementedError("TODO: roc_auc")
