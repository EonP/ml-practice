"""Drill 07: decision-tree split search (target: 25 min). NumPy only."""
import numpy as np


def gini(y):
    """Gini impurity 1 - sum_k p_k^2 of an int label array. Empty array -> 0.0."""
    raise NotImplementedError("TODO: gini")


def entropy(y):
    """Entropy -sum_k p_k log2 p_k (skip p_k = 0). Empty array -> 0.0."""
    raise NotImplementedError("TODO: entropy")


def best_split(X, y, criterion="gini"):
    """Best single split over all features and thresholds.

    X: (N, D) float, y: (N,) int. Candidate thresholds for a feature are the
    midpoints between consecutive distinct sorted values; samples with
    x[f] <= threshold go left.
    gain = impurity(parent) - (n_L/N) impurity(left) - (n_R/N) impurity(right)
    Returns (feature, threshold, gain) for the largest gain, or
    (None, None, 0.0) if no split has positive gain.
    """
    raise NotImplementedError("TODO: best_split")
