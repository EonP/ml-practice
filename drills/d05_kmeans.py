"""Drill 05: k-means with k-means++ initialisation (target: 30 min). NumPy only."""
import numpy as np


def kmeans_pp_init(X, k, rng):
    """k-means++: pick the first centre uniformly at random, then each next
    centre with probability proportional to its squared distance to the
    nearest centre chosen so far. Returns (k, D).
    """
    raise NotImplementedError("TODO: kmeans_pp_init")


def kmeans(X, k, n_iter=100, seed=0, tol=1e-8):
    """Lloyd's algorithm.

    X: (N, D). Returns (centers (k, D), labels (N,), inertias) where
    `inertias` is the list of the sum of squared distances to the assigned
    centre, recorded after each assignment step. Stop early when the inertia
    improves by less than `tol`. If a cluster becomes empty, keep its old centre.
    Use np.random.default_rng(seed) and `kmeans_pp_init`.
    """
    raise NotImplementedError("TODO: kmeans")
