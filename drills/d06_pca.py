"""Drill 06: PCA via SVD (target: 15 min). NumPy only."""
import numpy as np


def pca(X, n_components):
    """Principal component analysis.

    X: (N, D). Returns
      components: (n_components, D), orthonormal rows, largest variance first
      X_proj:     (N, n_components), the centred data projected onto them
      explained_variance_ratio: (n_components,)
    Hint: SVD of the centred data, np.linalg.svd(Xc, full_matrices=False).
    How do the singular values relate to the variances?
    """
    raise NotImplementedError("TODO: pca")
