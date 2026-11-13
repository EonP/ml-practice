import numpy as np
from sklearn.decomposition import PCA

from drills.d06_pca import pca

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5)) @ rng.normal(size=(5, 5)) + 3.0


def test_matches_sklearn_up_to_sign():
    comps, proj, ratio = pca(X, 3)
    ref = PCA(n_components=3).fit(X)
    assert comps.shape == (3, 5) and proj.shape == (100, 3) and ratio.shape == (3,)
    signs = np.sign((comps * ref.components_).sum(1))
    np.testing.assert_allclose(comps * signs[:, None], ref.components_, atol=1e-8)
    np.testing.assert_allclose(proj * signs, ref.transform(X), atol=1e-8)
    np.testing.assert_allclose(ratio, ref.explained_variance_ratio_, atol=1e-10)


def test_components_orthonormal():
    comps, _, _ = pca(X, 4)
    np.testing.assert_allclose(comps @ comps.T, np.eye(4), atol=1e-10)
