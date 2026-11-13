import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from drills.d05_kmeans import kmeans, kmeans_pp_init

X, y = make_blobs(n_samples=300, centers=[[-5, -5], [5, 5], [-5, 5], [5, -5]], cluster_std=1.0, random_state=0)


def test_init_picks_k_distinct_data_points():
    C = kmeans_pp_init(X, 4, np.random.default_rng(0))
    assert C.shape == (4, 2)
    assert len({tuple(c) for c in C}) == 4
    for c in C:
        assert (np.abs(X - c).sum(1) < 1e-12).any(), "centres must be actual data points"


def test_recovers_blobs():
    for seed in range(5):
        centers, labels, _ = kmeans(X, 4, seed=seed)
        assert centers.shape == (4, 2) and labels.shape == (300,)
        assert adjusted_rand_score(y, labels) > 0.99, f"seed {seed}"


def test_inertia_never_increases():
    _, _, inertias = kmeans(X, 4, seed=1)
    assert len(inertias) >= 2
    assert all(b <= a + 1e-9 for a, b in zip(inertias, inertias[1:]))


def test_inertia_close_to_sklearn():
    _, _, inertias = kmeans(X, 4, seed=0)
    ref = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X).inertia_
    assert inertias[-1] <= ref * 1.01
