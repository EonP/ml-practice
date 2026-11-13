import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from drills.d04_knn import knn_predict, pairwise_sq_dists

rng = np.random.default_rng(0)


def test_pairwise_sq_dists():
    A, B = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    expected = np.array([[((a - b) ** 2).sum() for b in B] for a in A])
    out = pairwise_sq_dists(A, B)
    assert out.shape == (7, 5)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_pairwise_self_distance_not_negative():
    A = rng.normal(size=(50, 10)) * 1e3
    assert (pairwise_sq_dists(A, A) >= 0).all()


def test_knn_matches_sklearn():
    Xtr = rng.normal(size=(200, 4))
    ytr = (Xtr[:, 0] + Xtr[:, 1] ** 2 > 0.5).astype(int)
    Xte = rng.normal(size=(50, 4))
    ref = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr).predict(Xte)
    np.testing.assert_array_equal(knn_predict(Xtr, ytr, Xte, k=5), ref)


def test_knn_tie_goes_to_smallest_label():
    Xtr = np.array([[0.0], [1.0], [-1.0], [5.0]])
    ytr = np.array([2, 1, 1, 2])
    # k=2 nearest to 0.4 are x=0 (label 2) and x=1 (label 1): tie -> 1
    assert knn_predict(Xtr, ytr, np.array([[0.4]]), k=2)[0] == 1
