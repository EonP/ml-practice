import numpy as np
from sklearn.tree import DecisionTreeClassifier

from drills.d07_tree_split import best_split, entropy, gini


def test_impurities():
    assert gini(np.array([0, 0, 0])) == 0.0
    np.testing.assert_allclose(gini(np.array([0, 1, 0, 1])), 0.5)
    np.testing.assert_allclose(entropy(np.array([0, 1, 0, 1])), 1.0)
    np.testing.assert_allclose(entropy(np.array([0, 0, 1, 2])), 1.5)
    assert gini(np.array([], dtype=int)) == 0.0


def test_hand_example():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    y = np.array([0, 0, 1, 1])
    f, t, g = best_split(X, y)
    assert f == 0 and t == 2.5
    np.testing.assert_allclose(g, 0.5)


def test_no_useful_split():
    X = np.array([[1.0], [1.0], [1.0]])
    assert best_split(X, np.array([0, 1, 0])) == (None, None, 0.0)


def test_matches_sklearn_stump():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = (X[:, 1] > 0.3).astype(int)
    y[:5] = 1 - y[:5]
    for crit in ("gini", "entropy"):
        stump = DecisionTreeClassifier(max_depth=1, criterion=crit).fit(X, y)
        f, t, _ = best_split(X, y, criterion=crit)
        assert f == stump.tree_.feature[0]
        np.testing.assert_allclose(t, stump.tree_.threshold[0], atol=1e-6)
