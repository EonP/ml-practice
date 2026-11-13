import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from drills.d14_classification_metrics import precision_recall_f1, roc_auc

rng = np.random.default_rng(0)


def test_prf_matches_sklearn():
    y, p = rng.integers(0, 2, 100), rng.integers(0, 2, 100)
    np.testing.assert_allclose(precision_recall_f1(y, p), (precision_score(y, p), recall_score(y, p), f1_score(y, p)))


def test_prf_zero_division():
    assert precision_recall_f1(np.array([1, 1]), np.array([0, 0])) == (0.0, 0.0, 0.0)


def test_auc_matches_sklearn():
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 1, 500)
    np.testing.assert_allclose(roc_auc(y, s), roc_auc_score(y, s))


def test_auc_with_ties():
    y = rng.integers(0, 2, 300)
    s = np.round(y * 0.5 + rng.random(300), 1)   # lots of tied scores
    np.testing.assert_allclose(roc_auc(y, s), roc_auc_score(y, s))


def test_auc_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
