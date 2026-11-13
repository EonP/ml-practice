import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from drills.d03_logistic_regression import bce_grad, bce_loss, fit, predict_proba, sigmoid

X, y = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0, class_sep=1.5, random_state=1)
rng = np.random.default_rng(0)


def test_sigmoid_values():
    np.testing.assert_allclose(sigmoid(np.array([0.0, 2.0, -2.0])), [0.5, 1 / (1 + np.exp(-2)), 1 / (1 + np.exp(2))])


def test_sigmoid_no_overflow_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sigmoid(np.array([-1000.0, 1000.0]))
    np.testing.assert_allclose(out, [0.0, 1.0])


def test_bce_loss_at_zero_is_log2():
    np.testing.assert_allclose(bce_loss(np.zeros(4), 0.0, X, y), np.log(2))


def test_bce_loss_stable_for_huge_logits():
    w = np.array([1000.0, 0, 0, 0])
    assert np.isfinite(bce_loss(w, 0.0, X, y))


def test_gradient_matches_finite_differences():
    w, b = rng.normal(size=4), 0.3
    dw, db = bce_grad(w, b, X, y)
    h = 1e-6
    num_dw = np.array([(bce_loss(w + h * e, b, X, y) - bce_loss(w - h * e, b, X, y)) / (2 * h) for e in np.eye(4)])
    num_db = (bce_loss(w, b + h, X, y) - bce_loss(w, b - h, X, y)) / (2 * h)
    np.testing.assert_allclose(dw, num_dw, atol=1e-6)
    np.testing.assert_allclose(db, num_db, atol=1e-6)


def test_fit_reaches_good_accuracy():
    w, b = fit(X, y)
    acc = ((predict_proba(w, b, X) > 0.5) == y).mean()
    ref = LogisticRegression().fit(X, y).score(X, y)
    assert acc > 0.9 and acc >= ref - 0.02
