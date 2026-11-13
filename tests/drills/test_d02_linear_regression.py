import numpy as np
from sklearn.linear_model import Ridge

from drills.d02_linear_regression import fit_closed_form, fit_gd

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))
w_true = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
y = X @ w_true + 4.0 + 0.1 * rng.normal(size=200)


def test_ols_recovers_weights():
    w, b = fit_closed_form(X, y)
    assert w.shape == (5,)
    np.testing.assert_allclose(w, w_true, atol=0.05)
    np.testing.assert_allclose(b, 4.0, atol=0.05)


def test_ridge_matches_sklearn():
    w, b = fit_closed_form(X, y, l2=10.0)
    ref = Ridge(alpha=10.0).fit(X, y)
    np.testing.assert_allclose(w, ref.coef_, atol=1e-8)
    np.testing.assert_allclose(b, ref.intercept_, atol=1e-8)


def test_gd_converges_to_closed_form():
    w, b, losses = fit_gd(X, y, lr=0.1, n_steps=1000)
    w_cf, b_cf = fit_closed_form(X, y)
    assert len(losses) == 1000
    assert losses[-1] < losses[0]
    assert all(l2 <= l1 + 1e-12 for l1, l2 in zip(losses, losses[1:])), "loss should never go up"
    np.testing.assert_allclose(w, w_cf, atol=1e-4)
    np.testing.assert_allclose(b, b_cf, atol=1e-4)
