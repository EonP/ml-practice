import warnings

import numpy as np

from numpy_drills.n08_masks_by_axis import drop_nan_rows, fill_nan_with_column_mean, high_variance_columns, masked_row_mean

rng = np.random.default_rng(0)


def test_drop_nan_rows():
    x = np.arange(15.0).reshape(5, 3)
    np.testing.assert_array_equal(drop_nan_rows(x), x)
    x[1, 2], x[3, 0] = np.nan, np.nan
    np.testing.assert_array_equal(drop_nan_rows(x), x[[0, 2, 4]])
    assert drop_nan_rows(np.full((4, 3), np.nan)).shape == (0, 3)


def test_high_variance_columns():
    x = rng.normal(size=(50, 6)) * np.array([0.1, 2, 0.1, 3, 0.5, 2])
    v = x.var(axis=0)
    keep = [j for j in range(6) if v[j] > 1.0]
    np.testing.assert_array_equal(high_variance_columns(x, 1.0), x[:, keep])
    assert high_variance_columns(x, 1e9).shape == (50, 0)


def test_fill_nan_with_column_mean():
    x = np.array([[1.0, np.nan, np.nan], [3.0, 4.0, np.nan], [np.nan, 8.0, np.nan]])
    before = x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = fill_nan_with_column_mean(x)
    np.testing.assert_array_equal(out, [[1, 6, 0], [3, 4, 0], [2, 8, 0]])
    np.testing.assert_array_equal(x, before, err_msg="x must not be modified")


def test_masked_row_mean():
    x = rng.normal(size=(6, 5))
    mask = rng.random((6, 5)) < 0.5
    mask[2] = False
    mask[4] = True
    expected = [x[i][mask[i]].mean() if mask[i].any() else 0.0 for i in range(6)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = masked_row_mean(x, mask)
    assert out.shape == (6,)
    np.testing.assert_allclose(out, expected)


def test_no_python_loops(no_loops):
    no_loops(drop_nan_rows, high_variance_columns, fill_nan_with_column_mean, masked_row_mean)
