import numpy as np

from numpy_drills.n06_views_inplace import crop_copy, crop_view, normalize_rows_, zero_border_

rng = np.random.default_rng(0)


def test_crop_view_writes_through():
    x = np.arange(30.0).reshape(5, 6)
    out = crop_view(x, 1, 2, 3, 2)
    np.testing.assert_array_equal(out, [[8, 9], [14, 15], [20, 21]])
    out[0, 0] = -1
    assert x[1, 2] == -1, "writing into the view must change x"


def test_crop_copy_is_independent():
    x = np.arange(30.0).reshape(5, 6)
    out = crop_copy(x, 1, 2, 3, 2)
    np.testing.assert_array_equal(out, [[8, 9], [14, 15], [20, 21]])
    out[0, 0] = -1
    assert x[1, 2] == 8, "writing into the copy must not change x"


def test_zero_border_in_place():
    x = np.ones((5, 6))
    assert zero_border_(x, 1) is None, "in-place functions return None"
    expected = np.zeros((5, 6))
    expected[1:4, 1:5] = 1
    np.testing.assert_array_equal(x, expected)

    x = np.ones((6, 7))
    zero_border_(x, 2)
    expected = np.zeros((6, 7))
    expected[2:4, 2:5] = 1
    np.testing.assert_array_equal(x, expected)


def test_zero_border_through_a_view():
    big = np.ones((8, 8))
    zero_border_(big[2:6, 2:6], 1)
    expected = np.ones((8, 8))
    expected[2:6, 2:6] = 0
    expected[3:5, 3:5] = 1
    np.testing.assert_array_equal(big, expected)


def test_normalize_rows_in_place():
    x = rng.random((4, 5)) + 0.1
    expected = x / x.sum(axis=1, keepdims=True)
    assert normalize_rows_(x) is None, "in-place functions return None"
    np.testing.assert_allclose(x, expected, err_msg="x itself must change: `x = x / ...` only rebinds the local name")


def test_no_python_loops(no_loops):
    no_loops(crop_view, crop_copy, zero_border_, normalize_rows_)
