import warnings

import numpy as np

from numpy_drills.n18_add_at import bincount, scatter_add_rows, segment_mean

rng = np.random.default_rng(0)


def test_bincount_with_repeats():
    out = bincount(np.array([0, 2, 2, 2, 1, 4, 4]), 6)
    assert np.issubdtype(out.dtype, np.integer)
    np.testing.assert_array_equal(out, [1, 1, 3, 0, 2, 0], err_msg="counts[x] += 1 counts each repeated index once")


def test_bincount_random():
    x = rng.integers(0, 10, size=1000)
    np.testing.assert_array_equal(bincount(x, 12), np.bincount(x, minlength=12))


def test_scatter_add_rows():
    src, idx = rng.normal(size=(30, 4)), rng.integers(0, 6, size=30)
    idx[:3] = 5
    out = scatter_add_rows(src, idx, 7)
    assert out.shape == (7, 4)
    expected = np.zeros((7, 4))
    for i in range(30):
        expected[idx[i]] += src[i]
    np.testing.assert_allclose(out, expected)


def test_segment_mean():
    values, seg = rng.normal(size=(20, 3)), rng.integers(0, 5, size=20)
    seg[seg == 3] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = segment_mean(values, seg, 6)
    expected = [values[seg == j].mean(axis=0) if (seg == j).any() else np.zeros(3) for j in range(6)]
    np.testing.assert_allclose(out, expected)


def test_no_python_loops(no_loops):
    no_loops(bincount, ban=("bincount", "unique", "histogram"))
    no_loops(scatter_add_rows, segment_mean)
