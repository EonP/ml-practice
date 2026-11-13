import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from numpy_drills.n23_as_strided import sliding_window_1d, sliding_window_2d


def _check_1d(x):
    for k, step in [(3, 1), (4, 2), (len(x), 1), (1, 3), (3, 4)]:
        out = sliding_window_1d(x, k, step)
        n = (len(x) - k) // step + 1
        assert out.shape == (n, k), f"k={k}, step={step}: shape {out.shape}"
        np.testing.assert_array_equal(out, [x[i * step:i * step + k] for i in range(n)], err_msg=f"k={k}, step={step}")
        assert np.shares_memory(out, x), "must be a view"


def test_sliding_window_1d_contiguous():
    _check_1d(np.arange(10.0))


def test_sliding_window_1d_other_itemsize():
    _check_1d(np.arange(10, dtype=np.int32))


def test_sliding_window_1d_non_contiguous():
    _check_1d(np.arange(30.0)[::3])


def test_sliding_window_2d():
    inputs = [np.arange(20.0).reshape(4, 5), np.arange(42.0).reshape(6, 7).T, np.arange(24, dtype=np.int16).reshape(4, 6)[:, ::2]]
    for x in inputs:
        for kh, kw in [(2, 2), (1, 3), (3, 2)]:
            out = sliding_window_2d(x, kh, kw)
            np.testing.assert_array_equal(out, sliding_window_view(x, (kh, kw)), err_msg=f"x.strides={x.strides}, kh={kh}, kw={kw}")
            assert np.shares_memory(out, x), "must be a view"


def test_no_python_loops(no_loops):
    no_loops(sliding_window_1d, sliding_window_2d, ban=("sliding_window_view",))
