import numpy as np

from numpy_drills.n01_slicing_1d import drop_ends, every_kth, last_k, odd_positions_reversed

x = np.arange(10) * 10


def test_last_k():
    np.testing.assert_array_equal(last_k(x, 3), [70, 80, 90])
    np.testing.assert_array_equal(last_k(x, 1), [90])
    np.testing.assert_array_equal(last_k(x, 10), x)


def test_last_k_zero_is_empty():
    assert last_k(x, 0).shape == (0,), "x[-0:] is x[0:], the whole array"


def test_drop_ends():
    np.testing.assert_array_equal(drop_ends(x, 2), [20, 30, 40, 50, 60, 70])
    assert drop_ends(x, 5).shape == (0,)


def test_drop_ends_zero_drops_nothing():
    np.testing.assert_array_equal(drop_ends(x, 0), x, err_msg="x[0:-0] is x[0:0], which is empty")


def test_every_kth():
    for k in (1, 2, 3, 4, 11):
        for start in (0, 1, 3):
            expected = [x[i] for i in range(start, len(x), k)]
            np.testing.assert_array_equal(every_kth(x, k, start), expected, err_msg=f"k={k}, start={start}")


def test_odd_positions_reversed():
    for n in range(8):
        a = np.arange(n) * 10
        expected = [a[i] for i in range(n - 1, -1, -1) if i % 2 == 1]
        np.testing.assert_array_equal(odd_positions_reversed(a), expected, err_msg=f"len(x) = {n}")


def test_results_are_views():
    for out in (last_k(x, 3), drop_ends(x, 2), every_kth(x, 3, 1), odd_positions_reversed(x)):
        assert np.shares_memory(out, x), "slicing never copies: don't build a new array"


def test_no_python_loops(no_loops):
    no_loops(last_k, drop_ends, every_kth, odd_positions_reversed, ban=("flip",))
