import numpy as np

from numpy_drills.n12_paired_indexing import gather_pixels, pick_per_row, set_per_row_, swap_rows_

rng = np.random.default_rng(0)


def test_pick_per_row():
    x = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(pick_per_row(x, np.array([3, 0, 2])), [3, 4, 10])
    y, cols = rng.normal(size=(20, 7)), rng.integers(0, 7, size=20)
    np.testing.assert_array_equal(pick_per_row(y, cols), [y[i, cols[i]] for i in range(20)])


def test_set_per_row_in_place():
    x = np.zeros((3, 4))
    assert set_per_row_(x, np.array([1, 3, 1]), 7.0) is None
    np.testing.assert_array_equal(x, [[0, 7, 0, 0], [0, 0, 0, 7], [0, 7, 0, 0]])


def test_gather_pixels():
    img = rng.integers(0, 256, size=(5, 6, 3))
    pts = np.array([[0, 0], [4, 5], [2, 1], [2, 1]])
    out = gather_pixels(img, pts)
    assert out.shape == (4, 3)
    np.testing.assert_array_equal(out, [img[r, c] for r, c in pts])


def test_swap_rows_in_place():
    x = np.arange(12).reshape(4, 3)
    assert swap_rows_(x, 0, 2) is None
    np.testing.assert_array_equal(
        x, [[6, 7, 8], [3, 4, 5], [0, 1, 2], [9, 10, 11]], err_msg="x[i], x[j] = x[j], x[i] swaps views, not data"
    )
    swap_rows_(x, 1, 1)
    np.testing.assert_array_equal(x[1], [3, 4, 5])


def test_no_python_loops(no_loops):
    no_loops(pick_per_row, set_per_row_, gather_pixels, swap_rows_)
