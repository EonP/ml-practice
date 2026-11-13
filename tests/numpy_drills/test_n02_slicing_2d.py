import numpy as np

from numpy_drills.n02_slicing_2d import column, corners, interior, tile

x = np.arange(30).reshape(5, 6)


def test_column():
    out = column(x, 2)
    assert out.shape == (5,)
    np.testing.assert_array_equal(out, [2, 8, 14, 20, 26])
    np.testing.assert_array_equal(column(x, -1), [5, 11, 17, 23, 29])


def test_column_keepdims():
    for j in (0, 2, -2, -1):
        out = column(x, j, keepdims=True)
        assert out.shape == (5, 1), f"j={j}: got shape {out.shape} (what is j + 1 when j = -1?)"
        np.testing.assert_array_equal(out[:, 0], [x[i][j] for i in range(5)])


def test_interior():
    np.testing.assert_array_equal(interior(x), [[7, 8, 9, 10], [13, 14, 15, 16], [19, 20, 21, 22]])
    np.testing.assert_array_equal(interior(x, 2), [[14, 15]])


def test_corners():
    np.testing.assert_array_equal(corners(x), [[0, 5], [24, 29]])
    np.testing.assert_array_equal(corners(np.arange(4).reshape(2, 2)), [[0, 1], [2, 3]])
    np.testing.assert_array_equal(corners(np.arange(21).reshape(7, 3)), [[0, 2], [18, 20]])


def test_tile():
    y = np.arange(36).reshape(6, 6)
    np.testing.assert_array_equal(tile(y, 0, 0, 2), [[0, 1], [6, 7]])
    np.testing.assert_array_equal(tile(y, 1, 2, 2), [[16, 17], [22, 23]])
    np.testing.assert_array_equal(tile(y, 1, 0, 3), [[18, 19, 20], [24, 25, 26], [30, 31, 32]])


def test_results_are_views():
    for out in (column(x, 1), column(x, -1, keepdims=True), interior(x), corners(x), tile(x, 1, 1, 2)):
        assert np.shares_memory(out, x), "use basic indexing (ints, slices, None): no index lists, no copies"


def test_no_python_loops(no_loops):
    no_loops(column, interior, corners, tile)
