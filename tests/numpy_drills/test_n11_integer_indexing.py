import numpy as np

from numpy_drills.n11_integer_indexing import colorize, invert_permutation, one_hot, shuffle_together

rng = np.random.default_rng(0)


def test_shuffle_together():
    x = np.arange(50)[:, None] * np.array([1, 10])
    y = np.arange(50) * 100
    x0, y0 = x.copy(), y.copy()
    xs, ys = shuffle_together(x, y, np.random.default_rng(0))
    np.testing.assert_array_equal(x, x0, err_msg="don't modify x")
    np.testing.assert_array_equal(y, y0, err_msg="don't modify y")
    np.testing.assert_array_equal(np.sort(ys), y, err_msg="y must be permuted, not changed")
    np.testing.assert_array_equal(xs[:, 1], xs[:, 0] * 10, err_msg="rows of x must stay intact")
    np.testing.assert_array_equal(xs[:, 0] * 100, ys, err_msg="row i of x must still go with y[i]")
    assert not np.array_equal(ys, y), "nothing was shuffled"


def test_one_hot():
    out = one_hot(np.array([2, 0, 1, 2]), 3)
    assert np.issubdtype(out.dtype, np.floating)
    np.testing.assert_array_equal(out, [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_one_hot_any_shape():
    y = rng.integers(0, 5, size=(2, 3, 4))
    out = one_hot(y, 5)
    assert out.shape == (2, 3, 4, 5)
    np.testing.assert_array_equal(out.sum(axis=-1), 1)
    np.testing.assert_array_equal(out.argmax(axis=-1), y)


def test_colorize():
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    labels = rng.integers(0, 3, size=(4, 5))
    out = colorize(labels, palette)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8
    for r in range(4):
        for c in range(5):
            np.testing.assert_array_equal(out[r, c], palette[labels[r, c]])


def test_invert_permutation():
    p = rng.permutation(20)
    inv = invert_permutation(p)
    np.testing.assert_array_equal(p[inv], np.arange(20))
    np.testing.assert_array_equal(inv[p], np.arange(20))


def test_no_python_loops(no_loops):
    no_loops(shuffle_together, one_hot, colorize)
    no_loops(invert_permutation, ban=("argsort", "sort", "lexsort"))
