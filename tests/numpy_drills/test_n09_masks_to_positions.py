import numpy as np

from numpy_drills.n09_masks_to_positions import bbox, coords_of, first_true, last_true

rng = np.random.default_rng(0)


def _first(mask, reverse=False):
    out = []
    for row in mask:
        cols = [j for j, v in enumerate(row) if v]
        out.append((cols[-1] if reverse else cols[0]) if cols else -1)
    return out


def test_coords_of():
    x = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 1]])
    np.testing.assert_array_equal(coords_of(x, 1), [[0, 0], [0, 2], [2, 1], [2, 2]])
    assert coords_of(x, 7).shape == (0, 2)


def test_first_true():
    mask = rng.random((50, 8)) < 0.15
    mask[0], mask[1] = False, True
    out = first_true(mask)
    assert out.shape == (50,) and np.issubdtype(out.dtype, np.integer)
    np.testing.assert_array_equal(out, _first(mask))


def test_first_true_empty_row_is_minus_one():
    out = first_true(np.array([[False, False, False], [False, True, False]]))
    np.testing.assert_array_equal(out, [-1, 1], err_msg="argmax of an all-False row is 0, not -1")


def test_last_true():
    mask = rng.random((50, 8)) < 0.15
    mask[0], mask[1] = False, True
    np.testing.assert_array_equal(last_true(mask), _first(mask, reverse=True))
    np.testing.assert_array_equal(last_true(np.array([[False] * 3, [True, False, True]])), [-1, 2])


def test_bbox():
    m = np.zeros((6, 7), dtype=bool)
    m[2, 3] = True
    assert bbox(m) == (2, 3, 2, 3)
    m[4, 1] = m[3, 5] = True
    box = bbox(m)
    assert box == (2, 1, 4, 5)
    assert all(type(v) is int for v in box), "return Python ints, not NumPy scalars"
    assert bbox(np.zeros((3, 3), dtype=bool)) is None


def test_no_python_loops(no_loops):
    no_loops(coords_of, first_true, last_true, bbox)
