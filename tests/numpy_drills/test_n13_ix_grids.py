import numpy as np

from numpy_drills.n13_ix_grids import masked_submatrix, set_submatrix_, subgraph, submatrix

rng = np.random.default_rng(0)
x = np.arange(30).reshape(5, 6)


def test_submatrix():
    np.testing.assert_array_equal(submatrix(x, np.array([4, 0]), np.array([1, 1, 5])), [[25, 25, 29], [1, 1, 5]])


def test_submatrix_square_is_still_a_grid():
    rows, cols = np.array([0, 2, 3]), np.array([5, 1, 0])
    out = submatrix(x, rows, cols)
    assert out.shape == (3, 3), "x[rows, cols] pairs the indices up: you get (R,), not (R, S)"
    np.testing.assert_array_equal(out, [[x[r, c] for c in cols] for r in rows])


def test_set_submatrix_in_place():
    y = np.zeros((4, 5), dtype=int)
    assert set_submatrix_(y, np.array([0, 2]), np.array([1, 4]), np.array([[1, 2], [3, 4]])) is None
    expected = np.zeros((4, 5), dtype=int)
    expected[0, 1], expected[0, 4], expected[2, 1], expected[2, 4] = 1, 2, 3, 4
    np.testing.assert_array_equal(y, expected)


def test_subgraph():
    adj = (rng.random((8, 8)) < 0.3).astype(int)
    nodes = np.array([5, 1, 6])
    np.testing.assert_array_equal(subgraph(adj, nodes), [[adj[a, b] for b in nodes] for a in nodes])


def test_masked_submatrix():
    rm = np.array([True, False, True, True, False])
    cm = np.array([False, True, False, False, True, True])
    expected = [[x[i, j] for j in range(6) if cm[j]] for i in range(5) if rm[i]]
    np.testing.assert_array_equal(masked_submatrix(x, rm, cm), expected)
    rm2 = np.array([True, True, False, False, False])
    np.testing.assert_array_equal(masked_submatrix(x, rm2, cm), [[1, 4, 5], [7, 10, 11]])


def test_no_python_loops(no_loops):
    no_loops(submatrix, set_submatrix_, subgraph, masked_submatrix)
