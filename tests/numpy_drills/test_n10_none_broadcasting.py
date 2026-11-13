import numpy as np

from numpy_drills.n10_none_broadcasting import add_per_channel, causal_mask, outer, pairwise_diff, scale_rows

rng = np.random.default_rng(0)


def test_outer():
    a, b = rng.normal(size=4), rng.normal(size=6)
    out = outer(a, b)
    assert out.shape == (4, 6)
    np.testing.assert_allclose(out, np.outer(a, b))


def test_scale_rows():
    for N, D in [(3, 5), (4, 4)]:
        x, s = rng.normal(size=(N, D)), rng.normal(size=N)
        np.testing.assert_allclose(scale_rows(x, s), np.diag(s) @ x, err_msg=f"N={N}, D={D} (x * s scales COLUMNS)")


def test_add_per_channel():
    x, bias = rng.normal(size=(2, 3, 4, 3)), rng.normal(size=3)
    expected = x.copy()
    for c in range(3):
        expected[:, c] += bias[c]
    np.testing.assert_allclose(add_per_channel(x, bias), expected, err_msg="x + bias lines bias up with W, not C")


def test_causal_mask():
    for n in (1, 2, 5):
        out = causal_mask(n)
        assert out.dtype == bool and out.shape == (n, n)
        np.testing.assert_array_equal(out, np.tril(np.ones((n, n), dtype=bool)))


def test_pairwise_diff():
    x = rng.normal(size=(4, 3))
    out = pairwise_diff(x)
    assert out.shape == (4, 4, 3)
    for i in range(4):
        for j in range(4):
            np.testing.assert_allclose(out[i, j], x[i] - x[j])


def test_no_python_loops(no_loops):
    no_loops(outer, scale_rows, add_per_channel, pairwise_diff, ban=("outer", "einsum", "diag"))
    no_loops(causal_mask, ban=("tril", "triu", "tri", "tril_indices", "triu_indices"))
