import numpy as np

from numpy_drills.n16_ragged_masks import pad_ragged, radial_distance, sequence_mask, unpad

rng = np.random.default_rng(0)


def test_sequence_mask():
    out = sequence_mask(np.array([2, 0, 3]), 4)
    assert out.dtype == bool
    np.testing.assert_array_equal(out, [[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0]])


def test_pad_ragged():
    out = pad_ragged(np.array([1, 2, 3, 4, 5, 6]), np.array([3, 0, 1, 2]), fill=-1)
    assert out.dtype == np.array([1]).dtype
    np.testing.assert_array_equal(out, [[1, 2, 3], [-1, -1, -1], [4, -1, -1], [5, 6, -1]])


def test_pad_ragged_random():
    lengths = rng.integers(1, 6, size=10)
    flat = rng.normal(size=lengths.sum())
    out = pad_ragged(flat, lengths)
    assert out.shape == (10, lengths.max()) and out.dtype == flat.dtype
    start = 0
    for i, n in enumerate(lengths):
        np.testing.assert_array_equal(out[i, :n], flat[start:start + n])
        np.testing.assert_array_equal(out[i, n:], 0)
        start += n


def test_unpad():
    padded = np.array([[1, 2, 9], [3, 9, 9], [9, 9, 9], [4, 5, 6]])
    np.testing.assert_array_equal(unpad(padded, np.array([2, 1, 0, 3])), [1, 2, 3, 4, 5, 6])


def test_radial_distance():
    for H, W, cy, cx in [(4, 5, 1, 3), (3, 3, 1.5, 0.5)]:
        out = radial_distance(H, W, cy, cx)
        assert out.shape == (H, W)
        expected = [[np.hypot(r - cy, c - cx) for c in range(W)] for r in range(H)]
        np.testing.assert_allclose(out, expected)


def test_no_python_loops(no_loops):
    no_loops(sequence_mask, pad_ragged, unpad)
    no_loops(radial_distance, ban=("meshgrid", "indices", "mgrid"))
