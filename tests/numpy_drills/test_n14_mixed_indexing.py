import numpy as np

from numpy_drills.n14_mixed_indexing import channel_per_sample, columns_of_sample, pixel_per_sample, select_channels

rng = np.random.default_rng(0)
N, C, H, W = 2, 3, 5, 7
x = rng.normal(size=(N, C, H, W))


def test_select_channels():
    ch = np.array([2, 0, 2, 1])
    out = select_channels(x, ch)
    assert out.shape == (N, 4, H, W)
    np.testing.assert_array_equal(out, np.stack([x[:, c] for c in ch], axis=1))


def test_channel_per_sample():
    ch = np.array([2, 0])
    out = channel_per_sample(x, ch)
    assert out.shape == (N, H, W)
    np.testing.assert_array_equal(out, np.stack([x[n, ch[n]] for n in range(N)]))


def test_pixel_per_sample():
    ys, xs = np.array([4, 0]), np.array([1, 6])
    out = pixel_per_sample(x, ys, xs)
    assert out.shape == (N, C), f"expected (N, C) = {(N, C)}, got {out.shape}"
    np.testing.assert_array_equal(out, np.stack([x[n, :, ys[n], xs[n]] for n in range(N)]))


def test_columns_of_sample():
    y = rng.normal(size=(3, 5, 7))
    cols = np.array([6, 0, 3])
    out = columns_of_sample(y, 1, cols)
    assert out.shape == (5, 3), (
        f"expected (H, K) = (5, 3), got {out.shape}: a slice between two advanced indices sends the broadcast axis to the front"
    )
    np.testing.assert_array_equal(out, [[y[1, h, c] for c in cols] for h in range(5)])


def test_no_python_loops(no_loops):
    no_loops(select_channels, channel_per_sample, pixel_per_sample, columns_of_sample)
