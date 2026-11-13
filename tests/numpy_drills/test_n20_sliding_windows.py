import numpy as np

from numpy_drills.n20_sliding_windows import frames, is_local_max, moving_average, windows_by_index

rng = np.random.default_rng(0)
x = rng.normal(size=20)


def test_windows_by_index():
    for k in (1, 3, 20):
        out = windows_by_index(x, k)
        assert out.shape == (21 - k, k)
        np.testing.assert_array_equal(out, [x[i:i + k] for i in range(21 - k)])


def test_frames():
    for L, hop in [(5, 2), (4, 4), (6, 3), (20, 1), (7, 5)]:
        out = frames(x, L, hop)
        n = 1 + (20 - L) // hop
        assert out.shape == (n, L), f"frame_len={L}, hop={hop}"
        np.testing.assert_array_equal(out, [x[i * hop:i * hop + L] for i in range(n)])
        assert np.shares_memory(out, x), "frames must be a view of x"


def test_moving_average():
    for k in (1, 4, 20):
        np.testing.assert_allclose(moving_average(x, k), np.convolve(x, np.ones(k) / k, "valid"), atol=1e-12)


def test_is_local_max():
    y = rng.permutation(30).astype(float)
    for r in (1, 2, 5):
        out = is_local_max(y, r)
        assert out.dtype == bool and out.shape == (30,)
        np.testing.assert_array_equal(out, [y[i] == y[max(0, i - r):i + r + 1].max() for i in range(30)], err_msg=f"r={r}")


def test_no_python_loops(no_loops):
    no_loops(windows_by_index, ban=("sliding_window_view", "as_strided"))
    no_loops(frames, moving_average, is_local_max)
