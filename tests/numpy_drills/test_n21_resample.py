import numpy as np
from scipy.ndimage import map_coordinates

from numpy_drills.n21_resample import bilinear_sample, nearest_resize

rng = np.random.default_rng(0)


def test_nearest_resize():
    img = rng.integers(0, 256, size=(5, 7, 3))
    for H2, W2 in [(10, 14), (3, 4), (5, 7), (8, 3), (15, 7)]:
        out = nearest_resize(img, H2, W2)
        assert out.shape == (H2, W2, 3)
        for i in range(H2):
            for j in range(W2):
                np.testing.assert_array_equal(out[i, j], img[i * 5 // H2, j * 7 // W2])


def test_nearest_resize_doubling_repeats_pixels():
    out = nearest_resize(np.arange(4).reshape(2, 2, 1), 4, 4)
    np.testing.assert_array_equal(out[..., 0], [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])


def test_bilinear_on_pixel_centres_is_exact():
    img = rng.normal(size=(5, 6))
    ys, xs = np.array([0, 2, 4, 4, 0]), np.array([0, 3, 5, 0, 5])
    np.testing.assert_allclose(bilinear_sample(img, ys.astype(float), xs.astype(float)), img[ys, xs], atol=1e-12)


def test_bilinear_matches_scipy():
    img = rng.normal(size=(6, 8))
    ys = np.r_[rng.uniform(0, 5, size=50), 5.0, 5.0, 0.0, 2.5]
    xs = np.r_[rng.uniform(0, 7, size=50), 7.0, 3.3, 7.0, 7.0]
    expected = map_coordinates(img, [ys, xs], order=1, mode="nearest")
    np.testing.assert_allclose(bilinear_sample(img, ys, xs), expected, atol=1e-10, err_msg="check points on the last row/column")


def test_no_python_loops(no_loops):
    no_loops(nearest_resize, bilinear_sample, ban=("map_coordinates", "interp", "zoom", "resize"))
