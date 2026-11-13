import numpy as np

from numpy_drills.n04_ellipsis import bgr_to_rgb, center_crop, last_channel, split_halves

rng = np.random.default_rng(0)
SHAPES = [(3,), (5, 3), (4, 5, 3), (2, 4, 5, 3), (2, 2, 4, 5, 3)]


def test_last_channel():
    for s in SHAPES:
        x = rng.normal(size=s)
        out = last_channel(x)
        assert out.shape == s[:-1]
        np.testing.assert_array_equal(out, np.moveaxis(x, -1, 0)[-1])


def test_bgr_to_rgb():
    for s in SHAPES:
        x = rng.normal(size=s)
        np.testing.assert_array_equal(bgr_to_rgb(x), np.flip(x, axis=-1))


def test_split_halves():
    for lead in [(), (3,), (2, 3)]:
        x = rng.normal(size=lead + (8,))
        a, b = split_halves(x)
        np.testing.assert_array_equal(a, np.split(x, 2, axis=-1)[0])
        np.testing.assert_array_equal(b, np.split(x, 2, axis=-1)[1])


def test_center_crop():
    x = np.arange(7 * 9).reshape(7, 9)
    np.testing.assert_array_equal(center_crop(x, 3, 4), x[2:5, 2:6])
    np.testing.assert_array_equal(center_crop(x, 4, 4), x[1:5, 2:6], err_msg="odd leftover: the extra row goes at the bottom")
    np.testing.assert_array_equal(center_crop(x, 7, 9), x)


def test_center_crop_any_leading_axes():
    xb = rng.normal(size=(2, 3, 7, 9))
    np.testing.assert_array_equal(center_crop(xb, 3, 4), xb[:, :, 2:5, 2:6])
    np.testing.assert_array_equal(center_crop(xb[0, 0], 3, 4), xb[0, 0, 2:5, 2:6])


def test_results_are_views():
    x = rng.normal(size=(2, 7, 9, 4))
    for out in (last_channel(x), bgr_to_rgb(x[..., :3]), *split_halves(x), center_crop(x, 3, 4)):
        assert np.shares_memory(out, x)


def test_no_python_loops(no_loops):
    no_loops(last_channel, bgr_to_rgb, split_halves, center_crop, ban=("flip",))
