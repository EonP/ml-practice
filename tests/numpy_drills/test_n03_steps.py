import numpy as np

from numpy_drills.n03_steps import checkerboard, downsample, flip_lr, interleave, rot180

rng = np.random.default_rng(0)
img = rng.integers(0, 256, size=(7, 10, 3), dtype=np.uint8)


def test_downsample():
    for f in (1, 2, 3, 4):
        rows = [i for i in range(7) if i % f == 0]
        cols = [j for j in range(10) if j % f == 0]
        np.testing.assert_array_equal(downsample(img, f), img[rows][:, cols], err_msg=f"f={f}")


def test_flip_lr():
    np.testing.assert_array_equal(flip_lr(img), np.flip(img, axis=1))


def test_rot180():
    np.testing.assert_array_equal(rot180(img), np.rot90(img, 2))


def test_results_are_views():
    for out in (downsample(img, 2), flip_lr(img), rot180(img)):
        assert np.shares_memory(out, img), "negative and positive steps both give views"


def test_interleave():
    np.testing.assert_array_equal(interleave(np.arange(4), np.arange(4) + 100), [0, 100, 1, 101, 2, 102, 3, 103])


def test_interleave_keeps_trailing_axes_and_dtype():
    a, b = rng.integers(0, 9, size=(3, 2, 2)), rng.integers(0, 9, size=(3, 2, 2))
    out = interleave(a, b)
    assert out.shape == (6, 2, 2) and out.dtype == a.dtype
    np.testing.assert_array_equal(out, np.stack([a, b], axis=1).reshape(6, 2, 2))
    a32, b32 = rng.random(5, dtype=np.float32), rng.random(5, dtype=np.float32)
    assert interleave(a32, b32).dtype == np.float32, "keep the input dtype"


def test_checkerboard():
    for h, w in [(1, 1), (2, 3), (4, 4), (5, 2)]:
        out = checkerboard(h, w)
        assert out.shape == (h, w) and np.issubdtype(out.dtype, np.integer)
        np.testing.assert_array_equal(out, np.add.outer(np.arange(h), np.arange(w)) % 2, err_msg=f"{h}x{w}")


def test_no_python_loops(no_loops):
    no_loops(downsample, flip_lr, rot180, ban=("flip", "fliplr", "flipud", "rot90"))
    no_loops(interleave, ban=("stack", "vstack", "hstack", "dstack", "column_stack", "concatenate", "insert", "repeat"))
    no_loops(checkerboard, ban=("arange", "indices", "meshgrid", "mgrid", "ogrid", "tile", "kron", "outer"))
