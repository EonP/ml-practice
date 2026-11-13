import numpy as np

from numpy_drills.n15_take_along_axis import sort_rows_by, target_logits, value_at_argmax

rng = np.random.default_rng(0)


def test_sort_rows_by():
    keys, values = rng.random((5, 6)), rng.normal(size=(5, 6))
    expected = np.array([values[i][np.argsort(keys[i])] for i in range(5)])
    np.testing.assert_array_equal(sort_rows_by(keys, values), expected)


def test_value_at_argmax():
    for shape in [(3, 4, 5), (3, 1, 5)]:
        a, b = rng.random(shape), rng.normal(size=shape)
        for axis in (0, 1, 2, -1):
            out = value_at_argmax(a, b, axis)
            out_shape = tuple(s for d, s in enumerate(shape) if d != axis % 3)
            assert out.shape == out_shape, f"shape={shape}, axis={axis}: expected {out_shape}, got {out.shape}"
            for idx in np.ndindex(out_shape):
                sl = list(idx)
                sl.insert(axis % 3, slice(None))
                sl = tuple(sl)
                assert out[idx] == b[sl][np.argmax(a[sl])], f"shape={shape}, axis={axis}, position {idx}"


def test_target_logits():
    N, K, H, W = 2, 4, 3, 5
    logits = rng.normal(size=(N, K, H, W))
    target = rng.integers(0, K, size=(N, H, W))
    out = target_logits(logits, target)
    assert out.shape == (N, H, W)
    for n in range(N):
        for h in range(H):
            for w in range(W):
                assert out[n, h, w] == logits[n, target[n, h, w], h, w]


def test_no_python_loops(no_loops):
    no_loops(sort_rows_by, value_at_argmax, target_logits)
