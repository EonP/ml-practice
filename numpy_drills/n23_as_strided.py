"""NumPy drill 23: sliding windows by hand with as_strided (target: 15 min). No loops.

sliding_window_view is off limits here. np.lib.stride_tricks.as_strided(x,
shape=..., strides=...) makes a view with whatever shape and strides you ask
for. Strides are BYTES per step along each axis: take them from x.strides
rather than assuming x is contiguous. Nothing is bounds-checked, so a wrong
shape silently reads memory outside x.
"""
import numpy as np


def sliding_window_1d(x, k, step=1):
    """(N,) -> ((N - k) // step + 1, k) view, with window i starting at x[i * step].

    Must work on non-contiguous input such as a[::3].
    """
    raise NotImplementedError("TODO: sliding_window_1d")


def sliding_window_2d(x, kh, kw):
    """(H, W) -> (H - kh + 1, W - kw + 1, kh, kw) view with out[i, j] = x[i:i + kh, j:j + kw].

    Must work on non-contiguous input such as a.T.
    """
    raise NotImplementedError("TODO: sliding_window_2d")
