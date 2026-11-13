"""NumPy drill 20: sliding windows in 1D (target: 10 min). No loops.

np.lib.stride_tricks.sliding_window_view(x, k) returns every length-k window
as a VIEW of shape (N - k + 1, k): no data is copied. Index arithmetic builds
the same thing as a copy: window starts as a column plus offsets as a row.
"""
import numpy as np


def windows_by_index(x, k):
    """(N,) -> (N - k + 1, k) with out[i] = x[i:i + k].

    Build it with broadcast index arrays (no sliding_window_view or as_strided here).
    """
    raise NotImplementedError("TODO: windows_by_index")


def frames(x, frame_len, hop):
    """Audio-style framing: (N,) -> (1 + (N - frame_len) // hop, frame_len).

    Frame i starts at i * hop. Must be a view of x.
    """
    raise NotImplementedError("TODO: frames")


def moving_average(x, k):
    """(N,) -> (N - k + 1,): the mean of each length-k window."""
    raise NotImplementedError("TODO: moving_average")


def is_local_max(x, r):
    """x (N,) floats with no ties -> (N,) bool, True where x[i] is the max of x[max(0, i - r):i + r + 1].

    Hint: pad so that every position gets a full window.
    """
    raise NotImplementedError("TODO: is_local_max")
