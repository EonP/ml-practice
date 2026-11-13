"""Given: a tiny synthetic segmentation dataset for quick checks without downloads.

Noisy dark background (class 0), a red disc (class 1) and a blue square (class 2).
"""
import numpy as np


def make_toy_pair(rng, size=32):
    """One (uint8 image (size, size, 3), int64 mask (size, size)) pair."""
    img = rng.integers(0, 60, size=(size, size, 3)).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.int64)
    rr, cc = np.mgrid[:size, :size]
    cy, cx = rng.integers(size // 4, size - size // 4, 2)
    r = int(rng.integers(size // 8, size // 5 + 1))
    disc = (rr - cy) ** 2 + (cc - cx) ** 2 <= r ** 2
    mask[disc], img[disc] = 1, [220, 40, 40]
    s = max(size // 5, 3)
    y0, x0 = rng.integers(0, size - s, 2)
    sq = (rr >= y0) & (rr < y0 + s) & (cc >= x0) & (cc < x0 + s) & ~disc
    mask[sq], img[sq] = 2, [40, 40, 220]
    return img, mask


def make_toy_dataset(n, size=32, seed=0):
    """Lists (images, masks) of n toy pairs."""
    rng = np.random.default_rng(seed)
    pairs = [make_toy_pair(rng, size) for _ in range(n)]
    return [p[0] for p in pairs], [p[1] for p in pairs]
