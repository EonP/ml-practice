"""NumPy drill 22: non-overlapping patches with reshape + transpose (target: 15 min).
No loops, no stride tricks.

Splitting an axis of length H into (H // p, p) is a free reshape. The real work
is moving the axes into the right order afterwards, remembering that reshape
reads and writes elements in row-major (C) order.
"""
import numpy as np


def block_mean(img, f):
    """Average pooling: (H, W) -> (H // f, W // f), the mean of each f x f block. H and W are multiples of f."""
    raise NotImplementedError("TODO: block_mean")


def patchify(img, p):
    """(H, W, C) -> (H // p, W // p, p, p, C) with out[i, j] = img[i*p:(i+1)*p, j*p:(j+1)*p].

    H and W are multiples of p.
    """
    raise NotImplementedError("TODO: patchify")


def unpatchify(patches):
    """Inverse of patchify: (Hp, Wp, p, p, C) -> (Hp * p, Wp * p, C)."""
    raise NotImplementedError("TODO: unpatchify")


def vit_tokens(x, p):
    """ViT patch tokens: x (N, C, H, W) -> (N, L, C * p * p) with L = (H // p) * (W // p).

    Patches come in row-major order over the patch grid, and each patch is
    flattened in (C, p, p) order. This is what
    torch.nn.functional.unfold(x, p, stride=p).transpose(1, 2) gives.
    """
    raise NotImplementedError("TODO: vit_tokens")
