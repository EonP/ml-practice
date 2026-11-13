"""Stage 1 (Day 1): NumPy for images and masks.

Conventions used throughout the project:
  image: (H, W, 3) uint8 as loaded from disk (HWC); models want (3, H, W) float (CHW)
  mask:  (H, W) integer class labels in [0, K)
"""
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def hwc_to_chw(img):
    """(H, W, C) -> (C, H, W). One call, no copy needed."""
    raise NotImplementedError("TODO: hwc_to_chw")


def chw_to_hwc(img):
    """(C, H, W) -> (H, W, C)."""
    raise NotImplementedError("TODO: chw_to_hwc")


def normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """uint8 (H, W, 3) -> float32 (H, W, 3): scale to [0, 1], then (x - mean) / std
    per channel (broadcasting, no loops)."""
    raise NotImplementedError("TODO: normalize")


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Inverse of `normalize`, returning floats clipped to [0, 1] (ready for imshow)."""
    raise NotImplementedError("TODO: denormalize")


def one_hot(mask, K):
    """(H, W) int -> (K, H, W) float32 one-hot. No loops. Hint: np.eye or comparison with np.arange(K)[:, None, None]."""
    raise NotImplementedError("TODO: one_hot")


def class_pixel_counts(masks, K):
    """Total number of pixels of each class over a list (or array) of masks -> (K,) ints.
    Useful for spotting class imbalance before you train anything."""
    raise NotImplementedError("TODO: class_pixel_counts")


def hflip_pair(img, mask):
    """Horizontally flip an (H, W, C) image and its (H, W) mask together."""
    raise NotImplementedError("TODO: hflip_pair")


def random_crop_pair(img, mask, size, rng):
    """Crop the SAME random (size x size) window from the image and the mask.
    rng is a np.random.Generator. Assumes H, W >= size."""
    raise NotImplementedError("TODO: random_crop_pair")


def overlay(ax, img, mask, alpha=0.5):
    """Matplotlib: draw the image, then the mask on top with transparency, on the
    given Axes. Turn the axis ticks off. Returns ax.
    Hint: two ax.imshow calls; use interpolation='nearest' and a discrete cmap for the mask."""
    raise NotImplementedError("TODO: overlay")
