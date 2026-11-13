"""Stage 1 (Day 1): NumPy for images and masks.

Conventions used throughout the project:
  image: (H, W, 3) uint8 as loaded from disk (HWC); models want (3, H, W) float (CHW)
  mask:  (H, W) integer class labels in [0, K)
"""
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def hwc_to_chw(img: np.ndarray):
    """(H, W, C) -> (C, H, W). One call, no copy needed."""
    return img.transpose((2, 0, 1))


def chw_to_hwc(img: np.ndarray):
    """(C, H, W) -> (H, W, C)."""
    return img.transpose((1, 2, 0))


def normalize(img: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """uint8 (H, W, 3) -> float32 (H, W, 3): scale to [0, 1], then (x - mean) / std
    per channel (broadcasting, no loops)."""
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    return x.astype(np.float32)


def denormalize(x: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Inverse of `normalize`, returning floats clipped to [0, 1] (ready for imshow)."""
    return np.clip((x * std) + mean, 0.0, 1.0)


def one_hot(mask: np.ndarray, K):
    """(H, W) int -> (K, H, W) float32 one-hot. No loops. Hint: np.eye or comparison with np.arange(K)[:, None, None]."""
    # mask: segmentation labels, with values in [0, k - 1]

    # ## version 1
    # new_mask = mask[np.newaxis, :] # (H, W) -> (1, H, W)
    # classes = np.arange(K)[:, np.newaxis, np.newaxis] # (K,) -> (K, 1, 1)
    # return (new_mask == classes).astype(np.float32)

    # ## version 2
    # return (mask[None] == np.arange(K)[:, None, None]).astype(np.float32)

    return np.eye(K, dtype=np.float32)[mask].transpose((2, 0, 1))


def class_pixel_counts(masks, K):
    """Total number of pixels of each class over a list (or array) of masks -> (K,) ints.
    Useful for spotting class imbalance before you train anything."""
    # return np.sum([np.bincount(mask.flatten(), minlength=K) for mask in masks], axis=0)
    return np.bincount(np.concatenate([mask.ravel() for mask in masks]), minlength=K)


def hflip_pair(img, mask):
    """Horizontally flip an (H, W, C) image and its (H, W) mask together."""
    # return img[:, ::-1], mask[:, ::-1]
    return np.flip(img, axis=1), np.flip(mask, axis=1)


def random_crop_pair(img, mask, size, rng):
    """Crop the SAME random (size x size) window from the image and the mask.
    rng is a np.random.Generator. Assumes H, W >= size."""
    h, w = mask.shape
    top = rng.integers(0, h - size + 1)
    left = rng.integers(0, w - size + 1)
    return img[top:top+size, left:left+size], mask[top:top+size, left:left+size]


def overlay(ax, img, mask, alpha=0.5):
    """Matplotlib: draw the image, then the mask on top with transparency, on the
    given Axes. Turn the axis ticks off. Returns ax.
    Hint: two ax.imshow calls; use interpolation='nearest' and a discrete cmap for the mask."""
    ax.imshow(img)
    ax.imshow(mask, alpha=alpha, cmap='tab10', vmin=0, vmax=9, interpolation="nearest")
    ax.axis("off")
    return ax