"""Stage 4 (Day 4): Dataset and joint augmentation.

The #1 segmentation bug: a random transform applied to the image but not the
mask (or with different random parameters). These classes make that impossible.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from seg.s1_images import IMAGENET_MEAN, IMAGENET_STD, hflip_pair, normalize, random_crop_pair


class JointTransform:
    """Callable applied to a NumPy (image (H, W, 3) uint8, mask (H, W)) pair.

    __init__(crop_size=None, hflip_p=0.5, seed=None): store the settings and
    create self.rng = np.random.default_rng(seed).
    __call__(img, mask): if crop_size is set, random-crop both (same window);
    then with probability hflip_p flip both. Returns (img, mask).
    Reuse the stage 1 helpers.
    """

    def __init__(self, crop_size=None, hflip_p=0.5, seed=None):
        raise NotImplementedError("TODO: __init__")

    def __call__(self, img, mask):
        raise NotImplementedError("TODO: __call__")


class SegDataset(Dataset):
    """In-memory segmentation dataset.

    __init__(images, masks, transform=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
      images: list of uint8 (H, W, 3) arrays, masks: list of (H, W) int arrays.
    __len__: number of samples.
    __getitem__(i): apply `transform` (if any) to the pair, normalise the image,
      and return (image float32 tensor (3, H, W), mask int64 tensor (H, W)).
      Careful: flipped arrays have negative strides and torch.from_numpy
      rejects them. np.ascontiguousarray fixes that.
    """

    def __init__(self, images, masks, transform=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        raise NotImplementedError("TODO: __init__")

    def __len__(self):
        raise NotImplementedError("TODO: __len__")

    def __getitem__(self, i):
        raise NotImplementedError("TODO: __getitem__")
