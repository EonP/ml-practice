"""Oxford-IIIT Pet segmentation data (Day 6).

torchvision.datasets.OxfordIIITPet(root, split="trainval" or "test",
target_types="segmentation", download=True) yields (PIL RGB image, PIL trimap)
where the trimap pixels are 1 = pet, 2 = background, 3 = border.
Images have different sizes, so each pair is resized to a fixed square.
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

from seg.s4_dataset import SegDataset

N_CLASSES = 3
CLASS_NAMES = ["pet", "background", "border"]


def prepare_pair(pil_img, pil_trimap, size):
    """PIL image + PIL trimap -> (uint8 (size, size, 3) array, int64 (size, size) mask in {0, 1, 2}).

    Resize the image with Image.BILINEAR but the mask with Image.NEAREST
    (interpolating labels invents classes that do not exist), and subtract 1
    from the trimap so the classes start at 0."""
    raise NotImplementedError("TODO: prepare_pair")


class PetsSeg(Dataset):
    """Wraps OxfordIIITPet: download it under `root`, run `prepare_pair` on each
    item, and delegate augmentation + tensor conversion to a SegDataset-style
    pipeline. Returns (float32 (3, size, size), int64 (size, size)).

    Hint: the simplest correct version loads everything into lists once in
    __init__ (about 3.7k images, ~180 MB at 128 px) and wraps a SegDataset.
    """

    def __init__(self, root, split="trainval", size=128, transform=None):
        raise NotImplementedError("TODO: __init__")

    def __len__(self):
        raise NotImplementedError("TODO: __len__")

    def __getitem__(self, i):
        raise NotImplementedError("TODO: __getitem__")
