import numpy as np
import torch
from torch.utils.data import DataLoader

from seg.s1_images import denormalize
from seg.s4_dataset import JointTransform, SegDataset


def test_dataset_types_and_shapes(toy_data):
    ds = SegDataset(*toy_data)
    assert len(ds) == 12
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert y.shape == (32, 32) and y.dtype == torch.int64
    np.testing.assert_array_equal(y.numpy(), toy_data[1][0])


def test_joint_transform_keeps_image_and_mask_aligned(toy_data):
    ds = SegDataset(*toy_data, transform=JointTransform(crop_size=20, hflip_p=0.5, seed=0))
    for i in range(len(ds)):
        for _ in range(5):
            x, y = ds[i]
            assert x.shape == (3, 20, 20) and y.shape == (20, 20)
            img = denormalize(x.permute(1, 2, 0).numpy())
            red = img[..., 0] > 0.8
            np.testing.assert_array_equal(red, y.numpy() == 1, err_msg="image and mask were transformed differently")


def test_flip_actually_happens(toy_data):
    t = JointTransform(hflip_p=1.0, seed=0)
    img, mask = toy_data[0][0], toy_data[1][0]
    fi, fm = t(img, mask)
    np.testing.assert_array_equal(fm, mask[:, ::-1])
    t = JointTransform(hflip_p=0.0, seed=0)
    np.testing.assert_array_equal(t(img, mask)[1], mask)


def test_dataloader_batches(toy_data):
    ds = SegDataset(*toy_data, transform=JointTransform(crop_size=16, seed=1))
    x, y = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
    assert x.shape == (4, 3, 16, 16) and y.shape == (4, 16, 16)
