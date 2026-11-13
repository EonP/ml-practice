import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seg.s1_images import (
    chw_to_hwc,
    class_pixel_counts,
    denormalize,
    hflip_pair,
    hwc_to_chw,
    normalize,
    one_hot,
    overlay,
    random_crop_pair,
)


def test_layout_roundtrip(toy_data):
    img = toy_data[0][0]
    chw = hwc_to_chw(img)
    assert chw.shape == (3, 32, 32)
    np.testing.assert_array_equal(chw[0], img[..., 0])
    np.testing.assert_array_equal(chw_to_hwc(chw), img)


def test_normalize_roundtrip(toy_data):
    img = toy_data[0][0]
    x = normalize(img)
    assert x.dtype == np.float32 and x.shape == img.shape
    np.testing.assert_allclose(denormalize(x), img / 255.0, atol=1e-6)
    white = normalize(np.full((1, 1, 3), 255, np.uint8))[0, 0]
    np.testing.assert_allclose(white, (1 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]), rtol=1e-5)


def test_one_hot(toy_data):
    mask = toy_data[1][0]
    oh = one_hot(mask, 3)
    assert oh.shape == (3, 32, 32) and oh.dtype == np.float32
    np.testing.assert_array_equal(oh.sum(0), 1)
    np.testing.assert_array_equal(oh.argmax(0), mask)


def test_class_counts(toy_data):
    masks = toy_data[1]
    counts = class_pixel_counts(masks, 3)
    assert counts.shape == (3,) and counts.sum() == 12 * 32 * 32
    assert counts[1] == sum((m == 1).sum() for m in masks)


def test_flip_and_crop_keep_alignment(toy_data):
    img, mask = toy_data[0][0], toy_data[1][0]
    fi, fm = hflip_pair(img, mask)
    np.testing.assert_array_equal(fi[:, 0], img[:, -1])
    np.testing.assert_array_equal(fm[:, 0], mask[:, -1])
    ci, cm = random_crop_pair(img, mask, 16, np.random.default_rng(3))
    assert ci.shape == (16, 16, 3) and cm.shape == (16, 16)
    # the red pixels must still be exactly the class-1 pixels
    np.testing.assert_array_equal(ci[..., 0] == 220, cm == 1)


def test_overlay_draws_two_layers(toy_data):
    fig, ax = plt.subplots()
    out = overlay(ax, toy_data[0][0], toy_data[1][0])
    assert out is ax and len(ax.images) == 2
    assert 0 < ax.images[1].get_alpha() < 1
    plt.close(fig)
