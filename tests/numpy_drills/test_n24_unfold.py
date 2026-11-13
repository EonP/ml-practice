import numpy as np
import pytest
import torch
import torch.nn.functional as F

from numpy_drills.n24_unfold import crop_patches, unfold

rng = np.random.default_rng(0)


@pytest.mark.parametrize("k,stride,padding", [(3, 1, 0), (3, 2, 0), (3, 1, 1), (2, 2, 0), (3, 2, 1), (5, 3, 2), (1, 1, 0)])
def test_unfold_matches_torch(k, stride, padding):
    x = rng.normal(size=(2, 3, 9, 8))
    expected = F.unfold(torch.tensor(x), k, stride=stride, padding=padding).numpy()
    out = unfold(x, k, stride, padding)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


def test_crop_patches():
    img = rng.normal(size=(7, 9))
    centers = np.array([[3, 4], [0, 0], [6, 8], [1, 7], [6, 0]])
    for size in (1, 3, 5):
        out = crop_patches(img, centers, size)
        assert out.shape == (len(centers), size, size)
        r = size // 2
        for k, (cy, cx) in enumerate(centers):
            expected = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    y, x = cy - r + i, cx - r + j
                    if 0 <= y < 7 and 0 <= x < 9:
                        expected[i, j] = img[y, x]
            np.testing.assert_array_equal(out[k], expected, err_msg=f"size={size}, centre={(cy, cx)}")


def test_no_python_loops(no_loops):
    no_loops(unfold, crop_patches)
