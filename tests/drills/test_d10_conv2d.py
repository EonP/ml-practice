import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from drills.d10_conv2d import conv2d_fast, conv2d_naive, maxpool2d

rng = np.random.default_rng(0)
CASES = [(1, 0), (1, 1), (2, 1), (2, 0)]


def _ref(x, w, b, stride, padding):
    return F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=stride, padding=padding).numpy()


@pytest.mark.parametrize("stride,padding", CASES)
def test_naive_matches_torch(stride, padding):
    x, w, b = rng.normal(size=(2, 3, 9, 8)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    np.testing.assert_allclose(conv2d_naive(x, w, b, stride, padding), _ref(x, w, b, stride, padding), atol=1e-10)


@pytest.mark.parametrize("stride,padding", CASES)
def test_fast_matches_torch(stride, padding):
    x, w, b = rng.normal(size=(2, 3, 9, 8)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    np.testing.assert_allclose(conv2d_fast(x, w, b, stride, padding), _ref(x, w, b, stride, padding), atol=1e-10)


def test_fast_non_square_kernel():
    x, w, b = rng.normal(size=(1, 2, 7, 7)), rng.normal(size=(3, 2, 1, 3)), rng.normal(size=3)
    np.testing.assert_allclose(conv2d_fast(x, w, b), _ref(x, w, b, 1, 0), atol=1e-10)


def test_fast_is_actually_fast():
    x, w, b = rng.normal(size=(16, 16, 64, 64)), rng.normal(size=(32, 16, 3, 3)), rng.normal(size=32)
    t = time.perf_counter()
    conv2d_fast(x, w, b, 1, 1)
    assert time.perf_counter() - t < 3.0, "too slow: are there Python loops over N, H or W?"


def test_maxpool_matches_torch():
    x = rng.normal(size=(2, 3, 8, 6))
    np.testing.assert_allclose(maxpool2d(x), F.max_pool2d(torch.tensor(x), 2, 2).numpy())
    np.testing.assert_allclose(maxpool2d(x, 3, 1), F.max_pool2d(torch.tensor(x), 3, 1).numpy())
