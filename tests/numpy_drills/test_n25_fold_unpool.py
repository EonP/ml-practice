import numpy as np
import pytest
import torch
import torch.nn.functional as F

from numpy_drills.n25_fold_unpool import fold, max_unpool, maxpool_with_indices

rng = np.random.default_rng(0)


@pytest.mark.parametrize("k,stride", [(3, 1), (2, 2), (3, 2), (4, 3), (1, 1)])
def test_fold_matches_torch(k, stride):
    N, C, H, W = 2, 3, 9, 10
    L = ((H - k) // stride + 1) * ((W - k) // stride + 1)
    cols = rng.normal(size=(N, C * k * k, L))
    expected = F.fold(torch.tensor(cols), (H, W), k, stride=stride).numpy()
    np.testing.assert_allclose(fold(cols, H, W, k, stride), expected, atol=1e-12)


def test_fold_of_ones_counts_overlaps():
    out = fold(np.ones((1, 9, 9)), 5, 5, 3)[0, 0]
    np.testing.assert_array_equal(out, [[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]])


@pytest.mark.parametrize("H,W,k", [(8, 6, 2), (9, 7, 2), (9, 9, 3)])
def test_maxpool_with_indices_matches_torch(H, W, k):
    x = rng.normal(size=(2, 3, H, W))
    values, indices = maxpool_with_indices(x, k)
    tv, ti = F.max_pool2d(torch.tensor(x), k, return_indices=True)
    np.testing.assert_array_equal(values, tv.numpy())
    np.testing.assert_array_equal(indices, ti.numpy())
    assert indices.dtype == np.int64


def test_max_unpool_matches_torch():
    x = rng.normal(size=(2, 3, 8, 6))
    tv, ti = F.max_pool2d(torch.tensor(x), 2, return_indices=True)
    expected = F.max_unpool2d(tv, ti, 2, output_size=(8, 6)).numpy()
    np.testing.assert_array_equal(max_unpool(tv.numpy(), ti.numpy(), 8, 6), expected)


def test_no_python_loops(no_loops):
    no_loops(fold, maxpool_with_indices, max_unpool)
