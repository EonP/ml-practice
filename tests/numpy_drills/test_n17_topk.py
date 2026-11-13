import numpy as np
import torch

from numpy_drills.n17_topk import topk, topk_accuracy, topk_mask

rng = np.random.default_rng(0)


def test_topk_matches_torch():
    x = rng.normal(size=(6, 10))
    for k in (1, 3, 10):
        vals, idx = topk(x, k)
        tv, ti = torch.topk(torch.tensor(x), k, dim=1)
        assert vals.shape == idx.shape == (6, k), f"k={k}"
        np.testing.assert_array_equal(idx, ti.numpy(), err_msg=f"k={k}")
        np.testing.assert_array_equal(vals, tv.numpy(), err_msg=f"k={k}")


def test_topk_mask():
    x = rng.normal(size=(5, 8))
    for k in (1, 3, 8):
        m = topk_mask(x, k)
        assert m.dtype == bool and m.shape == x.shape
        expected = np.zeros_like(m)
        for i, row in enumerate(torch.topk(torch.tensor(x), k, dim=1).indices.numpy()):
            expected[i, row] = True
        np.testing.assert_array_equal(m, expected, err_msg=f"k={k}")


def test_topk_accuracy():
    logits, y = rng.normal(size=(200, 10)), rng.integers(0, 10, size=200)
    for k in (1, 5, 10):
        out = topk_accuracy(logits, y, k)
        expected = np.mean([y[i] in np.argsort(-logits[i])[:k] for i in range(200)])
        assert isinstance(out, float)
        np.testing.assert_allclose(out, expected, err_msg=f"k={k}")


def test_no_python_loops(no_loops):
    no_loops(topk, topk_mask, topk_accuracy)
