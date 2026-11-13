import warnings

import numpy as np
import torch
import torch.nn.functional as F

from numpy_drills.n07_boolean_masks import clip_outliers, count_in_range, masked_mean_loss, positives

rng = np.random.default_rng(0)


def test_positives():
    x = np.array([[1.0, -2.0, 0.0], [3.0, 0.5, -1.0]])
    np.testing.assert_array_equal(positives(x), [1.0, 3.0, 0.5])
    y = rng.normal(size=(3, 4, 5))
    np.testing.assert_array_equal(positives(y), [v for v in y.ravel() if v > 0])


def test_count_in_range():
    x = np.arange(10).reshape(2, 5)
    out = count_in_range(x, 3, 7)
    assert out == 4 and type(out) is int, f"expected the Python int 4, got {out!r}"
    assert count_in_range(x, 20, 30) == 0
    y = rng.normal(size=(50, 4))
    assert count_in_range(y, -0.5, 1.0) == sum(1 for v in y.ravel() if -0.5 <= v < 1.0)


def test_clip_outliers():
    x = rng.normal(size=(20, 5))
    x[3, 2], x[7, 0] = 50.0, -40.0
    before = x.copy()
    out = clip_outliers(x, 2.0)
    np.testing.assert_array_equal(x, before, err_msg="x must not be modified")
    m, s = x.mean(), x.std()
    expected = np.array([v if abs(v - m) <= 2 * s else m for v in x.ravel()]).reshape(x.shape)
    np.testing.assert_allclose(out, expected)


def test_masked_mean_loss_matches_torch_ignore_index():
    logits = rng.normal(size=(2, 4, 5, 6))
    target = rng.integers(0, 4, size=(2, 5, 6))
    target[0, :2] = 255
    lt, tt = torch.tensor(logits), torch.tensor(target)
    loss = F.cross_entropy(lt, tt, ignore_index=255, reduction="none").numpy()
    expected = F.cross_entropy(lt, tt, ignore_index=255).item()
    out = masked_mean_loss(loss, target, 255)
    assert isinstance(out, float)
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_masked_mean_loss_all_ignored():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = masked_mean_loss(np.ones((1, 2, 2)), np.full((1, 2, 2), 255), 255)
    assert out == 0.0


def test_no_python_loops(no_loops):
    no_loops(positives, count_in_range, clip_outliers, masked_mean_loss)
