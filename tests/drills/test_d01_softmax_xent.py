import numpy as np
import torch
import torch.nn.functional as F

from drills.d01_softmax_xent import cross_entropy, cross_entropy_grad, log_softmax, softmax

rng = np.random.default_rng(0)


def test_softmax_matches_torch():
    z = rng.normal(size=(4, 7))
    np.testing.assert_allclose(softmax(z), F.softmax(torch.tensor(z), -1).numpy(), atol=1e-10)


def test_softmax_other_axis():
    z = rng.normal(size=(3, 5, 2))
    np.testing.assert_allclose(softmax(z, axis=1), F.softmax(torch.tensor(z), 1).numpy(), atol=1e-10)


def test_softmax_is_stable():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    out = softmax(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(), 1.0)


def test_log_softmax_stable_for_very_negative_logits():
    z = np.array([[0.0, -1000.0]])
    out = log_softmax(z)
    assert np.all(np.isfinite(out)), "np.log(softmax(z)) gives -inf here: use log-sum-exp"
    np.testing.assert_allclose(out, [[0.0, -1000.0]], atol=1e-8)


def test_cross_entropy_matches_torch():
    z = rng.normal(size=(8, 5)) * 3
    y = rng.integers(0, 5, size=8)
    expected = F.cross_entropy(torch.tensor(z), torch.tensor(y)).item()
    assert isinstance(cross_entropy(z, y), float)
    np.testing.assert_allclose(cross_entropy(z, y), expected, rtol=1e-10)


def test_cross_entropy_grad_matches_autograd():
    z = rng.normal(size=(6, 4))
    y = rng.integers(0, 4, size=6)
    zt = torch.tensor(z, requires_grad=True)
    F.cross_entropy(zt, torch.tensor(y)).backward()
    np.testing.assert_allclose(cross_entropy_grad(z, y), zt.grad.numpy(), atol=1e-10)
