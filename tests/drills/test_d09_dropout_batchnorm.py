import numpy as np
import torch

from drills.d09_dropout_batchnorm import (
    batchnorm_backward,
    batchnorm_eval,
    batchnorm_forward,
    dropout,
    update_running_stats,
)

rng = np.random.default_rng(0)


def test_dropout_eval_is_identity():
    x = rng.normal(size=(10, 10))
    out, mask = dropout(x, 0.5, train=False, rng=rng)
    np.testing.assert_array_equal(out, x)
    assert mask is None


def test_dropout_train_keeps_expectation():
    x = np.ones((1000, 100))
    out, mask = dropout(x, 0.3, train=True, rng=np.random.default_rng(1))
    assert mask.dtype == bool and mask.shape == x.shape
    np.testing.assert_allclose(out[mask], 1 / 0.7)
    assert (out[~mask] == 0).all()
    np.testing.assert_allclose(out.mean(), 1.0, atol=0.02)


def _torch_bn(gamma, beta):
    bn = torch.nn.BatchNorm1d(len(gamma), eps=1e-5, momentum=0.1).double()
    with torch.no_grad():
        bn.weight.copy_(torch.tensor(gamma))
        bn.bias.copy_(torch.tensor(beta))
    return bn


def test_batchnorm_forward_and_backward_match_torch():
    x, gamma, beta = rng.normal(2, 3, size=(16, 5)), rng.normal(size=5), rng.normal(size=5)
    dout = rng.normal(size=(16, 5))
    bn = _torch_bn(gamma, beta).train()
    xt = torch.tensor(x, requires_grad=True)
    ref = bn(xt)
    ref.backward(torch.tensor(dout))

    out, cache = batchnorm_forward(x, gamma, beta)
    np.testing.assert_allclose(out, ref.detach().numpy(), atol=1e-8)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    np.testing.assert_allclose(dx, xt.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(dgamma, bn.weight.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(dbeta, bn.bias.grad.numpy(), atol=1e-8)


def test_running_stats_and_eval_match_torch():
    gamma, beta = rng.normal(size=4), rng.normal(size=4)
    bn = _torch_bn(gamma, beta).train()
    rm, rv = np.zeros(4), np.ones(4)
    for _ in range(3):
        x = rng.normal(1, 2, size=(8, 4))
        bn(torch.tensor(x))
        rm, rv = update_running_stats(rm, rv, x)
    np.testing.assert_allclose(rm, bn.running_mean.numpy(), atol=1e-10)
    np.testing.assert_allclose(rv, bn.running_var.numpy(), atol=1e-10)

    x = rng.normal(size=(5, 4))
    bn.eval()
    np.testing.assert_allclose(batchnorm_eval(x, gamma, beta, rm, rv), bn(torch.tensor(x)).detach().numpy(), atol=1e-8)
