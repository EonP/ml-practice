import numpy as np
import torch
import torch.nn.functional as F

from drills.d08_mlp_backprop import backward, forward, init_params, train


def spiral(n=100, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.0, 1, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(0, 0.2, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n, c))
    return np.concatenate(X), np.concatenate(y)


def test_forward_shape():
    p = init_params(2, 16, 3)
    logits, _ = forward(p, np.zeros((5, 2)))
    assert logits.shape == (5, 3)


def test_grads_match_autograd():
    X, y = spiral(20)
    p = init_params(2, 16, 3, seed=1)
    logits, cache = forward(p, X)
    loss, grads = backward(p, cache, y)

    t = {k: torch.tensor(v, requires_grad=True) for k, v in p.items()}
    out = torch.relu(torch.tensor(X) @ t["W1"] + t["b1"]) @ t["W2"] + t["b2"]
    ref = F.cross_entropy(out, torch.tensor(y))
    ref.backward()

    np.testing.assert_allclose(loss, ref.item(), rtol=1e-10)
    for k in p:
        assert grads[k].shape == p[k].shape, k
        np.testing.assert_allclose(grads[k], t[k].grad.numpy(), atol=1e-10, err_msg=k)


def test_trains_on_spiral():
    X, y = spiral()
    params, losses = train(X, y, H=64, lr=1.0, n_steps=3000)
    acc = (forward(params, X)[0].argmax(1) == y).mean()
    assert losses[-1] < losses[0]
    assert acc > 0.95
