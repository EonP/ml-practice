import pytest
import torch
from torch import nn

from torch_drills.t08_optimizers import SGD, Adam, AdamW

f64 = torch.float64


def _model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)).double()
    model.unused = nn.Parameter(torch.ones(3, dtype=f64))  # never gets a gradient
    return model


def _train(model, opt, steps):
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(16, 4, generator=g, dtype=f64)
        y = torch.randn(16, 2, generator=g, dtype=f64)
        opt.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        opt.step()


def _compare(mine, ref, steps):
    a, b = _model(), _model()
    _train(a, mine(a.parameters()), steps)  # passed as a generator, like model.parameters() usually is
    _train(b, ref(b.parameters()), steps)
    for (name, p), q in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(p, q, msg=f"{name} differs from torch.optim after {steps} steps")
    torch.testing.assert_close(a.unused, torch.ones(3, dtype=f64), msg="skip parameters whose .grad is None")


@pytest.mark.parametrize(
    "kw", [dict(lr=0.1), dict(lr=0.1, momentum=0.9), dict(lr=0.05, momentum=0.8, weight_decay=0.01)]
)
def test_sgd_matches_torch(kw):
    _compare(lambda ps: SGD(ps, **kw), lambda ps: torch.optim.SGD(ps, **kw), steps=5)


def test_zero_grad_sets_none():
    model = _model()
    model(torch.randn(2, 4, dtype=f64)).sum().backward()
    for opt in (SGD(model.parameters(), lr=0.1), Adam(model.parameters()), AdamW(model.parameters())):
        model(torch.randn(2, 4, dtype=f64)).sum().backward()
        opt.zero_grad()
        assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("kw", [dict(), dict(lr=0.01, betas=(0.8, 0.99)), dict(lr=0.01, weight_decay=0.1)])
def test_adam_matches_torch(kw):
    _compare(lambda ps: Adam(ps, **kw), lambda ps: torch.optim.Adam(ps, **kw), steps=10)


@pytest.mark.parametrize("kw", [dict(), dict(lr=0.01, weight_decay=0.1)])
def test_adamw_matches_torch(kw):
    _compare(lambda ps: AdamW(ps, **kw), lambda ps: torch.optim.AdamW(ps, **kw), steps=10)


def test_no_torch_optim(no_shortcuts):
    no_shortcuts(SGD, Adam, AdamW, ban=("optim",))
