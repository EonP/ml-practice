import torch

from torch_drills.t03_autograd import accumulated_grad, grad_at, hvp, numerical_grad, straight_through_round

f64 = torch.float64


def test_grad_at():
    x = torch.randn(3, 4, dtype=f64)
    g = grad_at(lambda t: (t**3).sum() + t[0, 1] * t[2, 3], x)
    expected = 3 * x**2
    expected[0, 1] += x[2, 3]
    expected[2, 3] += x[0, 1]
    torch.testing.assert_close(g, expected)
    assert not x.requires_grad and x.grad is None, "work on a detached copy of x"


def test_numerical_grad_matches_autograd():
    x = torch.randn(2, 3, dtype=f64)
    x0 = x.clone()
    f = lambda t: (t.sin() * t.exp()).sum() + t.sum() ** 3  # every partial depends on every entry
    num = numerical_grad(f, x)
    assert num.shape == x.shape
    xr = x.clone().requires_grad_()
    f(xr).backward()
    torch.testing.assert_close(num, xr.grad, atol=1e-6, rtol=1e-6)
    # with a larger eps, forgetting to put each entry back before moving on shows up clearly
    torch.testing.assert_close(numerical_grad(f, x, eps=1e-4), xr.grad, atol=1e-6, rtol=1e-6)
    assert torch.equal(x, x0), "x changed (x += eps followed by x -= eps can leave rounding errors)"


def test_accumulated_grad_matches_full_batch():
    torch.manual_seed(0)
    X, Y = torch.randn(10, 3, dtype=f64), torch.randn(10, dtype=f64)
    loss_fn = lambda w, x, y: ((x @ w - y) ** 2).mean()
    w = torch.randn(3, dtype=f64, requires_grad=True)
    w.grad = torch.full((3,), 100.0, dtype=f64)  # stale gradient from an earlier step
    batches = [(X[:2], Y[:2]), (X[2:9], Y[2:9]), (X[9:], Y[9:])]
    g = accumulated_grad(w, batches, loss_fn)
    full = torch.autograd.grad(loss_fn(w, X, Y), w)[0]
    torch.testing.assert_close(g, full)
    torch.testing.assert_close(w.grad, full)


def test_straight_through_round():
    x = torch.tensor([0.2, 1.7, -2.6, 3.4], requires_grad=True)
    y = straight_through_round(x)
    torch.testing.assert_close(y.detach(), torch.round(x.detach()))
    (y * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_hvp():
    torch.manual_seed(0)
    A = torch.randn(4, 4, dtype=f64)
    A = A + A.T
    f = lambda t: 0.5 * t @ A @ t + t.sin().sum()
    x, v = torch.randn(4, dtype=f64), torch.randn(4, dtype=f64)
    out = hvp(f, x, v)
    torch.testing.assert_close(out, A @ v - x.sin() * v)
    assert not x.requires_grad and x.grad is None


def test_no_shortcuts(no_shortcuts):
    no_shortcuts(grad_at, numerical_grad, accumulated_grad, straight_through_round)
    no_shortcuts(hvp, ban=("functional", "hessian", "jacobian"))
