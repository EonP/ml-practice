import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from torch_drills.t10_custom_autograd import GradReverse, LinearFunction, Sigmoid, StraightThroughClamp

f64 = torch.float64


def test_sigmoid():
    torch.manual_seed(0)
    x = torch.randn(5, 3, dtype=f64, requires_grad=True)
    torch.testing.assert_close(Sigmoid.apply(x), torch.sigmoid(x))
    assert gradcheck(Sigmoid.apply, (x,))


def test_linear_function():
    torch.manual_seed(0)
    x = torch.randn(4, 3, dtype=f64, requires_grad=True)
    W = torch.randn(2, 3, dtype=f64, requires_grad=True)
    b = torch.randn(2, dtype=f64, requires_grad=True)
    torch.testing.assert_close(LinearFunction.apply(x, W, b), F.linear(x, W, b))
    assert gradcheck(LinearFunction.apply, (x, W, b))


def test_straight_through_clamp():
    x = torch.tensor([-3.0, -0.5, 0.2, 2.5], requires_grad=True)
    y = StraightThroughClamp.apply(x, -1.0, 1.0)
    torch.testing.assert_close(y.detach(), torch.tensor([-1.0, -0.5, 0.2, 1.0]))
    (y * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_grad_reverse():
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    y = GradReverse.apply(x, 0.5)
    torch.testing.assert_close(y.detach(), x.detach())
    (y * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([-0.5, -1.0, -1.5]))


def test_no_autograd_inside(no_shortcuts):
    no_shortcuts(Sigmoid, LinearFunction, StraightThroughClamp, GradReverse, ban=("autograd", "enable_grad"))
