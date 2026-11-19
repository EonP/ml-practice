import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torch_drills.t12_true_or_false import STATEMENTS


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def _ptr(t):
    return t.untyped_storage().data_ptr()


def _w():
    return torch.ones(2, requires_grad=True)


def _from_numpy():
    a = np.zeros(3, dtype=np.float32)
    torch.from_numpy(a)[0] = 1
    return a[0] == 1


def _tensor_from_numpy():
    a = np.zeros(3, dtype=np.float32)
    torch.tensor(a)[0] = 1
    return a[0] == 1


def _numpy_of_tensor():
    x = torch.zeros(3)
    x.numpy()[0] = 1
    return x[0].item() == 1


def _backward_twice():
    w = _w()
    loss = (w * 3).sum()
    loss.backward(retain_graph=True)
    loss.backward()
    return torch.equal(w.grad, torch.full((2,), 6.0))


def _clone_cuts_graph():
    w = _w()
    w.clone().sum().backward()
    return w.grad is None


def _total_keeps_graph():
    w, total = _w(), 0
    for _ in range(3):
        total = total + (w * 2).sum()
    return total.grad_fn is not None


def _no_grad_in_no_grad():
    w = _w()
    with torch.no_grad():
        y = w * 2
    return y.requires_grad


def _eval_disables_grad():
    return not nn.Linear(2, 1).eval()(torch.ones(1, 2)).requires_grad


def _no_grad_disables_dropout():
    torch.manual_seed(0)
    x = torch.ones(1000)
    with torch.no_grad():
        y = nn.Dropout(0.5).train()(x)
    return torch.equal(y, x)


class _ListModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 2)]


class _BufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("stat", torch.zeros(2))


def _state_dict_copies():
    model = nn.Linear(2, 1)
    sd = model.state_dict()
    with torch.no_grad():
        model.weight.add_(1.0)
    return not torch.equal(sd["weight"], model.weight)


def _zero_grad_none():
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()
    opt.zero_grad()
    return all(p.grad is None for p in model.parameters())


def _ce_expects_probabilities():
    torch.manual_seed(0)
    p, y = torch.randn(4, 3).softmax(1), torch.tensor([0, 1, 2, 0])
    as_if_probabilities = -p.log().gather(1, y[:, None]).mean()
    return torch.allclose(F.cross_entropy(p, y), as_if_probabilities)


def _seeded_twice():
    torch.manual_seed(0)
    return torch.equal(torch.randn(3), torch.randn(3))


CHECKS = {
    "torch.from_numpy(a) shares memory with the NumPy array a": _from_numpy,
    "torch.tensor(a) shares memory with the NumPy array a": _tensor_from_numpy,
    "x.numpy() shares memory with x": _numpy_of_tensor,
    "x.detach() shares memory with x": lambda: (lambda x: _ptr(x.detach()) == _ptr(x))(_w()),
    "x.T.view(-1) works for x of shape (3, 4)": lambda: not _raises(lambda: torch.zeros(3, 4).T.view(-1)),
    "x.T.reshape(-1) shares memory with x, for x of shape (3, 4)": lambda: (lambda x: _ptr(x.T.reshape(-1)) == _ptr(x))(torch.zeros(3, 4)),
    "x.contiguous() makes a copy even when x is already contiguous": lambda: (lambda x: _ptr(x.contiguous()) != _ptr(x))(torch.zeros(3, 4)),
    "x.float() returns x itself (the same object) when x is already float32": lambda: (lambda x: x.float() is x)(torch.zeros(3)),
    "x.expand(4, 3) copies the data 4 times, for x of shape (1, 3)": lambda: (lambda x: _ptr(x.expand(4, 3)) != _ptr(x))(torch.zeros(1, 3)),
    "torch.zeros(1, 3, 1).squeeze() has shape (3,)": lambda: torch.zeros(1, 3, 1).squeeze().shape == (3,),
    "w * 2 has requires_grad=True": lambda: (_w() * 2).requires_grad,
    "w * 2 computed inside torch.no_grad() has requires_grad=True": _no_grad_in_no_grad,
    "Calling backward() twice (with retain_graph=True) without zeroing doubles w.grad": _backward_twice,
    "x.clone() of a tensor that requires grad cuts the graph: no gradient flows back through it": _clone_cuts_graph,
    "torch.argmax(w) has requires_grad=True, so you can backpropagate through it": lambda: torch.argmax(torch.randn(4, requires_grad=True)).requires_grad,
    "torch.tensor([1, 2], requires_grad=True) works": lambda: not _raises(lambda: torch.tensor([1, 2], requires_grad=True)),
    "After total = total + loss (rather than loss.item()), total still holds the graph (it has a grad_fn)": _total_keeps_graph,
    "model.eval() turns off gradient tracking": _eval_disables_grad,
    "torch.no_grad() makes dropout layers stop dropping": _no_grad_disables_dropout,
    "Layers kept in a plain Python list attribute are included in model.parameters()": lambda: len(list(_ListModel().parameters())) > 0,
    "A tensor saved with register_buffer appears in model.parameters()": lambda: len(list(_BufferModel().parameters())) > 0,
    "A tensor saved with register_buffer appears in model.state_dict()": lambda: "stat" in _BufferModel().state_dict(),
    "model.state_dict() returns copies, so training afterwards doesn't change it": _state_dict_copies,
    "optimizer.zero_grad() sets every .grad to None (by default)": _zero_grad_none,
    "model.parameters() returns a list, so you can loop over it twice": lambda: isinstance(nn.Linear(2, 1).parameters(), list),
    "F.cross_entropy expects probabilities (softmax already applied) as its input": _ce_expects_probabilities,
    "F.cross_entropy accepts float class probabilities of shape (N, K) as the target": lambda: not _raises(lambda: F.cross_entropy(torch.randn(4, 3), torch.rand(4, 3).softmax(1))),
    "F.cross_entropy raises an error if the class-index target is int32 instead of int64": lambda: _raises(lambda: F.cross_entropy(torch.randn(4, 3), torch.tensor([0, 1, 2, 0], dtype=torch.int32))),
    "nn.BatchNorm1d in training mode raises an error on a batch of one sample": lambda: _raises(lambda: nn.BatchNorm1d(3).train()(torch.randn(1, 3))),
    "After one torch.manual_seed(0), two calls to torch.randn(3) return the same values": _seeded_twice,
}


def test_every_prediction_is_filled_in():
    assert set(STATEMENTS) == set(CHECKS), "only replace the None values; don't edit the statements"
    blanks = [s for s, v in STATEMENTS.items() if not isinstance(v, bool)]
    assert not blanks, f"replace None with True or False for: {blanks}"


@pytest.mark.parametrize("statement", list(CHECKS))
def test_statement(statement):
    guess = STATEMENTS.get(statement)
    assert isinstance(guess, bool), "fill in True or False first"
    actual = bool(CHECKS[statement]())
    assert guess == actual, f"{'TRUE' if actual else 'FALSE'}: {statement}"
