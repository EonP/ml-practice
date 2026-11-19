import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_drills.t07_training_loop import evaluate, fit, overfit_one_batch, train_one_epoch


def _data(n=37, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 5, generator=g)
    y = (X[:, 0] > 0).long() + (X[:, 1] > 0.5).long()  # 3 classes
    return X, y


def _loader(X, y, batch_size=8):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)  # 37 samples: last batch has 5


def test_train_one_epoch_matches_reference():
    X, y = _data()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 3))
    ref = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    model.eval()
    out = train_one_epoch(model, _loader(X, y), nn.CrossEntropyLoss(), opt)
    assert isinstance(out, float)
    assert model.training, "call model.train()"
    total = 0.0
    for xb, yb in _loader(X, y):
        ref_opt.zero_grad()
        loss = F.cross_entropy(ref(xb), yb)
        loss.backward()
        ref_opt.step()
        total += loss.item() * len(xb)
    for p, q in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(p, q, msg="parameters differ from a correct loop: check zero_grad and the order")
    assert out == pytest.approx(total / len(X)), "average over samples, not over batches"


class Spy(nn.Module):
    """A linear layer with dropout that records (grad enabled?, training?) on every call."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 3)
        self.drop = nn.Dropout(0.5)
        self.calls = []

    def forward(self, x):
        self.calls.append((torch.is_grad_enabled(), self.training))
        return self.drop(self.lin(x))


def test_evaluate():
    X, y = _data()
    torch.manual_seed(0)
    model = Spy().train()
    loss, acc = evaluate(model, _loader(X, y), nn.CrossEntropyLoss())
    assert isinstance(loss, float) and isinstance(acc, float)
    assert model.calls, "evaluate never called the model"
    assert all(grad_on is False for grad_on, _ in model.calls), "use torch.no_grad()"
    assert all(training is False for _, training in model.calls), "use model.eval()"
    with torch.no_grad():
        logits = model.lin(X)
    assert loss == pytest.approx(F.cross_entropy(logits, y).item())
    assert acc == pytest.approx((logits.argmax(1) == y).float().mean().item())
    assert all(p.grad is None for p in model.parameters())


def test_overfit_one_batch():
    torch.manual_seed(0)
    X, y = _data(n=16)
    model = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 3))
    losses = overfit_one_batch(model, X, y, nn.CrossEntropyLoss(), steps=300, lr=1e-2)
    assert len(losses) == 300 and all(isinstance(l, float) for l in losses)
    assert losses[-1] < 0.05 * losses[0]


def _toy_regression():
    """One weight w, starting at 0. Training pulls w towards 2, validation is best at w = 1.
    With SGD(lr=0.1) on MSE, w goes 0 -> 0.4 -> 0.72 -> 0.976 -> 1.1808 -> 1.34464 -> ..."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    train = DataLoader(TensorDataset(torch.ones(1, 1), torch.full((1, 1), 2.0)), batch_size=1)
    val = DataLoader(TensorDataset(torch.ones(1, 1), torch.ones(1, 1)), batch_size=1)
    return model, train, val, torch.optim.SGD(model.parameters(), lr=0.1)


def test_fit_early_stopping_restores_best_weights():
    model, train, val, opt = _toy_regression()
    hist = fit(model, train, val, nn.MSELoss(), opt, max_epochs=50, patience=2)
    assert set(hist) == {"train_loss", "val_loss"}
    assert len(hist["val_loss"]) == 5, "epochs 4 and 5 don't improve on epoch 3, so stop after epoch 5"
    assert hist["train_loss"][0] == pytest.approx(4.0)
    assert hist["val_loss"][:3] == pytest.approx([0.36, 0.0784, 0.000576], abs=1e-5)
    assert model.weight.item() == pytest.approx(0.976, abs=1e-5), (
        "load the best epoch's weights back: state_dict() holds references to the live tensors, so copy it"
    )


def test_fit_stops_at_max_epochs():
    model, train, val, opt = _toy_regression()
    hist = fit(model, train, val, nn.MSELoss(), opt, max_epochs=3, patience=10)
    assert len(hist["train_loss"]) == len(hist["val_loss"]) == 3
    assert model.weight.item() == pytest.approx(0.976, abs=1e-5)


def test_no_stubs_left(no_shortcuts):
    no_shortcuts(train_one_epoch, evaluate, overfit_one_batch, fit)
