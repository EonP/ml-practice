import math

import pytest
import torch
from torch import nn

from torch_drills.t11_init_hooks_freezing import capture_activations, freeze_except, init_weights_, param_groups


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.LayerNorm(16 * 4 * 4), nn.Linear(16 * 4 * 4, 10))

    def forward(self, x):  # x (N, 3, 4, 4)
        return self.head(self.encoder(x))


def _no_hooks_left(model):
    return all(len(m._forward_hooks) == 0 for m in model.modules())


def test_init_weights_statistics():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Conv2d(64, 64, 3))
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)
    assert init_weights_(model) is None
    lin, conv = model[0], model[2]
    assert lin.weight.std().item() == pytest.approx(math.sqrt(2 / 1000), rel=0.05)
    assert conv.weight.std().item() == pytest.approx(math.sqrt(2 / (64 * 9)), rel=0.05), "fan_in of a conv is C_in * kH * kW"
    assert abs(lin.weight.mean().item()) < 1e-3
    assert (lin.bias == 0).all() and (conv.bias == 0).all()


def test_init_weights_norm_layers_and_missing_bias():
    model = Net()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)
    init_weights_(model)
    bn, ln = model.encoder[1], model.head[1]
    assert (bn.weight == 1).all() and (bn.bias == 0).all()
    assert (ln.weight == 1).all() and (ln.bias == 0).all()
    assert model.encoder[3].bias is None and not (model.encoder[3].weight == 7).any()


def test_capture_activations():
    torch.manual_seed(0)
    model = Net().train()
    x = torch.randn(2, 3, 4, 4)
    acts = capture_activations(model, x, ["encoder.2", "head.1"])
    assert set(acts) == {"encoder.2", "head.1"}
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(acts["encoder.2"], model.encoder[:3](x), msg="run in eval mode")
        torch.testing.assert_close(acts["head.1"], model.head[:2](model.encoder(x)))
    assert not acts["head.1"].requires_grad, "run without building a graph"
    assert _no_hooks_left(model), "remove your hooks"


def test_capture_activations_unknown_name():
    model = Net()
    with pytest.raises(KeyError):
        capture_activations(model, torch.randn(1, 3, 4, 4), ["encoder.0", "encoder.9"])
    assert _no_hooks_left(model), "no hooks may be left behind, even when raising"


def test_freeze_except():
    model = Net()
    assert freeze_except(model, ["head.2"]) == 16 * 4 * 4 * 10 + 10
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {"head.2.weight", "head.2.bias"}
    freeze_except(model, ["encoder.1", "head"])
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {
        "encoder.1.weight", "encoder.1.bias", "head.1.weight", "head.1.bias", "head.2.weight", "head.2.bias",
    }


def test_param_groups():
    model = Net()
    for p in model.encoder[0].parameters():
        p.requires_grad_(False)
    groups = param_groups(model, 0.05)
    assert [g["weight_decay"] for g in groups] == [0.05, 0.0]
    decay, no_decay = ({id(p) for p in g["params"]} for g in groups)
    assert decay == {id(model.encoder[3].weight), id(model.head[2].weight)}
    assert no_decay == {
        id(model.encoder[1].weight), id(model.encoder[1].bias),
        id(model.head[1].weight), id(model.head[1].bias), id(model.head[2].bias),
    }
    torch.optim.AdamW(groups, lr=1e-3)  # a real optimiser accepts them


def test_no_stubs_left(no_shortcuts):
    no_shortcuts(init_weights_, capture_activations, freeze_except, param_groups)
