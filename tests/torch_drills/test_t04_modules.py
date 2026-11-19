import torch
from torch import nn

from torch_drills.t04_modules import MLP, LayerNorm, RunningCenter


def _linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def test_mlp_layers_are_registered():
    model = MLP([10, 32, 16, 3], dropout=0.5)
    assert [(l.in_features, l.out_features) for l in _linears(model)] == [(10, 32), (32, 16), (16, 3)], (
        "layers kept in a plain Python list aren't registered: use nn.ModuleList or nn.Sequential"
    )
    assert sum(p.numel() for p in model.parameters()) == 11 * 32 + 33 * 16 + 17 * 3


def test_mlp_forward_matches_manual():
    torch.manual_seed(0)
    model = MLP([10, 32, 16, 3], dropout=0.5).eval()
    x = torch.randn(32, 10)
    l1, l2, l3 = _linears(model)
    with torch.no_grad():
        l3.bias.copy_(torch.tensor([-1.0, 0.0, 1.0]))  # make sure some logits are negative
    expected = l3(torch.relu(l2(torch.relu(l1(x)))))
    torch.testing.assert_close(model(x), expected, msg="no ReLU or dropout after the last layer: logits can be negative")


def test_mlp_dropout_only_in_train_mode():
    torch.manual_seed(0)
    model = MLP([10, 64, 3], dropout=0.5)
    x = torch.randn(8, 10)
    model.train()
    assert not torch.equal(model(x), model(x)), "dropout should be active in training mode"
    model.eval()
    assert torch.equal(model(x), model(x)), "dropout should be off in eval mode"


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    ln = LayerNorm(6)
    assert set(dict(ln.named_parameters())) == {"weight", "bias"}
    torch.testing.assert_close(ln.weight.detach(), torch.ones(6))
    torch.testing.assert_close(ln.bias.detach(), torch.zeros(6))
    with torch.no_grad():
        ln.weight.normal_()
        ln.bias.normal_()
    ref = nn.LayerNorm(6)
    ref.load_state_dict(ln.state_dict())
    x = torch.randn(2, 5, 6) * 3 + 1
    torch.testing.assert_close(ln(x), ref(x))
    ln(x).sum().backward()
    assert ln.weight.grad is not None and ln.bias.grad is not None


def test_layernorm_eps():
    x = torch.randn(3, 4) * 0.01
    torch.testing.assert_close(LayerNorm(4, eps=1e-3)(x), nn.LayerNorm(4, eps=1e-3)(x))


def test_running_center_train_and_eval():
    rc = RunningCenter(3, momentum=0.5)
    assert "running_mean" in rc.state_dict(), "use register_buffer"
    assert len(list(rc.parameters())) == 0, "a buffer is not a parameter"
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    torch.testing.assert_close(rc(x), x - x.mean(0))
    torch.testing.assert_close(rc.running_mean, 0.5 * x.mean(0))
    rc.eval()
    torch.testing.assert_close(rc(x), x - 0.5 * x.mean(0))
    torch.testing.assert_close(rc.running_mean, 0.5 * x.mean(0), msg="eval mode must not update the buffer")


def test_running_center_buffer_follows_module_and_has_no_grad():
    rc = RunningCenter(3).double()
    assert rc.running_mean.dtype == torch.float64, "a plain tensor attribute doesn't follow .double() / .to()"
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    rc(x).sum().backward()
    assert not rc.running_mean.requires_grad, "update the buffer without tracking gradients"


def test_no_shortcuts(no_shortcuts):
    no_shortcuts(MLP, RunningCenter)
    no_shortcuts(LayerNorm, ban=("layer_norm", "LayerNorm"))
