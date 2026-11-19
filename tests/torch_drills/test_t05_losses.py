import pytest
import torch
import torch.nn.functional as F

from torch_drills.t05_losses import bce_with_logits, cross_entropy, label_smoothing_ce, seg_cross_entropy

f64 = torch.float64


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(8, 5, dtype=f64)
    y = torch.randint(0, 5, (8,))
    out = cross_entropy(logits, y)
    assert isinstance(out, torch.Tensor) and out.dim() == 0
    torch.testing.assert_close(out, F.cross_entropy(logits, y))


def test_cross_entropy_gradient():
    torch.manual_seed(1)
    logits = torch.randn(6, 4, dtype=f64, requires_grad=True)
    y = torch.randint(0, 4, (6,))
    g = torch.autograd.grad(cross_entropy(logits, y), logits)[0]
    torch.testing.assert_close(g, torch.autograd.grad(F.cross_entropy(logits, y), logits)[0])


def test_cross_entropy_stable_with_ignore_index():
    logits = torch.tensor([[1000.0, 0.0, -1000.0], [0.0, 1000.0, 0.0], [5.0, 5.0, 5.0]], dtype=f64)
    y = torch.tensor([2, 1, -100])
    out = cross_entropy(logits, y)
    assert torch.isfinite(out)
    torch.testing.assert_close(out, F.cross_entropy(logits, y))
    y2 = torch.tensor([255, 0, 255])
    torch.testing.assert_close(cross_entropy(logits, y2, ignore_index=255), F.cross_entropy(logits, y2, ignore_index=255))


def test_seg_cross_entropy():
    torch.manual_seed(0)
    N, H, W, K = 2, 3, 5, 4
    logits = torch.randn(N, H, W, K, dtype=f64)
    target = torch.randint(0, K, (N, H, W))
    expected = F.cross_entropy(logits.reshape(-1, K), target.reshape(-1))
    torch.testing.assert_close(seg_cross_entropy(logits, target), expected)


def test_bce_matches_torch():
    torch.manual_seed(0)
    z = torch.randn(4, 3, dtype=f64) * 3
    y = torch.rand(4, 3, dtype=f64)  # soft labels are allowed
    torch.testing.assert_close(bce_with_logits(z, y), F.binary_cross_entropy_with_logits(z, y))
    pw = torch.tensor([1.0, 2.0, 5.0], dtype=f64)
    torch.testing.assert_close(bce_with_logits(z, y, pos_weight=pw), F.binary_cross_entropy_with_logits(z, y, pos_weight=pw))


def test_bce_stable_for_huge_logits():
    z = torch.tensor([1000.0, -1000.0, 1000.0, -1000.0], dtype=f64, requires_grad=True)
    y = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=f64)
    out = bce_with_logits(z, y)
    assert torch.isfinite(out), "never take log(sigmoid(z))"
    torch.testing.assert_close(out, F.binary_cross_entropy_with_logits(z, y))
    out.backward()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.3])
def test_label_smoothing(eps):
    torch.manual_seed(0)
    logits = torch.randn(7, 5, dtype=f64)
    y = torch.randint(0, 5, (7,))
    torch.testing.assert_close(label_smoothing_ce(logits, y, eps), F.cross_entropy(logits, y, label_smoothing=eps))


def test_no_shortcuts(no_shortcuts):
    banned = (
        "cross_entropy", "nll_loss", "CrossEntropyLoss", "NLLLoss",
        "binary_cross_entropy_with_logits", "binary_cross_entropy", "BCEWithLogitsLoss", "BCELoss",
    )
    no_shortcuts(cross_entropy, bce_with_logits, label_smoothing_ce, ban=banned)
    no_shortcuts(seg_cross_entropy)
