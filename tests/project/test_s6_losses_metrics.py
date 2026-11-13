import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

from seg.s6_losses_metrics import ce_dice_loss, mean_iou, soft_dice_loss

torch.manual_seed(0)
target = torch.randint(0, 3, (2, 16, 16))


def test_dice_loss_near_zero_for_perfect_logits():
    logits = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
    assert soft_dice_loss(logits, target).item() < 1e-3


def test_dice_loss_range_and_gradient():
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    loss = soft_dice_loss(logits, target)
    assert loss.ndim == 0 and 0 < loss.item() < 1
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_dice_loss_value():
    logits = torch.zeros(1, 2, 1, 2)                 # uniform probs 0.5
    t = torch.tensor([[[0, 0]]])
    # class 0: 2*1 / (1 + 2) = 2/3 ; class 1: 0 / (1 + 0) = 0  -> loss = 1 - 1/3
    np.testing.assert_allclose(soft_dice_loss(logits, t, eps=0).item(), 2 / 3, rtol=1e-6)


def test_ce_dice():
    logits = torch.randn(2, 3, 16, 16)
    expected = F.cross_entropy(logits, target) + 0.5 * soft_dice_loss(logits, target)
    torch.testing.assert_close(ce_dice_loss(logits, target, dice_weight=0.5), expected)


def test_mean_iou_matches_sklearn():
    pred = torch.where(torch.rand(2, 16, 16) < 0.7, target, torch.randint(0, 3, (2, 16, 16)))
    ref = jaccard_score(target.flatten().numpy(), pred.flatten().numpy(), average="macro")
    out = mean_iou(pred, target, 3)
    assert isinstance(out, float)
    np.testing.assert_allclose(out, ref, rtol=1e-6)


def test_mean_iou_skips_absent_classes():
    t = torch.tensor([[[0, 0, 1, 1]]])
    np.testing.assert_allclose(mean_iou(t.clone(), t, 5), 1.0)
