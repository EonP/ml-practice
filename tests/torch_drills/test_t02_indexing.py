import torch
import torch.nn.functional as F

from torch_drills.t02_indexing import (
    causal_fill,
    one_hot,
    pick_target_logits,
    pixel_target_logits,
    segment_sum,
    topk_mask,
)


def test_pick_target_logits():
    logits = torch.randn(6, 4)
    y = torch.tensor([3, 0, 1, 1, 2, 3])
    torch.testing.assert_close(pick_target_logits(logits, y), logits[torch.arange(6), y])


def test_pixel_target_logits():
    logits = torch.randn(2, 5, 3, 4)
    target = torch.randint(0, 5, (2, 3, 4))
    out = pixel_target_logits(logits, target)
    assert out.shape == (2, 3, 4)
    expected = torch.tensor(
        [[[logits[n, target[n, h, w], h, w].item() for w in range(4)] for h in range(3)] for n in range(2)]
    )
    torch.testing.assert_close(out, expected)


def test_one_hot():
    y = torch.tensor([[2, 0], [1, 2]])
    out = one_hot(y, 3)
    assert out.dtype == torch.float32 and out.shape == (2, 2, 3)
    torch.testing.assert_close(out, F.one_hot(y, 3).float())


def test_causal_fill():
    s = torch.randn(2, 4, 4)
    out = causal_fill(s)
    assert out.shape == s.shape
    lower = torch.ones(4, 4, dtype=torch.bool).tril()
    torch.testing.assert_close(out[:, lower], s[:, lower])
    assert torch.isneginf(out[:, ~lower]).all()
    assert not torch.isinf(s).any(), "return a copy: don't fill `scores` in place"
    w = out.softmax(-1)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 4))
    assert (w[:, ~lower] == 0).all()


def test_segment_sum_adds_repeated_ids():
    values = torch.randn(7, 3)
    ids = torch.tensor([0, 2, 2, 0, 4, 2, 0])
    out = segment_sum(values, ids, 6)
    assert out.shape == (6, 3)
    for s in range(6):
        torch.testing.assert_close(out[s], values[ids == s].sum(0))


def test_topk_mask():
    x = torch.randn(5, 6)
    m = topk_mask(x, 2)
    assert m.dtype == torch.bool and m.shape == (5, 6)
    assert (m.sum(1) == 2).all()
    smallest_kept = torch.where(m, x, torch.inf).min(1).values
    largest_dropped = torch.where(m, -torch.inf, x).max(1).values
    assert (smallest_kept > largest_dropped).all()
    assert not topk_mask(x, 0).any()
    assert topk_mask(x, 6).all()


def test_no_python_loops(no_shortcuts):
    no_shortcuts(pick_target_logits, pixel_target_logits, causal_fill, segment_sum, topk_mask, loops=False)
    no_shortcuts(one_hot, loops=False, ban=("one_hot", "eye"))
