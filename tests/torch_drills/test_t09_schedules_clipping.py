import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torch_drills.t09_schedules_clipping import clip_grad_norm, make_warmup_cosine, train_step, warmup_cosine_lr


def test_warmup_cosine_lr():
    lrs = [warmup_cosine_lr(s, 1.0, 10, 110, min_lr=0.1) for s in range(130)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0) and lrs[10] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10])), "warmup should go up"
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:])), "after warmup it should never go up"
    assert lrs[60] == pytest.approx(0.55), "halfway through the decay, the lr is halfway between base and min"
    assert lrs[110] == pytest.approx(0.1) and lrs[129] == pytest.approx(0.1)


def test_warmup_cosine_lr_without_warmup():
    assert warmup_cosine_lr(0, 0.5, 0, 100) == pytest.approx(0.5)
    assert warmup_cosine_lr(50, 0.5, 0, 100) == pytest.approx(0.25)


def test_make_warmup_cosine():
    p1, p2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([{"params": [p1], "lr": 0.5}, {"params": [p2], "lr": 0.05}])
    sched = make_warmup_cosine(opt, warmup_steps=5, total_steps=30, min_lr_ratio=0.1)
    assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)
    seen = []
    for _ in range(35):
        seen.append([g["lr"] for g in opt.param_groups])
        opt.step()
        sched.step()
    for s, (lr1, lr2) in enumerate(seen):
        assert lr1 == pytest.approx(warmup_cosine_lr(s, 0.5, 5, 30, 0.05))
        assert lr2 == pytest.approx(warmup_cosine_lr(s, 0.05, 5, 30, 0.005))


def _params_with_grads():
    torch.manual_seed(0)
    ps = [nn.Parameter(torch.randn(3, 4)), nn.Parameter(torch.randn(5)), nn.Parameter(torch.randn(2))]
    ps[0].grad = torch.randn(3, 4) * 3
    ps[1].grad = torch.randn(5) * 3  # ps[2] has no gradient
    return ps


@pytest.mark.parametrize("max_norm", [0.5, 1000.0])
def test_clip_grad_norm_matches_torch(max_norm):
    mine, ref = _params_with_grads(), _params_with_grads()
    norm = clip_grad_norm(mine, max_norm)
    ref_norm = torch.nn.utils.clip_grad_norm_(ref, max_norm)
    assert isinstance(norm, float) and norm == pytest.approx(ref_norm.item())
    for p, q in zip(mine, ref):
        if q.grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, msg="clip by the norm of ALL gradients together")


def _linear_with_grads():
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    model(torch.randn(4, 3) * 10).sum().backward()
    return model


def test_clip_grad_norm_accepts_a_generator():
    model, ref = _linear_with_grads(), _linear_with_grads()
    clip_grad_norm(model.parameters(), 0.1)
    torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.1)
    for p, q in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad, msg="a generator can only be read once")


def test_train_step_order():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    ref = copy.deepcopy(model)

    def setup(m):
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5**s)

    (opt, sched), (ref_opt, ref_sched) = setup(model), setup(ref)
    x, y = torch.randn(8, 4) * 10, torch.randint(0, 3, (8,))
    for _ in range(3):
        loss, norm = train_step(model, x, y, nn.CrossEntropyLoss(), opt, sched, max_norm=0.1)
        ref_opt.zero_grad()
        ref_loss = F.cross_entropy(ref(x), y)
        ref_loss.backward()
        ref_norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.1)
        ref_opt.step()
        ref_sched.step()
        assert isinstance(loss, float) and loss == pytest.approx(ref_loss.item())
        assert isinstance(norm, float) and norm == pytest.approx(ref_norm.item())
        for p, q in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(p, q)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.125)


def test_no_shortcuts(no_shortcuts):
    no_shortcuts(warmup_cosine_lr, make_warmup_cosine, train_step)
    no_shortcuts(clip_grad_norm, ban=("clip_grad_norm_", "clip_grad_norm", "clip_grads_with_norm_", "get_total_norm"))
