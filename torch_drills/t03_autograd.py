"""PyTorch drill 03: autograd (target: 15 min).

- loss.backward() ADDS into the .grad of every leaf tensor with requires_grad=True.
  It never overwrites, which is why training loops call zero_grad().
- torch.autograd.grad(out, inputs) returns the gradients instead of storing them.
- x.detach() shares x's data but is cut out of the graph; torch.no_grad() stops
  recording the graph altogether.
- create_graph=True records the backward pass too, so you can differentiate again.
"""
import torch


def grad_at(f, x):
    """Gradient of the scalar function f at x (any shape) -> a tensor shaped like x.

    Leave x untouched: don't switch on x.requires_grad, and leave nothing in x.grad.
    """
    raise NotImplementedError("TODO: grad_at")


def numerical_grad(f, x, eps=1e-6):
    """Central finite differences of the scalar function f at x (float64, any shape):
    out[i] = (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps). Loops are fine here.

    x must be exactly unchanged afterwards. This is how you check a hand-written backward.
    """
    raise NotImplementedError("TODO: numerical_grad")


def accumulated_grad(w, micro_batches, loss_fn):
    """Gradient accumulation: the gradient of the MEAN loss over ALL examples in
    micro_batches, built by calling backward() once per micro-batch.

    w: leaf tensor with requires_grad=True. Its .grad may hold an old value: start fresh.
    micro_batches: list of (x, y) pairs, NOT all the same size.
    loss_fn(w, x, y) returns the MEAN loss over one micro-batch.
    Leave the result in w.grad and return it.
    """
    raise NotImplementedError("TODO: accumulated_grad")


def straight_through_round(x):
    """Forward: x rounded to the nearest integer. Backward: act as if it were the identity,
    so the gradient passes straight through (round itself has zero gradient almost everywhere).
    Hint: x + (something).detach()."""
    raise NotImplementedError("TODO: straight_through_round")


def hvp(f, x, v):
    """Hessian-vector product H v of the scalar function f at x, without building H.
    v has x's shape. Leave x untouched, as in grad_at.
    Hint: differentiate sum(grad_f(x) * v) with respect to x."""
    raise NotImplementedError("TODO: hvp")
