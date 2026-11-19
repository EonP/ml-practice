"""PyTorch drill 10: custom autograd Functions (target: 20 min).

A torch.autograd.Function has two static methods:
    forward(ctx, *inputs)    computes the output. Save what backward needs with
                             ctx.save_for_backward(tensor, ...) for tensors, or
                             ctx.some_name = value for anything else.
    backward(ctx, grad_out)  receives dLoss/dOutput and returns dLoss/dInput for EVERY
                             input of forward, in the same order (None for an input
                             that doesn't need a gradient, such as a Python float).
Call it as MyFunction.apply(...). torch.autograd.gradcheck compares your backward with
finite differences, using float64 inputs.
"""
import torch
from torch.autograd import Function


class Sigmoid(Function):
    """y = 1 / (1 + exp(-x)). Save the OUTPUT for backward: dy/dx = y * (1 - y)."""

    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError("TODO: Sigmoid.forward")

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("TODO: Sigmoid.backward")


class LinearFunction(Function):
    """y = x @ W.T + b, for x (N, in), W (out, in), b (out,). backward returns (dx, dW, db)."""

    @staticmethod
    def forward(ctx, x, W, b):
        raise NotImplementedError("TODO: LinearFunction.forward")

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("TODO: LinearFunction.backward")


class StraightThroughClamp(Function):
    """forward(ctx, x, lo, hi): x clamped to [lo, hi], where lo and hi are Python floats.
    backward passes grad_out through UNCHANGED, as if the clamp were the identity."""

    @staticmethod
    def forward(ctx, x, lo, hi):
        raise NotImplementedError("TODO: StraightThroughClamp.forward")

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("TODO: StraightThroughClamp.backward")


class GradReverse(Function):
    """forward(ctx, x, lam): the identity (return x.view_as(x) rather than x itself).
    backward multiplies the incoming gradient by -lam. Used in domain-adversarial training,
    where it makes a feature extractor work AGAINST a domain classifier."""

    @staticmethod
    def forward(ctx, x, lam):
        raise NotImplementedError("TODO: GradReverse.forward")

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("TODO: GradReverse.backward")
