"""PyTorch drill 08: optimisers from scratch (target: 25 min).

An optimiser holds the list of parameters, plus some state for each one (a momentum
buffer; Adam's two moving averages). step() updates every parameter IN PLACE from its
.grad, inside torch.no_grad(), and skips parameters whose .grad is None. `params` may
be a generator such as model.parameters(), which can only be read once.
Don't use torch.optim here: the tests compare your updates against it.
"""
import torch


class SGD:
    """SGD with momentum and L2 weight decay, matching torch.optim.SGD (no dampening, no Nesterov).

    For each parameter p that has a gradient:
        g = p.grad + weight_decay * p
        buf = g on the first step, then momentum * buf + g    (no buffer needed when momentum == 0)
        p <- p - lr * buf
    """

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        raise NotImplementedError("TODO: SGD.__init__")

    def zero_grad(self):
        """Set every parameter's .grad to None."""
        raise NotImplementedError("TODO: SGD.zero_grad")

    def step(self):
        raise NotImplementedError("TODO: SGD.step")


class Adam:
    """Adam, matching torch.optim.Adam. With t = the number of updates this parameter has had,
    counting the current one (so t = 1 on the first step), and m = v = 0 to start:
        g = p.grad + weight_decay * p
        m <- beta1 * m + (1 - beta1) * g
        v <- beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t),   v_hat = v / (1 - beta2 ** t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        raise NotImplementedError("TODO: Adam.__init__")

    def zero_grad(self):
        raise NotImplementedError("TODO: Adam.zero_grad")

    def step(self):
        raise NotImplementedError("TODO: Adam.step")


class AdamW(Adam):
    """AdamW, matching torch.optim.AdamW: the weight decay is DECOUPLED from the gradient.

    Before the Adam update, shrink each weight directly, p <- p * (1 - lr * weight_decay),
    and then do the Adam update with g = p.grad (no weight_decay term in g).
    Note the different default weight_decay.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        raise NotImplementedError("TODO: AdamW.__init__")

    def step(self):
        raise NotImplementedError("TODO: AdamW.step")
