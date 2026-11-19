"""PyTorch drill 11: initialisation, hooks and fine-tuning (target: 15 min).

model.named_modules() and model.named_parameters() walk the whole tree of submodules,
with dotted names like "encoder.0.weight". The nn.init functions change a tensor in
place without recording gradients. A forward hook is fn(module, inputs, output),
registered with module.register_forward_hook(fn); that call returns a handle whose
.remove() unregisters the hook. Hooks left behind keep running (and keep whatever
they store alive), which is a classic memory leak.
"""
import torch
from torch import nn


def init_weights_(model):
    """In place, for every submodule:
    nn.Linear and nn.Conv2d: Kaiming-normal weights for ReLU (fan_in mode); zero bias, if it has one.
    nn.BatchNorm2d and nn.LayerNorm: weight 1, bias 0.
    Returns None.
    """
    raise NotImplementedError("TODO: init_weights_")


def capture_activations(model, x, names):
    """Run model on x once (eval mode, no graph) and return {name: output} for each submodule
    named in `names`, using the names from model.named_modules().

    Raise KeyError for a name that isn't a submodule. Leave no hooks behind, even then.
    """
    raise NotImplementedError("TODO: capture_activations")


def freeze_except(model, prefixes):
    """Fine-tuning: set requires_grad = True on every parameter whose name starts with one of
    `prefixes`, and False on every other parameter. Returns the number of trainable values
    (scalars, not tensors)."""
    raise NotImplementedError("TODO: freeze_except")


def param_groups(model, weight_decay):
    """Two param groups for AdamW. Parameters with 2 or more dims (weight matrices, conv kernels)
    get weight_decay; parameters with 1 dim (biases, normalisation weights) get none.
    Frozen parameters (requires_grad False) are left out of both.
    Returns [{"params": [...], "weight_decay": weight_decay}, {"params": [...], "weight_decay": 0.0}].
    """
    raise NotImplementedError("TODO: param_groups")
