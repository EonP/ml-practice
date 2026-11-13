import pytest
import torch

from seg.s5_unet import DoubleConv, UNet


def test_double_conv_keeps_size():
    out = DoubleConv(3, 8)(torch.randn(2, 3, 17, 13))
    assert out.shape == (2, 8, 17, 13)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_unet_output_shape(depth):
    model = UNet(in_channels=3, n_classes=5, base=8, depth=depth)
    out = model(torch.randn(2, 3, 64, 48))
    assert out.shape == (2, 5, 64, 48)


def test_parameters_are_registered():
    """A plain Python list of layers would make these invisible to the optimiser."""
    small = sum(p.numel() for p in UNet(base=8, depth=1).parameters())
    big = sum(p.numel() for p in UNet(base=8, depth=3).parameters())
    assert big > 10 * small


def test_every_parameter_gets_a_gradient():
    model = UNet(base=8, depth=2)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} gets no gradient: is it used in forward?"


def test_has_batchnorm():
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in UNet(base=8, depth=1).modules())
