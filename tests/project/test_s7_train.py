import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from seg.pets import prepare_pair
from seg.s4_dataset import JointTransform, SegDataset
from seg.s5_unet import UNet
from seg.s6_losses_metrics import ce_dice_loss
from seg.s7_train import evaluate, overfit_one_batch, plot_curves, plot_predictions, train_one_epoch


def test_overfit_one_batch(toy_data):
    torch.manual_seed(0)
    x, y = next(iter(DataLoader(SegDataset(*toy_data), batch_size=4)))
    losses = overfit_one_batch(UNet(base=8, depth=2, n_classes=3), x, y, ce_dice_loss, steps=150)
    assert len(losses) == 150
    assert losses[-1] < 0.15 * losses[0]


def test_train_and_evaluate(toy_data):
    torch.manual_seed(0)
    ds = SegDataset(*toy_data, transform=JointTransform(hflip_p=0.5, seed=0))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = UNet(base=8, depth=2, n_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = train_one_epoch(model, loader, opt, ce_dice_loss, "cpu")
    for _ in range(14):
        last = train_one_epoch(model, loader, opt, ce_dice_loss, "cpu")
    assert isinstance(first, float) and last < first
    metrics = evaluate(model, DataLoader(SegDataset(*toy_data), batch_size=4), ce_dice_loss, 3, "cpu")
    assert set(metrics) == {"loss", "miou"}
    assert not model.training, "evaluate must switch the model to eval mode"
    assert metrics["miou"] > 0.6


def test_plots(toy_data):
    x, y = next(iter(DataLoader(SegDataset(*toy_data), batch_size=4)))
    fig = plot_predictions(x, y, y, n=3)
    assert len(fig.axes) == 9
    assert fig.axes[0].get_title() == "image"
    plt.close(fig)
    fig = plot_curves({"train_loss": [3, 2, 1], "val_loss": [3, 2.5, 2], "val_miou": [0.1, 0.3, 0.5]})
    assert len(fig.axes) == 2
    assert fig.axes[0].get_xlabel() == "epoch" and fig.axes[0].get_legend() is not None
    plt.close(fig)


def test_prepare_pets_pair():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (50, 70, 3), dtype=np.uint8))
    tri = Image.fromarray(rng.integers(1, 4, (50, 70), dtype=np.uint8))
    a, m = prepare_pair(img, tri, 32)
    assert a.shape == (32, 32, 3) and a.dtype == np.uint8
    assert m.shape == (32, 32) and m.dtype == np.int64
    assert set(np.unique(m)) <= {0, 1, 2}, "use NEAREST for masks and shift labels to start at 0"
