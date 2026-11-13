"""Train a U-Net end to end (given; it only works once stages 1-7 pass).

    python train.py --toy                 # synthetic data, ~1 min on CPU: check the pipeline
    python train.py --overfit             # sanity check: overfit one batch of real data
    python train.py --epochs 20           # Oxford-IIIT Pet (downloads ~800 MB the first time)

Outputs go to runs/: curves.png, predictions.png, model.pt.
Read this file line by line once your stages pass: it is what a typical
training script looks like, and you should be able to write one from memory.
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader, Subset

from seg.s4_dataset import JointTransform, SegDataset
from seg.s5_unet import UNet
from seg.s6_losses_metrics import ce_dice_loss
from seg.s7_train import evaluate, overfit_one_batch, plot_curves, plot_predictions, train_one_epoch
from seg.toy import make_toy_dataset


def get_data(args):
    if args.toy:
        train_ds = SegDataset(*make_toy_dataset(256, size=64, seed=0), transform=JointTransform(hflip_p=0.5, seed=0))
        val_ds = SegDataset(*make_toy_dataset(64, size=64, seed=1))
        return train_ds, val_ds, 3
    from seg.pets import N_CLASSES, PetsSeg

    train_ds = PetsSeg(args.data, "trainval", args.size, transform=JointTransform(hflip_p=0.5, seed=0))
    val_ds = PetsSeg(args.data, "test", args.size)
    if args.limit:
        train_ds, val_ds = Subset(train_ds, range(args.limit)), Subset(val_ds, range(args.limit // 4))
    return train_ds, val_ds, N_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--toy", action="store_true", help="use synthetic data instead of Oxford Pets")
    p.add_argument("--overfit", action="store_true", help="only overfit a single batch")
    p.add_argument("--data", default="data")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="use only this many training images")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    train_ds, val_ds, n_classes = get_data(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    model = UNet(in_channels=3, n_classes=n_classes, base=args.base, depth=args.depth).to(device)
    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    if args.overfit:
        x, y = next(iter(train_loader))
        losses = overfit_one_batch(model, x.to(device), y.to(device), ce_dice_loss, steps=200)
        print(f"overfit one batch: loss {losses[0]:.3f} -> {losses[-1]:.3f}  (should approach 0)")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best = -1.0
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, opt, ce_dice_loss, device)
        va = evaluate(model, val_loader, ce_dice_loss, n_classes, device)
        sched.step()
        history["train_loss"].append(tr)
        history["val_loss"].append(va["loss"])
        history["val_miou"].append(va["miou"])
        print(f"epoch {epoch + 1:3d}  train {tr:.4f}  val {va['loss']:.4f}  mIoU {va['miou']:.3f}")
        if va["miou"] > best:
            best = va["miou"]
            torch.save(model.state_dict(), out / "model.pt")

    plot_curves(history).savefig(out / "curves.png", dpi=120)
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    model.eval()
    x, y = next(iter(val_loader))
    with torch.no_grad():
        pred = model(x.to(device)).argmax(1).cpu()
    plot_predictions(x, y, pred, n=4).savefig(out / "predictions.png", dpi=120)
    print(f"best val mIoU {best:.3f}; plots and checkpoint in {out}/")


if __name__ == "__main__":
    main()
