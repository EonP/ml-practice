import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_drills.t06_data import PairDataset, channel_mean_std, make_loaders, pad_collate


def test_pair_dataset():
    X, y = torch.randn(10, 3), torch.arange(10)
    ds = PairDataset(X, y)
    assert len(ds) == 10
    x5, y5 = ds[5]
    assert torch.equal(x5, X[5]) and y5 == 5
    xb, yb = next(iter(DataLoader(ds, batch_size=4)))
    assert xb.shape == (4, 3) and torch.equal(yb, torch.arange(4))


def test_pair_dataset_transform():
    X, y = torch.randn(10, 3), torch.arange(10)
    x3, y3 = PairDataset(X, y, transform=lambda t: t * 2)[3]
    torch.testing.assert_close(x3, X[3] * 2)
    assert y3 == 3


def test_pad_collate():
    seqs = [torch.randn(3, 2), torch.randn(5, 2), torch.randn(1, 2)]
    padded, lengths, labels = pad_collate([(s, i) for i, s in enumerate(seqs)], pad_value=-1.0)
    assert padded.shape == (3, 5, 2)
    assert lengths.dtype == torch.long and lengths.tolist() == [3, 5, 1]
    assert labels.dtype == torch.long and labels.tolist() == [0, 1, 2]
    for i, s in enumerate(seqs):
        torch.testing.assert_close(padded[i, : len(s)], s)
        assert (padded[i, len(s) :] == -1).all()


def test_pad_collate_in_a_dataloader():
    data = [(torch.randn(n, 4), n % 2) for n in (2, 7, 3, 3, 5)]
    loader = DataLoader(data, batch_size=2, collate_fn=pad_collate)
    assert [tuple(p.shape) for p, _, _ in loader] == [(2, 7, 4), (2, 3, 4), (1, 5, 4)]


def _ids(loader):
    return torch.cat([yb for _, yb in loader]).tolist()


def test_make_loaders_split():
    X, y = torch.arange(50.0)[:, None], torch.arange(50)
    train, val = make_loaders(X, y, batch_size=8, val_frac=0.2, seed=0)
    train_ids, val_ids = _ids(train), _ids(val)
    assert len(val_ids) == 10 and len(train_ids) == 40
    assert set(train_ids) | set(val_ids) == set(range(50))
    assert sorted(val_ids) != list(range(40, 50)), "split at random, not by position"
    assert _ids(make_loaders(X, y, 8, 0.2, seed=0)[1]) == val_ids, "the same seed must give the same split"
    assert sorted(_ids(make_loaders(X, y, 8, 0.2, seed=1)[1])) != sorted(val_ids)
    xb, yb = next(iter(train))
    assert xb.shape == (8, 1) and torch.equal(xb[:, 0].long(), yb)


def test_make_loaders_shuffling():
    torch.manual_seed(0)
    X, y = torch.arange(50.0)[:, None], torch.arange(50)
    train, val = make_loaders(X, y, batch_size=8, val_frac=0.2, seed=0)
    epoch1, epoch2 = _ids(train), _ids(train)
    assert epoch1 != epoch2, "the train loader should reshuffle every epoch"
    assert sorted(epoch1) == sorted(epoch2)
    assert _ids(val) == _ids(val), "the validation loader should not shuffle"


def test_channel_mean_std():
    torch.manual_seed(0)
    scale = torch.tensor([1.0, 2.0, 3.0])[:, None, None]
    shift = torch.tensor([0.0, -1.0, 5.0])[:, None, None]
    imgs = torch.rand(23, 3, 4, 5) * scale + shift
    loader = DataLoader(TensorDataset(imgs, torch.zeros(23)), batch_size=5)  # last batch has 3
    mean, std = channel_mean_std(loader)
    ref = imgs.double().transpose(0, 1).reshape(3, -1)
    assert mean.dtype == torch.float64 and std.dtype == torch.float64
    torch.testing.assert_close(mean, ref.mean(1))
    torch.testing.assert_close(std, ref.std(1, correction=0))


def test_no_stubs_left(no_shortcuts):
    no_shortcuts(PairDataset, pad_collate, make_loaders, channel_mean_std)
