import numpy as np
import torch

from torch_drills.t01_tensors import (
    batch_images,
    concat_sequences,
    flatten_features,
    like,
    merge_heads,
    split_heads,
    squeeze_last,
    to_channels_last,
)


def shares_memory(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def test_like_copies_dtype_and_device():
    for dtype in (torch.float64, torch.int32, torch.float16):
        x = torch.zeros(2, dtype=dtype)
        out = like(x, [[1, 2], [3, 4]])
        assert out.dtype == dtype and out.device == x.device
        assert out.tolist() == [[1, 2], [3, 4]]
    x = torch.zeros(3, dtype=torch.float64)
    out = like(x, np.array([1.5, 2.5], dtype=np.float32))
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, torch.tensor([1.5, 2.5], dtype=torch.float64))


def test_batch_images():
    imgs = [torch.randn(3, 4, 5) for _ in range(6)]
    out = batch_images(imgs)
    assert out.shape == (6, 3, 4, 5)
    for i, img in enumerate(imgs):
        torch.testing.assert_close(out[i], img)
    assert batch_images(imgs[:1]).shape == (1, 3, 4, 5)


def test_concat_sequences():
    seqs = [torch.randn(n, 4) for n in (2, 5, 1)]
    out = concat_sequences(seqs)
    assert out.shape == (8, 4)
    torch.testing.assert_close(out[2:7], seqs[1])


def test_flatten_features():
    x = torch.randn(2, 3, 4, 5)
    out = flatten_features(x)
    assert out.shape == (2, 60)
    assert shares_memory(out, x), "x is contiguous, so this should be a view"
    torch.testing.assert_close(out[1], x[1].reshape(-1))
    assert flatten_features(torch.randn(1, 3, 4, 5)).shape == (1, 60)


def test_flatten_features_non_contiguous():
    x = torch.randn(2, 4, 5, 3).permute(0, 3, 1, 2)  # (2, 3, 4, 5) but not contiguous
    torch.testing.assert_close(flatten_features(x), x.contiguous().view(2, -1))


def test_to_channels_last():
    x = torch.randn(2, 3, 4, 5)
    out = to_channels_last(x)
    assert out.shape == (2, 4, 5, 3) and out.is_contiguous()
    torch.testing.assert_close(out[1, 2, 3], x[1, :, 2, 3])


def test_split_and_merge_heads():
    x = torch.randn(2, 5, 12)
    out = split_heads(x, 3)
    assert out.shape == (2, 3, 5, 4)
    for h in range(3):
        torch.testing.assert_close(out[:, h], x[:, :, 4 * h : 4 * (h + 1)])
    back = merge_heads(out)
    assert back.shape == (2, 5, 12)
    torch.testing.assert_close(back, x)


def test_merge_heads_on_a_fresh_tensor():
    y = torch.randn(2, 3, 5, 4)  # e.g. the output of attention: contiguous in (B, h, L, dh)
    out = merge_heads(y)
    assert out.shape == (2, 5, 12)
    torch.testing.assert_close(out[:, :, 4:8], y[:, 1])


def test_squeeze_last():
    assert squeeze_last(torch.randn(7, 1)).shape == (7,)
    assert squeeze_last(torch.randn(1, 1)).shape == (1,)


def test_no_stubs_left(no_shortcuts):
    no_shortcuts(like, batch_images, concat_sequences, flatten_features, to_channels_last, split_heads, merge_heads, squeeze_last)
