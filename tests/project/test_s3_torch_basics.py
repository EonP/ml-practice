import numpy as np
import torch
from sklearn.datasets import make_blobs

from seg.s3_torch_basics import count_params, image_to_tensor, mask_to_tensor, train_softmax_regression


def test_image_to_tensor(toy_data):
    img = toy_data[0][0]
    t = image_to_tensor(img)
    assert t.shape == (3, 32, 32) and t.dtype == torch.float32
    np.testing.assert_allclose(t[2].numpy(), img[..., 2] / 255.0, atol=1e-6)


def test_image_to_tensor_handles_flipped_views(toy_data):
    t = image_to_tensor(toy_data[0][0][:, ::-1])
    assert t.shape == (3, 32, 32)


def test_mask_to_tensor(toy_data):
    t = mask_to_tensor(toy_data[1][0].astype(np.uint8))
    assert t.dtype == torch.int64 and t.shape == (32, 32)


def test_count_params():
    m = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 2))
    assert count_params(m) == 10 * 5 + 5 + 5 * 2 + 2
    m[0].weight.requires_grad_(False)
    assert count_params(m) == 5 + 5 * 2 + 2


def test_softmax_regression_trains():
    X, y = make_blobs(300, centers=3, random_state=0)
    X = (X - X.mean(0)) / X.std(0)
    model, losses = train_softmax_regression(torch.tensor(X, dtype=torch.float32), torch.tensor(y), 3)
    assert isinstance(model, torch.nn.Linear)
    assert all(isinstance(l, float) for l in losses)
    assert losses[-1] < 0.5 * losses[0]
    with torch.no_grad():
        acc = (model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy() == y).mean()
    assert acc > 0.9
