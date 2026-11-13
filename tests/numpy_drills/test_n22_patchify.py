import numpy as np
import torch
import torch.nn.functional as F

from numpy_drills.n22_patchify import block_mean, patchify, unpatchify, vit_tokens

rng = np.random.default_rng(0)


def test_block_mean_matches_avg_pool():
    img = rng.normal(size=(6, 8))
    for f in (1, 2):
        expected = F.avg_pool2d(torch.tensor(img)[None, None], f)[0, 0].numpy()
        np.testing.assert_allclose(block_mean(img, f), expected, atol=1e-12)
    np.testing.assert_allclose(block_mean(np.arange(16.0).reshape(4, 4), 2), [[2.5, 4.5], [10.5, 12.5]])


def test_patchify():
    img = rng.normal(size=(6, 9, 2))
    out = patchify(img, 3)
    assert out.shape == (2, 3, 3, 3, 2)
    for i in range(2):
        for j in range(3):
            np.testing.assert_array_equal(out[i, j], img[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])


def test_unpatchify():
    patches = rng.normal(size=(2, 3, 4, 4, 3))
    out = unpatchify(patches)
    assert out.shape == (8, 12, 3)
    for i in range(2):
        for j in range(3):
            np.testing.assert_array_equal(out[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4], patches[i, j])


def test_vit_tokens_matches_torch_unfold():
    x = rng.normal(size=(2, 3, 8, 12))
    out = vit_tokens(x, 4)
    assert out.shape == (2, 6, 48)
    np.testing.assert_array_equal(out, F.unfold(torch.tensor(x), 4, stride=4).transpose(1, 2).numpy())


def test_no_python_loops(no_loops):
    no_loops(block_mean, patchify, unpatchify, vit_tokens, ban=("sliding_window_view", "as_strided"))
