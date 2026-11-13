import numpy as np
import torch
import torch.nn.functional as F

from drills.d13_attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention

rng = np.random.default_rng(0)


def test_causal_mask():
    np.testing.assert_array_equal(causal_mask(3), [[1, 0, 0], [1, 1, 0], [1, 1, 1]])


def test_sdpa_matches_torch():
    q, k, v = rng.normal(size=(2, 3, 5, 8)), rng.normal(size=(2, 3, 7, 8)), rng.normal(size=(2, 3, 7, 4))
    out, w = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(*map(torch.tensor, (q, k, v))).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-10)
    assert w.shape == (2, 3, 5, 7)
    np.testing.assert_allclose(w.sum(-1), 1.0)


def test_sdpa_causal_matches_torch():
    q, k, v = (rng.normal(size=(2, 6, 8)) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v, mask=causal_mask(6))
    ref = F.scaled_dot_product_attention(*map(torch.tensor, (q, k, v)), is_causal=True).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-10)
    assert np.all(np.triu(w[0], 1) == 0)


def _reference(mha, d, h):
    ref = torch.nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight]))
        ref.in_proj_bias.copy_(torch.cat([mha.q_proj.bias, mha.k_proj.bias, mha.v_proj.bias]))
        ref.out_proj.weight.copy_(mha.out_proj.weight)
        ref.out_proj.bias.copy_(mha.out_proj.bias)
    return ref


def test_mha_matches_torch():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4)
    ref = _reference(mha, 16, 4)
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(mha(x), ref(x, x, x, need_weights=False)[0], atol=1e-5, rtol=1e-5)


def test_mha_causal_matches_torch():
    torch.manual_seed(1)
    mha = MultiHeadAttention(16, 2)
    ref = _reference(mha, 16, 2)
    x = torch.randn(2, 6, 16)
    blocked = torch.ones(6, 6, dtype=torch.bool).triu(1)   # True = NOT allowed, for nn.MultiheadAttention
    torch.testing.assert_close(mha(x, causal=True), ref(x, x, x, attn_mask=blocked, need_weights=False)[0], atol=1e-5, rtol=1e-5)
