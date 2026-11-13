import numpy as np
import torch
import torch.nn.functional as F

from numpy_drills.n19_scatter_2d import embedding_grad, scatter_max, splat

rng = np.random.default_rng(0)


def test_splat_accumulates_repeats():
    points = np.array([[0, 0], [2, 3], [2, 3], [1, 1], [2, 3]])
    out = splat(points, np.array([1.0, 0.5, 0.25, 2.0, 1.0]), 3, 4)
    expected = np.zeros((3, 4))
    expected[0, 0], expected[2, 3], expected[1, 1] = 1.0, 1.75, 2.0
    np.testing.assert_allclose(out, expected)


def test_splat_random():
    points = np.c_[rng.integers(0, 5, size=100), rng.integers(0, 6, size=100)]
    weights = rng.random(100)
    expected = np.zeros((5, 6))
    for (r, c), w in zip(points, weights):
        expected[r, c] += w
    np.testing.assert_allclose(splat(points, weights, 5, 6), expected)


def test_embedding_grad_matches_autograd():
    N, T, D, V = 4, 6, 3, 5
    ids, grad_out = rng.integers(0, V, size=(N, T)), rng.normal(size=(N, T, D))
    table = torch.zeros(V, D, dtype=torch.float64, requires_grad=True)
    (F.embedding(torch.tensor(ids), table) * torch.tensor(grad_out)).sum().backward()
    np.testing.assert_allclose(embedding_grad(grad_out, ids, V), table.grad.numpy(), atol=1e-12)


def test_scatter_max():
    out = scatter_max(np.array([1.0, 5.0, -2.0, 3.0, 4.0]), np.array([0, 0, 2, 2, 0]), 4)
    np.testing.assert_array_equal(out, [5.0, -np.inf, 3.0, -np.inf])
    src, idx = rng.normal(size=50), rng.integers(0, 8, size=50)
    expected = [max((s for s, i in zip(src, idx) if i == j), default=-np.inf) for j in range(10)]
    np.testing.assert_array_equal(scatter_max(src, idx, 10), expected)


def test_no_python_loops(no_loops):
    no_loops(splat, embedding_grad, scatter_max)
