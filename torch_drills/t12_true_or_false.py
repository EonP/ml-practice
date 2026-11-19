"""PyTorch drill 12: true or false? (target: 10 min). No code: predictions only.

Replace every None with True or False BEFORE running anything. Then run the tests:
each failure names a statement you got wrong. Work out why before moving on. Most of
these are bugs that people ship.

Unless a statement says otherwise, x is a float32 CPU tensor, w is a float32 tensor
with requires_grad=True, and model is an nn.Module.
"""

STATEMENTS = {
    # memory: views and copies
    "torch.from_numpy(a) shares memory with the NumPy array a": None,
    "torch.tensor(a) shares memory with the NumPy array a": None,
    "x.numpy() shares memory with x": None,
    "x.detach() shares memory with x": None,
    "x.T.view(-1) works for x of shape (3, 4)": None,
    "x.T.reshape(-1) shares memory with x, for x of shape (3, 4)": None,
    "x.contiguous() makes a copy even when x is already contiguous": None,
    "x.float() returns x itself (the same object) when x is already float32": None,
    "x.expand(4, 3) copies the data 4 times, for x of shape (1, 3)": None,
    "torch.zeros(1, 3, 1).squeeze() has shape (3,)": None,
    # autograd
    "w * 2 has requires_grad=True": None,
    "w * 2 computed inside torch.no_grad() has requires_grad=True": None,
    "Calling backward() twice (with retain_graph=True) without zeroing doubles w.grad": None,
    "x.clone() of a tensor that requires grad cuts the graph: no gradient flows back through it": None,
    "torch.argmax(w) has requires_grad=True, so you can backpropagate through it": None,
    "torch.tensor([1, 2], requires_grad=True) works": None,
    "After total = total + loss (rather than loss.item()), total still holds the graph (it has a grad_fn)": None,
    # modules and optimisers
    "model.eval() turns off gradient tracking": None,
    "torch.no_grad() makes dropout layers stop dropping": None,
    "Layers kept in a plain Python list attribute are included in model.parameters()": None,
    "A tensor saved with register_buffer appears in model.parameters()": None,
    "A tensor saved with register_buffer appears in model.state_dict()": None,
    "model.state_dict() returns copies, so training afterwards doesn't change it": None,
    "optimizer.zero_grad() sets every .grad to None (by default)": None,
    "model.parameters() returns a list, so you can loop over it twice": None,
    # losses and layers
    "F.cross_entropy expects probabilities (softmax already applied) as its input": None,
    "F.cross_entropy accepts float class probabilities of shape (N, K) as the target": None,
    "F.cross_entropy raises an error if the class-index target is int32 instead of int64": None,
    "nn.BatchNorm1d in training mode raises an error on a batch of one sample": None,
    "After one torch.manual_seed(0), two calls to torch.randn(3) return the same values": None,
}
