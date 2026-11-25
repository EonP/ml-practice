"""PyTorch drill 02: gather, scatter and friends (target: 12 min). No loops.

Slices, boolean masks and x[rows, cols] work just as in NumPy. The torch-specific tools:
- x.gather(dim, index): along `dim`, pick the entry that `index` names, e.g. for dim=1
  out[i, j] = x[i, index[i, j]]. `index` must have the SAME number of dims as x, so
  you often unsqueeze it first and squeeze the result.
- out.scatter_(dim, index, src): the reverse, writing src into out at those positions.
- out.index_add_(0, idx, src): out[idx[i]] += src[i], with repeated idx adding up
  (np.add.at in NumPy).
- x.masked_fill(mask, value): value where mask is True, x elsewhere.
"""
import torch


def pick_target_logits(logits: torch.Tensor, y):
    """logits (N, K), y (N,) long -> (N,) with out[i] = logits[i, y[i]]. Use gather."""
    return logits[torch.arange(logits.shape[0]), y]
    # return torch.gather(logits, 1, y[:, None]).squeeze(-1)


def pixel_target_logits(logits: torch.Tensor, target):
    """logits (N, K, H, W), target (N, H, W) long -> (N, H, W): at every pixel, the logit
    of that pixel's target class."""
    return torch.gather(logits, 1, target[:, None]).squeeze(1)


def one_hot(y, K):
    """y long of any shape -> float32 tensor of shape y.shape + (K,).
    Use scatter_, not F.one_hot or torch.eye."""
    # return torch.eye(K)[y]
    out = torch.zeros(*y.shape, K, device=y.device)
    out.scatter_(-1, y[..., None], 1.0)
    return out

def causal_fill(scores):
    """scores (..., L, L) -> a copy with every entry ABOVE the diagonal (key j > query i)
    set to -inf, so a softmax over the last axis ignores the future."""
    L = scores.shape[-1]
    i = torch.arange(L, device=scores.device)
    mask = i[None, :] > i[:, None]
    return torch.where(mask, float("-inf"), scores)

def segment_sum(values, segment_ids, n_segments):
    """values (N, D), segment_ids (N,) long in [0, n_segments) -> (n_segments, D), where
    row s is the sum of the rows of values whose id is s (zeros if there are none)."""
    _, D = values.shape
    out = torch.zeros((n_segments, D), dtype=values.dtype, device=values.device)
    out.index_add_(0, segment_ids, values)
    return out


def topk_mask(x: torch.Tensor, k):
    """x (N, D) -> bool (N, D), True at the k largest entries of each row. k may be 0."""
    mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    indices = x.topk(k, dim=1).indices # (N, k)
    mask.scatter_(1, indices, True)
    return mask


