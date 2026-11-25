"""PyTorch drill 01: tensor basics (target: 12 min).

Most PyTorch bugs are shape bugs. Four facts to keep straight:
- view() never copies, so it needs memory laid out the right way (contiguous).
  reshape() returns a view when it can and copies when it must.
- permute() and transpose() only reorder strides: the result is usually NOT
  contiguous, so a view() on it raises.
- torch.stack adds a NEW axis; torch.cat joins along an EXISTING one.
- squeeze() with no argument removes EVERY size-1 axis, including a batch of 1.
"""
import torch


def like(x, data):
    """A new tensor holding `data` (nested lists or a NumPy array), with x's dtype and device."""
    return torch.tensor(data, dtype=x.dtype, device=x.device)


def batch_images(images):
    """List of N tensors, each (C, H, W) -> one batch (N, C, H, W)."""
    return torch.stack(images)


def concat_sequences(seqs):
    """List of tensors (L_i, D) with different lengths L_i -> one tensor (sum of the L_i, D)."""
    return torch.cat(seqs, dim=0)


def flatten_features(x):
    """(N, C, H, W) -> (N, C * H * W), keeping the batch axis even when N == 1.

    Return a view when x is contiguous, and still work when it isn't.
    """
    return torch.flatten(x, start_dim=1)


def to_channels_last(x):
    """(N, C, H, W) -> (N, H, W, C), returned as a contiguous tensor."""
    return torch.permute(x, (0, 2, 3, 1)).contiguous()


def split_heads(x: torch.Tensor, n_heads):
    """(B, L, D) -> (B, n_heads, L, D // n_heads). Head h gets features h * dh to (h + 1) * dh - 1."""
    B, L, D = x.shape
    y = x.reshape(B, L, n_heads, D // n_heads)
    return y.transpose(1, 2)

def merge_heads(x: torch.Tensor):
    """Inverse of split_heads: (B, n_heads, L, dh) -> (B, L, n_heads * dh).

    x may be any tensor of that shape (e.g. the output of attention), not only a view
    straight out of split_heads.
    """
    B, n_heads, L, dh = x.shape
    return x.transpose(1, 2).reshape(B, L, n_heads * dh)


def squeeze_last(x):
    """(N, 1) -> (N,). The result must keep the batch axis when N == 1."""
    return x.squeeze(dim=-1)
