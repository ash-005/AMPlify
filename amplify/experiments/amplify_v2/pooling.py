"""Pooling layers for aggregating sequence features."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolingLayer(nn.Module):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError()


class SingleHeadAttentionPooling(PoolingLayer):
    def __init__(self, input_dim: int):
        super().__init__()
        self.context = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (batch, seq_len, dim)
        scores = self.context(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        out = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return out, weights


class MultiHeadPooling(PoolingLayer):
    def __init__(self, input_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.head_dim = input_dim // num_heads
        self.attn = nn.ModuleList([nn.Linear(self.head_dim, 1) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch, seq_len, dim = x.size()
        h = x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        outs = []
        weights_list = []
        for i in range(self.num_heads):
            scores = self.attn[i](h[:, i]) .squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            w = torch.softmax(scores, dim=-1)
            pooled = torch.bmm(w.unsqueeze(1), h[:, i]).squeeze(1)  # (batch, head_dim)
            outs.append(pooled)
            weights_list.append(w)
        out = torch.cat(outs, dim=-1)
        weights = torch.stack(weights_list, dim=1)
        return out, weights


class CLSPooling(PoolingLayer):
    def __init__(self, input_dim: int, cls_index: int = 0):
        super().__init__()
        self.cls_index = cls_index

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Take a fixed-position token as CLS
        out = x[:, self.cls_index, :]
        return out, None
