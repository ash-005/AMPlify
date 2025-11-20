"""Modular sequence backbones for AMPlify-v2.

Lightweight, dependency-minimal implementations suitable for research.
"""
from typing import Optional
import torch
import torch.nn as nn


class SequenceBackbone(nn.Module):
    """Abstract backbone interface."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        raise NotImplementedError()


class BiLSTMBackbone(SequenceBackbone):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        # x: (batch, seq_len, input_dim)
        outputs, _ = self.rnn(x)
        return outputs  # (batch, seq_len, 2*hidden_dim)


class TransformerBackbone(SequenceBackbone):
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        # x: (batch, seq_len, input_dim)
        h = self.input_proj(x)
        out = self.encoder(h)
        return out  # (batch, seq_len, d_model)


class ConformerBlock(nn.Module):
    """A tiny Conformer-like block (feed-forward + attention + conv)."""

    def __init__(self, d_model: int, nhead: int = 4, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size // 2), groups=1),
            nn.GLU(dim=1) if False else nn.Identity(),
            nn.LayerNorm(d_model),
        )
        self.ff2 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        residual = x
        x = residual + 0.5 * self.ff1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        # conv expects (batch, channels, seq)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv[0](x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        x = x + 0.5 * self.ff2(x)
        return x


class ConformerBackbone(SequenceBackbone):
    def __init__(self, input_dim: int, d_model: int = 256, num_blocks: int = 2, nhead: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, nhead=nhead) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        h = self.input_proj(x)
        for b in self.blocks:
            h = b(h)
        return h
