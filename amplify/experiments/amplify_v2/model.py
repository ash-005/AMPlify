"""AMPlify-v2 model wrapper that composes backbone + pooling.

Designed for dependency injection so backbones and pooling are pluggable.
"""
from typing import Optional
import torch
import torch.nn as nn


class AMPlifyV2(nn.Module):
    def __init__(self, backbone: nn.Module, pooling: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.classifier = None

    def build_classifier(self, pooled_dim: int, output_dim: int = 1, global_feat_dim: Optional[int] = None, esm_embed_dim: Optional[int] = None):
        add_dim = 0
        if global_feat_dim:
            add_dim += global_feat_dim
        if esm_embed_dim:
            add_dim += esm_embed_dim
        in_dim = pooled_dim + add_dim
        self.classifier = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, max(64, in_dim // 2)), nn.ReLU(), nn.Dropout(0.2), nn.Linear(max(64, in_dim // 2), output_dim))

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, global_feats: Optional[torch.Tensor] = None, esm_emb: Optional[torch.Tensor] = None):
        # x: (batch, seq_len, input_dim)
        h = self.backbone(x, lengths)
        pooled, attn = self.pooling(h)
        parts = [pooled]
        if global_feats is not None:
            parts.append(global_feats)
        if esm_emb is not None:
            parts.append(esm_emb)
        feat = torch.cat(parts, dim=-1)
        if self.classifier is None:
            raise RuntimeError("Classifier not built. Call build_classifier(...) with proper dimensions before training.")
        out = self.classifier(feat)
        return out.squeeze(-1), attn
