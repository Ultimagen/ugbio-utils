from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CNNReadClassifier(nn.Module):
    def __init__(
        self,
        base_vocab_size: int,
        t0_vocab_size: int,
        numeric_channels: int,
        base_embed_dim: int = 8,
        ref_embed_dim: int = 8,
        t0_embed_dim: int = 8,
        hidden_channels: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.read_base_emb = nn.Embedding(base_vocab_size, base_embed_dim, padding_idx=0)
        self.ref_base_emb = nn.Embedding(base_vocab_size, ref_embed_dim, padding_idx=0)
        self.t0_emb = nn.Embedding(t0_vocab_size, t0_embed_dim, padding_idx=0)
        in_ch = base_embed_dim + ref_embed_dim + t0_embed_dim + numeric_channels

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock1D(hidden_channels) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        read_base_idx: torch.Tensor,
        ref_base_idx: torch.Tensor,
        t0_idx: torch.Tensor,
        x_num: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Inputs:
        # - read_base_idx: [B, L]
        # - ref_base_idx: [B, L]
        # - t0_idx: [B, L]
        # - x_num: [B, C_num, L]
        # - mask: [B, L]
        read_e = self.read_base_emb(read_base_idx).transpose(1, 2)
        ref_e = self.ref_base_emb(ref_base_idx).transpose(1, 2)
        t0_e = self.t0_emb(t0_idx).transpose(1, 2)
        x = torch.cat([read_e, ref_e, t0_e, x_num], dim=1)
        x = self.stem(x)
        x = self.blocks(x)

        # masked mean pooling over length
        mask = mask.unsqueeze(1)
        x = x * mask
        pooled = x.sum(dim=2) / torch.clamp(mask.sum(dim=2), min=1.0)
        logits = self.head(pooled).squeeze(1)
        return logits
