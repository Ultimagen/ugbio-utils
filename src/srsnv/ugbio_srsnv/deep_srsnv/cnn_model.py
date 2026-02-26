from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as f_nn

FOCUS_CHANNEL_IDX = 3


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CNNReadClassifier(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        base_vocab_size: int,
        t0_vocab_size: int,
        numeric_channels: int,
        tm_vocab_size: int = 1,
        st_vocab_size: int = 1,
        et_vocab_size: int = 1,
        base_embed_dim: int = 16,
        ref_embed_dim: int = 16,
        t0_embed_dim: int = 16,
        cat_embed_dim: int = 4,
        hidden_channels: int = 128,
        n_blocks: int = 6,
        dropout: float = 0.3,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        self.read_base_emb = nn.Embedding(base_vocab_size, base_embed_dim, padding_idx=0)
        self.ref_base_emb = nn.Embedding(base_vocab_size, ref_embed_dim, padding_idx=0)
        self.t0_emb = nn.Embedding(t0_vocab_size, t0_embed_dim, padding_idx=0)

        cat_ch = 0
        self.has_cat_embeds = tm_vocab_size > 1 or st_vocab_size > 1 or et_vocab_size > 1
        if tm_vocab_size > 1:
            self.tm_emb = nn.Embedding(tm_vocab_size, cat_embed_dim, padding_idx=0)
            cat_ch += cat_embed_dim
        if st_vocab_size > 1:
            self.st_emb = nn.Embedding(st_vocab_size, cat_embed_dim, padding_idx=0)
            cat_ch += cat_embed_dim
        if et_vocab_size > 1:
            self.et_emb = nn.Embedding(et_vocab_size, cat_embed_dim, padding_idx=0)
            cat_ch += cat_embed_dim

        in_ch = base_embed_dim + ref_embed_dim + t0_embed_dim + numeric_channels + cat_ch

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        if dilations is None:
            dilations = [1, 1, 2, 2, 4, 4][:n_blocks]
            if len(dilations) < n_blocks:
                dilations.extend([dilations[-1]] * (n_blocks - len(dilations)))
        self.blocks = nn.Sequential(*[ResidualBlock1D(hidden_channels, dilation=d) for d in dilations])

        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 4),
            nn.Tanh(),
            nn.Linear(hidden_channels // 4, 1),
        )

        head_in = hidden_channels * 2
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        read_base_idx: torch.Tensor,
        ref_base_idx: torch.Tensor,
        t0_idx: torch.Tensor,
        x_num: torch.Tensor,
        mask: torch.Tensor,
        tm_idx: torch.Tensor | None = None,
        st_idx: torch.Tensor | None = None,
        et_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = read_base_idx.shape[0]
        seq_len = read_base_idx.shape[1]

        read_e = self.read_base_emb(read_base_idx).transpose(1, 2)
        ref_e = self.ref_base_emb(ref_base_idx).transpose(1, 2)
        t0_e = self.t0_emb(t0_idx).transpose(1, 2)

        parts = [read_e, ref_e, t0_e, x_num]

        if self.has_cat_embeds:
            if tm_idx is not None and hasattr(self, "tm_emb"):
                parts.append(self.tm_emb(tm_idx).unsqueeze(2).expand(-1, -1, seq_len))
            if st_idx is not None and hasattr(self, "st_emb"):
                parts.append(self.st_emb(st_idx).unsqueeze(2).expand(-1, -1, seq_len))
            if et_idx is not None and hasattr(self, "et_emb"):
                parts.append(self.et_emb(et_idx).unsqueeze(2).expand(-1, -1, seq_len))

        x = torch.cat(parts, dim=1)

        focus_pos = x_num[:, FOCUS_CHANNEL_IDX, :].argmax(dim=1)

        x = self.stem(x)
        x = self.blocks(x)

        mask_bool = mask.bool()
        x_t = x.transpose(1, 2)
        attn_scores = self.attn_proj(x_t).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask_bool, torch.finfo(attn_scores.dtype).min)
        attn_weights = f_nn.softmax(attn_scores, dim=1)
        attn_pooled = (x * attn_weights.unsqueeze(1)).sum(dim=2)

        batch_idx = torch.arange(batch_size, device=x.device)
        focus_feat = x[batch_idx, :, focus_pos]

        pooled = torch.cat([attn_pooled, focus_feat], dim=1)
        logits = self.head(pooled).squeeze(1)
        return logits
