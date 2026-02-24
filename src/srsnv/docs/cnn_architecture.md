# Deep SRSNV CNN Architecture

## Overview

- **~100K trainable parameters** (very compact model)
- **Input**: 36 channels × 300 positions per read (3 categorical embeddings + 12 numeric)
- **Receptive field**: stem kernel 7 + 4 blocks × 2 conv layers × kernel 5 = each output position "sees" ~47 positions of context
- **Masked mean pooling** handles variable-length reads by zeroing out padding positions before averaging

## Architecture Diagram

```
Input: one sequencing read (length L=300 positions)
─────────────────────────────────────────────────────────

┌─────────────────── Embedding Layer ───────────────────┐
│                                                       │
│  read_base_idx [B,300] ──→ Embedding(7,8)  ──→ [B,8,300]   (read bases: A,C,G,T,...)
│  ref_base_idx  [B,300] ──→ Embedding(7,8)  ──→ [B,8,300]   (reference bases)
│  t0_idx        [B,300] ──→ Embedding(11,8) ──→ [B,8,300]   (T0 flow signal)
│  x_num         [B,12,300]                               (numeric: qual, strand, mapq,
│                                                          rq, tp, focus, softclip, etc.)
│                                                       │
│  ──→ Concatenate along channels ──→ [B, 36, 300]      │
└───────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────── Stem Conv ────────────────────────┐
│  Conv1d(36 → 64, kernel=7, pad=3)                     │
│  BatchNorm1d(64)                                      │
│  ReLU                                                 │
│  ──→ [B, 64, 300]                                     │
└───────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────── 4 × Residual Blocks ─────────────────────┐
│                                                       │
│  Each block:                                          │
│  ┌─────────────────────────────────────┐              │
│  │  x ─────────────────────────┐       │              │
│  │  │                          │       │              │
│  │  Conv1d(64→64, k=5, pad=2)  │       │              │
│  │  BatchNorm1d(64)            │       │              │
│  │  ReLU                       │       │              │
│  │  Conv1d(64→64, k=5, pad=2)  │       │              │
│  │  BatchNorm1d(64)            │       │              │
│  │  │                          │       │              │
│  │  └──────── + (add) ─────────┘       │              │
│  │            │                        │              │
│  │           ReLU                       │              │
│  └─────────────────────────────────────┘              │
│  ──→ [B, 64, 300]                                     │
└───────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────── Masked Mean Pooling ────────────────────┐
│  mask [B, 1, 300]  (1 where read has data, 0=padding) │
│  x = x * mask                                        │
│  pooled = sum(x, dim=L) / sum(mask)  ──→ [B, 64]     │
└───────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────── Classification Head ─────────────────┐
│  Linear(64 → 64)                                      │
│  ReLU                                                 │
│  Dropout(0.2)                                         │
│  Linear(64 → 1)  ──→ logit                            │
│                                                       │
│  (sigmoid applied externally for probability)         │
└───────────────────────────────────────────────────────┘
                          │
                          ▼
              Output: P(true variant)
```

## Input Channels Detail

### 3 Categorical Channels (embedded)

| Channel | Vocabulary | Embedding Dim | Description |
|---------|-----------|---------------|-------------|
| **read_base_idx** | 7 | 8 | Read bases (A, C, G, T, N, gap, pad) |
| **ref_base_idx** | 7 | 8 | Reference bases |
| **t0_idx** | 11 | 8 | T0 flow signal (discretized) |

### 5 Per-position Numeric Channels

| Channel | Description |
|---------|-------------|
| **qual** | Base quality (normalized /50) |
| **tp** | T-prime flow signal value |
| **mask** | Valid position indicator (1=data, 0=padding) |
| **focus** | 1 at the variant position, 0 elsewhere |
| **softclip_mask** | 1 if position is soft-clipped |

### 7 Constant (read-level) Numeric Channels

| Channel | Description |
|---------|-------------|
| **strand** | Reverse strand (0/1) |
| **mapq** | Mapping quality (normalized /60) |
| **rq** | Read quality |
| **tm** | Trim mode (encoded as integer) |
| **st** | Sample type (encoded as integer) |
| **et** | End type (encoded as integer) |
| **mixed** | Mixed flag |
