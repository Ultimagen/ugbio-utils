"""Data preparation: backward-compatibility re-exports.

Vocabulary, alignment, and split utilities have been extracted to:
- ``utils.vocab`` — Encoders, load_vocab_config, channel constants
- ``utils.alignment`` — _to_numpy_tp, _to_string_t0, _build_gapped_channels
- ``utils.split_utils`` — _row_split_id, compute_split_ids

The standalone preprocessing path (BAM reading, tensor cache building,
and the DeepSRSNVDataset class) has been removed. The pipeline always
uses the fold-dir path via ``preprocessing.cram_to_tensors`` and
``preprocessing.combine_splits``.
"""

from __future__ import annotations

# Re-export extracted utilities for backward compatibility
from ugbio_srsnv.deep_srsnv.utils.alignment import (  # noqa: F401
    CIGAR_SOFT_CLIP,
    _build_gapped_channels,
    _compute_softclip_positions,
    _process_aligned_pair,
    _to_numpy_tp,
    _to_string_t0,
)
from ugbio_srsnv.deep_srsnv.utils.split_utils import (  # noqa: F401
    _row_split_id,
    compute_split_ids,
)
from ugbio_srsnv.deep_srsnv.utils.vocab import (  # noqa: F401
    CHANNEL_ORDER,
    NUM_CHANNELS_CONST,
    NUM_CHANNELS_POS,
    NUMERIC_CHANNELS,
    Encoders,
    load_vocab_config,
)
