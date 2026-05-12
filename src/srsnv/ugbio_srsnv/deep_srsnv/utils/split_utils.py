"""Split assignment utilities for deep SRSNV data preparation."""

from __future__ import annotations

import numpy as np
import torch
from ugbio_featuremap.featuremap_utils import FeatureMapFields

from ugbio_srsnv.split_manifest import (
    SPLIT_MODE_SINGLE_MODEL_CHROM_VAL,
    SPLIT_MODE_SINGLE_MODEL_READ_HASH,
    assign_single_model_chrom_val_role,
    assign_single_model_read_hash_role,
)

CHROM = FeatureMapFields.CHROM.value

_ROLE_TO_SPLIT_ID = {"test": -1, "val": 1}


def _row_split_id(row: dict, *, split_manifest: dict | None, chrom_to_fold: dict[str, int]) -> int:
    chrom_value = row.get(CHROM, row.get("chrom"))
    role: str | None = None
    if split_manifest and split_manifest.get("split_mode") == SPLIT_MODE_SINGLE_MODEL_CHROM_VAL:
        role = assign_single_model_chrom_val_role(chrom=str(chrom_value), manifest=split_manifest)
    elif split_manifest and split_manifest.get("split_mode") == SPLIT_MODE_SINGLE_MODEL_READ_HASH:
        role = assign_single_model_read_hash_role(
            chrom=str(chrom_value),
            rn=str(row["RN"]),
            manifest=split_manifest,
        )
    if role is not None:
        return _ROLE_TO_SPLIT_ID.get(role, 0)
    fold_id = chrom_to_fold.get(str(chrom_value))
    return -1 if fold_id is None else int(fold_id)


def compute_split_ids(
    chroms: list[str] | np.ndarray,
    rns: list[str] | np.ndarray | None,
    split_manifest: dict | None,
    chrom_to_fold: dict[str, int],
) -> torch.Tensor:
    """Compute split_id for each row based on CHROM (and optionally RN).

    For chrom-based split modes, this is O(unique_chroms) -- very fast.
    For read-hash mode, it falls back to per-row computation.
    """
    n = len(chroms)
    needs_per_row = split_manifest and split_manifest.get("split_mode") == SPLIT_MODE_SINGLE_MODEL_READ_HASH

    if needs_per_row:
        split_ids = torch.zeros(n, dtype=torch.int8)
        for i in range(n):
            row = {CHROM: chroms[i], "chrom": chroms[i], "RN": rns[i] if rns else ""}
            split_ids[i] = _row_split_id(row, split_manifest=split_manifest, chrom_to_fold=chrom_to_fold)
        return split_ids

    unique_chroms = sorted({str(c) for c in chroms})
    chrom_to_split: dict[str, int] = {}
    for chrom in unique_chroms:
        row = {CHROM: chrom, "chrom": chrom, "RN": ""}
        chrom_to_split[chrom] = _row_split_id(row, split_manifest=split_manifest, chrom_to_fold=chrom_to_fold)
    return torch.tensor([chrom_to_split[str(c)] for c in chroms], dtype=torch.int8)
