"""Alignment channel extraction utilities for deep SRSNV tensorization."""

from __future__ import annotations

import numpy as np
import pysam

CIGAR_SOFT_CLIP = 4


def _to_numpy_tp(tp_value, read_len: int) -> np.ndarray:
    arr = np.zeros(read_len, dtype=np.float32)
    if tp_value is None:
        return arr
    src = list(tp_value)
    n = min(read_len, len(src))
    arr[:n] = np.asarray(src[:n], dtype=np.float32)
    return arr


def _to_string_t0(t0_value, read_len: int) -> str:
    if not t0_value:
        return ""
    return str(t0_value)[:read_len]


def _compute_softclip_positions(cigar: list[tuple[int, int]]) -> set[int]:
    """Identify query positions that belong to soft-clipped CIGAR segments."""
    softclip_query_positions: set[int] = set()
    q_cursor = 0
    for op, length in cigar:
        if op == CIGAR_SOFT_CLIP:
            softclip_query_positions.update(range(q_cursor, q_cursor + length))
            q_cursor += length
        elif op in {0, 1, 7, 8}:  # M, I, =, X consume query
            q_cursor += length
        # D, N, H, P do not consume query
    return softclip_query_positions


def _process_aligned_pair(
    qpos,
    rpos,
    rbase,
    *,
    read_seq: str,
    read_quals,
    tp_raw: np.ndarray,
    t0_raw: str,
    snv_pos_1based: int,
    softclip_query_positions: set[int],
) -> tuple[str, str, float, float, float, float, float]:
    """Process a single aligned pair and return channel values.

    Returns (read_base, ref_base, qual, tp_val, t0_val, is_focus, is_softclip).
    """
    if qpos is None:
        read_base = "<GAP>"
        qual = 0.0
        tp_val = 0.0
        t0_val = 0.0
    else:
        read_base = read_seq[qpos].upper() if qpos < len(read_seq) else "N"
        qual = float(read_quals[qpos]) if qpos < len(read_quals) else 0.0
        tp_val = float(tp_raw[qpos]) if qpos < len(tp_raw) else 0.0
        t0_val = max(0.0, ord(t0_raw[qpos]) - 33) if qpos < len(t0_raw) else 0.0

    if rpos is None:
        ref_base = "<GAP>"
    elif rbase is None:
        ref_base = "N"
    else:
        ref_base = str(rbase).upper()
        if ref_base not in {"A", "C", "G", "T", "N"}:
            ref_base = "N"

    is_focus = 1.0 if (rpos is not None and (rpos + 1) == snv_pos_1based) else 0.0
    is_softclip = 1.0 if (qpos is not None and qpos in softclip_query_positions) else 0.0

    return read_base, ref_base, qual, tp_val, t0_val, is_focus, is_softclip


def _build_gapped_channels(
    rec: pysam.AlignedSegment,
    snv_pos_1based: int,
    tp_raw: np.ndarray,
    t0_raw: str,
    positive_focus_ref_override: str | None,
) -> dict:
    read_seq = rec.query_sequence or ""
    read_quals = rec.query_qualities if rec.query_qualities is not None else []
    aligned_pairs = rec.get_aligned_pairs(matches_only=False, with_seq=True)
    cigar = rec.cigartuples or []

    softclip_query_positions = _compute_softclip_positions(cigar)

    read_base_aln: list[str] = []
    ref_base_aln: list[str] = []
    qual_aln: list[float] = []
    tp_aln: list[float] = []
    t0_aln: list[float] = []
    focus_aln: list[float] = []
    softclip_mask_aln: list[float] = []
    focus_indices: list[int] = []

    for qpos, rpos, rbase in aligned_pairs:
        read_base, ref_base, qual, tp_val, t0_val, is_focus, is_softclip = _process_aligned_pair(
            qpos,
            rpos,
            rbase,
            read_seq=read_seq,
            read_quals=read_quals,
            tp_raw=tp_raw,
            t0_raw=t0_raw,
            snv_pos_1based=snv_pos_1based,
            softclip_query_positions=softclip_query_positions,
        )
        if is_focus == 1.0:
            focus_indices.append(len(focus_aln))

        read_base_aln.append(read_base)
        ref_base_aln.append(ref_base)
        qual_aln.append(qual)
        tp_aln.append(tp_val)
        t0_aln.append(t0_val)
        focus_aln.append(is_focus)
        softclip_mask_aln.append(is_softclip)

    # If the SNV position was not present on aligned reference positions, keep focus all-zero.
    # For positive records, mimic xgboost semantics by using X_ALT as the effective ref base at focus.
    if positive_focus_ref_override and focus_indices:
        focus_ref = positive_focus_ref_override.upper()
        if focus_ref in {"A", "C", "G", "T", "N"}:
            ref_base_aln[focus_indices[0]] = focus_ref

    return {
        "read_base_aln": read_base_aln,
        "ref_base_aln": ref_base_aln,
        "qual_aln": np.asarray(qual_aln, dtype=np.float32),
        "tp_aln": np.asarray(tp_aln, dtype=np.float32),
        "t0_aln": np.asarray(t0_aln, dtype=np.float32),
        "focus_aln": np.asarray(focus_aln, dtype=np.float32),
        "softclip_mask_aln": np.asarray(softclip_mask_aln, dtype=np.float32),
    }
