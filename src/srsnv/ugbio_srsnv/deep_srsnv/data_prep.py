from __future__ import annotations

import hashlib
import json
import os
import pickle
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pysam
import torch
from pyarrow import parquet as pq
from torch.utils.data import Dataset
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields

from ugbio_srsnv.split_manifest import SPLIT_MODE_SINGLE_MODEL_READ_HASH, assign_single_model_read_hash_role

CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value
REF = FeatureMapFields.REF.value
ALT = FeatureMapFields.ALT.value
X_ALT = FeatureMapFields.X_ALT.value


@dataclass
class Encoders:
    base_vocab: dict[str, int]
    t0_vocab: dict[str, int]
    tm_vocab: dict[str, int]
    st_vocab: dict[str, int]
    et_vocab: dict[str, int]


_WORKER_STATE: dict[str, object] = {}
_DEFAULT_T0_TOKENS = ["<GAP>", "D", ":", "-", "A", "C", "G", "T", "N"]
_DEFAULT_TM_VALUES = ["A", "Q", "Z", "AQ", "AZ", "QZ", "AQZ"]
_DEFAULT_ST_ET_VALUES = ["PLUS", "MINUS", "MIXED", "UNDETERMINED"]


def _default_base_vocab() -> dict[str, int]:
    return {"<PAD>": 0, "<GAP>": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6}


def _create_categorical_vocab(values: list[str], *, add_missing: bool = True) -> dict[str, int]:
    uniq = sorted({v for v in values if v is not None and v != ""})
    vocab = {"<PAD>": 0}
    if add_missing:
        vocab["<MISSING>"] = 1
    offset = len(vocab)
    for i, v in enumerate(uniq, start=offset):
        vocab[v] = i
    return vocab


def _record_tags(rec: pysam.AlignedSegment) -> dict:
    return dict(rec.get_tags(with_value_type=False))


def _get_worker_resources(positive_bam: str, negative_bam: str) -> dict[str, object]:
    cache_key = f"{positive_bam}::{negative_bam}"
    cached = _WORKER_STATE.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    bams = {
        True: pysam.AlignmentFile(positive_bam, "rb"),
        False: pysam.AlignmentFile(negative_bam, "rb"),
    }
    idx = {k: pysam.IndexedReads(v) for k, v in bams.items()}
    for value in idx.values():
        value.build()
    payload = {"bams": bams, "idx": idx}
    _WORKER_STATE[cache_key] = payload
    return payload


def _match_bam_read(
    bam: pysam.AlignmentFile,
    index: pysam.IndexedReads,
    chrom: str,
    pos: int,
    rn: str,
) -> pysam.AlignedSegment | None:
    try:
        reads = list(index.find(rn))
    except KeyError:
        return None
    if not reads:
        return None
    same_chrom = [r for r in reads if not r.is_unmapped and bam.get_reference_name(r.reference_id) == chrom]
    if not same_chrom:
        return None

    # Prefer reads that overlap the locus.
    overlap = [r for r in same_chrom if r.reference_start <= (pos - 1) < r.reference_end]
    if overlap:
        return overlap[0]
    # Fallback: closest start coordinate.
    return sorted(same_chrom, key=lambda r: abs((r.reference_start + 1) - pos))[0]


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


def _load_labeled_df(
    positive_parquet: str, negative_parquet: str, max_rows_per_class: int | None = None
) -> pl.DataFrame:
    pos_df = pl.read_parquet(positive_parquet).with_columns(pl.lit(value=True).alias("label"))
    neg_df = pl.read_parquet(negative_parquet).with_columns(pl.lit(value=False).alias("label"))
    if max_rows_per_class:
        pos_df = pos_df.head(max_rows_per_class)
        neg_df = neg_df.head(max_rows_per_class)
    return pl.concat([pos_df, neg_df], how="diagonal")


def _resource_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _build_cache_key(
    *,
    positive_parquet: str,
    negative_parquet: str,
    positive_bam: str,
    negative_bam: str,
    split_manifest: dict | None,
    chrom_to_fold: dict[str, int],
    tensor_length: int,
    batch_rows: int,
    cache_version: int = 5,
) -> str:
    payload = {
        "cache_version": cache_version,
        "positive_parquet": str(Path(positive_parquet).resolve()),
        "negative_parquet": str(Path(negative_parquet).resolve()),
        "positive_bam": str(Path(positive_bam).resolve()),
        "negative_bam": str(Path(negative_bam).resolve()),
        "positive_parquet_mtime_ns": os.stat(positive_parquet).st_mtime_ns,
        "negative_parquet_mtime_ns": os.stat(negative_parquet).st_mtime_ns,
        "positive_bam_mtime_ns": os.stat(positive_bam).st_mtime_ns,
        "negative_bam_mtime_ns": os.stat(negative_bam).st_mtime_ns,
        "split_manifest": split_manifest or {},
        "chrom_to_fold": chrom_to_fold,
        "tensor_spec": {
            "length": tensor_length,
            "channels": [
                "read_base_idx",
                "ref_base_idx",
                "t0_idx",
                "qual",
                "tp",
                "mask",
                "focus",
                "softclip_mask",
                "strand",
                "mapq",
                "rq",
                "tm",
                "st",
                "et",
                "mixed",
            ],
        },
        "batch_rows": int(batch_rows),
        "encoder_source": "known",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _iter_parquet_rows(
    parquet_path: str,
    *,
    label: bool,
    columns: list[str],
    batch_rows: int,
    max_rows: int | None,
):
    parquet = pq.ParquetFile(parquet_path)
    emitted = 0
    requested_cols = [c for c in columns if c in (parquet.schema.names or [])]
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=requested_cols):
        table = batch.to_pydict()
        if not table:
            continue
        n_rows = len(next(iter(table.values())))
        rows = []
        for i in range(n_rows):
            row = {k: table[k][i] for k in table}
            row["label"] = bool(label)
            rows.append(row)
        if max_rows is not None:
            remaining = max_rows - emitted
            if remaining <= 0:
                return
            rows = rows[:remaining]
        if rows:
            emitted += len(rows)
            yield rows
        if max_rows is not None and emitted >= max_rows:
            return


def _estimate_shard_rows(
    *,
    preprocess_max_ram_gb: float,
    preprocess_num_workers: int,
    requested_batch_rows: int,
) -> int:
    # Heuristic for row expansion after BAM lookup + aligned channels.
    bytes_per_row = 4096
    budget_per_worker_bytes = (preprocess_max_ram_gb * (1024**3)) / max(1, preprocess_num_workers)
    hard_cap_rows = int(max(1000, budget_per_worker_bytes / bytes_per_row))
    return max(1000, min(requested_batch_rows, hard_cap_rows))


def _row_split_id(row: dict, *, split_manifest: dict | None, chrom_to_fold: dict[str, int]) -> int:
    chrom_value = row.get(CHROM, row.get("chrom"))
    if split_manifest and split_manifest.get("split_mode") == SPLIT_MODE_SINGLE_MODEL_READ_HASH:
        role = assign_single_model_read_hash_role(
            chrom=str(chrom_value),
            rn=str(row["RN"]),
            manifest=split_manifest,
        )
        if role == "test":
            return -1
        if role == "val":
            return 1
        return 0
    fold_id = chrom_to_fold.get(str(chrom_value))
    return -1 if fold_id is None else int(fold_id)


def _compute_max_edist(parquet_path: str) -> int | None:
    edist_col = FeatureMapFields.EDIST.value
    pf = pq.ParquetFile(parquet_path)
    if edist_col not in (pf.schema.names or []):
        return None
    max_val = None
    for batch in pf.iter_batches(batch_size=100_000, columns=[edist_col]):
        col = batch.column(edist_col)
        batch_max = col.to_pylist()
        local_max = max(v for v in batch_max if v is not None)
        if max_val is None or local_max > max_val:
            max_val = local_max
    return max_val


def _build_split_buckets(
    *,
    positive_parquet: str,
    negative_parquet: str,
    columns: list[str],
    batch_rows: int,
    max_rows_per_class: int | None,
    split_manifest: dict | None,
    chrom_to_fold: dict[str, int],
) -> dict[int, dict[bool, list[dict]]]:
    max_edist = _compute_max_edist(positive_parquet)
    edist_col = FeatureMapFields.EDIST.value
    if max_edist is not None:
        logger.info("Positive EDIST filter: dropping rows with EDIST == %s (matches XGBoost pipeline)", max_edist)

    n_edist_dropped = 0
    buckets: dict[int, dict[bool, list[dict]]] = {}
    for label, parquet_path in ((True, positive_parquet), (False, negative_parquet)):
        for rows in _iter_parquet_rows(
            parquet_path,
            label=label,
            columns=columns,
            batch_rows=batch_rows,
            max_rows=max_rows_per_class,
        ):
            for row in rows:
                if label and max_edist is not None and row.get(edist_col) == max_edist:
                    n_edist_dropped += 1
                    continue
                sid = _row_split_id(row, split_manifest=split_manifest, chrom_to_fold=chrom_to_fold)
                if sid not in buckets:
                    buckets[sid] = {True: [], False: []}
                buckets[sid][bool(label)].append(row)
    if n_edist_dropped > 0:
        logger.info("Dropped %d positive rows with EDIST == %s", n_edist_dropped, max_edist)
    return buckets


def _build_balanced_shard_inputs(  # noqa: C901
    *,
    split_buckets: dict[int, dict[bool, list[dict]]],
    batch_rows: int,
    random_seed: int,
) -> tuple[list[tuple[int, list[dict], int]], dict[str, dict]]:
    shard_inputs: list[tuple[int, list[dict], int]] = []
    split_input_stats: dict[str, dict] = {}
    shard_id = 0
    for sid in sorted(split_buckets.keys()):
        pos_rows = list(split_buckets[sid].get(True, []))
        neg_rows = list(split_buckets[sid].get(False, []))
        split_seed = int(random_seed + (sid * 17) + 1337)
        rng = np.random.default_rng(split_seed)
        if pos_rows:
            rng.shuffle(pos_rows)
        if neg_rows:
            rng.shuffle(neg_rows)

        split_total = len(pos_rows) + len(neg_rows)
        split_input_stats[str(sid)] = {
            "split_id": int(sid),
            "rows": int(split_total),
            "positives": int(len(pos_rows)),
            "negatives": int(len(neg_rows)),
            "prevalence": (len(pos_rows) / split_total) if split_total else None,
        }
        if split_total == 0:
            continue

        p_idx = 0
        n_idx = 0
        while p_idx < len(pos_rows) or n_idx < len(neg_rows):
            remaining = batch_rows
            take_pos = min((batch_rows + 1) // 2, len(pos_rows) - p_idx)
            take_neg = min(batch_rows // 2, len(neg_rows) - n_idx)
            rows = []
            if take_pos > 0:
                rows.extend(pos_rows[p_idx : p_idx + take_pos])
                p_idx += take_pos
                remaining -= take_pos
            if take_neg > 0:
                rows.extend(neg_rows[n_idx : n_idx + take_neg])
                n_idx += take_neg
                remaining -= take_neg
            if remaining > 0:
                extra_pos = min(remaining, len(pos_rows) - p_idx)
                if extra_pos > 0:
                    rows.extend(pos_rows[p_idx : p_idx + extra_pos])
                    p_idx += extra_pos
                    remaining -= extra_pos
            if remaining > 0:
                extra_neg = min(remaining, len(neg_rows) - n_idx)
                if extra_neg > 0:
                    rows.extend(neg_rows[n_idx : n_idx + extra_neg])
                    n_idx += extra_neg
                    remaining -= extra_neg
            if not rows:
                break
            shard_rng = np.random.default_rng(split_seed + shard_id + 97)
            shard_rng.shuffle(rows)
            shard_inputs.append((shard_id, rows, int(sid)))
            shard_id += 1
    return shard_inputs, split_input_stats


def _build_gapped_channels(  # noqa: C901, PLR0912, PLR0915
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

    # Mark query indices that belong to soft-clipped CIGAR segments.
    # CIGAR op 4 = soft clip.
    softclip_query_positions: set[int] = set()
    q_cursor = 0
    for op, length in cigar:
        if op == 4:  # S  # noqa: PLR2004
            softclip_query_positions.update(range(q_cursor, q_cursor + length))
            q_cursor += length
        elif op in {0, 1, 7, 8}:  # M, I, =, X consume query
            q_cursor += length
        else:
            # D, N, H, P do not consume query
            continue

    read_base_aln: list[str] = []
    ref_base_aln: list[str] = []
    qual_aln: list[float] = []
    tp_aln: list[float] = []
    t0_aln: list[str] = []
    focus_aln: list[float] = []
    softclip_mask_aln: list[float] = []
    focus_indices: list[int] = []

    for qpos, rpos, rbase in aligned_pairs:
        # Read channel follows query sequence (or gap on deletion).
        if qpos is None:
            read_base = "<GAP>"
            qual = 0.0
            tp_val = 0.0
            t0_token = "<GAP>"  # noqa: S105
        else:
            read_base = read_seq[qpos].upper() if qpos < len(read_seq) else "N"
            qual = float(read_quals[qpos]) if qpos < len(read_quals) else 0.0
            tp_val = float(tp_raw[qpos]) if qpos < len(tp_raw) else 0.0
            t0_token = t0_raw[qpos] if qpos < len(t0_raw) else "<MISSING>"

        # Ref channel follows aligned reference (or gap on insertion).
        if rpos is None:
            ref_base = "<GAP>"
        elif rbase is None:
            ref_base = "N"
        else:
            # pysam can return lowercase for mismatches; normalize.
            ref_base = str(rbase).upper()
            if ref_base not in {"A", "C", "G", "T", "N"}:
                ref_base = "N"

        is_focus = 1.0 if (rpos is not None and (rpos + 1) == snv_pos_1based) else 0.0
        if is_focus == 1.0:
            focus_indices.append(len(focus_aln))
        is_softclip = 1.0 if (qpos is not None and qpos in softclip_query_positions) else 0.0

        read_base_aln.append(read_base)
        ref_base_aln.append(ref_base)
        qual_aln.append(qual)
        tp_aln.append(tp_val)
        t0_aln.append(t0_token)
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
        "t0_aln": t0_aln,
        "focus_aln": np.asarray(focus_aln, dtype=np.float32),
        "softclip_mask_aln": np.asarray(softclip_mask_aln, dtype=np.float32),
    }


def _process_rows_shard(
    *,
    shard_id: int,
    rows: list[dict],
    positive_bam: str,
    negative_bam: str,
    chrom_to_fold: dict[str, int],
) -> tuple[int, list[dict], dict]:
    t0 = time.perf_counter()
    missing = 0
    resources = _get_worker_resources(positive_bam, negative_bam)
    bams = resources["bams"]  # type: ignore[index]
    idx = resources["idx"]  # type: ignore[index]
    records: list[dict] = []

    for row in rows:
        label = bool(row["label"])
        bam = bams[label]
        rec = _match_bam_read(
            bam=bam,
            index=idx[label],
            chrom=row[CHROM],
            pos=int(row[POS]),
            rn=row["RN"],
        )
        if rec is None:
            missing += 1
            continue
        tags = _record_tags(rec)
        read_len = len(rec.query_sequence or "")
        tp_raw = _to_numpy_tp(tags.get("tp"), read_len=read_len)
        t0_raw = _to_string_t0(tags.get("t0"), read_len=read_len)
        is_positive = int(label) == 1
        positive_focus_ref_override = str(row.get(X_ALT) or "").upper() if is_positive else None
        aligned = _build_gapped_channels(
            rec=rec,
            snv_pos_1based=int(row[POS]),
            tp_raw=tp_raw,
            t0_raw=t0_raw,
            positive_focus_ref_override=positive_focus_ref_override,
        )
        st_value = tags.get("st", row.get("st", None))
        et_value = tags.get("et", row.get("et", None))
        mixed_flag = int((st_value == "MIXED") or (et_value == "MIXED"))
        records.append(
            {
                "chrom": row[CHROM],
                "pos": int(row[POS]),
                "ref": row[REF],
                "alt": row[ALT],
                "rn": row["RN"],
                "label": int(label),
                "fold_id": chrom_to_fold.get(row[CHROM]),
                "read_base_aln": aligned["read_base_aln"],
                "ref_base_aln": aligned["ref_base_aln"],
                "qual_aln": aligned["qual_aln"],
                "tp_aln": aligned["tp_aln"],
                "t0_aln": aligned["t0_aln"],
                "focus_aln": aligned["focus_aln"],
                "softclip_mask_aln": aligned["softclip_mask_aln"],
                "strand": int(rec.is_reverse),
                "mapq": float(rec.mapping_quality),
                "rq": float(tags.get("rq", row.get("rq", 0.0) or 0.0)),
                "tm": tags.get("tm", row.get("tm", None)),
                "st": st_value,
                "et": et_value,
                "mixed": mixed_flag,
                "index": int(row.get("INDEX", 0) or 0),
                "read_len": read_len,
            }
        )
    stats = {
        "shard_id": shard_id,
        "input_rows": len(rows),
        "output_rows": len(records),
        "missing_rows": missing,
        "wall_seconds": round(time.perf_counter() - t0, 4),
    }
    return shard_id, records, stats


def _process_rows_shard_to_tensors(
    *,
    shard_id: int,
    rows: list[dict],
    positive_bam: str,
    negative_bam: str,
    chrom_to_fold: dict[str, int],
    encoders: Encoders,
    tensor_length: int,
    split_manifest: dict | None,
) -> tuple[int, dict, dict]:
    """BAM reading + tensor encoding in a single worker step.

    Returns (shard_id, tensor_chunk_dict, stats_dict) where tensor_chunk_dict
    is in the same format written to the tensor cache file.
    """
    sid, records, stats = _process_rows_shard(
        shard_id=shard_id,
        rows=rows,
        positive_bam=positive_bam,
        negative_bam=negative_bam,
        chrom_to_fold=chrom_to_fold,
    )
    if not records:
        empty_chunk = {
            "cache_format_version": 4,
            "read_base_idx": torch.zeros(0, tensor_length, dtype=torch.int16),
            "ref_base_idx": torch.zeros(0, tensor_length, dtype=torch.int16),
            "t0_idx": torch.zeros(0, tensor_length, dtype=torch.int16),
            "tm_idx": torch.zeros(0, dtype=torch.int8),
            "st_idx": torch.zeros(0, dtype=torch.int8),
            "et_idx": torch.zeros(0, dtype=torch.int8),
            "x_num_pos": torch.zeros(0, 5, tensor_length, dtype=torch.float16),
            "x_num_const": torch.zeros(0, 4, dtype=torch.float16),
            "mask": torch.zeros(0, tensor_length, dtype=torch.uint8),
            "label": torch.zeros(0, dtype=torch.uint8),
            "split_id": torch.zeros(0, dtype=torch.int8),
            "chrom": np.array([], dtype=object),
            "pos": np.array([], dtype=np.int32),
            "rn": np.array([], dtype=object),
        }
        return sid, empty_chunk, stats

    ds = DeepSRSNVDataset(records, encoders=encoders, length=tensor_length)
    payload: dict[str, list] = {
        "read_base_idx": [],
        "ref_base_idx": [],
        "t0_idx": [],
        "tm_idx": [],
        "st_idx": [],
        "et_idx": [],
        "x_num_pos": [],
        "x_num_const": [],
        "mask": [],
        "label": [],
        "split_id": [],
        "chrom": [],
        "pos": [],
        "rn": [],
    }
    for i in range(len(ds)):
        item = ds[i]
        rec = records[i]
        x_num = item["x_num"]
        payload["read_base_idx"].append(item["read_base_idx"].to(dtype=torch.int16))
        payload["ref_base_idx"].append(item["ref_base_idx"].to(dtype=torch.int16))
        payload["t0_idx"].append(item["t0_idx"].to(dtype=torch.int16))
        payload["tm_idx"].append(int(item["tm_idx"].item()))
        payload["st_idx"].append(int(item["st_idx"].item()))
        payload["et_idx"].append(int(item["et_idx"].item()))
        payload["x_num_pos"].append(x_num[:5].to(dtype=torch.float16))
        payload["x_num_const"].append(x_num[5:, 0].to(dtype=torch.float16))
        payload["mask"].append(item["mask"].to(dtype=torch.uint8))
        payload["label"].append(int(item["label"].item()))
        split_id = _row_split_id(
            {"chrom": rec["chrom"], "RN": rec["rn"], CHROM: rec["chrom"]},
            split_manifest=split_manifest,
            chrom_to_fold=chrom_to_fold,
        )
        payload["split_id"].append(int(split_id))
        payload["chrom"].append(str(rec["chrom"]))
        payload["pos"].append(int(rec["pos"]))
        payload["rn"].append(str(rec["rn"]))

    chunk = {
        "cache_format_version": 4,
        "read_base_idx": torch.stack(payload["read_base_idx"], dim=0).to(dtype=torch.int16),
        "ref_base_idx": torch.stack(payload["ref_base_idx"], dim=0).to(dtype=torch.int16),
        "t0_idx": torch.stack(payload["t0_idx"], dim=0).to(dtype=torch.int16),
        "tm_idx": torch.tensor(payload["tm_idx"], dtype=torch.int8),
        "st_idx": torch.tensor(payload["st_idx"], dtype=torch.int8),
        "et_idx": torch.tensor(payload["et_idx"], dtype=torch.int8),
        "x_num_pos": torch.stack(payload["x_num_pos"], dim=0).to(dtype=torch.float16),
        "x_num_const": torch.stack(payload["x_num_const"], dim=0).to(dtype=torch.float16),
        "mask": torch.stack(payload["mask"], dim=0).to(dtype=torch.uint8),
        "label": torch.tensor(payload["label"], dtype=torch.uint8),
        "split_id": torch.tensor(payload["split_id"], dtype=torch.int8),
        "chrom": np.asarray(payload["chrom"], dtype=object),
        "pos": np.asarray(payload["pos"], dtype=np.int32),
        "rn": np.asarray(payload["rn"], dtype=object),
    }
    return sid, chunk, stats


def build_tensor_cache(  # noqa: C901, PLR0913, PLR0915
    *,
    positive_parquet: str,
    negative_parquet: str,
    positive_bam: str,
    negative_bam: str,
    chrom_to_fold: dict[str, int],
    split_manifest: dict | None,
    encoders: Encoders,
    cache_dir: str | Path,
    tensor_length: int = 300,
    max_rows_per_class: int | None = None,
    preprocess_num_workers: int = 1,
    preprocess_max_ram_gb: float = 48.0,
    preprocess_batch_rows: int = 25000,
    preprocess_dry_run: bool = False,
) -> dict:
    """Single-step preprocessing: BAM reading + tensor encoding + cache write."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Preprocess start: cache_dir=%s workers=%d ram_cap_gb=%.1f requested_batch_rows=%d",
        cache_root,
        int(preprocess_num_workers),
        float(preprocess_max_ram_gb),
        int(preprocess_batch_rows),
    )
    effective_workers = max(1, int(preprocess_num_workers))
    effective_batch_rows = _estimate_shard_rows(
        preprocess_max_ram_gb=float(preprocess_max_ram_gb),
        preprocess_num_workers=effective_workers,
        requested_batch_rows=int(preprocess_batch_rows),
    )
    cache_key = _build_cache_key(
        positive_parquet=positive_parquet,
        negative_parquet=negative_parquet,
        positive_bam=positive_bam,
        negative_bam=negative_bam,
        split_manifest=split_manifest,
        chrom_to_fold=chrom_to_fold,
        tensor_length=tensor_length,
        batch_rows=effective_batch_rows,
    )
    run_dir = cache_root / cache_key
    index_path = run_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index["cache_hit"] = True
        logger.info(
            "Cache hit: key=%s shards=%d rows=%d",
            cache_key,
            int(index.get("total_shards", 0)),
            int(index.get("total_output_rows", 0)),
        )
        return index

    if preprocess_dry_run:
        logger.info(
            "Preprocess dry-run: key=%s planned_workers=%d planned_batch_rows=%d",
            cache_key,
            effective_workers,
            effective_batch_rows,
        )
        return {
            "cache_key": cache_key,
            "cache_hit": False,
            "dry_run": True,
            "planned_batch_rows": effective_batch_rows,
            "planned_workers": effective_workers,
        }

    run_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        CHROM,
        POS,
        REF,
        ALT,
        X_ALT,
        "RN",
        "INDEX",
        "REV",
        "MAPQ",
        "rq",
        "tm",
        "st",
        "et",
        FeatureMapFields.EDIST.value,
    ]
    random_seed = int((split_manifest or {}).get("random_seed", 1))
    split_buckets = _build_split_buckets(
        positive_parquet=positive_parquet,
        negative_parquet=negative_parquet,
        columns=cols,
        batch_rows=effective_batch_rows,
        max_rows_per_class=max_rows_per_class,
        split_manifest=split_manifest,
        chrom_to_fold=chrom_to_fold,
    )
    shard_inputs, split_input_stats = _build_balanced_shard_inputs(
        split_buckets=split_buckets,
        batch_rows=effective_batch_rows,
        random_seed=random_seed,
    )
    logger.info(
        "Preprocess planned: key=%s shards=%d effective_workers=%d effective_batch_rows=%d split_ids=%s",
        cache_key,
        len(shard_inputs),
        effective_workers,
        effective_batch_rows,
        ",".join(str(k) for k in sorted(split_buckets.keys())),
    )

    t_all = time.perf_counter()
    shard_stats: list[dict] = []
    split_counts: dict[int, dict[str, int]] = {}
    chunk_split_stats: list[dict] = []
    peak_rss_gb = _resource_rss_gb()
    total_shards = len(shard_inputs)
    completed = 0
    input_rows_seen = 0
    output_rows_seen = 0
    missing_rows_seen = 0

    tensor_cache_path = run_dir / "tensor_cache.pkl"
    cache_handle = tensor_cache_path.open("wb")

    def _handle_result(sid: int, chunk: dict, stats: dict) -> None:
        nonlocal completed, input_rows_seen, output_rows_seen, missing_rows_seen, peak_rss_gb
        pickle.dump(chunk, cache_handle, protocol=pickle.HIGHEST_PROTOCOL)
        shard_stats.append(stats)
        peak_rss_gb = max(peak_rss_gb, _resource_rss_gb())
        completed += 1
        input_rows_seen += int(stats["input_rows"])
        output_rows_seen += int(stats["output_rows"])
        missing_rows_seen += int(stats["missing_rows"])

        chunk_stats: dict[str, dict] = {}
        labels = chunk["label"].numpy().tolist() if hasattr(chunk["label"], "numpy") else []
        split_ids = chunk["split_id"].numpy().tolist() if hasattr(chunk["split_id"], "numpy") else []
        for s_id, label in zip(split_ids, labels, strict=False):
            key = int(s_id)
            if key not in split_counts:
                split_counts[key] = {"rows": 0, "positives": 0, "negatives": 0}
            split_counts[key]["rows"] += 1
            if int(label) == 1:
                split_counts[key]["positives"] += 1
            else:
                split_counts[key]["negatives"] += 1
            skey = str(key)
            if skey not in chunk_stats:
                chunk_stats[skey] = {"rows": 0, "positives": 0, "negatives": 0, "prevalence": None}
            chunk_stats[skey]["rows"] = int(chunk_stats[skey]["rows"]) + 1
            if int(label) == 1:
                chunk_stats[skey]["positives"] = int(chunk_stats[skey]["positives"]) + 1
            else:
                chunk_stats[skey]["negatives"] = int(chunk_stats[skey]["negatives"]) + 1
        for _skey, cstats in chunk_stats.items():
            rows_count = int(cstats["rows"])
            pos_count = int(cstats["positives"])
            cstats["prevalence"] = (pos_count / rows_count) if rows_count > 0 else None
        chunk_split_stats.append({"chunk_id": sid, "split_stats": chunk_stats})

        if completed == 1 or completed % 10 == 0 or completed == total_shards:
            logger.info(
                "Preprocess progress: %d/%d shards, input_rows=%d output_rows=%d "
                "missing=%d peak_rss_gb=%.2f elapsed=%.1fs",
                completed,
                total_shards,
                input_rows_seen,
                output_rows_seen,
                missing_rows_seen,
                peak_rss_gb,
                time.perf_counter() - t_all,
            )

    if effective_workers == 1:
        for sid, rows, _sid_split in shard_inputs:
            _sid_out, chunk, stats = _process_rows_shard_to_tensors(
                shard_id=sid,
                rows=rows,
                positive_bam=positive_bam,
                negative_bam=negative_bam,
                chrom_to_fold=chrom_to_fold,
                encoders=encoders,
                tensor_length=tensor_length,
                split_manifest=split_manifest,
            )
            _handle_result(sid, chunk, stats)
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = [
                pool.submit(
                    _process_rows_shard_to_tensors,
                    shard_id=sid,
                    rows=rows,
                    positive_bam=positive_bam,
                    negative_bam=negative_bam,
                    chrom_to_fold=chrom_to_fold,
                    encoders=encoders,
                    tensor_length=tensor_length,
                    split_manifest=split_manifest,
                )
                for sid, rows, _sid_split in shard_inputs
            ]
            for fut in as_completed(futures):
                sid, chunk, stats = fut.result()
                _handle_result(sid, chunk, stats)

    cache_handle.close()
    shard_stats = sorted(shard_stats, key=lambda x: int(x["shard_id"]))
    index = {
        "cache_key": cache_key,
        "cache_hit": False,
        "cache_version": 4,
        "tensor_length": tensor_length,
        "batch_rows": effective_batch_rows,
        "num_workers": effective_workers,
        "preprocess_max_ram_gb": float(preprocess_max_ram_gb),
        "total_shards": total_shards,
        "peak_rss_gb": round(peak_rss_gb, 3),
        "wall_seconds": round(time.perf_counter() - t_all, 3),
        "total_input_rows": int(sum(s["input_rows"] for s in shard_stats)),
        "total_output_rows": int(sum(s["output_rows"] for s in shard_stats)),
        "total_missing_rows": int(sum(s["missing_rows"] for s in shard_stats)),
        "tensor_cache_path": str(tensor_cache_path),
        "shard_stats": shard_stats,
        "split_input_stats": split_input_stats,
        "split_counts": {str(k): v for k, v in sorted(split_counts.items(), key=lambda kv: kv[0])},
        "chunk_split_stats": chunk_split_stats,
    }
    index_path.write_text(json.dumps(index, indent=2))
    logger.info(
        "Preprocess completed: key=%s shards=%d output_rows=%d missing=%d wall=%.1fs peak_rss_gb=%.2f",
        cache_key,
        total_shards,
        index["total_output_rows"],
        index["total_missing_rows"],
        index["wall_seconds"],
        index["peak_rss_gb"],
    )
    return index


def build_encoders_from_schema(schema: dict | None) -> Encoders:
    categories = (schema or {}).get("category_values", {})
    t0_values = categories.get("t0", []) or []
    tm_values = categories.get("tm", []) or _DEFAULT_TM_VALUES
    st_values = categories.get("st", []) or _DEFAULT_ST_ET_VALUES
    et_values = categories.get("et", []) or _DEFAULT_ST_ET_VALUES

    t0_seed = _DEFAULT_T0_TOKENS + list(t0_values)
    logger.info(
        "Building encoders from known vocab dicts (schema): t0_seed=%d tm_seed=%d st_seed=%d et_seed=%d",
        len(t0_seed),
        len(tm_values),
        len(st_values),
        len(et_values),
    )
    return Encoders(
        base_vocab=_default_base_vocab(),
        t0_vocab=_create_categorical_vocab(t0_seed),
        tm_vocab=_create_categorical_vocab(tm_values),
        st_vocab=_create_categorical_vocab(st_values),
        et_vocab=_create_categorical_vocab(et_values),
    )


def build_training_records(
    positive_parquet: str,
    negative_parquet: str,
    positive_bam: str,
    negative_bam: str,
    chrom_to_fold: dict[str, int],
    max_rows_per_class: int | None = None,
) -> list[dict]:
    labeled_df = _load_labeled_df(positive_parquet, negative_parquet, max_rows_per_class=max_rows_per_class)
    cols = [CHROM, POS, REF, ALT, X_ALT, "RN", "INDEX", "REV", "MAPQ", "rq", "tm", "st", "et", "label"]
    rows = labeled_df.select([c for c in cols if c in labeled_df.columns]).to_dicts()

    bams = {
        True: pysam.AlignmentFile(positive_bam, "rb"),
        False: pysam.AlignmentFile(negative_bam, "rb"),
    }
    idx = {k: pysam.IndexedReads(v) for k, v in bams.items()}
    for value in idx.values():
        value.build()

    records: list[dict] = []
    missing = 0
    for row in rows:
        label = bool(row["label"])
        bam = bams[label]
        rec = _match_bam_read(
            bam=bam,
            index=idx[label],
            chrom=row[CHROM],
            pos=int(row[POS]),
            rn=row["RN"],
        )
        if rec is None:
            missing += 1
            continue
        tags = _record_tags(rec)
        query_seq = rec.query_sequence or ""
        read_len = len(query_seq)
        tp_raw = _to_numpy_tp(tags.get("tp"), read_len=read_len)
        t0_raw = _to_string_t0(tags.get("t0"), read_len=read_len)

        is_positive = int(label) == 1
        # XGBoost positive preprocessing semantics: REF is replaced by X_ALT.
        positive_focus_ref_override = str(row.get(X_ALT) or "").upper() if is_positive else None
        aligned = _build_gapped_channels(
            rec=rec,
            snv_pos_1based=int(row[POS]),
            tp_raw=tp_raw,
            t0_raw=t0_raw,
            positive_focus_ref_override=positive_focus_ref_override,
        )

        st_value = tags.get("st", row.get("st", None))
        et_value = tags.get("et", row.get("et", None))
        mixed_flag = int((st_value == "MIXED") or (et_value == "MIXED"))
        records.append(
            {
                "chrom": row[CHROM],
                "pos": int(row[POS]),
                "ref": row[REF],
                "alt": row[ALT],
                "rn": row["RN"],
                "label": int(label),
                "fold_id": chrom_to_fold.get(row[CHROM]),
                "read_base_aln": aligned["read_base_aln"],
                "ref_base_aln": aligned["ref_base_aln"],
                "qual_aln": aligned["qual_aln"],
                "tp_aln": aligned["tp_aln"],
                "t0_aln": aligned["t0_aln"],
                "focus_aln": aligned["focus_aln"],
                "softclip_mask_aln": aligned["softclip_mask_aln"],
                "strand": int(rec.is_reverse),
                "mapq": float(rec.mapping_quality),
                "rq": float(tags.get("rq", row.get("rq", 0.0) or 0.0)),
                "tm": tags.get("tm", row.get("tm", None)),
                "st": st_value,
                "et": et_value,
                "mixed": mixed_flag,
                "index": int(row.get("INDEX", 0) or 0),
                "read_len": read_len,
            }
        )

    for bam in bams.values():
        bam.close()
    if missing:
        print(f"[deep_srsnv] warning: {missing} rows had no BAM match and were dropped")
    return records


def build_encoders(records: list[dict]) -> Encoders:
    t0_tokens = []
    tm_values = []
    st_values = []
    et_values = []
    for r in records:
        t0_tokens.extend(r["t0_aln"])
        tm_values.append(r.get("tm"))
        st_values.append(r.get("st"))
        et_values.append(r.get("et"))
    return Encoders(
        base_vocab=_default_base_vocab(),
        t0_vocab=_create_categorical_vocab(t0_tokens),
        tm_vocab=_create_categorical_vocab(tm_values),
        st_vocab=_create_categorical_vocab(st_values),
        et_vocab=_create_categorical_vocab(et_values),
    )


class DeepSRSNVDataset(Dataset):
    def __init__(self, records: list[dict], encoders: Encoders, length: int = 300):
        self.records = records
        self.encoders = encoders
        self.length = length

    def __len__(self) -> int:
        return len(self.records)

    def _encode_seq(self, seq_tokens: list[str]) -> np.ndarray:
        arr = np.zeros(self.length, dtype=np.int64)
        n = min(self.length, len(seq_tokens))
        for i in range(n):
            arr[i] = self.encoders.base_vocab.get(seq_tokens[i], self.encoders.base_vocab["N"])
        return arr

    def _encode_t0(self, t0_tokens: list[str]) -> np.ndarray:
        arr = np.zeros(self.length, dtype=np.int64)
        miss = self.encoders.t0_vocab.get("<MISSING>", 0)
        n = min(self.length, len(t0_tokens))
        for i in range(self.length):
            if i < n:
                arr[i] = self.encoders.t0_vocab.get(t0_tokens[i], miss)
            else:
                arr[i] = 0
        return arr

    def _pad_float(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        out = np.full(self.length, fill, dtype=np.float32)
        n = min(self.length, len(values))
        if n > 0:
            out[:n] = values[:n]
        return out

    def __getitem__(self, idx: int):
        r = self.records[idx]
        valid = min(self.length, len(r["read_base_aln"]))
        mask = np.zeros(self.length, dtype=np.float32)
        mask[:valid] = 1.0

        read_base_idx = self._encode_seq(r["read_base_aln"])
        ref_base_idx = self._encode_seq(r["ref_base_aln"])
        t0_idx = self._encode_t0(r["t0_aln"])
        qual = self._pad_float(r["qual_aln"], fill=0.0) / 50.0
        tp = self._pad_float(r["tp_aln"], fill=0.0)
        focus = self._pad_float(r["focus_aln"], fill=0.0)
        softclip_mask = self._pad_float(r["softclip_mask_aln"], fill=0.0)

        tm_id = self.encoders.tm_vocab.get(r.get("tm") or "<MISSING>", self.encoders.tm_vocab.get("<MISSING>", 0))
        st_id = self.encoders.st_vocab.get(r.get("st") or "<MISSING>", self.encoders.st_vocab.get("<MISSING>", 0))
        et_id = self.encoders.et_vocab.get(r.get("et") or "<MISSING>", self.encoders.et_vocab.get("<MISSING>", 0))
        strand = float(r["strand"])
        mapq = float(r["mapq"]) / 60.0
        rq = float(r["rq"])
        mixed = float(r.get("mixed", 0))

        const = np.stack(
            [
                np.full(self.length, strand, dtype=np.float32),
                np.full(self.length, mapq, dtype=np.float32),
                np.full(self.length, rq, dtype=np.float32),
                np.full(self.length, mixed, dtype=np.float32),
            ],
            axis=0,
        )
        numeric = np.stack([qual, tp, mask, focus, softclip_mask], axis=0)
        x_num = np.concatenate([numeric, const], axis=0)

        return {
            "read_base_idx": torch.tensor(read_base_idx, dtype=torch.long),
            "ref_base_idx": torch.tensor(ref_base_idx, dtype=torch.long),
            "t0_idx": torch.tensor(t0_idx, dtype=torch.long),
            "tm_idx": torch.tensor(tm_id, dtype=torch.long),
            "st_idx": torch.tensor(st_id, dtype=torch.long),
            "et_idx": torch.tensor(et_id, dtype=torch.long),
            "x_num": torch.tensor(x_num, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "label": torch.tensor(r["label"], dtype=torch.float32),
            "fold_id": torch.tensor(-1 if r["fold_id"] is None else int(r["fold_id"]), dtype=torch.long),
            "chrom": r["chrom"],
            "pos": r["pos"],
            "rn": r["rn"],
        }


def iter_tensor_cache_chunks(tensor_cache_path: str):
    with Path(tensor_cache_path).open("rb") as handle:
        while True:
            try:
                yield pickle.load(handle)  # noqa: S301
            except EOFError:
                break


def load_full_tensor_cache(tensor_cache_path: str) -> dict:  # noqa: PLR0915
    """Load entire tensor cache into memory in compact form.

    Keeps the on-disk dtypes (int16, float16, uint8, int8) to minimize RAM.
    Expansion to training dtypes happens per-sample in TensorMapDataset.__getitem__.
    """
    t0 = time.perf_counter()
    all_read_base_idx: list[torch.Tensor] = []
    all_ref_base_idx: list[torch.Tensor] = []
    all_t0_idx: list[torch.Tensor] = []
    all_tm_idx: list[torch.Tensor] = []
    all_st_idx: list[torch.Tensor] = []
    all_et_idx: list[torch.Tensor] = []
    all_x_num_pos: list[torch.Tensor] = []
    all_x_num_const: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []
    all_label: list[torch.Tensor] = []
    all_split_id: list[torch.Tensor] = []
    all_chrom: list[str] = []
    all_pos: list[np.ndarray] = []
    all_rn: list[str] = []

    n_chunks = 0
    for chunk in iter_tensor_cache_chunks(tensor_cache_path):
        n_chunks += 1
        all_read_base_idx.append(chunk["read_base_idx"])
        all_ref_base_idx.append(chunk["ref_base_idx"])
        all_t0_idx.append(chunk["t0_idx"])
        all_mask.append(chunk["mask"])
        all_label.append(chunk["label"])
        all_split_id.append(chunk["split_id"])

        if "tm_idx" in chunk:
            all_tm_idx.append(chunk["tm_idx"])
            all_st_idx.append(chunk["st_idx"])
            all_et_idx.append(chunk["et_idx"])

        if "x_num" in chunk:
            x = chunk["x_num"]
            all_x_num_pos.append(x[:, :5, :])
            all_x_num_const.append(x[:, 5:, 0])
        else:
            all_x_num_pos.append(chunk["x_num_pos"])
            all_x_num_const.append(chunk["x_num_const"])

        if isinstance(chunk["chrom"], np.ndarray):
            all_chrom.extend(chunk["chrom"].tolist())
        else:
            all_chrom.extend(chunk["chrom"])
        all_pos.append(np.asarray(chunk["pos"], dtype=np.int32))
        if isinstance(chunk["rn"], np.ndarray):
            all_rn.extend(chunk["rn"].tolist())
        else:
            all_rn.extend(chunk["rn"])

    result = {
        "read_base_idx": torch.cat(all_read_base_idx, dim=0),
        "ref_base_idx": torch.cat(all_ref_base_idx, dim=0),
        "t0_idx": torch.cat(all_t0_idx, dim=0),
        "x_num_pos": torch.cat(all_x_num_pos, dim=0),
        "x_num_const": torch.cat(all_x_num_const, dim=0),
        "mask": torch.cat(all_mask, dim=0),
        "label": torch.cat(all_label, dim=0),
        "split_id": torch.cat(all_split_id, dim=0),
        "chrom": all_chrom,
        "pos": np.concatenate(all_pos),
        "rn": all_rn,
    }
    if all_tm_idx:
        result["tm_idx"] = torch.cat(all_tm_idx, dim=0)
        result["st_idx"] = torch.cat(all_st_idx, dim=0)
        result["et_idx"] = torch.cat(all_et_idx, dim=0)

    n_rows = int(result["label"].shape[0])
    mem_bytes = sum(t.element_size() * t.nelement() for t in result.values() if isinstance(t, torch.Tensor))
    logger.info(
        "Loaded full tensor cache: %d chunks, %d rows, %.2f GB compact memory, %.1fs",
        n_chunks,
        n_rows,
        mem_bytes / (1024**3),
        time.perf_counter() - t0,
    )
    return result
