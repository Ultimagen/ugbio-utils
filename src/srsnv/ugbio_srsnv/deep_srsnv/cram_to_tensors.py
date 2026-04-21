"""CRAM-to-tensor pipeline: stream reads directly from a CRAM file and produce tensor caches.

Replaces the previous workflow of extracting intermediate BAMs and then building
a tensor cache from them. Reads are fetched by region from the CRAM using the
coordinate index, matched by read name, and immediately tensorized.

Two entry points:
- ``cram_to_tensor_cache``  -- training: write sharded tensor cache to disk
- ``stream_cram_to_tensors`` -- inference: yield tensor batches without disk I/O
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pysam
import torch
from pyarrow import parquet as pq
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields

from ugbio_srsnv.deep_srsnv.data_prep import (
    NUM_CHANNELS_CONST,
    NUM_CHANNELS_POS,
    Encoders,
    _build_gapped_channels,
    _to_numpy_tp,
    _to_string_t0,
    load_vocab_config,
)

CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value
REF = FeatureMapFields.REF.value
ALT = FeatureMapFields.ALT.value
X_ALT = FeatureMapFields.X_ALT.value

_PARQUET_COLUMNS = [
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


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


@dataclass
class PrepProfile:
    """Profiling metrics collected during preprocessing."""

    wall_seconds: float = 0.0
    cpu_user_seconds: float = 0.0
    cpu_system_seconds: float = 0.0
    cpu_utilization: float = 0.0
    peak_rss_gb: float = 0.0
    total_input_rows: int = 0
    total_output_rows: int = 0
    rows_per_second: float = 0.0
    missing_rows: int = 0
    bytes_written: int = 0
    phase_seconds: dict = field(default_factory=dict)

    def log_summary(self, step_name: str) -> None:
        logger.info(
            "=== %s Profile ===\n"
            "  Wall time:        %.1fs\n"
            "  CPU utilization:  %.1f%% (user=%.1fs sys=%.1fs)\n"
            "  Peak RSS:         %.2f GB\n"
            "  Input rows:       %s\n"
            "  Output rows:      %s  (%d missing)\n"
            "  Throughput:       %s rows/sec\n"
            "  Bytes written:    %s",
            step_name,
            self.wall_seconds,
            self.cpu_utilization * 100,
            self.cpu_user_seconds,
            self.cpu_system_seconds,
            self.peak_rss_gb,
            f"{self.total_input_rows:,}",
            f"{self.total_output_rows:,}",
            self.missing_rows,
            f"{self.rows_per_second:,.0f}" if self.rows_per_second else "N/A",
            f"{self.bytes_written / (1024**3):.2f} GB" if self.bytes_written > 0 else "0",
        )


def _resource_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _cpu_times() -> tuple[float, float]:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime, r.ru_stime


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for f in path.iterdir():
        if f.is_file():
            total += f.stat().st_size
    return total


# ---------------------------------------------------------------------------
# CRAM read fetching
# ---------------------------------------------------------------------------


def _fetch_read_from_cram(
    cram: pysam.AlignmentFile,
    chrom: str,
    pos: int,
    rn: str,
) -> pysam.AlignedSegment | None:
    """Fetch a single read from a CRAM by coordinate region and read name.

    Used by the inference streaming path. For batch training, prefer
    ``_fetch_reads_batched`` which amortises decompression across many reads.
    """
    try:
        for rec in cram.fetch(chrom, pos - 1, pos):
            if rec.query_name == rn:
                return rec
    except ValueError:
        return None
    return None


def _fetch_reads_by_region(
    cram: pysam.AlignmentFile,
    rows: list[dict],
) -> dict[str, pysam.AlignedSegment]:
    """Single-range streaming fetch per chromosome segment.

    Groups rows by chromosome (they are already sorted by chr+pos) and
    issues **one** ``cram.fetch()`` per chromosome covering the full
    position range. This decompresses each CRAM slice exactly once
    instead of once per row, eliminating redundant index lookups and
    slice re-decompression.
    """
    from itertools import groupby  # noqa: PLC0415

    matched: dict[str, pysam.AlignedSegment] = {}

    for chrom, chrom_rows_iter in groupby(rows, key=lambda r: str(r[CHROM])):
        chrom_rows = list(chrom_rows_iter)
        chrom_names = {r["RN"] for r in chrom_rows}
        positions = [int(r[POS]) for r in chrom_rows]

        try:
            for rec in cram.fetch(chrom, max(0, min(positions) - 1), max(positions) + 1):
                if rec.query_name in chrom_names:
                    matched[rec.query_name] = rec
                    chrom_names.discard(rec.query_name)
                    if not chrom_names:
                        break
        except ValueError:
            continue

    return matched


def _fetch_reads_samtools_pipe(
    cram_path: str,
    reference_path: str | None,
    rows: list[dict],
) -> dict[str, pysam.AlignedSegment]:
    """Fetch reads via ``samtools view -N`` subprocess pipe.

    Writes the wanted read names to a temp file, launches samtools with
    ``-N`` (C-level hash-based name filtering) and region arguments, then
    reads only the matched records back via pysam from the BAM pipe.
    This avoids creating ~9.3M Python AlignedSegment objects per shard
    and instead creates objects only for the ~25K matches.
    """
    from itertools import groupby  # noqa: PLC0415

    if not rows:
        return {}

    names = {r["RN"] for r in rows}
    regions: list[str] = []
    for chrom, chrom_rows_iter in groupby(rows, key=lambda r: str(r[CHROM])):
        positions = [int(r[POS]) for r in chrom_rows_iter]
        start = max(1, min(positions))
        end = max(positions) + 1
        regions.append(f"{chrom}:{start}-{end}")

    names_fd, names_path = tempfile.mkstemp(prefix="samtools_names_", suffix=".txt")
    try:
        with os.fdopen(names_fd, "w") as f:
            f.write("\n".join(names))
            f.write("\n")

        cmd = ["samtools", "view", "-N", names_path, "-b", "--no-header"]
        if reference_path:
            cmd.extend(["--reference", reference_path])
        cmd.append(cram_path)
        cmd.extend(regions)

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        matched: dict[str, pysam.AlignedSegment] = {}
        try:
            with pysam.AlignmentFile(proc.stdout, "rb", check_sq=False) as bam_pipe:
                for rec in bam_pipe:
                    matched[rec.query_name] = rec
        finally:
            proc.stdout.close()
            proc.wait()
            if proc.returncode != 0:
                stderr_text = proc.stderr.read().decode(errors="replace").strip()
                logger.warning("samtools view exited %d: %s", proc.returncode, stderr_text[:500])
            proc.stderr.close()
    finally:
        os.unlink(names_path)

    return matched


def _samtools_available() -> bool:
    """Return True if samtools is on PATH."""
    return shutil.which("samtools") is not None


# ---------------------------------------------------------------------------
# EDIST filtering (same logic as existing pipeline)
# ---------------------------------------------------------------------------


def _compute_max_edist(parquet_path: str) -> int | None:
    edist_col = FeatureMapFields.EDIST.value
    pf = pq.ParquetFile(parquet_path)
    if edist_col not in (pf.schema.names or []):
        return None
    max_val = None
    for batch in pf.iter_batches(batch_size=100_000, columns=[edist_col]):
        col = batch.column(edist_col)
        batch_max = col.to_pylist()
        local_max = max((v for v in batch_max if v is not None), default=None)
        if local_max is not None and (max_val is None or local_max > max_val):
            max_val = local_max
    return max_val


# ---------------------------------------------------------------------------
# Per-shard tensorization (runs in worker processes)
# ---------------------------------------------------------------------------


def _worker_init():
    """Set torch multiprocessing sharing strategy to file_system.

    This avoids /dev/shm size limits on platforms like AWS HealthOmics
    where shared memory is restricted.  Must be called as a
    ``ProcessPoolExecutor`` initializer.
    """
    import torch.multiprocessing  # noqa: PLC0415

    torch.multiprocessing.set_sharing_strategy("file_system")


_WORKER_CRAM: pysam.AlignmentFile | None = None


def _init_cram_worker(cram_path: str, reference_path: str | None) -> None:
    global _WORKER_CRAM  # noqa: PLW0603
    _WORKER_CRAM = pysam.AlignmentFile(
        cram_path,
        "rc",
        reference_filename=reference_path,
    )


def _process_shard(  # noqa: PLR0915
    *,
    shard_id: int,
    rows: list[dict],
    cram_path: str,
    reference_path: str | None,
    encoders: Encoders,
    tensor_length: int,
    label: bool,  # noqa: FBT001
    max_edist: int | None,
    fetch_mode: str = "samtools",
) -> tuple[int, dict, dict]:
    """Process a shard of parquet rows into a tensor chunk.

    Parameters
    ----------
    fetch_mode
        ``"samtools"`` (default) uses ``samtools view -N`` subprocess pipe
        for C-level name filtering.  ``"pysam"`` falls back to pure-pysam
        region iteration.
    """
    t_start = time.perf_counter()

    edist_col = FeatureMapFields.EDIST.value

    if label and max_edist is not None:
        kept_rows = [r for r in rows if r.get(edist_col) != max_edist]
        edist_dropped = len(rows) - len(kept_rows)
    else:
        kept_rows = rows
        edist_dropped = 0

    t_f0 = time.perf_counter()
    if fetch_mode == "samtools":
        rn_to_rec = _fetch_reads_samtools_pipe(cram_path, reference_path, kept_rows)
    else:
        cram = _WORKER_CRAM
        if cram is None:
            cram = pysam.AlignmentFile(cram_path, "rc", reference_filename=reference_path)
        rn_to_rec = _fetch_reads_by_region(cram, kept_rows)
    t_cram_fetch = time.perf_counter() - t_f0

    # Tensorize matched reads
    t_t0 = time.perf_counter()

    base_vocab = encoders.base_vocab
    base_default = base_vocab["N"]
    tm_vocab = encoders.tm_vocab
    tm_default = tm_vocab.get("<MISSING>", 0)
    st_vocab = encoders.st_vocab
    st_default = st_vocab.get("<MISSING>", 0)
    et_vocab = encoders.et_vocab
    et_default = et_vocab.get("<MISSING>", 0)

    n = len(kept_rows)
    read_base_out = np.zeros((n, tensor_length), dtype=np.int16)
    ref_base_out = np.zeros((n, tensor_length), dtype=np.int16)
    x_num_pos_out = np.zeros((n, len(NUM_CHANNELS_POS), tensor_length), dtype=np.float16)
    x_num_const_out = np.zeros((n, len(NUM_CHANNELS_CONST)), dtype=np.float16)
    mask_out = np.zeros((n, tensor_length), dtype=np.uint8)
    label_out = np.zeros(n, dtype=np.uint8)
    tm_arr = np.zeros(n, dtype=np.int8)
    st_arr = np.zeros(n, dtype=np.int8)
    et_arr = np.zeros(n, dtype=np.int8)
    pos_out = np.zeros(n, dtype=np.int32)
    chrom_list: list[str] = []
    rn_list: list[str] = []

    missing = 0
    out_i = 0

    for row in kept_rows:
        chrom = row[CHROM]
        pos = int(row[POS])
        rn = row["RN"]

        rec = rn_to_rec.get(rn)
        if rec is None:
            missing += 1
            continue

        tags = dict(rec.get_tags(with_value_type=False))
        read_len = len(rec.query_sequence or "")
        tp_raw = _to_numpy_tp(tags.get("tp"), read_len)
        t0_raw = _to_string_t0(tags.get("t0"), read_len)
        positive_focus_ref_override = str(row.get(X_ALT) or "").upper() if label else None

        aligned = _build_gapped_channels(rec, pos, tp_raw, t0_raw, positive_focus_ref_override)
        aln_len = len(aligned["read_base_aln"])
        valid = min(tensor_length, aln_len)

        read_tokens = aligned["read_base_aln"][:valid]
        ref_tokens = aligned["ref_base_aln"][:valid]
        read_base_out[out_i, :valid] = [base_vocab.get(t, base_default) for t in read_tokens]
        ref_base_out[out_i, :valid] = [base_vocab.get(t, base_default) for t in ref_tokens]

        q = aligned["qual_aln"][:valid]
        x_num_pos_out[out_i, 0, :valid] = q / 10.0
        x_num_pos_out[out_i, 1, :valid] = aligned["tp_aln"][:valid]
        x_num_pos_out[out_i, 2, :valid] = 1.0
        x_num_pos_out[out_i, 3, :valid] = aligned["focus_aln"][:valid]
        x_num_pos_out[out_i, 4, :valid] = aligned["softclip_mask_aln"][:valid]
        x_num_pos_out[out_i, 5, :valid] = aligned["t0_aln"][:valid] / 10.0
        mask_out[out_i, :valid] = 1

        st_value = tags.get("st", row.get("st", None))
        et_value = tags.get("et", row.get("et", None))
        mixed_flag = float((st_value == "MIXED") or (et_value == "MIXED"))
        x_num_const_out[out_i, 0] = float(int(rec.is_reverse))
        x_num_const_out[out_i, 1] = float(rec.mapping_quality) / 60.0
        x_num_const_out[out_i, 2] = float(tags.get("rq", row.get("rq", 0.0) or 0.0))
        x_num_const_out[out_i, 3] = mixed_flag

        tm_arr[out_i] = tm_vocab.get(tags.get("tm", row.get("tm")) or "<MISSING>", tm_default)
        st_arr[out_i] = st_vocab.get(st_value or "<MISSING>", st_default)
        et_arr[out_i] = et_vocab.get(et_value or "<MISSING>", et_default)

        label_out[out_i] = int(label)
        chrom_list.append(str(chrom))
        pos_out[out_i] = pos
        rn_list.append(str(rn))
        out_i += 1

    t_tensorize = time.perf_counter() - t_t0

    chunk = {
        "cache_format_version": 6,
        "read_base_idx": read_base_out[:out_i].copy(),
        "ref_base_idx": ref_base_out[:out_i].copy(),
        "tm_idx": tm_arr[:out_i].copy(),
        "st_idx": st_arr[:out_i].copy(),
        "et_idx": et_arr[:out_i].copy(),
        "x_num_pos": x_num_pos_out[:out_i].copy(),
        "x_num_const": x_num_const_out[:out_i].copy(),
        "mask": mask_out[:out_i].copy(),
        "label": label_out[:out_i].copy(),
        "chrom": np.asarray(chrom_list, dtype=object),
        "pos": pos_out[:out_i].copy(),
        "rn": np.asarray(rn_list, dtype=object),
    }
    wall = time.perf_counter() - t_start
    n_windows = len(rn_to_rec)
    stats = {
        "shard_id": shard_id,
        "input_rows": len(rows),
        "output_rows": out_i,
        "missing_rows": missing,
        "edist_dropped": edist_dropped,
        "wall_seconds": round(wall, 4),
        "cram_fetch_seconds": round(t_cram_fetch, 4),
        "tensorize_seconds": round(t_tensorize, 4),
        "fetch_windows": n_windows,
    }
    return shard_id, chunk, stats


def _process_shard_from_parquet(
    *,
    shard_id: int,
    parquet_path: str,
    row_group_id: int,
    columns: list[str],
    **shard_kwargs,
) -> tuple[int, dict, dict]:
    """Read one row group from parquet, convert to list-of-dicts, then process as shard."""
    import pyarrow.parquet as pq_rg  # noqa: PLC0415

    pf = pq_rg.ParquetFile(parquet_path)
    table = pf.read_row_group(row_group_id, columns=columns)
    col_dict = table.to_pydict()
    rows = [{k: col_dict[k][i] for k in col_dict} for i in range(len(table))]
    del table, col_dict
    return _process_shard(shard_id=shard_id, rows=rows, **shard_kwargs)


_TORCH_TENSOR_KEYS = frozenset(
    {
        "read_base_idx",
        "ref_base_idx",
        "tm_idx",
        "st_idx",
        "et_idx",
        "x_num_pos",
        "x_num_const",
        "mask",
        "label",
    }
)


def _numpy_chunk_to_torch(chunk: dict) -> None:
    """Convert numpy arrays in a shard chunk to torch tensors in-place (for torch.save)."""
    for key in _TORCH_TENSOR_KEYS:
        if key in chunk and isinstance(chunk[key], np.ndarray):
            chunk[key] = torch.from_numpy(chunk[key])


def _save_shard(chunk: dict, path: Path, *, compress: bool = False) -> None:
    """Save a tensor shard to disk, optionally gzip-compressed."""
    if compress:
        import io  # noqa: PLC0415

        try:
            from isal import igzip as gzip  # noqa: PLC0415
        except ImportError:
            import gzip  # noqa: PLC0415

        buf = io.BytesIO()
        torch.save(chunk, buf)
        gz_path = path.with_suffix(".pt.gz") if not str(path).endswith(".gz") else path
        with gzip.open(gz_path, "wb", compresslevel=1) as f:
            f.write(buf.getvalue())
    else:
        torch.save(chunk, path)


# ---------------------------------------------------------------------------
# Main cache builder
# ---------------------------------------------------------------------------


def cram_to_tensor_cache(  # noqa: PLR0913, PLR0915, PLR0912, C901
    cram_path: str,
    parquet_path: str,
    encoders: Encoders,
    output_dir: str,
    label: bool,  # noqa: FBT001
    tensor_length: int = 300,
    reference_path: str | None = None,
    compress: bool = False,  # noqa: FBT001, FBT002
    num_workers: int = 1,
    shard_size: int = 25000,
    fetch_mode: str = "samtools",
    chromosomes: list[str] | None = None,
) -> dict:
    """Read parquet rows, fetch each read from CRAM, tensorize, and write sharded cache.

    Parameters
    ----------
    cram_path
        Path to the source CRAM file (must have .crai index).
    parquet_path
        Path to the feature map parquet (positive or negative).
    encoders
        Vocabulary encoders (from ``load_vocab_config``).
    output_dir
        Directory to write shard files and ``index.json``.
    label
        ``True`` for positive reads, ``False`` for negative.
    tensor_length
        Padded sequence length for tensors.
    reference_path
        Reference FASTA for CRAM decoding (optional if embedded).
    num_workers
        Number of parallel worker processes.
    shard_size
        Number of parquet rows per shard.
    fetch_mode
        ``"samtools"`` (default) delegates read filtering to a
        ``samtools view -N`` subprocess.  ``"pysam"`` uses pure-pysam
        region iteration as a fallback.
    chromosomes
        Optional list of chromosomes to include.  If provided, only rows
        whose CHROM is in this list are tensorized.  Used for per-fold
        inference tensorization.

    Returns
    -------
    dict
        Index metadata including profiling information.
    """
    if fetch_mode == "samtools" and not _samtools_available():
        logger.warning("samtools not found on PATH, falling back to pysam fetch mode")
        fetch_mode = "pysam"
    cpu_u0, cpu_s0 = _cpu_times()
    t_wall_start = time.perf_counter()
    peak_rss = _resource_rss_gb()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Phase 1: read parquet, filter by chromosomes, sort, and write sharded parquet
    # Workers will read individual row groups from the sharded file — no full dataset in memory.
    t_parquet = time.perf_counter()
    import pyarrow.compute as pc  # noqa: PLC0415

    pf = pq.ParquetFile(parquet_path)
    available_cols = [c for c in _PARQUET_COLUMNS if c in (pf.schema.names or [])]

    table = pf.read(columns=available_cols)
    if chromosomes is not None and CHROM in available_cols:
        chrom_set = set(chromosomes)
        mask = pc.is_in(table.column(CHROM), value_set=pa.array(list(chrom_set)))
        table = table.filter(mask)
        logger.info("Filtered parquet to chromosomes %s: %d rows", ", ".join(sorted(chrom_set)), len(table))

    sort_indices = pc.sort_indices(table, sort_keys=[(CHROM, "ascending"), (POS, "ascending")])
    table = table.take(sort_indices)
    n_rows = len(table)

    sharded_parquet = out_path / "_sharded_input.parquet"
    with pq.ParquetWriter(str(sharded_parquet), table.schema) as pw:
        for i in range(0, n_rows, shard_size):
            pw.write_table(table.slice(i, min(shard_size, n_rows - i)))
    del table, sort_indices  # Free pyarrow memory

    total_shards = pq.ParquetFile(str(sharded_parquet)).metadata.num_row_groups
    parquet_read_seconds = round(time.perf_counter() - t_parquet, 4)
    logger.info(
        "Prepared %d shards (%d rows) in %.1fs -> %s", total_shards, n_rows, parquet_read_seconds, sharded_parquet
    )

    max_edist = _compute_max_edist(parquet_path) if label else None
    if max_edist is not None:
        logger.info("Positive EDIST filter: will drop rows with EDIST == %s", max_edist)

    # Phase 2: process shards — workers read row groups from disk (low memory)
    shard_stats: list[dict] = []
    completed = 0
    total_output = 0
    total_missing = 0

    t_process = time.perf_counter()

    effective_workers = max(1, int(num_workers))

    shard_kwargs = {
        "cram_path": cram_path,
        "reference_path": reference_path,
        "encoders": encoders,
        "tensor_length": tensor_length,
        "label": label,
        "max_edist": max_edist,
        "fetch_mode": fetch_mode,
    }

    shard_args_list = [
        {
            "shard_id": rg,
            "parquet_path": str(sharded_parquet),
            "row_group_id": rg,
            "columns": available_cols,
            **shard_kwargs,
        }
        for rg in range(total_shards)
    ]
    logger.info("Processing %d shards (%d rows each, %d workers)", total_shards, shard_size, effective_workers)

    if effective_workers == 1:
        for shard_args in shard_args_list:
            _sid, chunk, stats = _process_shard_from_parquet(**shard_args)
            _numpy_chunk_to_torch(chunk)
            _save_shard(chunk, out_path / f"shard_{_sid:05d}.pt", compress=compress)
            shard_stats.append(stats)
            completed += 1
            total_output += stats["output_rows"]
            total_missing += stats["missing_rows"]
            peak_rss = max(peak_rss, _resource_rss_gb())
            if completed == 1 or completed % 10 == 0 or completed == total_shards:
                logger.info(
                    "Progress: %d/%d shards, output=%d missing=%d elapsed=%.1fs",
                    completed,
                    total_shards,
                    total_output,
                    total_missing,
                    time.perf_counter() - t_process,
                )
    else:
        max_pending = effective_workers * 2
        shard_iter = iter(shard_args_list)

        # Background writer thread pool for non-blocking shard saves (especially gzip)
        from queue import Queue as _Queue  # noqa: PLC0415

        write_queue: _Queue = _Queue(maxsize=4)
        write_errors: list[Exception] = []

        def _writer_loop() -> None:
            """Drain write_queue and save shards (possibly with gzip) off the main loop."""
            while True:
                item = write_queue.get()
                if item is None:
                    break
                _chunk, _path, _compress = item
                try:
                    _save_shard(_chunk, _path, compress=_compress)
                except Exception as exc:  # noqa: BLE001
                    write_errors.append(exc)
                    logger.exception("Background shard write failed: %s", _path)

        from threading import Thread as _Thread  # noqa: PLC0415

        writer_thread = _Thread(target=_writer_loop, daemon=True)
        writer_thread.start()

        with ProcessPoolExecutor(max_workers=effective_workers, initializer=_worker_init) as pool:
            pending = {
                pool.submit(_process_shard_from_parquet, **sa) for sa in itertools.islice(shard_iter, max_pending)
            }
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    _sid, chunk, stats = fut.result()
                    _numpy_chunk_to_torch(chunk)
                    write_queue.put((chunk, out_path / f"shard_{_sid:05d}.pt", compress))
                    shard_stats.append(stats)
                    completed += 1
                    total_output += stats["output_rows"]
                    total_missing += stats["missing_rows"]
                    peak_rss = max(peak_rss, _resource_rss_gb())
                    if completed == 1 or completed % 10 == 0 or completed == total_shards:
                        logger.info(
                            "Progress: %d/%d shards, output=%d missing=%d elapsed=%.1fs",
                            completed,
                            total_shards,
                            total_output,
                            total_missing,
                            time.perf_counter() - t_process,
                        )
                for sa in itertools.islice(shard_iter, len(done)):
                    pending.add(pool.submit(_process_shard_from_parquet, **sa))

        # Signal writer thread to finish and wait for it
        write_queue.put(None)
        writer_thread.join()
        if write_errors:
            raise write_errors[0]

    # Cleanup temp sharded parquet
    sharded_parquet.unlink(missing_ok=True)

    process_seconds = round(time.perf_counter() - t_process, 4)
    shard_stats.sort(key=lambda s: s["shard_id"])

    # Profiling summary
    cpu_u1, cpu_s1 = _cpu_times()
    wall = time.perf_counter() - t_wall_start

    profile = PrepProfile(
        wall_seconds=round(wall, 3),
        cpu_user_seconds=round(cpu_u1 - cpu_u0, 3),
        cpu_system_seconds=round(cpu_s1 - cpu_s0, 3),
        cpu_utilization=round(((cpu_u1 - cpu_u0) + (cpu_s1 - cpu_s0)) / max(wall, 0.001), 4),
        peak_rss_gb=round(peak_rss, 3),
        total_input_rows=n_rows,
        total_output_rows=total_output,
        rows_per_second=round(total_output / max(wall, 0.001), 1),
        missing_rows=total_missing,
        bytes_written=_dir_size_bytes(out_path),
        phase_seconds={
            "parquet_read": parquet_read_seconds,
            "process_and_write": process_seconds,
        },
    )

    # Worker load balance stats
    shard_walls = [s["wall_seconds"] for s in shard_stats]
    worker_stats = {
        "min_shard_wall": round(min(shard_walls), 4) if shard_walls else 0,
        "max_shard_wall": round(max(shard_walls), 4) if shard_walls else 0,
        "median_shard_wall": round(float(np.median(shard_walls)), 4) if shard_walls else 0,
    }

    index = {
        "cache_version": 5,
        "compressed": compress,
        "label": bool(label),
        "tensor_length": tensor_length,
        "cram_path": str(cram_path),
        "parquet_path": str(parquet_path),
        "total_shards": total_shards,
        "shard_size": shard_size,
        "num_workers": effective_workers,
        "fetch_mode": fetch_mode,
        "total_input_rows": n_rows,
        "total_output_rows": total_output,
        "total_missing_rows": total_missing,
        "shard_stats": shard_stats,
        "worker_stats": worker_stats,
        "profile": asdict(profile),
    }

    index_path = out_path / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    profile.log_summary("cram_to_tensor_cache")
    logger.info("Tensor cache written to %s (%d shards, %d rows)", out_path, total_shards, total_output)

    return index


# ---------------------------------------------------------------------------
# Inference streaming
# ---------------------------------------------------------------------------


def stream_cram_to_tensors(  # noqa: PLR0915
    cram_path: str,
    parquet_path: str,
    encoders: Encoders,
    tensor_length: int = 300,
    reference_path: str | None = None,
    batch_size: int = 256,
) -> Iterator[dict]:
    """Yield batches of tensors by streaming from CRAM -- no disk cache.

    Each yielded dict has the same structure as a tensor cache chunk,
    ready for direct model inference.
    """
    pf = pq.ParquetFile(parquet_path)
    available_cols = [c for c in _PARQUET_COLUMNS if c in (pf.schema.names or [])]

    cram = pysam.AlignmentFile(cram_path, "rc", reference_filename=reference_path)

    base_vocab = encoders.base_vocab
    base_default = base_vocab["N"]
    tm_vocab = encoders.tm_vocab
    tm_default = tm_vocab.get("<MISSING>", 0)
    st_vocab = encoders.st_vocab
    st_default = st_vocab.get("<MISSING>", 0)
    et_vocab = encoders.et_vocab
    et_default = et_vocab.get("<MISSING>", 0)

    batch_rows_buf: list[dict] = []

    def _flush_batch(buf: list[dict]) -> dict:
        n = len(buf)
        rb = np.zeros((n, tensor_length), dtype=np.int16)
        rfb = np.zeros((n, tensor_length), dtype=np.int16)
        xp = np.zeros((n, 6, tensor_length), dtype=np.float16)
        xc = np.zeros((n, 4), dtype=np.float16)
        mk = np.zeros((n, tensor_length), dtype=np.uint8)
        lb = np.zeros(n, dtype=np.uint8)
        tm_arr = np.zeros(n, dtype=np.int8)
        st_arr = np.zeros(n, dtype=np.int8)
        et_arr = np.zeros(n, dtype=np.int8)
        chroms = []
        poss = np.zeros(n, dtype=np.int32)
        rns = []

        for i, item in enumerate(buf):
            rb[i] = item["read_base_idx"]
            rfb[i] = item["ref_base_idx"]
            xp[i] = item["x_num_pos"]
            xc[i] = item["x_num_const"]
            mk[i] = item["mask"]
            lb[i] = item["label"]
            tm_arr[i] = item["tm_idx"]
            st_arr[i] = item["st_idx"]
            et_arr[i] = item["et_idx"]
            chroms.append(item["chrom"])
            poss[i] = item["pos"]
            rns.append(item["rn"])

        return {
            "read_base_idx": torch.from_numpy(rb),
            "ref_base_idx": torch.from_numpy(rfb),
            "tm_idx": torch.from_numpy(tm_arr),
            "st_idx": torch.from_numpy(st_arr),
            "et_idx": torch.from_numpy(et_arr),
            "x_num_pos": torch.from_numpy(xp),
            "x_num_const": torch.from_numpy(xc),
            "mask": torch.from_numpy(mk),
            "label": torch.from_numpy(lb),
            "chrom": np.asarray(chroms, dtype=object),
            "pos": poss,
            "rn": np.asarray(rns, dtype=object),
        }

    for pa_batch in pf.iter_batches(batch_size=5000, columns=available_cols):
        table_dict = pa_batch.to_pydict()
        batch_n = len(table_dict[available_cols[0]])
        for i in range(batch_n):
            row = {k: table_dict[k][i] for k in table_dict}
            chrom_val = row[CHROM]
            pos_val = int(row[POS])
            rn_val = row["RN"]

            rec = _fetch_read_from_cram(cram, chrom_val, pos_val, rn_val)
            if rec is None:
                continue

            tags = dict(rec.get_tags(with_value_type=False))
            read_len = len(rec.query_sequence or "")
            tp_raw = _to_numpy_tp(tags.get("tp"), read_len)
            t0_raw = _to_string_t0(tags.get("t0"), read_len)

            aligned = _build_gapped_channels(rec, pos_val, tp_raw, t0_raw, None)
            aln_len = len(aligned["read_base_aln"])
            valid = min(tensor_length, aln_len)

            read_arr = np.zeros(tensor_length, dtype=np.int16)
            ref_arr = np.zeros(tensor_length, dtype=np.int16)
            read_arr[:valid] = [base_vocab.get(t, base_default) for t in aligned["read_base_aln"][:valid]]
            ref_arr[:valid] = [base_vocab.get(t, base_default) for t in aligned["ref_base_aln"][:valid]]

            xp_row = np.zeros((6, tensor_length), dtype=np.float16)
            q = aligned["qual_aln"][:valid]
            xp_row[0, :valid] = q / 10.0
            xp_row[1, :valid] = aligned["tp_aln"][:valid]
            xp_row[2, :valid] = 1.0
            xp_row[3, :valid] = aligned["focus_aln"][:valid]
            xp_row[4, :valid] = aligned["softclip_mask_aln"][:valid]
            xp_row[5, :valid] = aligned["t0_aln"][:valid] / 10.0

            mask_row = np.zeros(tensor_length, dtype=np.uint8)
            mask_row[:valid] = 1

            st_value = tags.get("st", row.get("st", None))
            et_value = tags.get("et", row.get("et", None))
            mixed_flag = float((st_value == "MIXED") or (et_value == "MIXED"))
            xc_row = np.array(
                [
                    float(int(rec.is_reverse)),
                    float(rec.mapping_quality) / 60.0,
                    float(tags.get("rq", row.get("rq", 0.0) or 0.0)),
                    mixed_flag,
                ],
                dtype=np.float16,
            )

            batch_rows_buf.append(
                {
                    "read_base_idx": read_arr,
                    "ref_base_idx": ref_arr,
                    "tm_idx": tm_vocab.get(tags.get("tm", row.get("tm")) or "<MISSING>", tm_default),
                    "st_idx": st_vocab.get(st_value or "<MISSING>", st_default),
                    "et_idx": et_vocab.get(et_value or "<MISSING>", et_default),
                    "x_num_pos": xp_row,
                    "x_num_const": xc_row,
                    "mask": mask_row,
                    "label": 0,
                    "chrom": str(chrom_val),
                    "pos": int(pos_val),
                    "rn": str(rn_val),
                }
            )

            if len(batch_rows_buf) >= batch_size:
                yield _flush_batch(batch_rows_buf)
                batch_rows_buf = []

    if batch_rows_buf:
        yield _flush_batch(batch_rows_buf)

    cram.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CRAM + parquet -> tensor cache")
    ap.add_argument("--cram", required=True, help="Path to source CRAM file")
    ap.add_argument("--parquet", required=True, help="Path to feature map parquet")
    ap.add_argument("--label", required=True, choices=["positive", "negative", "inference"])
    ap.add_argument("--output", required=True, help="Output directory for tensor cache shards")
    ap.add_argument("--reference", default=None, help="Reference FASTA for CRAM decoding")
    ap.add_argument("--vocab-config", default=None, help="Path to vocab_config.json (default: bundled)")
    ap.add_argument("--tensor-length", type=int, default=300)
    ap.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--shard-size", type=int, default=25000)
    ap.add_argument(
        "--fetch-mode",
        choices=["samtools", "pysam"],
        default="samtools",
        help="Read fetch backend: 'samtools' (default) uses samtools -N pipe, 'pysam' uses pure-pysam iteration",
    )
    ap.add_argument("--chromosomes", nargs="*", default=None, help="Only tensorize these chromosomes")

    # Fold-based chromosome selection (alternative to --chromosomes)
    fold_group = ap.add_argument_group("fold-based chromosome selection")
    fold_group.add_argument("--fold-idx", type=int, default=None, help="Fold index for per-fold tensorization")
    fold_group.add_argument("--training-regions", default=None, help="Interval list for fold→chrom mapping")
    fold_group.add_argument("--num-folds", type=int, default=3, help="Number of cross-validation folds")
    fold_group.add_argument("--random-seed", type=int, default=0, help="Random seed for fold splitting")
    fold_group.add_argument(
        "--holdout-chromosomes", default=None, help="Comma-separated holdout chromosomes (e.g. chr21,chr22)"
    )
    ap.add_argument("--compress", action="store_true", default=False, help="Gzip-compress output shards (.pt.gz)")
    return ap.parse_args(argv)


def _resolve_chromosomes(args: argparse.Namespace) -> list[str] | None:
    """Resolve the chromosome list from --chromosomes or --fold-idx arguments."""
    if args.chromosomes is not None:
        return args.chromosomes

    if args.fold_idx is not None:
        if args.training_regions is None:
            raise ValueError("--training-regions is required when --fold-idx is specified")
        from ugbio_srsnv.split_manifest import build_split_manifest  # noqa: PLC0415

        holdout = args.holdout_chromosomes.split(",") if args.holdout_chromosomes else None
        manifest = build_split_manifest(
            training_regions=args.training_regions,
            k_folds=args.num_folds,
            random_seed=args.random_seed,
            holdout_chromosomes=holdout,
        )
        chrom_to_fold = manifest["chrom_to_fold"]
        fold_chroms = [c for c, f in chrom_to_fold.items() if f == args.fold_idx]
        # Fold 0 also includes holdout/test chromosomes
        if args.fold_idx == 0 and manifest.get("test_chromosomes"):
            fold_chroms.extend(manifest["test_chromosomes"])
        logger.info("Fold %d chromosomes: %s", args.fold_idx, ", ".join(sorted(fold_chroms)))
        return fold_chroms

    return None


def run(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    encoders = load_vocab_config(args.vocab_config)
    chromosomes = _resolve_chromosomes(args)
    return cram_to_tensor_cache(
        cram_path=args.cram,
        parquet_path=args.parquet,
        encoders=encoders,
        output_dir=args.output,
        label=(args.label == "positive"),
        tensor_length=args.tensor_length,
        reference_path=args.reference,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
        fetch_mode=args.fetch_mode,
        chromosomes=chromosomes,
        compress=args.compress,
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
