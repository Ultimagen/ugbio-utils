"""High-throughput DNN inference pipeline: parallel CRAM workers, multi-GPU
TensorRT (or PyTorch) inference, and annotated VCF output with ML_QUAL/SNVQ.

Supports two modes:
- **Single model**: ``--metadata srsnv_dnn_metadata.json``
- **K-fold ensemble**: ``--ensemble-manifest ensemble.json`` (fold-aware routing)

Usage::

    dnn_vcf_inference \\
        --featuremap-vcf input.vcf.gz \\
        --cram source.cram \\
        --reference ref.fa \\
        --metadata srsnv_dnn_metadata.json \\
        --output output.vcf.gz

    dnn_vcf_inference \\
        --featuremap-vcf input.vcf.gz \\
        --cram source.cram \\
        --ensemble-manifest ensemble.json \\
        --output output.vcf.gz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import numpy as np
import pysam
from pyarrow import parquet as pq
from ugbio_core.logger import logger
from ugbio_core.vcfbed.variant_annotation import VcfAnnotator

from ugbio_srsnv.deep_srsnv.cram_to_tensors import (
    _PARQUET_COLUMNS,
    CHROM,
    POS,
    _process_shard,
    _process_shard_from_parquet,
    _samtools_available,
    _worker_init,
)
from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config
from ugbio_srsnv.deep_srsnv.inference.trt_engine import load_inference_engine
from ugbio_srsnv.srsnv_utils import MAX_PHRED, prob_to_phred

ML_QUAL = "ML_QUAL"


# ---------------------------------------------------------------------------
# DNNQualAnnotator
# ---------------------------------------------------------------------------


class DNNQualAnnotator(VcfAnnotator):
    """VCF annotator that stamps pre-computed DNN predictions as ML_QUAL / SNVQ."""

    def __init__(  # noqa: PLR0913
        self,
        predictions: dict[tuple[str, int, str], float],
        quality_interpolation_fn=None,
        low_qual_threshold: float = 40.0,
        rn_format_key: str = "RN",
        rn_info_key: str | None = None,
    ):
        self.predictions = predictions
        self.quality_fn = quality_interpolation_fn
        self.threshold = low_qual_threshold
        self.rn_format_key = rn_format_key
        self.rn_info_key = rn_info_key

    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        header.info.add(ML_QUAL, "1", "Float", "DNN ML quality score (Phred-scaled)")
        header.filters.add("LowQual", None, None, "ML_QUAL below quality threshold")
        return header

    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        for rec in records:
            rn = self._extract_rn(rec)
            if rn is None:
                continue
            key = (rec.chrom, rec.pos, rn)
            prob = self.predictions.get(key)
            if prob is None:
                continue

            ml_qual = float(prob_to_phred(np.array([prob]), max_value=MAX_PHRED)[0])
            rec.info[ML_QUAL] = ml_qual

            if self.quality_fn is not None:
                snvq = float(np.around(self.quality_fn(ml_qual), decimals=2))
                rec.qual = snvq
                if snvq >= self.threshold:
                    rec.filter.add("PASS")
                else:
                    rec.filter.add("LowQual")
        return records

    def _extract_rn(self, rec: pysam.VariantRecord) -> str | None:
        """Get read name from FORMAT/RN or INFO field."""
        try:
            if rec.samples and self.rn_format_key:
                val = rec.samples[0].get(self.rn_format_key)
                if val is not None:
                    return str(val)
        except (KeyError, IndexError):
            pass
        if self.rn_info_key:
            try:
                return str(rec.info[self.rn_info_key])
            except KeyError:
                pass
        return None


# ---------------------------------------------------------------------------
# SNVQ recalibration LUT
# ---------------------------------------------------------------------------


class _QualityInterpolator:
    """Picklable MQUAL->SNVQ interpolation callable."""

    def __init__(self, x_lut: np.ndarray, y_lut: np.ndarray):
        self.x_lut = x_lut
        self.y_lut = y_lut

    def __call__(self, mqual):
        return np.interp(mqual, self.x_lut, self.y_lut, left=0, right=self.y_lut[-1])


def _build_quality_fn(metadata: dict) -> _QualityInterpolator | None:
    """Build an MQUAL->SNVQ interpolation function from metadata LUT."""
    table = metadata.get("quality_recalibration_table")
    min_lut_entries = 2
    if not table or len(table) < min_lut_entries:
        return None
    return _QualityInterpolator(np.array(table[0]), np.array(table[1]))


# ---------------------------------------------------------------------------
# Parallel CRAM fetch + multi-GPU inference
# ---------------------------------------------------------------------------


def _run_inference(  # noqa: PLR0913
    cram_path: str,
    parquet_path: str,
    metadata: dict,
    engines: list,
    *,
    num_cram_workers: int = 4,
    shard_size: int = 10000,
    batch_size: int = 512,
    tensor_length: int = 300,
    reference_path: str | None = None,
    fetch_mode: str = "samtools",
) -> dict[tuple[str, int, str], float]:
    """Phase 2: pipelined CRAM fetch + multi-GPU inference, returns predictions dict."""
    encoders = load_vocab_config()

    pf = pq.ParquetFile(parquet_path)
    available_cols = [c for c in _PARQUET_COLUMNS if c in (pf.schema.names or [])]
    table = pf.read(columns=available_cols)
    col_dict = table.to_pydict()
    n_rows = len(col_dict[available_cols[0]])
    rows = [{k: col_dict[k][i] for k in col_dict} for i in range(n_rows)]
    rows.sort(key=lambda r: (str(r.get(CHROM, "")), int(r.get(POS, 0))))

    shard_common = {
        "cram_path": cram_path,
        "reference_path": reference_path,
        "encoders": encoders,
        "tensor_length": tensor_length,
        "label": False,
        "max_edist": None,
        "fetch_mode": fetch_mode,
    }
    shard_args_list = [
        {"shard_id": sid, "rows": rows[i : i + shard_size], **shard_common}
        for sid, i in enumerate(range(0, len(rows), shard_size))
    ]
    total_shards = len(shard_args_list)
    logger.info(
        "Inference: %d rows -> %d shards (%d workers, %d GPUs)", n_rows, total_shards, num_cram_workers, len(engines)
    )

    t_start = time.perf_counter()
    predictions = _pipelined_inference(shard_args_list, engines, batch_size, num_cram_workers)
    logger.info("Inference complete: %d predictions in %.1fs", len(predictions), time.perf_counter() - t_start)
    return predictions


def _predict_shard(
    chunk: dict,
    engines: list,
    gpu_idx: int,
    batch_size: int,
    predictions: dict[tuple[str, int, str], float],
) -> None:
    """Run a single shard's tensors through a GPU engine and collect predictions."""
    engine = engines[gpu_idx % len(engines)]
    n = len(chunk["chrom"])
    if n == 0:
        return
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        s = slice(batch_start, batch_end)
        batch = {
            "read_base_idx": chunk["read_base_idx"][s],
            "ref_base_idx": chunk["ref_base_idx"][s],
            "t0_idx": chunk["t0_idx"][s],
            "tm_idx": chunk["tm_idx"][s],
            "st_idx": chunk["st_idx"][s],
            "et_idx": chunk["et_idx"][s],
            "x_num_pos": chunk["x_num_pos"][s],
            "x_num_const": chunk["x_num_const"][s],
            "mask": chunk["mask"][s],
        }
        probs = engine.predict_batch(batch)
        for j in range(len(probs)):
            idx = batch_start + j
            key = (str(chunk["chrom"][idx]), int(chunk["pos"][idx]), str(chunk["rn"][idx]))
            predictions[key] = float(probs[j])


def _gpu_consumer_thread(
    engine,
    queue: Queue,
    batch_size: int,
    local_preds: dict[tuple[str, int, str], float],
    done_event: Event,
) -> None:
    """Pull shard chunks from *queue* and run GPU inference until producers finish.

    Pushes the engine's CUDA context onto this thread's stack so PyCUDA
    operations target the correct GPU.
    """
    engine.push_context()
    try:
        while True:
            try:
                chunk = queue.get(timeout=1.0)
            except Empty:
                if done_event.is_set():
                    break
                continue
            if chunk is None:
                break
            _predict_shard(chunk, [engine], 0, batch_size, local_preds)
            queue.task_done()
    finally:
        engine.pop_context()


def _pipelined_inference(
    shard_args_list: list[dict],
    engines: list,
    batch_size: int,
    num_cram_workers: int,
    process_fn=None,
) -> dict[tuple[str, int, str], float]:
    """Run CPU shard processing and multi-GPU inference concurrently.

    CPU workers produce tensor chunks via ``ProcessPoolExecutor`` and push them
    into a ``Queue``.  Per-GPU consumer threads pull from the queue and run
    ``predict_batch`` in parallel, giving true multi-GPU utilisation.

    Parameters
    ----------
    shard_args_list
        List of keyword-argument dicts, one per shard.  Each dict is unpacked
        as ``process_fn(**shard_args)``.
    process_fn
        Callable that processes a shard.  Defaults to ``_process_shard``.
    """
    fn = process_fn or _process_shard
    total_shards = len(shard_args_list)
    effective_workers = max(1, num_cram_workers)

    prefetch = len(engines) * 2
    q: Queue = Queue(maxsize=prefetch)
    done_event = Event()
    local_preds: list[dict[tuple[str, int, str], float]] = [{} for _ in engines]

    gpu_threads: list[Thread] = []
    for i, engine in enumerate(engines):
        t = Thread(
            target=_gpu_consumer_thread,
            args=(engine, q, batch_size, local_preds[i], done_event),
            daemon=True,
        )
        t.start()
        gpu_threads.append(t)

    completed = 0
    t_start = time.perf_counter()

    try:
        if effective_workers == 1:
            for shard_args in shard_args_list:
                _sid, chunk, _stats = fn(**shard_args)
                q.put(chunk)
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total_shards:
                    logger.info(
                        "Inference progress: %d/%d shards (%.1fs)",
                        completed,
                        total_shards,
                        time.perf_counter() - t_start,
                    )
        else:
            with ProcessPoolExecutor(max_workers=effective_workers, initializer=_worker_init) as pool:
                futures = {pool.submit(fn, **shard_args): shard_args for shard_args in shard_args_list}
                for future in as_completed(futures):
                    _sid, chunk, _stats = future.result()
                    q.put(chunk)
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == total_shards:
                        logger.info(
                            "Inference progress: %d/%d shards (%.1fs)",
                            completed,
                            total_shards,
                            time.perf_counter() - t_start,
                        )
    finally:
        done_event.set()
        for t in gpu_threads:
            t.join()

    predictions: dict[tuple[str, int, str], float] = {}
    for lp in local_preds:
        predictions.update(lp)

    return predictions


# ---------------------------------------------------------------------------
# Ensemble manifest (k-fold)
# ---------------------------------------------------------------------------


def load_ensemble_manifest(path: str) -> dict:
    """Load and validate an ensemble manifest JSON for k-fold inference.

    Expected schema::

        {
            "k_folds": 3,
            "chrom_to_fold": {"chr1": 0, "chr2": 1, ...},
            "folds": [
                {"fold_idx": 0, "metadata_path": "/path/to/fold0/metadata.json"},
                ...
            ],
            "quality_recalibration_table": [[...], [...]]  // optional
        }
    """
    with open(path) as f:
        manifest = json.load(f)

    for key in ("k_folds", "chrom_to_fold", "folds"):
        if key not in manifest:
            raise ValueError(f"Ensemble manifest missing required key: {key!r}")

    k_folds = manifest["k_folds"]
    folds = manifest["folds"]
    if len(folds) != k_folds:
        raise ValueError(f"k_folds={k_folds} but {len(folds)} fold entries provided")

    for i, fold in enumerate(folds):
        if "fold_idx" not in fold or "metadata_path" not in fold:
            raise ValueError(f"Fold entry {i} missing 'fold_idx' or 'metadata_path'")
        meta_p = Path(fold["metadata_path"])
        if not meta_p.exists():
            raise FileNotFoundError(f"Fold {fold['fold_idx']} metadata not found: {meta_p}")

    return manifest


def _partition_rows_by_fold(
    rows: list[dict],
    chrom_to_fold: dict[str, int],
    k_folds: int,
) -> tuple[list[list[dict]], list[dict]]:
    """Partition rows into per-fold groups and a test group.

    Returns ``(fold_rows, test_rows)`` where ``fold_rows[k]`` contains rows
    for fold k and ``test_rows`` contains rows whose chromosome is not in
    ``chrom_to_fold`` (predicted by all models and aggregated).
    """
    fold_rows: list[list[dict]] = [[] for _ in range(k_folds)]
    test_rows: list[dict] = []
    for row in rows:
        chrom = str(row.get(CHROM, row.get("chrom", "")))
        fold_id = chrom_to_fold.get(chrom)
        if fold_id is not None and 0 <= fold_id < k_folds:
            fold_rows[fold_id].append(row)
        else:
            test_rows.append(row)
    return fold_rows, test_rows


def aggregate_fold_probabilities(prob_matrix: np.ndarray) -> np.ndarray:
    """Average K models' probabilities in logit space.

    Parameters
    ----------
    prob_matrix
        Shape ``(K, N)`` array of probabilities from K fold models for N reads.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` aggregated probabilities.
    """
    eps = 1e-7
    clipped = np.clip(prob_matrix, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    mean_logit = np.mean(logits, axis=0)
    return 1.0 / (1.0 + np.exp(-mean_logit))


# ---------------------------------------------------------------------------
# Fold-aware inference (k-fold ensemble)
# ---------------------------------------------------------------------------


def _run_fold_inference(  # noqa: PLR0913
    cram_path: str,
    parquet_path: str,
    n_rows: int,
    columns: list[str],
    engines: list,
    *,
    num_cram_workers: int = 4,
    batch_size: int = 512,
    tensor_length: int = 300,
    reference_path: str | None = None,
    fetch_mode: str = "samtools",
) -> dict[tuple[str, int, str], float]:
    """Run inference from a pre-sorted parquet file with row-group-per-shard layout.

    Each worker reads its own row group from disk — no row dicts in the main process.
    """
    if n_rows == 0:
        return {}

    encoders = load_vocab_config()

    pf = pq.ParquetFile(parquet_path)
    total_shards = pf.metadata.num_row_groups

    shard_common = {
        "cram_path": cram_path,
        "reference_path": reference_path,
        "encoders": encoders,
        "tensor_length": tensor_length,
        "label": False,
        "max_edist": None,
        "fetch_mode": fetch_mode,
    }

    shard_args_list = [
        {"shard_id": rg, "parquet_path": parquet_path, "row_group_id": rg, "columns": columns, **shard_common}
        for rg in range(total_shards)
    ]

    logger.info(
        "Fold inference: %d rows -> %d row-group shards (%d workers, %d GPUs)",
        n_rows,
        total_shards,
        num_cram_workers,
        len(engines),
    )

    t_start = time.perf_counter()
    predictions = _pipelined_inference(
        shard_args_list, engines, batch_size, num_cram_workers, process_fn=_process_shard_from_parquet
    )
    elapsed = time.perf_counter() - t_start

    logger.info(
        "Fold inference: %d rows, %d shards, %d predictions in %.1fs",
        n_rows,
        total_shards,
        len(predictions),
        elapsed,
    )
    return predictions


def _run_fold_inference_from_rows(  # noqa: PLR0913
    cram_path: str,
    rows: list[dict],
    engines: list,
    *,
    num_cram_workers: int = 4,
    shard_size: int = 10000,
    batch_size: int = 512,
    tensor_length: int = 300,
    reference_path: str | None = None,
    fetch_mode: str = "samtools",
) -> dict[tuple[str, int, str], float]:
    """Run inference on pre-loaded rows (used by ensemble path).

    Like ``_run_fold_inference`` but accepts in-memory rows instead of a parquet path.
    """
    if not rows:
        return {}

    encoders = load_vocab_config()

    rows_sorted = sorted(rows, key=lambda r: (str(r.get(CHROM, "")), int(r.get(POS, 0))))
    shard_common = {
        "cram_path": cram_path,
        "reference_path": reference_path,
        "encoders": encoders,
        "tensor_length": tensor_length,
        "label": False,
        "max_edist": None,
        "fetch_mode": fetch_mode,
    }
    shard_args_list = [
        {"shard_id": sid, "rows": rows_sorted[i : i + shard_size], **shard_common}
        for sid, i in enumerate(range(0, len(rows_sorted), shard_size))
    ]
    total_shards = len(shard_args_list)

    t_start = time.perf_counter()
    predictions = _pipelined_inference(shard_args_list, engines, batch_size, num_cram_workers)
    elapsed = time.perf_counter() - t_start

    logger.info(
        "Fold inference: %d rows, %d shards, %d predictions in %.1fs",
        len(rows),
        total_shards,
        len(predictions),
        elapsed,
    )
    return predictions


def _run_ensemble_inference(  # noqa: PLR0913, PLR0915
    manifest: dict,
    cram_path: str,
    parquet_path: str,
    *,
    backend: str = "pytorch",
    gpu_ids: list[int],
    num_cram_workers: int = 4,
    shard_size: int = 10000,
    batch_size: int = 512,
    tensor_length: int = 300,
    reference_path: str | None = None,
    fetch_mode: str = "samtools",
) -> dict[tuple[str, int, str], float]:
    """Fold-aware inference: iterate folds sequentially, all GPUs data-parallel per fold."""

    k_folds = manifest["k_folds"]
    chrom_to_fold = manifest["chrom_to_fold"]
    fold_entries = sorted(manifest["folds"], key=lambda f: f["fold_idx"])

    pf = pq.ParquetFile(parquet_path)
    available_cols = [c for c in _PARQUET_COLUMNS if c in (pf.schema.names or [])]
    table = pf.read(columns=available_cols)
    col_dict = table.to_pydict()
    n_rows = len(col_dict[available_cols[0]])
    all_rows = [{k: col_dict[k][i] for k in col_dict} for i in range(n_rows)]

    fold_rows, test_rows = _partition_rows_by_fold(all_rows, chrom_to_fold, k_folds)
    fold_sizes = [len(fr) for fr in fold_rows]
    logger.info(
        "Ensemble: %d total rows -> folds %s + %d test rows",
        n_rows,
        fold_sizes,
        len(test_rows),
    )

    all_predictions: dict[tuple[str, int, str], float] = {}
    t_start = time.perf_counter()

    infer_kwargs = {
        "cram_path": cram_path,
        "num_cram_workers": num_cram_workers,
        "shard_size": shard_size,
        "batch_size": batch_size,
        "tensor_length": tensor_length,
        "reference_path": reference_path,
        "fetch_mode": fetch_mode,
    }

    for fold_entry in fold_entries:
        fold_k = fold_entry["fold_idx"]
        meta_path = fold_entry["metadata_path"]
        rows_k = fold_rows[fold_k] if fold_k < len(fold_rows) else []
        if not rows_k:
            logger.info("Fold %d: 0 reads, skipping", fold_k)
            continue

        logger.info("Fold %d: loading model from %s (%d reads)", fold_k, meta_path, len(rows_k))
        engines = _create_engines(meta_path, backend=backend, gpu_ids=gpu_ids)
        try:
            fold_preds = _run_fold_inference_from_rows(rows=rows_k, engines=engines, **infer_kwargs)
            all_predictions.update(fold_preds)
        finally:
            for eng in engines:
                eng.close()

    if test_rows:
        logger.info("Processing %d test reads (aggregating across %d fold models)", len(test_rows), k_folds)
        test_keys = [
            (str(r.get(CHROM, r.get("chrom", ""))), int(r.get(POS, r.get("pos", 0))), str(r.get("RN", r.get("rn", ""))))
            for r in test_rows
        ]
        prob_matrix = np.zeros((k_folds, len(test_rows)), dtype=np.float64)

        for fold_entry in fold_entries:
            fold_k = fold_entry["fold_idx"]
            meta_path = fold_entry["metadata_path"]
            logger.info("Test reads: predicting with fold %d model", fold_k)
            engines = _create_engines(meta_path, backend=backend, gpu_ids=gpu_ids)
            try:
                test_preds = _run_fold_inference_from_rows(rows=test_rows, engines=engines, **infer_kwargs)
                for j, key in enumerate(test_keys):
                    prob_matrix[fold_k, j] = test_preds.get(key, 0.5)
            finally:
                for eng in engines:
                    eng.close()

        aggregated = aggregate_fold_probabilities(prob_matrix)
        for j, key in enumerate(test_keys):
            all_predictions[key] = float(aggregated[j])
        logger.info("Test reads aggregated: %d predictions", len(test_keys))

    elapsed = time.perf_counter() - t_start
    logger.info("Ensemble inference complete: %d predictions in %.1fs", len(all_predictions), elapsed)
    return all_predictions


def _create_engines(
    metadata_path: str,
    *,
    backend: str = "pytorch",
    gpu_ids: list[int],
) -> list:
    """Load one inference engine per GPU for a given metadata file.

    For TRT engines, each engine's CUDA context is popped after creation so it
    can be pushed onto consumer threads later.
    """
    engines = []
    for gid in gpu_ids:
        try:
            eng = load_inference_engine(metadata_path, backend=backend, device_id=gid)
            if hasattr(eng, "pop_context"):
                eng.pop_context()
            engines.append(eng)
        except Exception:
            logger.warning("Failed to create engine on GPU:%d for %s, skipping", gid, metadata_path, exc_info=True)
    if not engines:
        raise RuntimeError(f"No inference engines could be created from {metadata_path}")
    return engines


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_inference_pipeline(  # noqa: PLR0913, C901, PLR0912
    featuremap_vcf: str,
    cram_path: str,
    output_vcf: str,
    *,
    metadata_path: str | None = None,
    ensemble_manifest_path: str | None = None,
    reference_path: str | None = None,
    backend: str = "trt",
    engine_path: str | None = None,
    checkpoint_path: str | None = None,
    gpu_ids: list[int] | None = None,
    num_cram_workers: int = 4,
    shard_size: int = 10000,
    batch_size: int = 512,
    tensor_length: int = 300,
    low_qual_threshold: float = 40.0,
    parquet_path: str | None = None,
    fetch_mode: str = "samtools",
) -> str:
    """Run the full DNN VCF inference pipeline.

    Supports two modes:
    - **Single model**: provide ``metadata_path``
    - **K-fold ensemble**: provide ``ensemble_manifest_path``

    Returns the path to the output VCF.
    """
    if not metadata_path and not ensemble_manifest_path:
        raise ValueError("Either metadata_path or ensemble_manifest_path must be provided")

    import torch.multiprocessing  # noqa: PLC0415

    torch.multiprocessing.set_sharing_strategy("file_system")

    t0 = time.perf_counter()

    # ── Phase 1: setup ──
    if fetch_mode == "samtools" and not _samtools_available():
        logger.warning("samtools not found; falling back to pysam fetch mode")
        fetch_mode = "pysam"

    if gpu_ids is None:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = [0]

    # ── Parquet preparation ──
    if parquet_path is None:
        import tempfile  # noqa: PLC0415

        tmp_dir = tempfile.mkdtemp(prefix="dnn_infer_")
        parquet_path = str(Path(tmp_dir) / "featuremap.parquet")
        logger.info("Converting featuremap VCF to parquet: %s", parquet_path)
        _vcf_to_parquet(featuremap_vcf, parquet_path)

    # ── Dispatch: single model vs k-fold ensemble ──
    if ensemble_manifest_path:
        manifest = load_ensemble_manifest(ensemble_manifest_path)
        quality_fn = _build_quality_fn(manifest)

        predictions = _run_ensemble_inference(
            manifest=manifest,
            cram_path=cram_path,
            parquet_path=parquet_path,
            backend=backend,
            gpu_ids=gpu_ids,
            num_cram_workers=num_cram_workers,
            shard_size=shard_size,
            batch_size=batch_size,
            tensor_length=tensor_length,
            reference_path=reference_path,
            fetch_mode=fetch_mode,
        )
    else:
        with open(metadata_path) as f:
            metadata = json.load(f)
        quality_fn = _build_quality_fn(metadata)

        engines = []
        for gid in gpu_ids:
            try:
                eng = load_inference_engine(
                    metadata_path,
                    backend=backend,
                    device_id=gid,
                    engine_path=engine_path,
                    checkpoint_path=checkpoint_path,
                )
                if hasattr(eng, "pop_context"):
                    eng.pop_context()
                engines.append(eng)
            except Exception:
                logger.warning("Failed to create engine on GPU:%d, skipping", gid, exc_info=True)
        if not engines:
            raise RuntimeError("No inference engines could be created")
        logger.info("Created %d %s engine(s) on GPUs %s", len(engines), backend, gpu_ids[: len(engines)])

        predictions = _run_inference(
            cram_path=cram_path,
            parquet_path=parquet_path,
            metadata=metadata,
            engines=engines,
            num_cram_workers=num_cram_workers,
            shard_size=shard_size,
            batch_size=batch_size,
            tensor_length=tensor_length,
            reference_path=reference_path,
            fetch_mode=fetch_mode,
        )

        for eng in engines:
            eng.close()

    # ── Phase 3: VCF annotation ──
    annotator = DNNQualAnnotator(
        predictions=predictions,
        quality_interpolation_fn=quality_fn,
        low_qual_threshold=low_qual_threshold,
    )
    VcfAnnotator.process_vcf(
        annotators=[annotator],
        input_path=featuremap_vcf,
        output_path=output_vcf,
    )

    wall = time.perf_counter() - t0
    logger.info("DNN VCF inference complete in %.1fs: %d predictions -> %s", wall, len(predictions), output_vcf)
    return output_vcf


def _vcf_to_parquet(vcf_path: str, parquet_path: str) -> None:
    """Lightweight VCF->parquet for inference (extracts CHROM, POS, RN, etc.)."""
    try:
        from ugbio_featuremap.featuremap_to_dataframe import vcf_to_parquet  # noqa: PLC0415

        vcf_to_parquet(vcf_path, parquet_path, drop_format={"AD"})
    except ImportError:
        raise ImportError("ugbio_featuremap is required for VCF-to-parquet conversion") from None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DNN VCF inference: CRAM -> TensorRT/PyTorch -> annotated VCF",
    )
    ap.add_argument("--featuremap-vcf", required=True, help="Input featuremap VCF")
    ap.add_argument("--cram", required=True, help="Source CRAM file")
    ap.add_argument("--reference", default=None, help="Reference FASTA for CRAM decoding")
    ap.add_argument("--output", required=True, help="Output annotated VCF path")
    ap.add_argument("--parquet", default=None, help="Pre-computed featuremap parquet (skip VCF conversion)")

    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--metadata", default=None, help="Path to srsnv_dnn_metadata.json (single model)")
    model_group.add_argument("--ensemble-manifest", default=None, help="Path to ensemble manifest JSON (k-fold)")

    ap.add_argument("--engine-path", default=None, help="Override TRT engine path (single model only)")
    ap.add_argument("--checkpoint", default=None, help="Override model checkpoint path (single model only)")
    ap.add_argument("--backend", default="trt", choices=["trt", "pytorch"], help="Inference backend (default: trt)")
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU IDs (default: all visible)")
    ap.add_argument("--num-cram-workers", type=int, default=None, help="CPU workers for CRAM fetch (default: all CPUs)")
    ap.add_argument("--shard-size", type=int, default=10000, help="Rows per shard")
    ap.add_argument("--batch-size", type=int, default=512, help="GPU inference batch size")
    ap.add_argument("--tensor-length", type=int, default=300, help="Padded sequence length")
    ap.add_argument("--low-qual-threshold", type=float, default=40.0, help="SNVQ threshold for PASS filter")
    ap.add_argument("--fetch-mode", default="samtools", choices=["samtools", "pysam"], help="CRAM fetch strategy")
    return ap.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    num_workers = args.num_cram_workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    run_inference_pipeline(
        featuremap_vcf=args.featuremap_vcf,
        cram_path=args.cram,
        metadata_path=args.metadata,
        ensemble_manifest_path=args.ensemble_manifest,
        output_vcf=args.output,
        reference_path=args.reference,
        backend=args.backend,
        engine_path=args.engine_path,
        checkpoint_path=args.checkpoint,
        gpu_ids=gpu_ids,
        num_cram_workers=num_workers,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        tensor_length=args.tensor_length,
        low_qual_threshold=args.low_qual_threshold,
        parquet_path=args.parquet,
        fetch_mode=args.fetch_mode,
    )


def main() -> None:
    run()


# ---------------------------------------------------------------------------
# Per-fold inference CLI  (dnn_fold_inference)
# ---------------------------------------------------------------------------


def _parse_fold_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DNN single-fold inference: predict one fold's chromosomes",
    )
    ap.add_argument("--featuremap-vcf", required=True, help="Input featuremap VCF")
    ap.add_argument("--cram", required=True, help="Source CRAM file")
    ap.add_argument("--reference", default=None, help="Reference FASTA for CRAM decoding")
    ap.add_argument("--fold-metadata", required=True, help="Path to this fold's srsnv_dnn_metadata.json")
    ap.add_argument("--split-manifest", required=True, help="Path to split_manifest.json (chrom->fold mapping)")
    ap.add_argument("--fold-idx", type=int, required=True, help="Fold index (0-based)")
    ap.add_argument("--output", required=True, help="Output predictions parquet path")
    ap.add_argument("--parquet", default=None, help="Pre-computed featuremap parquet (skip VCF conversion)")
    ap.add_argument("--backend", default="trt", choices=["trt", "pytorch"], help="Inference backend")
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU IDs (default: all visible)")
    ap.add_argument("--num-cram-workers", type=int, default=None, help="CPU workers for CRAM fetch")
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--tensor-length", type=int, default=300)
    ap.add_argument("--fetch-mode", default="samtools", choices=["samtools", "pysam"])
    return ap.parse_args(argv)


def run_fold(argv: list[str] | None = None) -> None:  # noqa: PLR0915, C901, PLR0912
    """Run inference for a single fold's chromosomes and write predictions parquet."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.compute  # noqa: PLC0415
    import torch.multiprocessing  # noqa: PLC0415

    torch.multiprocessing.set_sharing_strategy("file_system")

    if argv is None:
        argv = sys.argv[1:]
    args = _parse_fold_args(argv)

    fold_idx = args.fold_idx
    t0 = time.perf_counter()

    # GPU setup
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    if gpu_ids is None:
        import torch  # noqa: PLC0415

        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]

    num_workers = args.num_cram_workers
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 2)

    fetch_mode = args.fetch_mode
    if fetch_mode == "samtools" and not _samtools_available():
        logger.warning("samtools not found; falling back to pysam fetch mode")
        fetch_mode = "pysam"

    # Parquet preparation
    parquet_path = args.parquet
    if parquet_path is None:
        import tempfile  # noqa: PLC0415

        tmp_dir = tempfile.mkdtemp(prefix="dnn_fold_")
        parquet_path = str(Path(tmp_dir) / "featuremap.parquet")
        logger.info("Converting featuremap VCF to parquet: %s", parquet_path)
        _vcf_to_parquet(args.featuremap_vcf, parquet_path)

    # Read split manifest
    with open(args.split_manifest) as f:
        manifest = json.load(f)
    chrom_to_fold: dict[str, int] = manifest.get("chrom_to_fold", {})

    # Read parquet and filter to this fold's chromosomes (memory-efficient)
    import pyarrow.parquet as pq_mod  # noqa: PLC0415

    pf = pq_mod.ParquetFile(parquet_path)
    available_cols = [c for c in _PARQUET_COLUMNS if c in (pf.schema.names or [])]

    # Build set of chromosomes belonging to this fold
    my_chroms: set[str] = set()
    test_chroms: set[str] = set()
    for chrom, fid in chrom_to_fold.items():
        if fid == fold_idx:
            my_chroms.add(chrom)
    # Fold 0 also handles test chromosomes (not in chrom_to_fold)
    if fold_idx == 0:
        # Read just the CHROM column to discover test chroms
        chrom_col_arr = pf.read(columns=[CHROM]).column(CHROM)
        all_chroms_in_data = set(chrom_col_arr.to_pylist())
        test_chroms = all_chroms_in_data - set(chrom_to_fold.keys())
        del chrom_col_arr
    target_chroms = my_chroms | test_chroms

    # Read only rows matching target chromosomes using row-group filtering
    total_rows = pf.metadata.num_rows
    batches = []
    for batch in pf.iter_batches(batch_size=500_000, columns=available_cols):
        chrom_arr = batch.column(CHROM)
        mask = pa.compute.is_in(chrom_arr, value_set=pa.array(list(target_chroms)))
        filtered_batch = batch.filter(mask)
        if len(filtered_batch) > 0:
            batches.append(filtered_batch)
    if batches:
        filtered = pa.concat_tables([pa.Table.from_batches([b]) for b in batches])
    else:
        filtered = pa.table({c: [] for c in available_cols})
    del batches

    # Sort by chrom, pos using Arrow (no Python dicts)
    n_rows = len(filtered)
    logger.info("Fold %d: %d rows (total %d, filtered %d)", fold_idx, n_rows, total_rows, n_rows)

    if n_rows == 0:
        logger.info("Fold %d: no rows, writing empty predictions", fold_idx)
        empty_table = pa.table({"chrom": [], "pos": [], "rn": [], "prob": []})
        pq_mod.write_table(empty_table, args.output)
        return

    sort_indices = pa.compute.sort_indices(filtered, sort_keys=[(CHROM, "ascending"), (POS, "ascending")])
    sorted_table = filtered.take(sort_indices)
    del filtered, sort_indices

    # Write to temp parquet with row_group_size = shard_size for per-worker reading
    fold_parquet = str(Path(args.output).parent / f"fold_{fold_idx}_sorted.parquet")
    available_cols_final = [c for c in available_cols if c in sorted_table.column_names]
    pq_mod.write_table(sorted_table, fold_parquet, row_group_size=args.shard_size)
    del sorted_table

    # Load engines
    engines = _create_engines(args.fold_metadata, backend=args.backend, gpu_ids=gpu_ids)
    logger.info("Fold %d: %d %s engine(s) on GPUs %s", fold_idx, len(engines), args.backend, gpu_ids[: len(engines)])

    try:
        predictions = _run_fold_inference(
            cram_path=args.cram,
            parquet_path=fold_parquet,
            n_rows=n_rows,
            columns=available_cols_final,
            engines=engines,
            num_cram_workers=num_workers,
            batch_size=args.batch_size,
            tensor_length=args.tensor_length,
            reference_path=args.reference,
            fetch_mode=fetch_mode,
        )
    finally:
        for eng in engines:
            eng.close()

    # Write predictions parquet
    chroms = [k[0] for k in predictions]
    poss = [k[1] for k in predictions]
    rns = [k[2] for k in predictions]
    probs = [predictions[k] for k in predictions]
    out_table = pa.table({"chrom": chroms, "pos": poss, "rn": rns, "prob": probs})
    pq_mod.write_table(out_table, args.output)

    elapsed = time.perf_counter() - t0
    logger.info("Fold %d: wrote %d predictions to %s in %.1fs", fold_idx, len(predictions), args.output, elapsed)


def main_fold() -> None:
    run_fold()


# ---------------------------------------------------------------------------
# Merge + annotate CLI  (dnn_merge_and_annotate)
# ---------------------------------------------------------------------------


def _parse_merge_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge per-fold DNN predictions and annotate VCF",
    )
    ap.add_argument("--featuremap-vcf", required=True, help="Input featuremap VCF to annotate")
    ap.add_argument("--fold-predictions", nargs="+", required=True, help="Per-fold prediction parquet files")
    ap.add_argument("--fold-metadata", required=True, help="Fold 0 metadata JSON (for quality recalibration LUT)")
    ap.add_argument("--output", required=True, help="Output annotated VCF path")
    ap.add_argument("--low-qual-threshold", type=float, default=40.0, help="SNVQ threshold for PASS filter")
    ap.add_argument("--process-number", type=int, default=-2, help="Parallel processes for VCF annotation")
    return ap.parse_args(argv)


def run_merge(argv: list[str] | None = None) -> None:
    """Merge per-fold predictions and annotate the featuremap VCF."""
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_merge_args(argv)

    t0 = time.perf_counter()

    # Load all fold predictions into a unified dict
    predictions: dict[tuple[str, int, str], float] = {}
    for pred_path in args.fold_predictions:
        pred_table = pq.read_table(pred_path)
        chroms = pred_table.column("chrom").to_pylist()
        poss = pred_table.column("pos").to_pylist()
        rns = pred_table.column("rn").to_pylist()
        probs = pred_table.column("prob").to_pylist()
        for c, p, r, prob in zip(chroms, poss, rns, probs, strict=False):
            predictions[(str(c), int(p), str(r))] = float(prob)

    logger.info("Loaded %d predictions from %d fold files", len(predictions), len(args.fold_predictions))

    # Build quality recalibration function from fold metadata
    with open(args.fold_metadata) as f:
        metadata = json.load(f)
    quality_fn = _build_quality_fn(metadata)

    # Annotate VCF
    annotator = DNNQualAnnotator(
        predictions=predictions,
        quality_interpolation_fn=quality_fn,
        low_qual_threshold=args.low_qual_threshold,
    )
    VcfAnnotator.process_vcf(
        annotators=[annotator],
        input_path=args.featuremap_vcf,
        output_path=args.output,
        process_number=args.process_number,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Merge + annotate complete: %d predictions -> %s in %.1fs", len(predictions), args.output, elapsed)


def main_merge() -> None:
    run_merge()


if __name__ == "__main__":
    main()
