"""DNN fold inference from pre-computed tensor cache.

Reads shard_*.pt(.gz) files produced by ``cram_to_tensors`` and feeds
them through a 3-stage pipeline (disk I/O → CPU prep → GPU inference)
to produce a predictions parquet file.

Usage::

    dnn_fold_inference_from_cache \\
        --tensor-cache tensor_cache/ \\
        --fold-metadata srsnv_dnn_metadata.json \\
        --output predictions.parquet \\
        --backend trt \\
        --batch-size 512
"""

from __future__ import annotations

import argparse
import importlib
import io
import itertools
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import numpy as np
import pyarrow as pa
from isal import igzip
from pyarrow import parquet as pq
from ugbio_core.logger import logger

from ugbio_srsnv.deep_srsnv.inference.trt_engine import (
    _compose_x_num,
    _to_numpy,
    _to_numpy_long,
    load_inference_engine,
)

# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------


def _create_engines(
    metadata_path: str,
    *,
    backend: str = "pytorch",
    gpu_ids: list[int],
) -> list:
    """Load one inference engine per GPU for a given metadata file."""
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
# 3-stage pipeline: disk loader → CPU prep → GPU inference
# ---------------------------------------------------------------------------


def _load_tensor_shard(shard_path: str) -> dict:
    """Load a single tensor shard and convert torch tensors to numpy."""
    _torch = importlib.import_module("torch")

    if shard_path.endswith(".gz"):
        with igzip.open(shard_path, "rb") as f:
            chunk = _torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
    else:
        chunk = _torch.load(shard_path, map_location="cpu", weights_only=False)
    result = {}
    for key, val in chunk.items():
        if isinstance(val, _torch.Tensor):
            result[key] = val.numpy()
        else:
            result[key] = val
    return result


def _cpu_prep_worker_fn(
    raw_queue: Queue,
    gpu_queue: Queue,
    error_event: Event,
    error_holder: list[BaseException],
) -> None:
    """Stage 2: prepare GPU-ready arrays from raw tensor chunks."""
    try:
        while True:
            chunk = raw_queue.get()
            if chunk is None:
                break
            x_num = _compose_x_num(chunk)
            prepared = {
                "read_base_idx": _to_numpy_long(chunk["read_base_idx"]),
                "ref_base_idx": _to_numpy_long(chunk["ref_base_idx"]),
                "x_num": x_num,
                "mask": _to_numpy(chunk["mask"]),
            }
            for key in ("tm_idx", "st_idx", "et_idx"):
                if key in chunk:
                    prepared[key] = _to_numpy_long(chunk[key])
            prepared["_chrom"] = chunk["chrom"]
            prepared["_pos"] = chunk["pos"]
            prepared["_rn"] = chunk["rn"]
            gpu_queue.put(prepared)
    except Exception as exc:
        logger.error("_cpu_prep_worker failed: %s", exc, exc_info=True)
        error_holder.append(exc)
        error_event.set()
    finally:
        gpu_queue.put(None)


def _gpu_worker_fn(
    engine,
    gpu_queue: Queue,
    result_queue: Queue,
    batch_size: int,
    error_event: Event,
    error_holder: list[BaseException],
) -> None:
    """Stage 3: GPU inference only."""
    engine.push_context()
    try:
        while True:
            prepared = gpu_queue.get()
            if prepared is None:
                break
            n = prepared["x_num"].shape[0]
            all_probs = []
            for batch_start in range(0, n, batch_size):
                s = slice(batch_start, min(batch_start + batch_size, n))
                batch = {
                    "read_base_idx": prepared["read_base_idx"][s],
                    "ref_base_idx": prepared["ref_base_idx"][s],
                    "x_num": prepared["x_num"][s],
                    "mask": prepared["mask"][s],
                }
                for key in ("tm_idx", "st_idx", "et_idx"):
                    if key in prepared:
                        batch[key] = prepared[key][s]
                all_probs.append(engine.predict_batch_prepared(batch))
            result_queue.put((prepared["_chrom"], prepared["_pos"], prepared["_rn"], np.concatenate(all_probs)))
            gpu_queue.task_done()
    except Exception as exc:
        logger.error("_gpu_worker failed: %s", exc, exc_info=True)
        error_holder.append(exc)
        error_event.set()
    finally:
        engine.pop_context()
        result_queue.put(None)


def _shard_loader_thread_fn(
    shard_files: list,
    raw_queue: Queue,
    num_loader_threads: int,
    error_event: Event,
    error_holder: list[BaseException],
) -> None:
    """Stage 1: load tensor shards from disk into *raw_queue*."""
    try:
        with ThreadPoolExecutor(max_workers=num_loader_threads) as loader_pool:
            max_pending = raw_queue.maxsize
            shard_iter = iter(shard_files)
            pending_loads = {
                loader_pool.submit(_load_tensor_shard, str(p)) for p in itertools.islice(shard_iter, max_pending)
            }
            while pending_loads:
                done_loads, pending_loads = wait(pending_loads, return_when=FIRST_COMPLETED)
                for fut in done_loads:
                    raw_queue.put(fut.result())
                for p in itertools.islice(shard_iter, len(done_loads)):
                    pending_loads.add(loader_pool.submit(_load_tensor_shard, str(p)))
    except Exception as exc:
        logger.error("_loader_thread failed: %s", exc, exc_info=True)
        error_holder.append(exc)
        error_event.set()
    finally:
        raw_queue.put(None)


def _collect_cache_results(
    result_queue: Queue,
    total_shards: int,
    error_event: Event,
    t_start: float,
    *,
    streaming: bool,
    writer=None,
    predictions: dict[tuple[str, int, str], float] | None = None,
) -> tuple[int, int]:
    """Collect GPU results from *result_queue*, returning (total_preds, collected)."""
    total_preds = 0
    collected = 0
    while collected < total_shards:
        try:
            item = result_queue.get(timeout=5.0)
        except Empty:
            if error_event.is_set():
                break
            continue
        if item is None:
            break
        chroms, positions, rns, probs = item
        n = len(probs)
        if streaming:
            table = pa.table(
                {
                    "chrom": chroms[:n].tolist() if hasattr(chroms, "tolist") else list(chroms[:n]),
                    "pos": positions[:n].tolist() if hasattr(positions, "tolist") else list(positions[:n]),
                    "rn": rns[:n].tolist() if hasattr(rns, "tolist") else list(rns[:n]),
                    "prob": probs.tolist(),
                }
            )
            writer.write_table(table)
        else:
            for j in range(n):
                predictions[(str(chroms[j]), int(positions[j]), str(rns[j]))] = float(probs[j])
        total_preds += n
        collected += 1
        if collected == 1 or collected % 10 == 0 or collected == total_shards:
            logger.info(
                "Cache inference progress: %d/%d shards (%.1fs)",
                collected,
                total_shards,
                time.perf_counter() - t_start,
            )
    return total_preds, collected


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _run_fold_inference_from_cache(
    cache_dir: str,
    engines: list,
    batch_size: int = 512,
    output_path: str | None = None,
    num_loader_threads: int = 4,
) -> dict[tuple[str, int, str], float] | int:
    """Run GPU inference from pre-computed tensor cache shards.

    Returns predictions dict or count (if *output_path* streaming).
    """
    cache_path = Path(cache_dir)
    shard_files = sorted(cache_path.glob("shard_*.pt.gz")) or sorted(cache_path.glob("shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.pt files found in {cache_dir}")

    total_shards = len(shard_files)
    logger.info("Loading %d tensor shards from %s (%d loader threads)", total_shards, cache_dir, num_loader_threads)

    raw_queue: Queue = Queue(maxsize=num_loader_threads + 1)
    gpu_queue: Queue = Queue(maxsize=len(engines) * 2)
    result_queue: Queue = Queue()
    error_event = Event()
    error_holder: list[BaseException] = []

    engine = engines[0]

    prep_thread = Thread(
        target=_cpu_prep_worker_fn, args=(raw_queue, gpu_queue, error_event, error_holder), daemon=True
    )
    prep_thread.start()

    gpu_thread = Thread(
        target=_gpu_worker_fn,
        args=(engine, gpu_queue, result_queue, batch_size, error_event, error_holder),
        daemon=True,
    )
    gpu_thread.start()

    loader_t = Thread(
        target=_shard_loader_thread_fn,
        args=(shard_files, raw_queue, num_loader_threads, error_event, error_holder),
        daemon=True,
    )
    loader_t.start()

    streaming = output_path is not None
    writer = None
    if streaming:
        schema = pa.schema([("chrom", pa.string()), ("pos", pa.int64()), ("rn", pa.string()), ("prob", pa.float64())])
        writer = pq.ParquetWriter(output_path, schema)

    predictions: dict[tuple[str, int, str], float] = {}
    t_start = time.perf_counter()

    try:
        total_preds, _collected = _collect_cache_results(
            result_queue,
            total_shards,
            error_event,
            t_start,
            streaming=streaming,
            writer=writer,
            predictions=predictions,
        )

        if error_event.is_set() and error_holder:
            raise RuntimeError(f"Worker thread failed: {error_holder[0]}") from error_holder[0]

        loader_t.join(timeout=30.0)
        prep_thread.join(timeout=30.0)
        gpu_thread.join(timeout=30.0)
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - t_start
    logger.info("Cache inference complete: %d predictions in %.1fs", total_preds, elapsed)
    return total_preds if streaming else predictions


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_fold_cache_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DNN fold inference from pre-computed tensor cache (no CRAM access needed)",
    )
    ap.add_argument("--tensor-cache", required=True, help="Directory with shard_*.pt files")
    ap.add_argument("--fold-metadata", required=True, help="Path to this fold's srsnv_dnn_metadata.json")
    ap.add_argument("--output", required=True, help="Output predictions parquet path")
    ap.add_argument("--backend", default="trt", choices=["trt", "pytorch"], help="Inference backend")
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU IDs (default: all visible)")
    ap.add_argument("--batch-size", type=int, default=512)
    return ap.parse_args(argv)


def run_fold_from_cache(argv: list[str] | None = None) -> None:
    """Run inference for a single fold from a pre-computed tensor cache."""
    torch_mp = importlib.import_module("torch.multiprocessing")

    torch_mp.set_sharing_strategy("file_system")

    if argv is None:
        argv = sys.argv[1:]
    args = _parse_fold_cache_args(argv)

    t0 = time.perf_counter()

    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    if gpu_ids is None:
        _torch = importlib.import_module("torch")

        gpu_ids = list(range(_torch.cuda.device_count())) if _torch.cuda.is_available() else [0]

    engines = _create_engines(args.fold_metadata, backend=args.backend, gpu_ids=gpu_ids)
    logger.info("Loaded %d %s engine(s)", len(engines), args.backend)

    try:
        n_preds = _run_fold_inference_from_cache(
            cache_dir=args.tensor_cache,
            engines=engines,
            batch_size=args.batch_size,
            output_path=args.output,
        )
    finally:
        for eng in engines:
            eng.close()

    elapsed = time.perf_counter() - t0
    logger.info("Fold from cache: wrote %d predictions to %s in %.1fs", n_preds, args.output, elapsed)


def main_fold_from_cache() -> None:
    run_fold_from_cache()


if __name__ == "__main__":
    main_fold_from_cache()
