from __future__ import annotations

import argparse
import copy
import functools
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields

from ugbio_srsnv.deep_srsnv.bam_schema import discover_bam_schema
from ugbio_srsnv.deep_srsnv.cnn_model import CNNReadClassifier
from ugbio_srsnv.deep_srsnv.data_prep import (
    build_encoders_from_schema,
    build_tensor_cache,
    load_full_tensor_cache,
)
from ugbio_srsnv.split_manifest import (
    SPLIT_MODE_SINGLE_MODEL_READ_HASH,
    build_single_model_read_hash_manifest,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
    validate_manifest_against_regions,
)
from ugbio_srsnv.srsnv_utils import MAX_PHRED, prob_to_phred

CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train BAM-native CNN for SRSNV")
    ap.add_argument("--positive-bam", required=True)
    ap.add_argument("--negative-bam", required=True)
    ap.add_argument("--positive-parquet", required=True)
    ap.add_argument("--negative-parquet", required=True)
    ap.add_argument("--training-regions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--basename", default="")
    ap.add_argument("--k-folds", type=int, default=3)
    ap.add_argument("--split-manifest-in", default=None)
    ap.add_argument("--split-manifest-out", default=None)
    ap.add_argument("--holdout-chromosomes", default="chr21,chr22")
    ap.add_argument("--single-model-split", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--split-hash-key", default="RN")
    ap.add_argument("--max-rows-per-class", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience (epochs without val AUC improvement)"
    )
    ap.add_argument("--min-epochs", type=int, default=3, help="Minimum epochs before early stopping can trigger")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--predict-batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--random-seed", type=int, default=1)
    ap.add_argument("--length", type=int, default=300)
    ap.add_argument("--preprocess-cache-dir", default=None)
    ap.add_argument("--preprocess-num-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 2, 16)))
    ap.add_argument("--preprocess-max-ram-gb", type=float, default=48.0)
    ap.add_argument("--preprocess-batch-rows", type=int, default=25000)
    ap.add_argument("--loader-num-workers", type=int, default=max(1, min((os.cpu_count() or 4) // 2, 8)))
    ap.add_argument("--loader-prefetch-factor", type=int, default=4)
    ap.add_argument("--loader-pin-memory", action="store_true")
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--use-tf32", action="store_true")
    ap.add_argument("--autotune-batch-size", action="store_true")
    ap.add_argument("--gpu-telemetry-interval-steps", type=int, default=500)
    ap.add_argument("--preprocess-dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _parse_holdout(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return list(dict.fromkeys(items)) if items else None


def _make_out_base(out_dir: Path, basename: str) -> str:
    if basename and not basename.endswith("."):
        basename = basename + "."
    return basename


def _evaluate_loader(
    model: CNNReadClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                read_base_idx=batch["read_base_idx"].to(device),
                ref_base_idx=batch["ref_base_idx"].to(device),
                t0_idx=batch["t0_idx"].to(device),
                x_num=batch["x_num"].to(device),
                mask=batch["mask"].to(device),
            )
            if criterion is not None:
                labels_t = batch["label"].to(device)
                batch_loss = criterion(logits, labels_t)
                rows = int(labels_t.shape[0])
                total_loss += float(batch_loss.item()) * rows
                total_rows += rows
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
    if not all_probs:
        return np.array([]), np.array([]), None
    avg_loss = (total_loss / total_rows) if (criterion is not None and total_rows > 0) else None
    return np.concatenate(all_probs), np.concatenate(all_labels), avg_loss


def _safe_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {}
    if len(y_true) == 0:
        return {"auc": None, "aupr": None, "logloss": None}
    if len(np.unique(y_true)) >= 2:  # noqa: PLR2004
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["aupr"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auc"] = None
        metrics["aupr"] = None
    metrics["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    return metrics


def _split_name(split_id: int) -> str:
    if split_id == 0:
        return "train"
    if split_id == 1:
        return "val"
    if split_id == -1:
        return "test"
    return f"fold_{split_id}"


def _normalize_split_counts(raw_counts: dict | None) -> dict[str, dict]:
    if not raw_counts:
        return {}
    out: dict[str, dict] = {}
    for split_id_raw, values in raw_counts.items():
        sid = int(split_id_raw)
        rows = int(values.get("rows", 0))
        positives = int(values.get("positives", 0))
        negatives = int(values.get("negatives", 0))
        prevalence = (positives / rows) if rows > 0 else None
        out[_split_name(sid)] = {
            "split_id": sid,
            "rows": rows,
            "positives": positives,
            "negatives": negatives,
            "prevalence": prevalence,
        }
    return out


def _summarize_chunk_prevalence(chunk_split_stats: list[dict] | None, split_id: int) -> dict | None:
    if not chunk_split_stats:
        return None
    values: list[float] = []
    sid_key = str(split_id)
    for entry in chunk_split_stats:
        split_stats = entry.get("split_stats", {})
        stats = split_stats.get(sid_key)
        if not stats:
            continue
        prevalence = stats.get("prevalence")
        if prevalence is None:
            continue
        values.append(float(prevalence))
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_chunks": int(arr.shape[0]),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "near_pure_count": int(np.sum((arr < 0.01) | (arr > 0.99))),  # noqa: PLR2004
    }


class TensorMapDataset(Dataset):
    """Map-style dataset backed by a shared compact cache via index indirection.

    All instances reference the same underlying tensors in `full_cache`;
    each split only stores a small int32 index array (~5-22 MB).
    Returns raw compact tensors from __getitem__; dtype casting and x_num
    expansion happen once per batch in _compact_collate_fn.
    """

    def __init__(
        self,
        full_cache: dict,
        split_id_keep: set[int],
        *,
        include_meta: bool = False,
    ):
        self._cache = full_cache
        self.include_meta = include_meta

        split_ids = torch.tensor(sorted(split_id_keep), dtype=torch.long)
        keep_mask = torch.isin(full_cache["split_id"], split_ids)
        self._idx = torch.nonzero(keep_mask, as_tuple=False).flatten().to(dtype=torch.int32)

    def __len__(self) -> int:
        return int(self._idx.shape[0])

    def __getitem__(self, idx: int) -> tuple:
        gi = int(self._idx[idx])
        if self.include_meta:
            return (gi, self._cache["chrom"][gi], self._cache["rn"][gi])
        return (gi,)


def _compact_collate_fn(batch: list[tuple], cache: dict, *, include_meta: bool) -> dict:
    """Batch-level collate: index into compact cache and cast once per batch."""
    if include_meta:
        gis, chroms, rns = zip(*batch, strict=False)
    else:
        gis = [b[0] for b in batch]
    idx = torch.tensor(gis, dtype=torch.long)

    x_pos = cache["x_num_pos"][idx].to(dtype=torch.float32)
    x_const = cache["x_num_const"][idx].to(dtype=torch.float32)
    x_num = torch.cat([x_pos, x_const.unsqueeze(-1).expand(-1, -1, x_pos.shape[-1])], dim=1)

    result = {
        "read_base_idx": cache["read_base_idx"][idx].to(dtype=torch.long),
        "ref_base_idx": cache["ref_base_idx"][idx].to(dtype=torch.long),
        "t0_idx": cache["t0_idx"][idx].to(dtype=torch.long),
        "x_num": x_num,
        "mask": cache["mask"][idx].to(dtype=torch.float32),
        "label": cache["label"][idx].to(dtype=torch.float32),
        "fold_id": cache["split_id"][idx].to(dtype=torch.long),
    }
    if include_meta:
        result["chrom"] = list(chroms)
        result["pos"] = torch.from_numpy(cache["pos"][torch.tensor(gis, dtype=torch.long).numpy()]).to(dtype=torch.long)
        result["rn"] = list(rns)
    return result


def _build_loader(
    dataset: TensorMapDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    include_meta = dataset.include_meta
    cache = dataset._cache
    collate = functools.partial(_compact_collate_fn, cache=cache, include_meta=include_meta)
    kwargs: dict = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": 0,
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "collate_fn": collate,
    }
    return DataLoader(dataset, **kwargs)


def _gpu_snapshot(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"gpu_util": None, "gpu_mem_mib": None}
    try:
        mem = float(torch.cuda.memory_allocated(device) / (1024 * 1024))
    except Exception:
        mem = None
    return {"gpu_util": None, "gpu_mem_mib": mem}


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = _cli()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _make_out_base(out_dir, args.basename)
    logger.info("deep_srsnv run started: output=%s basename=%s", out_dir, base or "<none>")

    # Split manifest parity with XGBoost trainer.
    requested_single_model = bool(args.single_model_split)
    if args.split_manifest_in:
        split_manifest = load_split_manifest(args.split_manifest_in)
        validate_manifest_against_regions(split_manifest, args.training_regions)
    else:
        holdout_chromosomes = _parse_holdout(args.holdout_chromosomes) or ["chr21", "chr22"]
        if requested_single_model:
            split_manifest = build_single_model_read_hash_manifest(
                training_regions=args.training_regions,
                random_seed=args.random_seed,
                holdout_chromosomes=holdout_chromosomes,
                val_fraction=args.val_fraction,
                hash_key=args.split_hash_key,
            )
        else:
            split_manifest = build_split_manifest(
                training_regions=args.training_regions,
                k_folds=args.k_folds,
                random_seed=args.random_seed,
                holdout_chromosomes=holdout_chromosomes,
            )
        if args.split_manifest_out:
            save_split_manifest(split_manifest, args.split_manifest_out)

    single_model_split = split_manifest.get("split_mode") == SPLIT_MODE_SINGLE_MODEL_READ_HASH
    chrom_to_fold = {} if single_model_split else {k: int(v) for k, v in split_manifest["chrom_to_fold"].items()}
    logger.info(
        "Split manifest ready: mode=%s k_folds=%s holdout_chromosomes=%s",
        split_manifest.get("split_mode", "chromosome_kfold"),
        split_manifest.get("k_folds", 1),
        ",".join(split_manifest.get("test_chromosomes", [])),
    )

    # Discover BAM schema + build encoders before preprocessing.
    schema_t0 = time.perf_counter()
    schema = discover_bam_schema([args.positive_bam, args.negative_bam], sample_reads_per_bam=20000)
    schema_path = out_dir / f"{base}feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    logger.info("BAM schema discovered and saved in %.1fs: %s", time.perf_counter() - schema_t0, schema_path)

    encoders = build_encoders_from_schema(schema)
    logger.info(
        "Encoders built: base=%d t0=%d tm=%d st=%d et=%d",
        len(encoders.base_vocab),
        len(encoders.t0_vocab),
        len(encoders.tm_vocab),
        len(encoders.st_vocab),
        len(encoders.et_vocab),
    )
    numeric_channels = 12

    preprocess_t0 = time.perf_counter()
    preprocess_cache_dir = args.preprocess_cache_dir or str(out_dir / "deep_srsnv_cache")
    logger.info("Preprocess phase started: cache_dir=%s", preprocess_cache_dir)
    preprocess_index = build_tensor_cache(
        positive_parquet=args.positive_parquet,
        negative_parquet=args.negative_parquet,
        positive_bam=args.positive_bam,
        negative_bam=args.negative_bam,
        chrom_to_fold=chrom_to_fold,
        split_manifest=split_manifest,
        encoders=encoders,
        cache_dir=preprocess_cache_dir,
        tensor_length=args.length,
        max_rows_per_class=args.max_rows_per_class,
        preprocess_num_workers=args.preprocess_num_workers,
        preprocess_max_ram_gb=args.preprocess_max_ram_gb,
        preprocess_batch_rows=args.preprocess_batch_rows,
        preprocess_dry_run=args.preprocess_dry_run,
    )
    preprocess_wall_seconds = round(time.perf_counter() - preprocess_t0, 3)
    logger.info(
        "Preprocess phase finished in %.1fs (cache_hit=%s, shards=%d, rows=%d)",
        preprocess_wall_seconds,
        bool(preprocess_index.get("cache_hit", False)),
        int(preprocess_index.get("total_shards", 0)),
        int(preprocess_index.get("total_output_rows", 0)),
    )
    if args.preprocess_dry_run:
        metadata_path = out_dir / f"{base}srsnv_dnn_metadata.json"
        metadata_path.write_text(json.dumps({"preprocess": preprocess_index}, indent=2))
        logger.info("Preprocess dry-run written to %s", metadata_path)
        return

    tensor_cache_path = preprocess_index.get("tensor_cache_path")
    if not tensor_cache_path:
        raise ValueError("No tensor cache path was produced by preprocessing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.use_amp))

    split_prevalence = _normalize_split_counts(preprocess_index.get("split_counts"))
    if split_prevalence:
        logger.info("Split stats (pre-training):")
        for split_name in ["train", "val", "test"]:
            stats = split_prevalence.get(split_name)
            if not stats:
                continue
            logger.info(
                "  %s: rows=%d positives=%d negatives=%d prevalence=%s",
                split_name,
                stats["rows"],
                stats["positives"],
                stats["negatives"],
                "n/a" if stats["prevalence"] is None else f"{stats['prevalence']:.6f}",
            )
    train_chunk_mix = _summarize_chunk_prevalence(preprocess_index.get("chunk_split_stats"), split_id=0)
    if train_chunk_mix:
        logger.info(
            "Train chunk prevalence (pre-training): chunks=%d min=%.4f median=%.4f max=%.4f near_pure_chunks=%d",
            train_chunk_mix["n_chunks"],
            train_chunk_mix["min"],
            train_chunk_mix["median"],
            train_chunk_mix["max"],
            train_chunk_mix["near_pure_count"],
        )
        if int(train_chunk_mix["near_pure_count"]) > 0:
            logger.warning(
                "Detected near-pure train chunks (<1%% or >99%% positives). "
                "This can destabilize optimization and hurt validation."
            )

    tuned_batch_size = int(args.batch_size)
    if args.autotune_batch_size and device.type == "cuda":
        tuned_batch_size = min(max(int(args.batch_size), 128), 512)
        logger.info("Autotune batch size enabled, selected=%d", tuned_batch_size)
    eval_batch_size_arg = getattr(args, "eval_batch_size", None)
    predict_batch_size_arg = getattr(args, "predict_batch_size", None)
    eval_batch_size = int(eval_batch_size_arg) if eval_batch_size_arg else int(tuned_batch_size * 2)
    predict_batch_size = int(predict_batch_size_arg) if predict_batch_size_arg else int(eval_batch_size * 2)
    eval_batch_size = max(1, eval_batch_size)
    predict_batch_size = max(1, predict_batch_size)

    full_cache = load_full_tensor_cache(tensor_cache_path)

    pred_dataset = TensorMapDataset(
        full_cache=full_cache,
        split_id_keep={-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        include_meta=True,
    )

    pred_loader = _build_loader(
        pred_dataset,
        predict_batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        prefetch_factor=args.loader_prefetch_factor,
        pin_memory=args.loader_pin_memory,
    )
    logger.info(
        "Training deep_srsnv CNN on device=%s with %d shards (%d records), "
        "train_batch=%d eval_batch=%d predict_batch=%d workers=%d amp=%s",
        device,
        int(preprocess_index.get("total_shards", 0)),
        int(preprocess_index.get("total_output_rows", 0)),
        tuned_batch_size,
        eval_batch_size,
        predict_batch_size,
        args.loader_num_workers,
        bool(args.use_amp),
    )

    models: list[CNNReadClassifier] = []
    training_results = []
    training_runtime_metrics: list[dict] = []
    model_arch_summary: dict | None = None
    n_models = 1 if single_model_split else args.k_folds

    for fold_idx in range(n_models):
        fold_t0 = time.perf_counter()
        logger.info("Fold %d/%d started", fold_idx + 1, n_models)
        if single_model_split:
            train_keep = {0}
            val_keep = {1}
        else:
            train_keep = {i for i in range(args.k_folds) if i != fold_idx}
            val_keep = {fold_idx}
        train_loader = _build_loader(
            TensorMapDataset(full_cache=full_cache, split_id_keep=train_keep),
            tuned_batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            prefetch_factor=args.loader_prefetch_factor,
            pin_memory=args.loader_pin_memory,
        )
        val_loader = _build_loader(
            TensorMapDataset(full_cache=full_cache, split_id_keep=val_keep),
            eval_batch_size,
            shuffle=False,
            num_workers=args.loader_num_workers,
            prefetch_factor=args.loader_prefetch_factor,
            pin_memory=args.loader_pin_memory,
        )
        model = CNNReadClassifier(
            base_vocab_size=len(encoders.base_vocab),
            t0_vocab_size=len(encoders.t0_vocab),
            numeric_channels=numeric_channels,
        ).to(device)
        if fold_idx == 0:
            n_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            model_arch_summary = {
                "class_name": model.__class__.__name__,
                "trainable_parameters": n_trainable_params,
                "structure": str(model),
            }
            logger.info("Model architecture: %s", model.__class__.__name__)
            logger.info("Model trainable parameters: %d", n_trainable_params)
            logger.info("Model structure:\n%s", model)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_val_auc = -1.0
        best_epoch = 0
        best_state_dict = None
        epochs_without_improvement = 0
        stopped_early = False

        for _epoch in range(args.epochs):
            epoch_num = _epoch + 1
            epoch_t0 = time.perf_counter()
            model.train()
            epoch_batches = 0
            epoch_rows = 0
            epoch_loss = 0.0
            wait_time_s = 0.0
            compute_time_s = 0.0
            train_probs_parts: list[np.ndarray] = []
            train_labels_parts: list[np.ndarray] = []
            batch_iter = iter(train_loader)
            step = 0
            while True:
                t_wait = time.perf_counter()
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                wait_time_s += time.perf_counter() - t_wait

                t_compute = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.use_amp)):
                    logits = model(
                        read_base_idx=batch["read_base_idx"].to(device, non_blocking=True),
                        ref_base_idx=batch["ref_base_idx"].to(device, non_blocking=True),
                        t0_idx=batch["t0_idx"].to(device, non_blocking=True),
                        x_num=batch["x_num"].to(device, non_blocking=True),
                        mask=batch["mask"].to(device, non_blocking=True),
                    )
                    loss = criterion(logits, batch["label"].to(device, non_blocking=True))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                train_probs_parts.append(torch.sigmoid(logits).detach().cpu().numpy())
                train_labels_parts.append(batch["label"].detach().cpu().numpy())

                compute_time_s += time.perf_counter() - t_compute
                epoch_batches += 1
                epoch_rows += int(batch["label"].shape[0])
                epoch_loss += float(loss.item())
                step += 1
                if step == 1 or step % max(1, args.gpu_telemetry_interval_steps) == 0:
                    snap = _gpu_snapshot(device)
                    logger.info(
                        "Fold %d epoch %d step %d: rows=%d loss=%.5f gpu_mem_mib=%s",
                        fold_idx,
                        epoch_num,
                        step,
                        epoch_rows,
                        float(loss.item()),
                        "n/a" if snap["gpu_mem_mib"] is None else f"{snap['gpu_mem_mib']:.1f}",
                    )
            avg_loss = (epoch_loss / epoch_batches) if epoch_batches else None
            train_probs = np.concatenate(train_probs_parts) if train_probs_parts else np.array([])
            train_labels = np.concatenate(train_labels_parts) if train_labels_parts else np.array([])
            train_metrics = _safe_binary_metrics(train_labels, train_probs)
            val_probs, val_labels, val_loss = _evaluate_loader(model, val_loader, device, criterion=criterion)
            val_metrics = _safe_binary_metrics(val_labels, val_probs)
            logger.info(
                "Fold %d epoch %d finished in %.1fs (batches=%d rows=%d "
                "train_loss=%s train_auc=%s train_aupr=%s train_logloss=%s "
                "val_loss=%s val_auc=%s val_aupr=%s val_logloss=%s "
                "auc_gap=%s aupr_gap=%s wait_s=%.1f compute_s=%.1f samples_per_s=%.1f)",
                fold_idx,
                epoch_num,
                time.perf_counter() - epoch_t0,
                epoch_batches,
                epoch_rows,
                "n/a" if avg_loss is None else f"{avg_loss:.5f}",
                "n/a" if train_metrics["auc"] is None else f"{train_metrics['auc']:.6f}",
                "n/a" if train_metrics["aupr"] is None else f"{train_metrics['aupr']:.6f}",
                "n/a" if train_metrics["logloss"] is None else f"{train_metrics['logloss']:.6f}",
                "n/a" if val_loss is None else f"{val_loss:.5f}",
                "n/a" if val_metrics["auc"] is None else f"{val_metrics['auc']:.6f}",
                "n/a" if val_metrics["aupr"] is None else f"{val_metrics['aupr']:.6f}",
                "n/a" if val_metrics["logloss"] is None else f"{val_metrics['logloss']:.6f}",
                (
                    "n/a"
                    if (train_metrics["auc"] is None or val_metrics["auc"] is None)
                    else f"{(train_metrics['auc'] - val_metrics['auc']):.6f}"
                ),
                (
                    "n/a"
                    if (train_metrics["aupr"] is None or val_metrics["aupr"] is None)
                    else f"{(train_metrics['aupr'] - val_metrics['aupr']):.6f}"
                ),
                wait_time_s,
                compute_time_s,
                (epoch_rows / max(1e-6, (time.perf_counter() - epoch_t0))),
            )
            training_runtime_metrics.append(
                {
                    "fold": fold_idx,
                    "epoch": epoch_num,
                    "batches": int(epoch_batches),
                    "rows": int(epoch_rows),
                    "train_loss": None if avg_loss is None else float(avg_loss),
                    "train_auc": train_metrics["auc"],
                    "train_aupr": train_metrics["aupr"],
                    "train_logloss": train_metrics["logloss"],
                    "val_loss": None if val_loss is None else float(val_loss),
                    "val_auc": val_metrics["auc"],
                    "val_aupr": val_metrics["aupr"],
                    "val_logloss": val_metrics["logloss"],
                    "train_val_auc_gap": None
                    if (train_metrics["auc"] is None or val_metrics["auc"] is None)
                    else float(train_metrics["auc"] - val_metrics["auc"]),
                    "train_val_aupr_gap": None
                    if (train_metrics["aupr"] is None or val_metrics["aupr"] is None)
                    else float(train_metrics["aupr"] - val_metrics["aupr"]),
                    "wait_seconds": float(wait_time_s),
                    "compute_seconds": float(compute_time_s),
                    "samples_per_second": float(epoch_rows / max(1e-6, (time.perf_counter() - epoch_t0))),
                }
            )

            current_val_auc = val_metrics["auc"]
            if current_val_auc is not None and current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                best_epoch = epoch_num
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            logger.info(
                "Fold %d epoch %d early-stop status: best_epoch=%d best_val_auc=%.6f patience=%d/%d",
                fold_idx,
                epoch_num,
                best_epoch,
                best_val_auc,
                epochs_without_improvement,
                args.patience,
            )

            if epochs_without_improvement >= args.patience and epoch_num >= args.min_epochs:
                logger.info(
                    "Fold %d early stopping triggered at epoch %d (best_epoch=%d best_val_auc=%.6f)",
                    fold_idx,
                    epoch_num,
                    best_epoch,
                    best_val_auc,
                )
                stopped_early = True
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logger.info(
                "Fold %d restored best model weights from epoch %d (val_auc=%.6f)", fold_idx, best_epoch, best_val_auc
            )

        fold_epoch_metrics = [m for m in training_runtime_metrics if m["fold"] == fold_idx]
        last_epoch_metrics = fold_epoch_metrics[-1] if fold_epoch_metrics else {}
        best_epoch_metrics = next((m for m in fold_epoch_metrics if m["epoch"] == best_epoch), last_epoch_metrics)
        training_results.append(
            {
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "stopped_early": stopped_early,
                "total_epochs": len(fold_epoch_metrics),
                "best_val_auc": best_val_auc if best_val_auc > -1.0 else None,
                "val_auc": best_epoch_metrics.get("val_auc"),
                "val_aupr": best_epoch_metrics.get("val_aupr"),
                "val_logloss": best_epoch_metrics.get("val_logloss"),
            }
        )
        logger.info(
            "Fold %d complete in %.1fs (best_epoch=%d/%d stopped_early=%s best_val_auc=%s val_aupr=%s val_logloss=%s)",
            fold_idx,
            time.perf_counter() - fold_t0,
            best_epoch,
            len(fold_epoch_metrics),
            stopped_early,
            "n/a" if best_epoch_metrics.get("val_auc") is None else f"{best_epoch_metrics['val_auc']:.6f}",
            "n/a" if best_epoch_metrics.get("val_aupr") is None else f"{best_epoch_metrics['val_aupr']:.6f}",
            "n/a" if best_epoch_metrics.get("val_logloss") is None else f"{best_epoch_metrics['val_logloss']:.6f}",
        )

        model_path = out_dir / f"{base}dnn_model_fold_{fold_idx}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info("Saved fold model: %s", model_path)
        models.append(model)

    logger.info("Prediction/export phase started")
    for model in models:
        model.eval()
    chrom_out: list[str] = []
    pos_out: list[int] = []
    rn_out: list[str] = []
    labels_out: list[int] = []
    fold_ids_out: list[int] = []
    probs_out: list[float] = []

    pred_steps = 0
    for batch in pred_loader:
        with torch.no_grad():
            per_model_probs = [
                torch.sigmoid(
                    m(
                        read_base_idx=batch["read_base_idx"].to(device, non_blocking=True),
                        ref_base_idx=batch["ref_base_idx"].to(device, non_blocking=True),
                        t0_idx=batch["t0_idx"].to(device, non_blocking=True),
                        x_num=batch["x_num"].to(device, non_blocking=True),
                        mask=batch["mask"].to(device, non_blocking=True),
                    )
                )
                .detach()
                .cpu()
                .numpy()
                for m in models
            ]
        batch_fold_ids = batch["fold_id"].cpu().numpy().astype(int)
        batch_labels = batch["label"].cpu().numpy().astype(int)
        batch_pos = batch["pos"].cpu().numpy().astype(int)
        batch_chrom = list(batch["chrom"])
        batch_rn = list(batch["rn"])
        for j, fold_id in enumerate(batch_fold_ids):
            if single_model_split:
                prob = float(per_model_probs[0][j])
            elif fold_id == -1:
                prob = float(np.mean([p[j] for p in per_model_probs]))
            else:
                prob = float(per_model_probs[fold_id][j])
            probs_out.append(prob)
            fold_ids_out.append(int(fold_id))
            labels_out.append(int(batch_labels[j]))
            chrom_out.append(str(batch_chrom[j]))
            pos_out.append(int(batch_pos[j]))
            rn_out.append(str(batch_rn[j]))
        pred_steps += 1
        if pred_steps == 1 or pred_steps % max(1, args.gpu_telemetry_interval_steps) == 0:
            logger.info("Prediction progress: steps=%d rows=%d", pred_steps, len(labels_out))

    probs = np.asarray(probs_out, dtype=np.float64)
    labels = np.asarray(labels_out, dtype=int)
    fold_ids = np.asarray(fold_ids_out, dtype=int)

    mqual = prob_to_phred(probs, max_value=MAX_PHRED)
    snvq = mqual.copy()

    df_out = pl.DataFrame(
        {
            CHROM: chrom_out,
            POS: pos_out,
            "RN": rn_out,
            "label": labels,
            "fold_id": fold_ids,
            "prob_orig": probs,
            "MQUAL": mqual,
            "SNVQ": snvq,
        }
    )
    df_path = out_dir / f"{base}featuremap_df.parquet"
    df_out.write_parquet(df_path)
    logger.info("Saved prediction dataframe: %s", df_path)

    # Holdout metrics (chr21/22 via fold=-1).
    holdout_mask = fold_ids == -1
    holdout_metrics = {}
    if holdout_mask.any():
        holdout_metrics = _safe_binary_metrics(labels[holdout_mask], probs[holdout_mask])

    metadata = {
        "model_type": "deep_srsnv_cnn",
        "split_manifest": split_manifest,
        "encoders": {
            "base_vocab": encoders.base_vocab,
            "ref_base_vocab": encoders.base_vocab,
            "t0_vocab": encoders.t0_vocab,
            "tm_vocab": encoders.tm_vocab,
            "st_vocab": encoders.st_vocab,
            "et_vocab": encoders.et_vocab,
        },
        "channel_order": [
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
        "training_results": training_results,
        "training_runtime_metrics": training_runtime_metrics,
        "holdout_metrics": holdout_metrics,
        "split_prevalence": split_prevalence,
        "chunk_composition": {
            "train_chunk_prevalence": train_chunk_mix,
        },
        "model_architecture": model_arch_summary,
        "schema_path": str(schema_path),
        "data_paths": {
            "positive_bam": args.positive_bam,
            "negative_bam": args.negative_bam,
            "positive_parquet": args.positive_parquet,
            "negative_parquet": args.negative_parquet,
        },
        "training_parameters": {
            "k_folds": n_models,
            "epochs": args.epochs,
            "patience": args.patience,
            "min_epochs": args.min_epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": eval_batch_size_arg,
            "predict_batch_size": predict_batch_size_arg,
            "learning_rate": args.learning_rate,
            "length": args.length,
            "max_rows_per_class": args.max_rows_per_class,
            "loader_num_workers": args.loader_num_workers,
            "loader_prefetch_factor": args.loader_prefetch_factor,
            "loader_pin_memory": bool(args.loader_pin_memory),
            "use_amp": bool(args.use_amp),
            "use_tf32": bool(args.use_tf32),
            "autotune_batch_size": bool(args.autotune_batch_size),
            "effective_train_batch_size": int(tuned_batch_size),
            "effective_eval_batch_size": int(eval_batch_size),
            "effective_predict_batch_size": int(predict_batch_size),
        },
        "split_summary": {
            "split_mode": split_manifest.get("split_mode", "chromosome_kfold"),
            "n_train": int(np.sum(fold_ids == 0)),
            "n_val": int(np.sum(fold_ids == 1)) if single_model_split else None,
            "n_test": int(np.sum(fold_ids == -1)),
            "val_fraction": split_manifest.get("val_fraction"),
            "hash_key": split_manifest.get("hash_key"),
        },
        "preprocess": {
            "cache_dir": preprocess_cache_dir,
            "wall_seconds": preprocess_wall_seconds,
            **preprocess_index,
        },
    }
    metadata_path = out_dir / f"{base}srsnv_dnn_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved metadata: %s", metadata_path)
    logger.info("Saved deep_srsnv outputs to %s", out_dir)


if __name__ == "__main__":
    main()
