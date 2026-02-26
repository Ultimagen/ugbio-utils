from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import lightning
import numpy as np
import polars as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_featuremap.filter_dataframe import read_filtering_stats_json

from ugbio_srsnv.deep_srsnv.bam_schema import discover_bam_schema
from ugbio_srsnv.deep_srsnv.data_module import SRSNVDataModule
from ugbio_srsnv.deep_srsnv.data_prep import (
    build_encoders_from_schema,
    build_tensor_cache,
    load_full_tensor_cache,
)
from ugbio_srsnv.deep_srsnv.lightning_module import LR_SCHEDULER_CHOICES, SRSNVLightningModule
from ugbio_srsnv.split_manifest import (
    SPLIT_MODE_SINGLE_MODEL_READ_HASH,
    build_single_model_read_hash_manifest,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
    validate_manifest_against_regions,
)
from ugbio_srsnv.srsnv_utils import MAX_PHRED, prob_to_phred, recalibrate_snvq

CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value


def _cli() -> argparse.Namespace:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description="Train BAM-native CNN for SRSNV (Lightning)")

    # Data / IO
    ap.add_argument("--positive-bam", required=True)
    ap.add_argument("--negative-bam", required=True)
    ap.add_argument("--positive-parquet", required=True)
    ap.add_argument("--negative-parquet", required=True)
    ap.add_argument("--training-regions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--basename", default="")

    # SNVQ recalibration (optional; when all four are supplied, MQUAL→SNVQ
    # recalibration is applied, matching the XGBoost pipeline)
    ap.add_argument("--stats-positive", default=None, help="Path to positive filtering-stats JSON")
    ap.add_argument("--stats-negative", default=None, help="Path to negative filtering-stats JSON")
    ap.add_argument("--stats-featuremap", default=None, help="Path to raw featuremap filtering-stats JSON")
    ap.add_argument("--mean-coverage", type=float, default=None, help="Mean sequencing coverage")

    # Split / CV
    ap.add_argument("--k-folds", type=int, default=3)
    ap.add_argument("--split-manifest-in", default=None)
    ap.add_argument("--split-manifest-out", default=None)
    ap.add_argument("--holdout-chromosomes", default="chr21,chr22")
    ap.add_argument("--single-model-split", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--split-hash-key", default="RN")
    ap.add_argument("--max-rows-per-class", type=int, default=None)

    # Training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience (epochs without val AUC improvement)"
    )
    ap.add_argument("--min-epochs", type=int, default=3, help="Minimum epochs before early stopping can trigger")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--predict-batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    ap.add_argument("--random-seed", type=int, default=1)
    ap.add_argument("--length", type=int, default=300)

    # Architecture
    ap.add_argument("--hidden-channels", type=int, default=128, help="Hidden channels in CNN residual blocks")
    ap.add_argument("--n-blocks", type=int, default=6, help="Number of residual blocks")
    ap.add_argument("--base-embed-dim", type=int, default=16, help="Read/ref base embedding dimension")
    ap.add_argument("--t0-embed-dim", type=int, default=16, help="T0 token embedding dimension")
    ap.add_argument("--cat-embed-dim", type=int, default=4, help="Categorical (tm/st/et) embedding dimension")
    ap.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in classification head")

    # LR scheduler
    ap.add_argument("--lr-scheduler", choices=LR_SCHEDULER_CHOICES, default="onecycle")
    ap.add_argument("--lr-warmup-epochs", type=int, default=1)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--lr-step-size", type=int, default=5)
    ap.add_argument("--lr-gamma", type=float, default=0.5)
    ap.add_argument("--lr-patience", type=int, default=3)

    # SWA
    ap.add_argument("--swa", action="store_true", help="Enable Stochastic Weight Averaging")
    ap.add_argument("--swa-lr", type=float, default=1e-4)
    ap.add_argument(
        "--swa-epoch-start", type=float, default=0.7, help="Fraction of epochs or absolute epoch to start SWA"
    )

    # Tuner
    ap.add_argument("--auto-lr-find", action="store_true", help="Run Lightning LR finder before training")
    ap.add_argument(
        "--auto-scale-batch-size", action="store_true", help="Run Lightning batch size finder before training"
    )

    # Hardware / precision
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--use-tf32", action="store_true")
    ap.add_argument("--gradient-clip-val", type=float, default=None)
    ap.add_argument("--accumulate-grad-batches", type=int, default=1)
    ap.add_argument("--devices", type=int, default=1, help="Number of GPUs (or 'auto')")
    ap.add_argument("--strategy", default="auto", help="Lightning strategy (auto, ddp, etc.)")

    # Preprocessing
    ap.add_argument("--preprocess-cache-dir", default=None)
    ap.add_argument("--preprocess-num-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 2, 16)))
    ap.add_argument("--preprocess-max-ram-gb", type=float, default=48.0)
    ap.add_argument("--preprocess-batch-rows", type=int, default=25000)
    ap.add_argument("--loader-num-workers", type=int, default=max(1, min((os.cpu_count() or 4) // 2, 8)))
    ap.add_argument("--loader-prefetch-factor", type=int, default=4)
    ap.add_argument("--loader-pin-memory", action="store_true")

    # Misc
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


def _build_callbacks(args: argparse.Namespace, out_dir: Path, base: str, fold_idx: int) -> list:
    callbacks = []

    callbacks.append(
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=args.patience,
            min_delta=0.0,
            verbose=True,
        )
    )

    callbacks.append(
        ModelCheckpoint(
            dirpath=out_dir,
            filename=f"{base}dnn_model_fold_{fold_idx}" + "-{epoch}-{val_auc:.4f}",
            monitor="val_auc",
            mode="max",
            save_top_k=1,
            save_last=False,
            verbose=True,
        )
    )

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if args.swa:
        swa_epoch_start_val = args.swa_epoch_start
        if 0 < swa_epoch_start_val < 1:
            swa_epoch_start_val = max(1, int(swa_epoch_start_val * args.epochs))
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=args.swa_lr,
                swa_epoch_start=int(swa_epoch_start_val),
            )
        )

    return callbacks


def _build_trainer(args: argparse.Namespace, out_dir: Path, base: str, fold_idx: int) -> lightning.Trainer:
    precision = "16-mixed" if args.use_amp else "32-true"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.devices if accelerator == "gpu" else 1
    strategy = args.strategy if devices > 1 else "auto"

    csv_logger = CSVLogger(save_dir=out_dir, name=f"{base}lightning_logs", version=f"fold_{fold_idx}")

    callbacks = _build_callbacks(args, out_dir, base, fold_idx)

    trainer = lightning.Trainer(
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=csv_logger,
        deterministic=False,
        enable_progress_bar=True,
        log_every_n_steps=50,
        default_root_dir=str(out_dir),
    )
    return trainer


def _extract_training_results(trainer: lightning.Trainer, fold_idx: int) -> dict:
    """Extract per-fold training results from the Trainer's callback metrics."""
    ckpt_callback = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_callback = cb
            break

    best_val_auc = float(ckpt_callback.best_model_score) if ckpt_callback and ckpt_callback.best_model_score else None
    stopped_early = trainer.early_stopping_callback is not None and trainer.early_stopping_callback.stopped_epoch > 0

    logged = trainer.logged_metrics
    return {
        "fold": fold_idx,
        "best_epoch": trainer.current_epoch + 1,
        "stopped_early": stopped_early,
        "total_epochs": trainer.current_epoch + 1,
        "best_val_auc": best_val_auc,
        "val_auc": float(logged.get("val_auc", 0)),
        "val_aupr": float(logged.get("val_aupr", 0)),
        "val_loss": float(logged.get("val_loss", 0)),
        "train_auc": float(logged.get("train_auc", 0)),
        "train_aupr": float(logged.get("train_aupr", 0)),
        "train_loss": float(logged.get("train_loss", 0)),
        "best_model_path": str(ckpt_callback.best_model_path) if ckpt_callback else None,
    }


def _collect_predictions(  # noqa: C901, PLR0912
    predictions: list[list[dict]], *, single_model_split: bool, n_models: int
) -> dict:
    """Aggregate predictions from predict_step outputs into arrays.

    For k-fold mode, each fold's model provides predictions for the whole
    dataset, and we pick the out-of-fold prediction for each sample.
    For single-model mode, we use the single model's predictions directly.
    """
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_fold_ids: list[np.ndarray] = []
    all_chroms: list[str] = []
    all_pos: list[np.ndarray] = []
    all_rns: list[str] = []

    if single_model_split or n_models == 1:
        fold_preds = predictions[0]
        for batch_out in fold_preds:
            all_probs.append(batch_out["probs"].cpu().numpy())
            all_labels.append(batch_out["label"].cpu().numpy().astype(int))
            all_fold_ids.append(batch_out["fold_id"].cpu().numpy().astype(int))
            if "chrom" in batch_out:
                all_chroms.extend(batch_out["chrom"])
            if "pos" in batch_out:
                all_pos.append(batch_out["pos"].cpu().numpy().astype(int))
            if "rn" in batch_out:
                all_rns.extend(batch_out["rn"])

        return {
            "probs": np.concatenate(all_probs),
            "labels": np.concatenate(all_labels),
            "fold_ids": np.concatenate(all_fold_ids),
            "chroms": all_chroms,
            "pos": np.concatenate(all_pos) if all_pos else np.array([]),
            "rns": all_rns,
        }

    n_samples = sum(b["probs"].shape[0] for b in predictions[0])
    per_model_probs = np.zeros((n_models, n_samples), dtype=np.float64)
    labels_arr = np.zeros(n_samples, dtype=int)
    fold_ids_arr = np.zeros(n_samples, dtype=int)
    chroms_list: list[str] = [""] * n_samples
    pos_arr = np.zeros(n_samples, dtype=int)
    rns_list: list[str] = [""] * n_samples

    for model_idx, fold_preds in enumerate(predictions):
        offset = 0
        for batch_out in fold_preds:
            bs = batch_out["probs"].shape[0]
            per_model_probs[model_idx, offset : offset + bs] = batch_out["probs"].cpu().numpy()
            if model_idx == 0:
                labels_arr[offset : offset + bs] = batch_out["label"].cpu().numpy().astype(int)
                fold_ids_arr[offset : offset + bs] = batch_out["fold_id"].cpu().numpy().astype(int)
                if "chrom" in batch_out:
                    chroms_list[offset : offset + bs] = batch_out["chrom"]
                if "pos" in batch_out:
                    pos_arr[offset : offset + bs] = batch_out["pos"].cpu().numpy().astype(int)
                if "rn" in batch_out:
                    rns_list[offset : offset + bs] = batch_out["rn"]
            offset += bs

    final_probs = np.zeros(n_samples, dtype=np.float64)
    for j in range(n_samples):
        fid = fold_ids_arr[j]
        if fid == -1:
            final_probs[j] = np.mean(per_model_probs[:, j])
        elif 0 <= fid < n_models:
            final_probs[j] = per_model_probs[fid, j]
        else:
            final_probs[j] = np.mean(per_model_probs[:, j])

    return {
        "probs": final_probs,
        "labels": labels_arr,
        "fold_ids": fold_ids_arr,
        "chroms": chroms_list,
        "pos": pos_arr,
        "rns": rns_list,
    }


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = _cli()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    lightning.seed_everything(args.random_seed, workers=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _make_out_base(out_dir, args.basename)
    logger.info("deep_srsnv Lightning run started: output=%s basename=%s", out_dir, base or "<none>")

    if args.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── Split manifest (same as original) ──
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

    # ── BAM schema + encoders ──
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
    numeric_channels = 9

    # ── Preprocessing (unchanged) ──
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

    # ── Log split prevalence ──
    split_prevalence = _normalize_split_counts(preprocess_index.get("split_counts"))
    if split_prevalence:
        logger.info("Split stats (pre-training):")
        for split_name_key in ["train", "val", "test"]:
            stats = split_prevalence.get(split_name_key)
            if not stats:
                continue
            logger.info(
                "  %s: rows=%d positives=%d negatives=%d prevalence=%s",
                split_name_key,
                stats["rows"],
                stats["positives"],
                stats["negatives"],
                "n/a" if stats["prevalence"] is None else f"{stats['prevalence']:.6f}",
            )
    train_chunk_mix = _summarize_chunk_prevalence(preprocess_index.get("chunk_split_stats"), split_id=0)
    if train_chunk_mix:
        logger.info(
            "Train chunk prevalence: chunks=%d min=%.4f median=%.4f max=%.4f near_pure_chunks=%d",
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

    # ── Load tensor cache ──
    full_cache = load_full_tensor_cache(tensor_cache_path)

    # ── Training loop (per fold) ──
    n_models = 1 if single_model_split else args.k_folds
    training_results: list[dict] = []
    all_predictions: list[list[dict]] = []
    model_arch_summary: dict | None = None
    best_ckpt_paths: list[str] = []

    for fold_idx in range(n_models):
        fold_t0 = time.perf_counter()
        logger.info("Fold %d/%d started", fold_idx + 1, n_models)

        if single_model_split:
            train_keep = {0}
            val_keep = {1}
        else:
            train_keep = {i for i in range(args.k_folds) if i != fold_idx}
            val_keep = {fold_idx}

        dm = SRSNVDataModule(
            full_cache=full_cache,
            train_split_ids=train_keep,
            val_split_ids=val_keep,
            test_split_ids={-1},
            train_batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            predict_batch_size=args.predict_batch_size,
            pin_memory=args.loader_pin_memory,
        )

        lit_model = SRSNVLightningModule(
            base_vocab_size=len(encoders.base_vocab),
            t0_vocab_size=len(encoders.t0_vocab),
            numeric_channels=numeric_channels,
            tm_vocab_size=len(encoders.tm_vocab),
            st_vocab_size=len(encoders.st_vocab),
            et_vocab_size=len(encoders.et_vocab),
            base_embed_dim=args.base_embed_dim,
            ref_embed_dim=args.base_embed_dim,
            t0_embed_dim=args.t0_embed_dim,
            cat_embed_dim=args.cat_embed_dim,
            hidden_channels=args.hidden_channels,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler=args.lr_scheduler,
            lr_warmup_epochs=args.lr_warmup_epochs,
            lr_min=args.lr_min,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            lr_patience=args.lr_patience,
        )

        if fold_idx == 0:
            n_trainable_params = int(sum(p.numel() for p in lit_model.model.parameters() if p.requires_grad))
            model_arch_summary = {
                "class_name": lit_model.model.__class__.__name__,
                "trainable_parameters": n_trainable_params,
                "structure": str(lit_model.model),
            }
            logger.info("Model architecture: %s", lit_model.model.__class__.__name__)
            logger.info("Model trainable parameters: %d", n_trainable_params)

        trainer = _build_trainer(args, out_dir, base, fold_idx)

        # ── Tuner: LR finder / batch size finder ──
        if args.auto_lr_find or args.auto_scale_batch_size:
            tuner = Tuner(trainer)
            if args.auto_scale_batch_size:
                tuner.scale_batch_size(lit_model, datamodule=dm, mode="power", max_trials=10)
                logger.info("Batch size finder result: %d", dm.train_batch_size)
            if args.auto_lr_find:
                lr_result = tuner.lr_find(lit_model, datamodule=dm)
                if lr_result and lr_result.suggestion():
                    suggested_lr = lr_result.suggestion()
                    lit_model.hparams.learning_rate = suggested_lr
                    logger.info("LR finder suggested: %e", suggested_lr)
                    fig = lr_result.plot(suggest=True)
                    fig.savefig(out_dir / f"{base}lr_finder_fold_{fold_idx}.png", dpi=100)
                    logger.info("LR finder plot saved to %s", out_dir / f"{base}lr_finder_fold_{fold_idx}.png")

        # ── Train ──
        trainer.fit(lit_model, datamodule=dm)
        fold_results = _extract_training_results(trainer, fold_idx)
        training_results.append(fold_results)
        best_ckpt_paths.append(fold_results.get("best_model_path", ""))

        logger.info(
            "Fold %d complete in %.1fs (best_val_auc=%s stopped_early=%s best_ckpt=%s)",
            fold_idx,
            time.perf_counter() - fold_t0,
            fold_results.get("best_val_auc"),
            fold_results.get("stopped_early"),
            fold_results.get("best_model_path"),
        )

        # ── Predict using best checkpoint ──
        best_path = fold_results.get("best_model_path")
        if best_path:
            fold_predictions = trainer.predict(lit_model, datamodule=dm, ckpt_path=best_path)
        else:
            fold_predictions = trainer.predict(lit_model, datamodule=dm)
        all_predictions.append(fold_predictions)

    # ── Aggregate predictions ──
    logger.info("Prediction/export phase started")
    collected = _collect_predictions(all_predictions, single_model_split=single_model_split, n_models=n_models)
    probs = collected["probs"]
    labels = collected["labels"]
    fold_ids = collected["fold_ids"]

    mqual = prob_to_phred(probs, max_value=MAX_PHRED)

    # ── SNVQ recalibration (when stats are provided) ──
    x_lut, y_lut = None, None
    has_recal_args = all(
        getattr(args, a, None) is not None
        for a in ("stats_positive", "stats_negative", "stats_featuremap", "mean_coverage")
    )
    if has_recal_args:
        from ugbio_srsnv.srsnv_training import count_bases_in_interval_list  # noqa: PLC0415

        pos_stats = read_filtering_stats_json(args.stats_positive)
        neg_stats = read_filtering_stats_json(args.stats_negative)
        raw_stats = read_filtering_stats_json(args.stats_featuremap)
        n_bases = count_bases_in_interval_list(args.training_regions, logger_fn=logger.debug)

        n_neg = int(np.sum(labels == 0))
        prior_train_error = n_neg / len(labels) if len(labels) > 0 else 0.5

        # In single-model mode, build the LUT from validation data only
        # to avoid overfitting bias from training-set predictions.
        lut_mask = (fold_ids == 1) if single_model_split else None

        snvq, x_lut, y_lut = recalibrate_snvq(
            mqual,
            labels,
            lut_mask=lut_mask,
            pos_stats=pos_stats,
            neg_stats=neg_stats,
            raw_stats=raw_stats,
            mean_coverage=args.mean_coverage,
            n_bases_in_region=n_bases,
            prior_train_error=prior_train_error,
        )
        logger.info(
            "Applied MQUAL→SNVQ recalibration (LUT %d pts, SNVQ [%.1f, %.1f])",
            len(x_lut),
            y_lut.min(),
            y_lut.max(),
        )
    else:
        logger.warning("Stats/coverage args not provided; SNVQ = MQUAL (no recalibration)")
        snvq = mqual.copy()

    df_out = pl.DataFrame(
        {
            CHROM: collected["chroms"],
            POS: collected["pos"].tolist(),
            "RN": collected["rns"],
            "label": labels.tolist(),
            "fold_id": fold_ids.tolist(),
            "prob_orig": probs.tolist(),
            "MQUAL": mqual.tolist(),
            "SNVQ": snvq.tolist(),
        }
    )
    df_path = out_dir / f"{base}featuremap_df.parquet"
    df_out.write_parquet(df_path)
    logger.info("Saved prediction dataframe: %s", df_path)

    # Holdout metrics
    holdout_mask = fold_ids == -1
    holdout_metrics = {}
    if holdout_mask.any():
        holdout_metrics = _safe_binary_metrics(labels[holdout_mask], probs[holdout_mask])

    # ── Metadata ──
    metadata = {
        "model_type": "deep_srsnv_cnn_lightning",
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
            "mixed",
        ],
        "training_results": training_results,
        "holdout_metrics": holdout_metrics,
        "split_prevalence": split_prevalence,
        "chunk_composition": {"train_chunk_prevalence": train_chunk_mix},
        "model_architecture": model_arch_summary,
        "schema_path": str(schema_path),
        "best_checkpoint_paths": best_ckpt_paths,
        "quality_recalibration_table": [x_lut.tolist(), y_lut.tolist()] if x_lut is not None else None,
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
            "eval_batch_size": args.eval_batch_size,
            "predict_batch_size": args.predict_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_channels": args.hidden_channels,
            "n_blocks": args.n_blocks,
            "base_embed_dim": args.base_embed_dim,
            "t0_embed_dim": args.t0_embed_dim,
            "cat_embed_dim": args.cat_embed_dim,
            "dropout": args.dropout,
            "lr_scheduler": args.lr_scheduler,
            "swa": args.swa,
            "swa_lr": args.swa_lr if args.swa else None,
            "swa_epoch_start": args.swa_epoch_start if args.swa else None,
            "gradient_clip_val": args.gradient_clip_val,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "length": args.length,
            "max_rows_per_class": args.max_rows_per_class,
            "loader_pin_memory": bool(args.loader_pin_memory),
            "use_amp": bool(args.use_amp),
            "use_tf32": bool(args.use_tf32),
            "devices": args.devices,
            "strategy": args.strategy,
            "auto_lr_find": args.auto_lr_find,
            "auto_scale_batch_size": args.auto_scale_batch_size,
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
    logger.info("Saved deep_srsnv Lightning outputs to %s", out_dir)


if __name__ == "__main__":
    main()
