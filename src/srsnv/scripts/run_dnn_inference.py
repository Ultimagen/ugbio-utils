#!/usr/bin/env python3
"""Standalone DNN inference: apply an existing SRSNV DNN model to new data.

Reuses the same preprocessing/tensorisation pipeline as training but skips
the training loop entirely.  Outputs a featuremap_df parquet and metadata
JSON that are compatible with ``compare_models_report``.

Usage
-----
    uv run python src/srsnv/scripts/run_dnn_inference.py \
        --checkpoint /path/to/model_swa.ckpt \
        --metadata   /path/to/srsnv_dnn_metadata.json \
        --positive-bam   inputs/positive_reads.bam \
        --negative-bam   inputs/negative_reads.bam \
        --positive-parquet inputs/positive.parquet \
        --negative-parquet inputs/negative.parquet \
        --training-regions inputs/training_regions.interval_list.gz \
        --stats-positive inputs/stats_positive.json \
        --stats-negative inputs/stats_negative.json \
        --stats-featuremap inputs/stats_featuremap.json \
        --mean-coverage 66 \
        --output /path/to/output_dir \
        --basename dnn_cross_inference
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightning
import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_featuremap.filter_dataframe import read_filtering_stats_json
from ugbio_srsnv.deep_srsnv.bam_schema import discover_bam_schema
from ugbio_srsnv.deep_srsnv.data_module import SRSNVDataModule
from ugbio_srsnv.deep_srsnv.data_prep import build_encoders_from_schema, build_tensor_cache, load_full_tensor_cache
from ugbio_srsnv.deep_srsnv.inference import load_dnn_model_from_swa_checkpoint
from ugbio_srsnv.deep_srsnv.lightning_module import SRSNVLightningModule
from ugbio_srsnv.srsnv_utils import MAX_PHRED, prob_to_phred, recalibrate_snvq

CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run DNN inference on new data with an existing model")

    ap.add_argument("--checkpoint", required=True, help="Path to the .ckpt file (SWA or best checkpoint)")
    ap.add_argument("--metadata", required=True, help="Path to the srsnv_dnn_metadata.json from original training")

    ap.add_argument("--positive-bam", required=True)
    ap.add_argument("--negative-bam", required=True)
    ap.add_argument("--positive-parquet", required=True)
    ap.add_argument("--negative-parquet", required=True)
    ap.add_argument("--training-regions", required=True)

    ap.add_argument("--stats-positive", default=None)
    ap.add_argument("--stats-negative", default=None)
    ap.add_argument("--stats-featuremap", default=None)
    ap.add_argument("--mean-coverage", type=float, default=None)

    ap.add_argument("--holdout-chromosomes", default="chr21,chr22")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--basename", default="dnn_cross_inference")

    ap.add_argument("--predict-batch-size", type=int, default=2048)
    ap.add_argument("--preprocess-cache-dir", default=None)
    ap.add_argument("--preprocess-num-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 2, 16)))
    ap.add_argument("--preprocess-max-ram-gb", type=float, default=48.0)
    ap.add_argument("--preprocess-batch-rows", type=int, default=25000)
    ap.add_argument("--max-rows-per-class", type=int, default=None)
    ap.add_argument("--length", type=int, default=None, help="Tensor read length (default: auto-read from metadata)")
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--use-tf32", action="store_true")

    return ap.parse_args()


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


def _collect_predictions(predictions: list[dict]) -> dict:
    """Aggregate batched predict_step outputs into numpy arrays."""
    all_probs, all_labels, all_fold_ids = [], [], []
    all_chroms, all_pos, all_rns = [], [], []

    for batch_out in predictions:
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


def _preprocess_data(args, out_dir, basename, split_manifest):
    """Discover BAM schema, build tensor cache, return (dm, preprocess_index, preprocess_wall)."""
    schema_t0 = time.perf_counter()
    schema = discover_bam_schema([args.positive_bam, args.negative_bam], sample_reads_per_bam=20000)
    schema_path = out_dir / f"{basename}feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    logger.info("BAM schema discovered in %.1fs: %s", time.perf_counter() - schema_t0, schema_path)

    encoders = build_encoders_from_schema(schema)
    logger.info(
        "Encoders: base=%d t0=%d tm=%d st=%d et=%d",
        len(encoders.base_vocab),
        len(encoders.t0_vocab),
        len(encoders.tm_vocab),
        len(encoders.st_vocab),
        len(encoders.et_vocab),
    )

    preprocess_cache_dir = args.preprocess_cache_dir or str(out_dir / "deep_srsnv_inference_cache")
    logger.info("Preprocessing started: cache_dir=%s", preprocess_cache_dir)
    preprocess_t0 = time.perf_counter()

    preprocess_index = build_tensor_cache(
        positive_parquet=args.positive_parquet,
        negative_parquet=args.negative_parquet,
        positive_bam=args.positive_bam,
        negative_bam=args.negative_bam,
        encoders=encoders,
        cache_dir=preprocess_cache_dir,
        tensor_length=args.length,
        max_rows_per_class=args.max_rows_per_class,
        preprocess_num_workers=args.preprocess_num_workers,
        preprocess_max_ram_gb=args.preprocess_max_ram_gb,
        preprocess_batch_rows=args.preprocess_batch_rows,
    )
    preprocess_wall = round(time.perf_counter() - preprocess_t0, 3)
    logger.info(
        "Preprocessing finished in %.1fs (cache_hit=%s, shards=%d, rows=%d)",
        preprocess_wall,
        bool(preprocess_index.get("cache_hit", False)),
        int(preprocess_index.get("total_shards", 0)),
        int(preprocess_index.get("total_output_rows", 0)),
    )

    tensor_cache_path = preprocess_index.get("tensor_cache_path")
    if not tensor_cache_path:
        raise ValueError("No tensor cache path produced by preprocessing")

    full_cache = load_full_tensor_cache(tensor_cache_path)
    dm = SRSNVDataModule(
        full_cache=full_cache,
        train_split_ids=set(),
        val_split_ids=set(),
        test_split_ids={-1, 0, 1},
        train_batch_size=args.predict_batch_size,
        eval_batch_size=args.predict_batch_size,
        predict_batch_size=args.predict_batch_size,
        pin_memory=False,
    )
    return dm, preprocess_index, preprocess_wall, preprocess_cache_dir


def _recalibrate(args, mqual, labels):
    """Compute SNVQ via recalibration if stats are available, else copy MQUAL."""
    has_recal = all(
        getattr(args, a, None) is not None
        for a in ("stats_positive", "stats_negative", "stats_featuremap", "mean_coverage")
    )
    if not has_recal:
        logger.warning("Stats/coverage not provided; SNVQ = MQUAL (no recalibration)")
        return mqual.copy(), None, None

    from ugbio_srsnv.srsnv_training import count_bases_in_interval_list  # noqa: PLC0415

    pos_stats = read_filtering_stats_json(args.stats_positive)
    neg_stats = read_filtering_stats_json(args.stats_negative)
    raw_stats = read_filtering_stats_json(args.stats_featuremap)
    n_bases = count_bases_in_interval_list(args.training_regions, logger_fn=logger.debug)
    n_neg = int(np.sum(labels == 0))
    prior_train_error = n_neg / len(labels) if len(labels) > 0 else 0.5

    snvq, x_lut, y_lut = recalibrate_snvq(
        mqual,
        labels,
        lut_mask=None,
        pos_stats=pos_stats,
        neg_stats=neg_stats,
        raw_stats=raw_stats,
        mean_coverage=args.mean_coverage,
        n_bases_in_region=n_bases,
        prior_train_error=prior_train_error,
    )
    logger.info("Applied MQUAL→SNVQ recalibration (LUT %d pts)", len(x_lut))
    return snvq, x_lut, y_lut


def _save_results(  # noqa: PLR0913
    args,
    out_dir,
    basename,
    collected,
    snvq,
    mqual,
    x_lut,
    y_lut,
    orig_metadata,
    split_manifest,
    is_swa,
    preprocess_cache_dir,
    preprocess_wall,
    preprocess_index,
    predict_wall,
):
    """Write the output parquet and metadata JSON."""
    probs, labels, fold_ids = collected["probs"], collected["labels"], collected["fold_ids"]

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
    df_path = out_dir / f"{basename}featuremap_df.parquet"
    df_out.write_parquet(df_path)
    logger.info("Saved prediction dataframe (%d rows): %s", len(df_out), df_path)

    holdout_mask = fold_ids == -1
    holdout_metrics = _safe_binary_metrics(labels[holdout_mask], probs[holdout_mask]) if holdout_mask.any() else {}
    all_metrics = _safe_binary_metrics(labels, probs)
    if holdout_metrics:
        fmt = {k: f"{v:.6f}" if v is not None else "N/A" for k, v in holdout_metrics.items()}
        logger.info("Holdout metrics: %s", fmt)
    logger.info(
        "All-data metrics: %s",
        {k: f"{v:.6f}" if v is not None else "N/A" for k, v in all_metrics.items()},
    )

    metadata = {
        "model_type": "deep_srsnv_cnn_lightning",
        "inference_mode": "cross_sample",
        "source_checkpoint": str(args.checkpoint),
        "source_metadata": str(args.metadata),
        "split_manifest": split_manifest,
        "encoders": orig_metadata.get("encoders", {}),
        "channel_order": orig_metadata.get("channel_order", []),
        "training_parameters": orig_metadata.get("training_parameters", {}),
        "holdout_metrics": holdout_metrics,
        "all_data_metrics": all_metrics,
        "prediction_model": orig_metadata.get("prediction_model", "unknown"),
        "quality_recalibration_table": [x_lut.tolist(), y_lut.tolist()] if x_lut is not None else None,
        "swa_checkpoint_paths": [str(args.checkpoint)] if is_swa else None,
        "best_checkpoint_paths": [] if is_swa else [str(args.checkpoint)],
        "data_paths": {
            "positive_bam": args.positive_bam,
            "negative_bam": args.negative_bam,
            "positive_parquet": args.positive_parquet,
            "negative_parquet": args.negative_parquet,
        },
        "preprocess": {"cache_dir": preprocess_cache_dir, "wall_seconds": preprocess_wall, **preprocess_index},
        "predict_wall_seconds": predict_wall,
    }
    metadata_path = out_dir / f"{basename}srsnv_dnn_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved metadata: %s", metadata_path)


def main() -> None:
    args = _cli()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    basename = args.basename
    if basename and not basename.endswith("."):
        basename += "."

    logger.info("DNN cross-inference started: output=%s basename=%s", out_dir, basename)

    if args.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    with open(args.metadata) as f:
        orig_metadata = json.load(f)

    if args.length is None:
        args.length = orig_metadata.get("training_parameters", {}).get("length", 300)
        logger.info("Auto-detected tensor_length=%d from training metadata", args.length)

    holdout_chroms = [c.strip() for c in args.holdout_chromosomes.split(",") if c.strip()]
    logger.info("Holdout chromosomes (test set): %s", holdout_chroms)
    split_manifest = {
        "split_version": 1,
        "split_mode": "single_model_chrom_val",
        "training_regions": args.training_regions,
        "test_chromosomes": holdout_chroms,
        "chrom_to_fold": {},
    }

    dm, preprocess_index, preprocess_wall, preprocess_cache_dir = _preprocess_data(
        args, out_dir, basename, split_manifest
    )

    # Load existing DNN model
    logger.info("Loading DNN model from checkpoint: %s", args.checkpoint)
    is_swa = orig_metadata.get("prediction_model") == "swa"
    if is_swa:
        lit_model = load_dnn_model_from_swa_checkpoint(args.checkpoint, orig_metadata)
    else:
        lit_model = SRSNVLightningModule.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
        lit_model.eval()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if args.use_amp else "32-true"
    trainer = lightning.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        enable_progress_bar=True,
        default_root_dir=str(out_dir),
    )

    logger.info("Running prediction...")
    predict_t0 = time.perf_counter()
    fold_predictions = trainer.predict(lit_model, datamodule=dm)
    predict_wall = round(time.perf_counter() - predict_t0, 3)
    logger.info("Prediction finished in %.1fs", predict_wall)

    collected = _collect_predictions(fold_predictions)
    probs, labels = collected["probs"], collected["labels"]
    mqual = prob_to_phred(probs, max_value=MAX_PHRED)
    snvq, x_lut, y_lut = _recalibrate(args, mqual, labels)

    _save_results(
        args,
        out_dir,
        basename,
        collected,
        snvq,
        mqual,
        x_lut,
        y_lut,
        orig_metadata,
        split_manifest,
        is_swa,
        preprocess_cache_dir,
        preprocess_wall,
        preprocess_index,
        predict_wall,
    )
    logger.info("DNN cross-inference complete.")


if __name__ == "__main__":
    main()
