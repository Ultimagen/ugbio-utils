"""Prepare DNN training outputs for consumption by srsnv_report.

Merges DNN predictions (prob_orig, MQUAL, SNVQ) into an XGBoost featuremap
parquet so that ``srsnv_report`` can generate quality reports on DNN models
without any changes to the report code.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl
from ugbio_core.logger import logger

from ugbio_srsnv.srsnv_utils import MAX_PHRED

MAX_ACCEPTABLE_DROP_PCT = 10


def _load_fold_csv_metrics(meta_path: Path) -> pd.DataFrame | None:
    """Load per-epoch metrics from a fold's Lightning CSV output."""
    meta = json.loads(meta_path.read_text())
    ckpt_paths = meta.get("best_checkpoint_paths", [])
    if not ckpt_paths:
        return None

    ckpt = Path(ckpt_paths[0])
    prefix = ckpt.name.split(".dnn_model_fold_")[0]
    logs_dir = ckpt.parent / f"{prefix}.lightning_logs"
    if not logs_dir.is_dir():
        candidates = list(ckpt.parent.glob("*.lightning_logs"))
        logs_dir = candidates[0] if candidates else None
    if logs_dir is None or not logs_dir.is_dir():
        return None

    csvs = sorted(logs_dir.glob("fold_*/metrics.csv"))
    if not csvs:
        return None

    try:
        metrics_df = pd.read_csv(csvs[0])
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    if "epoch" not in metrics_df.columns:
        return None

    return metrics_df.groupby("epoch").first().reset_index().sort_values("epoch")


def _build_dnn_training_results(fold_metadata_paths: list[str]) -> list[dict] | None:
    """Read per-fold Lightning CSV metrics and convert to XGBoost evals_result format.

    Returns a list of dicts (one per fold), each shaped like::

        {"validation_0": {"logloss": [...], "auc": [...]},   # train
         "validation_1": {"logloss": [...], "auc": [...]}}    # val
    """
    results: list[dict] = []
    for meta_path in fold_metadata_paths:
        mp = Path(meta_path)
        if not mp.exists():
            logger.warning("Fold metadata not found, skipping: %s", mp)
            continue

        epoch_df = _load_fold_csv_metrics(mp)
        if epoch_df is None:
            logger.warning("No Lightning CSV metrics found for %s", mp)
            continue

        train_loss = epoch_df["train_loss"].dropna().tolist() if "train_loss" in epoch_df else []
        train_auc = epoch_df["train_auc"].dropna().tolist() if "train_auc" in epoch_df else []
        val_loss = epoch_df["val_loss"].dropna().tolist() if "val_loss" in epoch_df else []
        val_auc = epoch_df["val_auc"].dropna().tolist() if "val_auc" in epoch_df else []

        if not train_loss and not val_loss:
            logger.warning("No loss metrics in epoch_df for %s", mp)
            continue

        results.append(
            {
                "validation_0": {"logloss": train_loss, "auc": train_auc},
                "validation_1": {"logloss": val_loss, "auc": val_auc},
            }
        )
        logger.info("Loaded training metrics from %s: %d epochs", mp, len(epoch_df))

    return results if results else None


def _extract_features_from_parquet(parquet: pl.DataFrame, feature_names: list[str] | None = None) -> list[dict]:
    """Extract feature schema from parquet columns.

    If *feature_names* is provided, only those columns are included.
    Otherwise falls back to scanning all parquet columns (excluding known non-feature columns).
    """
    if feature_names:
        cols_to_check = [c for c in feature_names if c in parquet.columns]
    else:
        skip_cols = {
            "label",
            "CHROM",
            "POS",
            "RN",
            "chrom",
            "pos",
            "rn",
            "fold",
            "fold_id",
            "prob",
            "prob_orig",
            "MQUAL",
            "SNVQ",
            "ID",
            "QUAL",
            "FILT_BITMAP",
            "MI",
            "X_ALT",
        }
        cols_to_check = [c for c in parquet.columns if c not in skip_cols and not c.startswith("prob_fold_")]

    features_meta = []
    for col in cols_to_check:
        if parquet[col].drop_nulls().n_unique() <= 1:
            logger.debug("Skipping constant column: %s", col)
            continue
        dtype = parquet[col].dtype
        if dtype in (pl.Categorical, pl.Utf8, pl.String) or isinstance(dtype, pl.Enum):
            unique_vals = sorted(str(v) for v in parquet[col].drop_nulls().unique().to_list())
            encoding = {v: i for i, v in enumerate(unique_vals)}
            features_meta.append({"name": col, "type": "c", "values": encoding})
        elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            features_meta.append({"name": col, "type": "int"})
        else:
            features_meta.append({"name": col, "type": "float"})
    return features_meta


def _detect_k_folds(metadata: dict) -> int:
    """Infer the number of CV folds from XGBoost metadata."""
    if "model_paths" in metadata:
        return len(metadata["model_paths"])
    split = metadata.get("split_summary", {})
    if "k_folds" in split:
        return split["k_folds"]
    return 1


def _merge_dnn_into_training(training_df: pl.DataFrame, dnn: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Join feature columns from training onto the DNN base DataFrame.

    The DNN parquet is used as the LEFT (base) table because it contains both
    TP and FP labels, prob_fold_* columns, MQUAL, SNVQ, etc.  Feature columns
    from the training parquet are joined onto it so the report can plot per-feature
    quality distributions.

    Returns the merged DataFrame and list of DNN prob_fold_* column names.
    """
    # Normalise DNN fold_id: -1 -> null to match XGBoost convention
    if "fold_id" in dnn.columns:
        dnn = dnn.with_columns(
            pl.when(pl.col("fold_id") == -1).then(None).otherwise(pl.col("fold_id")).alias("fold_id")
        )

    # De-duplicate test rows that appear in every fold
    test_mask = dnn["fold_id"].is_null()
    if test_mask.any():
        test_deduped = dnn.filter(test_mask).unique(subset=["CHROM", "POS", "RN"], keep="first")
        dnn = pl.concat([dnn.filter(~test_mask), test_deduped])
        logger.info("  After de-duplication: %d rows", len(dnn))

    dnn_prob_fold_cols = sorted(c for c in dnn.columns if c.startswith("prob_fold_"))

    # Select feature columns from training_df (everything except join keys
    # and columns already present in the DNN parquet)
    join_keys = ["CHROM", "POS", "RN"]
    dnn_existing_cols = set(dnn.columns)
    feature_cols = [c for c in training_df.columns if c not in dnn_existing_cols and c not in join_keys]

    if feature_cols:
        training_features = training_df.select(join_keys + feature_cols)
        merged = dnn.join(training_features, on=join_keys, how="left")
        n_matched = merged.filter(pl.col(feature_cols[0]).is_not_null()).height
        logger.info(
            "  DNN rows: %d, matched features from training: %d / %d",
            len(dnn),
            n_matched,
            len(training_df),
        )
    else:
        merged = dnn
        logger.info("  No additional feature columns to join from training parquet")

    return merged, dnn_prob_fold_cols


def prepare_dnn_report_data(  # noqa: C901, PLR0912, PLR0915
    training_parquet: str,
    dnn_parquet: str,
    training_metadata_path: str,
    dnn_metadata_path: str,
    output_dir: str,
    basename: str = "",
    dnn_fold_metadata_paths: list[str] | None = None,
    negative_parquet: str | None = None,
    feature_names: list[str] | None = None,
) -> tuple[Path, Path]:
    """Merge DNN predictions into an XGBoost parquet for srsnv_report.

    Parameters
    ----------
    xgb_parquet
        Path to the XGBoost featuremap_df.parquet (has all feature columns).
    dnn_parquet
        Path to the DNN featuremap_df.parquet (has prob_orig, MQUAL, SNVQ).
    xgb_metadata_path
        Path to the XGBoost srsnv_metadata.json.
    dnn_metadata_path
        Path to the DNN srsnv_dnn_metadata.json.
    output_dir
        Directory for output files.
    basename
        Prefix for output files.
    dnn_fold_metadata_paths
        Per-fold DNN metadata JSONs (used to load Lightning CSV training
        metrics for the training progress plot).

    Returns
    -------
    tuple[Path, Path]
        Paths to the merged parquet and metadata JSON.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    suffix = f"{basename}." if basename and not basename.endswith(".") else basename

    # ── Load parquets ───────────────────────────────────────────────
    logger.info("Loading positive training parquet: %s", training_parquet)
    pos_df = pl.read_parquet(training_parquet)
    logger.info("  Positive rows=%d, columns=%d", len(pos_df), len(pos_df.columns))

    # For positive (reference) reads, apply the same transformations as
    # XGBoost srsnv_training: swap REF/X_ALT, swap hmers, filter max EDIST,
    # and increment edit distance features
    if "X_ALT" in pos_df.columns:
        ref_dtype = pos_df["REF"].dtype
        pos_df = pos_df.with_columns(pl.col("X_ALT").cast(ref_dtype).alias("REF")).drop("X_ALT")
        if {"X_HMER_REF", "X_HMER_ALT"} <= set(pos_df.columns):
            pos_df = pos_df.rename({"X_HMER_REF": "__tmp_hmer", "X_HMER_ALT": "X_HMER_REF"}).rename(
                {"__tmp_hmer": "X_HMER_ALT"}
            )
        if "EDIST" in pos_df.columns:
            max_edist = pos_df.select(pl.max("EDIST")).item()
            pos_df = pos_df.filter(pl.col("EDIST") != max_edist)
        for feat in ("EDIST", "HAMDIST", "HAMDIST_FILT"):
            if feat in pos_df.columns:
                pos_df = pos_df.with_columns((pl.col(feat) + 1).alias(feat))
        logger.info("  Applied X_ALT→REF swap + edit distance adjustments for positive reads")

    training_df = pos_df
    if negative_parquet:
        logger.info("Loading negative training parquet: %s", negative_parquet)
        neg_df = pl.read_parquet(negative_parquet)
        if "X_ALT" in neg_df.columns:
            neg_df = neg_df.drop("X_ALT")
        logger.info("  Negative rows=%d", len(neg_df))
        training_df = pl.concat([training_df, neg_df], how="diagonal")
        logger.info("  Combined training rows=%d", len(training_df))

    logger.info("Loading DNN parquet: %s", dnn_parquet)
    dnn = pl.read_parquet(dnn_parquet)
    logger.info("  DNN rows=%d", len(dnn))

    # ── Join DNN scores onto XGBoost base ───────────────────────────
    merged, dnn_prob_fold_cols = _merge_dnn_into_training(training_df, dnn)

    with open(training_metadata_path) as f:
        training_metadata = json.load(f)

    if "features" not in training_metadata:
        logger.info("No 'features' in metadata — extracting schema from parquet columns")
        training_metadata["features"] = _extract_features_from_parquet(training_df, feature_names=feature_names)

    k_folds = _detect_k_folds(training_metadata)

    # If DNN didn't provide prob_fold_* columns, create them from prob_orig
    if not dnn_prob_fold_cols:
        for k in range(k_folds):
            merged = merged.with_columns(pl.col("prob_orig").alias(f"prob_fold_{k}"))
        dnn_prob_fold_cols = [f"prob_fold_{k}" for k in range(k_folds)]

    # ── Create ML_qual_* columns from prob_fold_* ──────────────────
    for c in dnn_prob_fold_cols:
        fold_idx = c.replace("prob_fold_", "")
        merged = merged.with_columns(
            (-10.0 * (1.0 - pl.col(c)).log(base=10)).clip(0, MAX_PHRED).alias(f"ML_qual_{fold_idx}")
        )

    # ── Write merged parquet ────────────────────────────────────────
    parquet_path = out / f"{suffix}featuremap_df.parquet"
    merged.write_parquet(parquet_path)
    logger.info("Merged parquet written: %s (%d rows)", parquet_path, len(merged))

    # ── Build compatible metadata ───────────────────────────────────
    with open(dnn_metadata_path) as f:
        dnn_metadata = json.load(f)

    metadata = copy.deepcopy(training_metadata)

    # Use DNN quality recalibration table
    dnn_recal = dnn_metadata.get("quality_recalibration_table")
    if dnn_recal is not None:
        metadata["quality_recalibration_table"] = dnn_recal

    # Convert DNN Lightning CSV metrics to XGBoost evals_result format
    if dnn_fold_metadata_paths:
        dnn_training_results = _build_dnn_training_results(dnn_fold_metadata_paths)
    else:
        dnn_training_results = None
    metadata["training_results"] = dnn_training_results

    # Populate model_paths with k_folds dummy entries so srsnv_report
    # creates the correct number of DummyClassifier models, which sets
    # num_CV_folds correctly for per-fold histogram lines.
    n_folds = len(dnn_prob_fold_cols) or k_folds
    metadata["model_paths"] = {str(i): "" for i in range(n_folds)}

    # Preserve user metadata and add DNN info
    user_meta = metadata.setdefault("metadata", {})
    user_meta["model_type"] = "dnn"
    user_meta["dnn_metadata_path"] = str(dnn_metadata_path)

    # Copy DNN holdout metrics if available
    holdout = dnn_metadata.get("holdout_metrics")
    if holdout:
        user_meta["dnn_holdout_metrics"] = json.dumps(holdout)

    metadata_path = out / f"{suffix}srsnv_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata written: %s", metadata_path)

    return parquet_path, metadata_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare DNN training outputs for srsnv_report",
    )
    ap.add_argument("--training-parquet", required=True, help="Positive training featuremap_df.parquet path")
    ap.add_argument("--negative-parquet", default=None, help="Negative training featuremap_df.parquet path")
    ap.add_argument("--dnn-parquet", required=True, help="DNN featuremap_df.parquet path")
    ap.add_argument("--training-metadata", required=True, help="Training metadata JSON path")
    ap.add_argument("--dnn-metadata", required=True, help="DNN srsnv_dnn_metadata.json path")
    ap.add_argument(
        "--dnn-fold-metadata",
        nargs="+",
        default=None,
        help="Per-fold DNN metadata JSONs (for training progress plot)",
    )
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--basename", default="", help="Basename prefix for output files")
    ap.add_argument("--features", default=None, help="Colon-separated feature names for report plots")
    return ap.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv[1:])
    feature_names = args.features.split(":") if args.features else None
    parquet_path, metadata_path = prepare_dnn_report_data(
        training_parquet=args.training_parquet,
        dnn_parquet=args.dnn_parquet,
        training_metadata_path=args.training_metadata,
        dnn_metadata_path=args.dnn_metadata,
        output_dir=args.output_dir,
        basename=args.basename,
        dnn_fold_metadata_paths=args.dnn_fold_metadata,
        negative_parquet=args.negative_parquet,
        feature_names=feature_names,
    )
    logger.info("Done. Use with srsnv_report:")
    logger.info(
        "  srsnv_report --featuremap-df %s --srsnv-metadata %s --report-path %s",
        parquet_path,
        metadata_path,
        args.output_dir,
    )


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
