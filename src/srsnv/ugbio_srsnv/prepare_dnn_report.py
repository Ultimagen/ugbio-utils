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

import polars as pl
from ugbio_core.logger import logger


def _build_dnn_training_results(fold_metadata_paths: list[str]) -> list[dict] | None:  # noqa: C901
    """Read per-fold Lightning CSV metrics and convert to XGBoost evals_result format.

    Returns a list of dicts (one per fold), each shaped like::

        {"validation_0": {"logloss": [...], "auc": [...]},   # train
         "validation_1": {"logloss": [...], "auc": [...]}}    # val
    """
    import pandas as pd  # noqa: PLC0415

    results: list[dict] = []
    for meta_path in fold_metadata_paths:
        mp = Path(meta_path)
        if not mp.exists():
            logger.warning("Fold metadata not found, skipping: %s", mp)
            continue
        meta = json.loads(mp.read_text())

        # Derive lightning_logs directory from checkpoint path
        ckpt_paths = meta.get("best_checkpoint_paths", [])
        csv_path = None
        if ckpt_paths:
            ckpt = Path(ckpt_paths[0])
            prefix = ckpt.name.split(".dnn_model_fold_")[0]
            logs_dir = ckpt.parent / f"{prefix}.lightning_logs"
            if not logs_dir.is_dir():
                candidates = list(ckpt.parent.glob("*.lightning_logs"))
                logs_dir = candidates[0] if candidates else None
            if logs_dir and logs_dir.is_dir():
                csvs = sorted(logs_dir.glob("fold_*/metrics.csv"))
                if csvs:
                    csv_path = csvs[0]

        if csv_path is None:
            logger.warning("No Lightning CSV found for %s", mp)
            continue

        try:
            metrics_df = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read CSV: %s", csv_path)
            continue

        if "epoch" not in metrics_df.columns:
            logger.warning("No 'epoch' column in %s", csv_path)
            continue

        epoch_df = metrics_df.groupby("epoch").first().reset_index().sort_values("epoch")

        train_loss = epoch_df["train_loss"].dropna().tolist() if "train_loss" in epoch_df else []
        train_auc = epoch_df["train_auc"].dropna().tolist() if "train_auc" in epoch_df else []
        val_loss = epoch_df["val_loss"].dropna().tolist() if "val_loss" in epoch_df else []
        val_auc = epoch_df["val_auc"].dropna().tolist() if "val_auc" in epoch_df else []

        if not train_loss and not val_loss:
            logger.warning("No loss metrics in %s", csv_path)
            continue

        results.append(
            {
                "validation_0": {"logloss": train_loss, "auc": train_auc},
                "validation_1": {"logloss": val_loss, "auc": val_auc},
            }
        )
        logger.info("Loaded training metrics from %s: %d epochs", csv_path, len(epoch_df))

    return results if results else None


def _detect_k_folds(metadata: dict) -> int:
    """Infer the number of CV folds from XGBoost metadata."""
    if "model_paths" in metadata:
        return len(metadata["model_paths"])
    split = metadata.get("split_summary", {})
    if "k_folds" in split:
        return split["k_folds"]
    return 1


def prepare_dnn_report_data(  # noqa: PLR0913, C901, PLR0915
    xgb_parquet: str,
    dnn_parquet: str,
    xgb_metadata_path: str,
    dnn_metadata_path: str,
    output_dir: str,
    basename: str = "",
    dnn_fold_metadata_paths: list[str] | None = None,
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
    logger.info("Loading XGBoost parquet: %s", xgb_parquet)
    xgb = pl.read_parquet(xgb_parquet)
    logger.info("  XGBoost rows=%d, columns=%d", len(xgb), len(xgb.columns))

    logger.info("Loading DNN parquet: %s", dnn_parquet)
    dnn = pl.read_parquet(dnn_parquet)
    logger.info("  DNN rows=%d", len(dnn))

    # Normalise DNN fold_id: -1 → null to match XGBoost convention
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

    # ── Join DNN scores onto XGBoost base ───────────────────────────
    # Carry over: prob_orig, MQUAL, SNVQ, fold_id, and any prob_fold_* columns
    dnn_join_cols = ["CHROM", "POS", "RN"]
    dnn_score_cols = [
        pl.col("prob_orig").alias("prob_orig_dnn"),
        pl.col("MQUAL").alias("MQUAL_dnn"),
        pl.col("SNVQ").alias("SNVQ_dnn"),
        pl.col("fold_id").alias("fold_id_dnn"),
    ]
    dnn_prob_fold_cols = sorted(c for c in dnn.columns if c.startswith("prob_fold_"))
    for c in dnn_prob_fold_cols:
        dnn_score_cols.append(pl.col(c).alias(f"{c}_dnn"))

    dnn_scores = dnn.select(dnn_join_cols + dnn_score_cols)

    merged = xgb.join(dnn_scores, on=["CHROM", "POS", "RN"], how="inner")
    logger.info("  Matched rows: %d / %d XGB, %d DNN", len(merged), len(xgb), len(dnn))

    drop_pct = (1 - len(merged) / len(xgb)) * 100
    if drop_pct > 10:  # noqa: PLR2004
        logger.warning(
            "%.1f%% of XGBoost rows dropped during join – " "ensure both pipelines trained on the same sample data.",
            drop_pct,
        )

    # Replace XGBoost quality columns and fold_id with DNN values
    cols_to_drop = [c for c in ("prob_orig", "MQUAL", "SNVQ", "fold_id") if c in merged.columns]
    merged = merged.drop(cols_to_drop)
    merged = merged.rename(
        {
            "prob_orig_dnn": "prob_orig",
            "MQUAL_dnn": "MQUAL",
            "SNVQ_dnn": "SNVQ",
            "fold_id_dnn": "fold_id",
        }
    )

    # Replace XGBoost prob_fold_* columns with DNN ones
    old_prob_fold_cols = [c for c in merged.columns if c.startswith("prob_fold_") and not c.endswith("_dnn")]
    if old_prob_fold_cols:
        merged = merged.drop(old_prob_fold_cols)
    for c in dnn_prob_fold_cols:
        merged = merged.rename({f"{c}_dnn": c})

    with open(xgb_metadata_path) as f:
        xgb_metadata = json.load(f)
    k_folds = _detect_k_folds(xgb_metadata)

    # If DNN didn't provide prob_fold_* columns, create them from prob_orig
    if not dnn_prob_fold_cols:
        for k in range(k_folds):
            merged = merged.with_columns(pl.col("prob_orig").alias(f"prob_fold_{k}"))

    # ── Write merged parquet ────────────────────────────────────────
    parquet_path = out / f"{suffix}featuremap_df.parquet"
    merged.write_parquet(parquet_path)
    logger.info("Merged parquet written: %s (%d rows)", parquet_path, len(merged))

    # ── Build compatible metadata ───────────────────────────────────
    with open(dnn_metadata_path) as f:
        dnn_metadata = json.load(f)

    metadata = copy.deepcopy(xgb_metadata)

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
    ap.add_argument("--xgb-parquet", required=True, help="XGBoost featuremap_df.parquet path")
    ap.add_argument("--dnn-parquet", required=True, help="DNN featuremap_df.parquet path")
    ap.add_argument("--xgb-metadata", required=True, help="XGBoost srsnv_metadata.json path")
    ap.add_argument("--dnn-metadata", required=True, help="DNN srsnv_dnn_metadata.json path")
    ap.add_argument(
        "--dnn-fold-metadata",
        nargs="+",
        default=None,
        help="Per-fold DNN metadata JSONs (for training progress plot)",
    )
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--basename", default="", help="Basename prefix for output files")
    return ap.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv[1:])
    parquet_path, metadata_path = prepare_dnn_report_data(
        xgb_parquet=args.xgb_parquet,
        dnn_parquet=args.dnn_parquet,
        xgb_metadata_path=args.xgb_metadata,
        dnn_metadata_path=args.dnn_metadata,
        output_dir=args.output_dir,
        basename=args.basename,
        dnn_fold_metadata_paths=args.dnn_fold_metadata,
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
