"""Combine per-fold DNN parquets and build a shared MQUAL-to-SNVQ lookup table.

After k independent DNN folds are trained (each producing its own parquet with
``prob_orig`` and ``MQUAL``), this script:

1. Collects val and test predictions from all folds.
2. Builds ``prob_fold_*`` columns on test rows (each fold predicts test data).
3. Constructs a single shared MQUAL-to-SNVQ LUT (counting-based or KDE),
   matching the XGBoost pipeline behaviour.
4. Writes a combined parquet with correct SNVQ and updates per-fold metadata
   with the shared ``quality_recalibration_table``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from ugbio_core.logger import logger

from ugbio_srsnv.srsnv_training import _extract_stats_from_unified
from ugbio_srsnv.srsnv_utils import MAX_PHRED, prob_to_phred, recalibrate_snvq, recalibrate_snvq_kde


def _load_fold_parquets(
    fold_parquets: list[str],
) -> tuple[list[pl.DataFrame], int]:
    """Load per-fold parquets and return them with the detected k_folds count."""
    frames = []
    for path in fold_parquets:
        fold_df = pl.read_parquet(path)
        logger.info("Loaded %s: %d rows, columns=%s", path, len(fold_df), fold_df.columns)
        frames.append(fold_df)
    return frames, len(frames)


def _build_combined_dataframe(
    frames: list[pl.DataFrame],
    k_folds: int,
) -> pl.DataFrame:
    """Merge per-fold predictions into a single DataFrame.

    Val rows (``fold_id == 1``): kept from their respective fold with
    ``prob_fold_{fold_idx}`` = ``prob_orig``.

    Test rows (``fold_id == -1``): joined across folds on ``(CHROM, POS, RN)``
    to produce ``prob_fold_0 .. prob_fold_{k-1}``.  Final ``prob_orig`` is the
    mean across all folds (matching ``_collect_predictions`` logic).
    """
    val_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []

    for fold_idx, df in enumerate(frames):
        prob_col = f"prob_fold_{fold_idx}"

        val_df = df.filter(pl.col("fold_id") != -1).with_columns(
            pl.col("prob_orig").alias(prob_col),
            pl.lit(fold_idx).alias("fold_id"),
        )
        val_parts.append(val_df)

        test_df = df.filter(pl.col("fold_id") == -1).select(
            [
                "CHROM",
                "POS",
                "RN",
                "label",
                pl.col("prob_orig").alias(prob_col),
            ]
        )
        test_parts.append(test_df)

    # -- Combine val rows (disjoint across folds) --
    combined_val = pl.concat(val_parts, how="diagonal_relaxed")

    # -- Combine test rows (same rows across all folds, different probs) --
    if test_parts:
        test_base = test_parts[0].rename({"label": "label"})
        for fold_idx in range(1, k_folds):
            prob_col = f"prob_fold_{fold_idx}"
            test_base = test_base.join(
                test_parts[fold_idx].select(["CHROM", "POS", "RN", prob_col]),
                on=["CHROM", "POS", "RN"],
                how="inner",
            )

        prob_fold_cols = [f"prob_fold_{i}" for i in range(k_folds)]
        test_base = test_base.with_columns(
            pl.mean_horizontal(*prob_fold_cols).alias("prob_orig"),
            pl.lit(None, dtype=pl.Int64).alias("fold_id"),
        )
        logger.info("Test rows after cross-fold join: %d", len(test_base))
    else:
        test_base = pl.DataFrame()

    if len(test_base) > 0:
        combined = pl.concat([combined_val, test_base], how="diagonal_relaxed")
    else:
        combined = combined_val

    logger.info(
        "Combined DataFrame: %d rows (%d val, %d test)",
        len(combined),
        len(combined_val),
        len(test_base) if len(test_base) > 0 else 0,
    )
    return combined


def recalibrate_dnn_folds(  # noqa: PLR0913, PLR0915
    fold_parquets: list[str],
    fold_metadata_paths: list[str],
    stats_file: str,
    training_regions: str,
    mean_coverage: float,
    output_dir: str,
    basename: str = "",
    *,
    use_kde: bool = False,
    kde_config_overrides: dict | None = None,
    featuremap_parquets: list[str] | None = None,
) -> tuple[Path, Path]:
    """Main entry point: combine folds, build shared LUT, write outputs.

    Returns
    -------
    tuple[Path, Path]
        Paths to the combined parquet and combined metadata JSON.
    """
    from ugbio_srsnv.srsnv_training import _count_bases_in_interval_list  # noqa: PLC0415

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"{basename}." if basename and not basename.endswith(".") else basename

    # -- Load data --
    frames, k_folds = _load_fold_parquets(fold_parquets)
    combined = _build_combined_dataframe(frames, k_folds)

    # -- Recompute MQUAL from (potentially updated) prob_orig --
    prob_orig = combined["prob_orig"].to_numpy()
    mqual = prob_to_phred(prob_orig, max_value=MAX_PHRED)
    combined = combined.with_columns(pl.Series("MQUAL", mqual))

    # -- Load stats from unified JSON --
    pos_stats, neg_stats = _extract_stats_from_unified(stats_file)
    raw_stats = neg_stats
    n_bases = _count_bases_in_interval_list(training_regions)

    labels = combined["label"].to_numpy().astype(bool)

    # lut_mask: val rows only (fold_id is not null and != -1)
    fold_id_arr = combined["fold_id"].to_numpy()
    lut_mask = ~np.isnan(fold_id_arr.astype(float))
    logger.info("LUT mask: %d val rows / %d total", int(lut_mask.sum()), len(lut_mask))

    if use_kde:
        pd_df = combined.to_pandas()
        pd_df["label"] = pd_df["label"].astype(bool)
        snvq, x_lut, y_lut = recalibrate_snvq_kde(
            pd_df,
            pos_stats=pos_stats,
            neg_stats=neg_stats,
            raw_stats=raw_stats,
            mean_coverage=mean_coverage,
            n_bases_in_region=n_bases,
            k_folds=k_folds,
            lut_mask=lut_mask,
        )
    else:
        snvq, x_lut, y_lut = recalibrate_snvq(
            mqual,
            labels,
            pos_stats=pos_stats,
            neg_stats=neg_stats,
            raw_stats=raw_stats,
            mean_coverage=mean_coverage,
            n_bases_in_region=n_bases,
            lut_mask=lut_mask,
        )

    combined = combined.with_columns(pl.Series("SNVQ", snvq))

    # -- Enrich with VCF feature columns from training featuremap parquets --
    if featuremap_parquets:
        fm_frames = [pl.read_parquet(p) for p in featuremap_parquets]
        fm_df = pl.concat(fm_frames, how="diagonal_relaxed")
        logger.info("Loaded %d featuremap parquets: %d total rows", len(fm_frames), len(fm_df))
        # Drop columns that already exist in combined (except join keys)
        join_keys = ["CHROM", "POS", "RN"]
        existing_cols = set(combined.columns) - set(join_keys)
        fm_cols_to_add = [c for c in fm_df.columns if c not in existing_cols]
        fm_df = fm_df.select(fm_cols_to_add)
        combined = combined.join(fm_df, on=join_keys, how="left")
        logger.info(
            "Enriched combined_featuremap_df with %d columns from %d parquets",
            len(fm_cols_to_add) - len(join_keys),
            len(featuremap_parquets),
        )

    # -- Write combined parquet --
    # Ensure label is boolean (MRD's calc_tumor_fraction_denominator_ratio uses .query('label'))
    combined = combined.with_columns(pl.col("label").cast(pl.Boolean))
    parquet_path = out / f"{suffix}featuremap_df.parquet"
    combined.write_parquet(parquet_path)
    logger.info("Combined parquet: %s (%d rows)", parquet_path, len(combined))

    # -- Update per-fold metadata with shared LUT --
    shared_lut = [x_lut.tolist(), y_lut.tolist()]
    for meta_path in fold_metadata_paths:
        mp = Path(meta_path)
        if mp.exists():
            meta = json.loads(mp.read_text())
            meta["quality_recalibration_table"] = shared_lut
            out_mp = out / mp.name
            out_mp.write_text(json.dumps(meta, indent=2))
            logger.info("Updated metadata with shared LUT: %s", out_mp)
        else:
            logger.warning("Metadata file not found, skipping: %s", mp)

    # -- Write standalone combined metadata --
    combined_meta = {
        "quality_recalibration_table": shared_lut,
        "filtering_stats": {
            "negative": neg_stats,
            "positive": pos_stats,
        },
        "k_folds": k_folds,
        "lut_method": "kde" if use_kde else "counting",
        "lut_points": len(x_lut),
        "snvq_range": [float(y_lut.min()), float(y_lut.max())],
        "fold_parquets": fold_parquets,
        "fold_metadata_paths": fold_metadata_paths,
    }
    meta_path = out / f"{suffix}shared_lut_metadata.json"
    meta_path.write_text(json.dumps(combined_meta, indent=2))
    logger.info("Shared LUT metadata: %s", meta_path)

    return parquet_path, meta_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Combine DNN fold parquets and build a shared MQUAL→SNVQ LUT",
    )
    ap.add_argument(
        "--fold-parquets",
        nargs="+",
        required=True,
        help="Per-fold featuremap_df.parquet files",
    )
    ap.add_argument(
        "--fold-metadata",
        nargs="+",
        required=True,
        help="Per-fold srsnv_dnn_metadata.json files (will be updated in-place with shared LUT)",
    )
    ap.add_argument("--stats-file", required=True, help="Unified stats JSON from snvfind -S")
    ap.add_argument("--training-regions", required=True, help="Training regions interval list")
    ap.add_argument("--mean-coverage", type=float, required=True, help="Mean sequencing coverage")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--basename", default="", help="Basename prefix for output files")
    ap.add_argument(
        "--use-kde",
        action="store_true",
        help="Use KDE-based smoothing (default: counting-based, matching XGBoost default)",
    )
    ap.add_argument(
        "--featuremap-parquets",
        nargs="+",
        default=None,
        help="Training featuremap parquets (positive + negative) with VCF feature columns. "
        "When provided, enriches the combined output with all VCF features for MRD compatibility.",
    )
    return ap.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv[1:])
    parquet_path, meta_path = recalibrate_dnn_folds(
        fold_parquets=args.fold_parquets,
        fold_metadata_paths=args.fold_metadata,
        stats_file=args.stats_file,
        training_regions=args.training_regions,
        mean_coverage=args.mean_coverage,
        output_dir=args.output_dir,
        basename=args.basename,
        use_kde=args.use_kde,
        featuremap_parquets=args.featuremap_parquets,
    )
    logger.info("Done. Combined parquet: %s, LUT metadata: %s", parquet_path, meta_path)


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
