#!/usr/bin/env python3
"""Profile the new CRAM-to-tensor preprocessing pipeline end-to-end.

Runs cram_to_tensor_cache for positive + negative, then combine_and_split
for 3-fold k-fold, and prints a detailed profiling summary.

Usage:
    cd /data/Runs/perchik/ugbio-utils-worktrees/deep-srsnv-improvements
    uv run python src/srsnv/scripts/profile_preprocessing.py
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ugbio_core.logger import logger

WORKSPACE = "/data/Runs/perchik/ppmseq_data/srsnv_training_workspace"
CRAM = f"{WORKSPACE}/inputs/source.cram"
POS_PARQUET = f"{WORKSPACE}/inputs/positive.parquet"
NEG_PARQUET = f"{WORKSPACE}/inputs/negative.parquet"
TRAINING_REGIONS = f"{WORKSPACE}/inputs/training_regions.interval_list.gz"

OUTPUT_ROOT = f"{WORKSPACE}/profile_preprocessing_run"
POS_CACHE = f"{OUTPUT_ROOT}/positive_cache"
NEG_CACHE = f"{OUTPUT_ROOT}/negative_cache"
FOLDS_DIR = f"{OUTPUT_ROOT}/folds"

NUM_WORKERS = max(1, min((os.cpu_count() or 4) - 2, 14))
SHARD_SIZE = 25000
TENSOR_LENGTH = 300
K_FOLDS = 3
HOLDOUT_CHROMS = ["chr21", "chr22"]


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def main():  # noqa: PLR0915
    from ugbio_srsnv.deep_srsnv.combine_splits import combine_and_split
    from ugbio_srsnv.deep_srsnv.cram_to_tensors import cram_to_tensor_cache
    from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("DNN PREPROCESSING PROFILING RUN")
    logger.info("=" * 72)
    logger.info("  CRAM:            %s", CRAM)
    logger.info("  POS parquet:     %s", POS_PARQUET)
    logger.info("  NEG parquet:     %s", NEG_PARQUET)
    logger.info("  Training regions:%s", TRAINING_REGIONS)
    logger.info("  Output:          %s", OUTPUT_ROOT)
    logger.info("  Workers:         %d", NUM_WORKERS)
    logger.info("  Shard size:      %d", SHARD_SIZE)
    logger.info("  Tensor length:   %d", TENSOR_LENGTH)
    logger.info("  K-folds:         %d", K_FOLDS)
    logger.info("  CPUs available:  %d", os.cpu_count() or 0)
    logger.info("=" * 72)

    overall_t0 = time.perf_counter()
    encoders = load_vocab_config()

    # --- Step 1: Positive tensor cache ---
    logger.info("\n>>> STEP 1/3: Building positive tensor cache...")
    step1_t0 = time.perf_counter()
    pos_index = cram_to_tensor_cache(
        cram_path=CRAM,
        parquet_path=POS_PARQUET,
        encoders=encoders,
        output_dir=POS_CACHE,
        label=True,
        tensor_length=TENSOR_LENGTH,
        num_workers=NUM_WORKERS,
        shard_size=SHARD_SIZE,
    )
    step1_wall = time.perf_counter() - step1_t0
    logger.info(
        "Step 1 done: %.1fs, output=%d rows, peak_rss=%.2f GB", step1_wall, pos_index["total_output_rows"], _rss_gb()
    )

    # --- Step 2: Negative tensor cache ---
    logger.info("\n>>> STEP 2/3: Building negative tensor cache...")
    step2_t0 = time.perf_counter()
    neg_index = cram_to_tensor_cache(
        cram_path=CRAM,
        parquet_path=NEG_PARQUET,
        encoders=encoders,
        output_dir=NEG_CACHE,
        label=False,
        tensor_length=TENSOR_LENGTH,
        num_workers=NUM_WORKERS,
        shard_size=SHARD_SIZE,
    )
    step2_wall = time.perf_counter() - step2_t0
    logger.info(
        "Step 2 done: %.1fs, output=%d rows, peak_rss=%.2f GB", step2_wall, neg_index["total_output_rows"], _rss_gb()
    )

    # --- Step 3: Combine + split ---
    logger.info("\n>>> STEP 3/3: Combining caches and splitting into %d folds...", K_FOLDS)
    step3_t0 = time.perf_counter()
    folds_index = combine_and_split(
        positive_cache_dir=POS_CACHE,
        negative_cache_dir=NEG_CACHE,
        training_regions=TRAINING_REGIONS,
        k_folds=K_FOLDS,
        holdout_chromosomes=HOLDOUT_CHROMS,
        random_seed=42,
        output_dir=FOLDS_DIR,
    )
    step3_wall = time.perf_counter() - step3_t0
    logger.info("Step 3 done: %.1fs, peak_rss=%.2f GB", step3_wall, _rss_gb())

    overall_wall = time.perf_counter() - overall_t0

    # --- Summary ---
    logger.info("\n" + "=" * 72)
    logger.info("PROFILING SUMMARY")
    logger.info("=" * 72)
    logger.info(
        "  Step 1 (positive cache):    %8.1fs  (%d -> %d rows, %d missing)",
        step1_wall,
        pos_index["total_input_rows"],
        pos_index["total_output_rows"],
        pos_index["total_missing_rows"],
    )
    logger.info(
        "  Step 2 (negative cache):    %8.1fs  (%d -> %d rows, %d missing)",
        step2_wall,
        neg_index["total_input_rows"],
        neg_index["total_output_rows"],
        neg_index["total_missing_rows"],
    )
    logger.info("  Step 3 (combine + split):   %8.1fs  (%d total rows)", step3_wall, folds_index["total_rows"])
    logger.info("  TOTAL:                      %8.1fs", overall_wall)
    logger.info("")
    logger.info("  Positive throughput:  %s rows/sec", f"{pos_index['profile']['rows_per_second']:,.0f}")
    logger.info("  Negative throughput:  %s rows/sec", f"{neg_index['profile']['rows_per_second']:,.0f}")
    logger.info("  Positive CPU util:    %.1f%%", pos_index["profile"]["cpu_utilization"] * 100)
    logger.info("  Negative CPU util:    %.1f%%", neg_index["profile"]["cpu_utilization"] * 100)
    logger.info("  Peak RSS:             %.2f GB", _rss_gb())
    logger.info("")
    logger.info("  Fold summary:")
    for f in folds_index["fold_summary"]:
        logger.info(
            "    Fold %d: train=%d val=%d test=%d  (pos: %d/%d/%d)",
            f["fold"],
            f["train_rows"],
            f["val_rows"],
            f["test_rows"],
            f["train_positives"],
            f["val_positives"],
            f["test_positives"],
        )
    logger.info("=" * 72)

    summary_path = Path(OUTPUT_ROOT) / "profiling_summary.json"
    summary = {
        "steps": {
            "positive_cache": {"wall_seconds": round(step1_wall, 1), **pos_index["profile"]},
            "negative_cache": {"wall_seconds": round(step2_wall, 1), **neg_index["profile"]},
            "combine_split": {"wall_seconds": round(step3_wall, 1), **folds_index["profile"]},
        },
        "total_wall_seconds": round(overall_wall, 1),
        "peak_rss_gb": round(_rss_gb(), 2),
        "fold_summary": folds_index["fold_summary"],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Profiling summary written to %s", summary_path)


if __name__ == "__main__":
    main()
