#!/env/python
# Copyright 2023 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Run single read SNV quality recalibration training
# CHANGELOG in reverse chronological order


from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from ugbio_core.logger import logger
from ugbio_featuremap.filter_dataframe import read_filtering_stats_json

from ugbio_srsnv.srsnv_training import (
    CHROM,
    FOLD_COL,
    LABEL_COL,
    MQUAL,
    PROB_FOLD_TMPL,
    PROB_ORIG,
    PROB_RECAL,
    PROB_RESCALED,
    SNVQ,
    SRSNVTrainer,
    _parse_interval_list,
    _parse_model_params,
    count_bases_in_interval_list,
    partition_into_folds,
)
from ugbio_srsnv.srsnv_utils import (
    MAX_PHRED,
    set_featuremap_df_dtypes,
)

pl.enable_string_cache()


# ───────────────────────── core logic ─────────────────────────────────────
class SRSNVTrainerFromFeatureMapDF(SRSNVTrainer):
    def __init__(self, args: argparse.Namespace):  # noqa: C901, PLR0915
        logger.debug("Initializing SRSNVTrainer")
        self.args = args
        self.out_dir = Path(args.output)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_qual = self.args.max_qual if self.args.max_qual is not None else MAX_PHRED
        self.eps = 10 ** (-self.max_qual / 10)  # small value to avoid division by zero

        # RNG
        self.seed = args.random_seed or int(datetime.now().timestamp())
        logger.debug("Using random seed: %d", self.seed)
        self.rng = np.random.default_rng(self.seed)

        # ─────────── read filtering-stats JSONs & compute priors ───────────
        self.pos_stats = read_filtering_stats_json(args.stats_positive)
        self.neg_stats = read_filtering_stats_json(args.stats_negative)
        self.raw_stats = read_filtering_stats_json(args.stats_featuremap)
        self.mean_coverage = args.mean_coverage  # TODO: Check what to do with multiple inputs
        if self.mean_coverage is None:
            raise ValueError("--mean-coverage is required if not present in stats-featuremap JSON")
        self.n_bases_in_region = count_bases_in_interval_list(args.training_regions)

        # sanity-check: identical “quality/region” filters in the two random-sample stats files
        def _quality_region_filters(st):
            return [f for f in st["filters"] if f.get("type") in {"quality", "region"}]

        pos_qr = _quality_region_filters(self.pos_stats)
        neg_qr = _quality_region_filters(self.neg_stats)
        if pos_qr != neg_qr:
            raise ValueError(
                "Mismatch between quality/region filters of "
                "--stats-positive and --stats-negative:\n"
                f" positive={pos_qr}\n negative={neg_qr}"
            )

        # helper: last entry that is *not* a down-sample operation
        def _last_non_downsample_rows(stats: dict) -> int:
            for f in reversed(stats["filters"]):
                if f.get("type") != "downsample":
                    return f["rows"]
            raise ValueError("stats JSON has no non-downsample filter entry")

        # new prior_real_error calculation
        pos_after_filter = _last_non_downsample_rows(self.pos_stats)
        neg_after_filter = _last_non_downsample_rows(self.neg_stats)
        raw_after_filter = _last_non_downsample_rows(self.raw_stats)
        self.prior_real_error = max(
            self.eps,
            min(1.0 - self.eps, neg_after_filter / (neg_after_filter + pos_after_filter)),
        )
        self.raw_featuremap_size_filtered = raw_after_filter

        self.k_folds = max(1, args.k_folds)

        # Data # TODO: Do I need to load here?
        logger.debug(
            "Loading data from featuremap_df=%s",
            args.featuremap_df,
        )
        self.data_frame = self._load_data_from_featuremap_df(args.featuremap_df)
        logger.debug("Data loaded. Shape: %s", self.data_frame.shape)

        # training-set prior
        self.n_neg = self.data_frame.filter(~pl.col(LABEL_COL)).height
        self.prior_train_error = self.n_neg / self.data_frame.height

        # Folds
        logger.debug("Parsing interval list from %s", args.training_regions)
        chrom_sizes, chrom_list = _parse_interval_list(args.training_regions)
        # partition_into_folds expects a pandas Series
        logger.debug("Partitioning %d chromosomes into %d folds", len(chrom_list), self.k_folds)
        self.chrom_to_fold: dict[str, int] = partition_into_folds(
            pd.Series({c: chrom_sizes[c] for c in chrom_list}),
            self.k_folds,
            n_chroms_leave_out=1,
        )
        logger.debug("Assigning folds to data")
        self.data_frame = self.data_frame.with_columns(
            pl.col(CHROM).map_elements(lambda c: self.chrom_to_fold.get(c), return_dtype=pl.Int64).alias(FOLD_COL)
        )
        logger.debug("Fold assignment complete")

        # Models
        logger.debug("Parsing model parameters from: %s", args.model_params)
        self.model_params = _parse_model_params(args.model_params)
        logger.debug(
            "Initializing %d XGBClassifier models with params: %s",
            self.k_folds,
            self.model_params,
        )
        self.models = [xgb.XGBClassifier(**self.model_params) for _ in range(self.k_folds)]

        # optional user-supplied feature subset
        self.feature_list: list[str] | None = args.features.split(":") if args.features else None
        logger.debug("Feature list from user: %s", self.feature_list)
        if self.feature_list is not None and self.feature_list_from_dtypes is not None:
            # ensure that the two lists are the same, in the same order
            if self.feature_list != self.feature_list_from_dtypes:
                raise ValueError("--features list does not match features in --features_dtypes_json")
        elif self.feature_list is None and self.feature_list_from_dtypes is not None:
            self.feature_list = self.feature_list_from_dtypes
            logger.debug("Using feature list from --features_dtypes_json: %s", self.feature_list)

        # Initialize containers for metadata
        self.categorical_encodings: dict[str, dict[str, int]] = {}
        self.feature_dtypes: dict[str, str] = {}

        # ─────────── user-supplied metadata ───────────
        self.user_metadata: dict[str, str] = {}
        for token in args.metadata or []:
            # Require exactly one '=' so that key and value are unambiguous
            if token.count("=") != 1:
                raise ValueError(f"--metadata token '{token}' must contain exactly one '=' (key=value)")
            k, v = token.split("=", 1)
            self.user_metadata[k] = v
        logger.debug("Parsed user metadata: %s", self.user_metadata)

    # ─────────────────────── data & features ────────────────────────────
    def _load_data_from_featuremap_df(
        self, featuremap_df_path: str, features_dtypes_json: str | None = None
    ) -> pl.DataFrame:
        """Read, validate and concatenate positive/negative dataframes."""
        featuremap_df = pl.read_parquet(featuremap_df_path)

        quality_columns = [PROB_FOLD_TMPL.format(k=k) for k in range(self.k_folds)] + [
            PROB_ORIG,
            PROB_RECAL,
            PROB_RESCALED,
            MQUAL,
            SNVQ,
        ]
        quality_columns_dict = {col: col + "_original" for col in quality_columns if col in featuremap_df.columns}
        featuremap_df = featuremap_df.rename(quality_columns_dict)
        self.feature_list_from_dtypes = None
        if features_dtypes_json is not None:
            with open(features_dtypes_json) as f:
                feature_dtypes = json.load(f)
            self.feature_list_from_dtypes = [f["name"] for f in feature_dtypes]
            featuremap_df = set_featuremap_df_dtypes(featuremap_df, feature_dtypes)
        return featuremap_df


# ───────────────────────── CLI helpers ────────────────────────────────────
def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train SingleReadSNV classifier", allow_abbrev=True)
    ap.add_argument("--featuremap_df", required=True, help="Parquet with labelled featuremap ready for training")
    ap.add_argument(
        "--training-regions",
        required=True,
        help="Picard interval_list file (supports .gz files)",
    )
    ap.add_argument("--k-folds", type=int, default=1, help="Number of CV folds (≥1)")
    ap.add_argument(
        "--model-params",
        help="XGBoost params as key=value tokens separated by ':' "
        "(e.g. 'eta=0.1:max_depth=8') or a path to a JSON file",
    )
    ap.add_argument("--features_dtypes_json", help="JSON file with feature dtypes", default=None)
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--basename", default="", help="Basename prefix for outputs")
    ap.add_argument(
        "--features",
        help="Colon-separated list of feature columns to use "
        "(e.g. 'X_HMER_REF:X_HMER_ALT:RAW_VAF') – if omitted, use all",
    )
    ap.add_argument("--random-seed", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")
    ap.add_argument(
        "--max-qual",
        type=float,
        default=100.0,
        help="Maximum Phred score for model quality",
    )
    ap.add_argument(
        "--stats-positive",
        required=True,
        help="JSON file with filtering stats for positive random-sample set",
    )
    ap.add_argument(
        "--stats-negative",
        required=True,
        help="JSON file with filtering stats for negative random-sample set",
    )
    ap.add_argument(
        "--stats-featuremap",
        required=True,
        help="JSON file with filtering stats for raw featuremap",
    )
    ap.add_argument(
        "--mean-coverage",
        type=float,
        help="Mean coverage of the sample",
    )
    ap.add_argument(
        "--quality-lut-size",
        type=int,
        default=1000,
        help="Number of points in the MQUAL→SNVQ lookup table " "(default 1000)",
    )
    ap.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Additional metadata key=value pairs (can be given multiple times)",
    )
    return ap.parse_args()


# ───────────────────────── main entry point ──────────────────────────────
def main() -> None:
    args = _cli()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    trainer = SRSNVTrainerFromFeatureMapDF(args)
    trainer.run()


if __name__ == "__main__":
    main()
