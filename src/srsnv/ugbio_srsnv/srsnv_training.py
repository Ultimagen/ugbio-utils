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
import gzip
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pysam
import xgboost as xgb
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_featuremap.filter_dataframe import read_filtering_stats_json

from ugbio_srsnv.srsnv_utils import EPS, MAX_PHRED, all_models_predict_proba, prob_to_phred, safe_roc_auc

FOLD_COL = "fold_id"
LABEL_COL = "label"
CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value
REF = FeatureMapFields.REF.value
X_ALT = FeatureMapFields.X_ALT.value
X_HMER_REF = FeatureMapFields.X_HMER_REF.value
X_HMER_ALT = FeatureMapFields.X_HMER_ALT.value
MQUAL = FeatureMapFields.MQUAL.value
SNVQ = FeatureMapFields.SNVQ.value
SNVQ_RAW = SNVQ + "_RAW"

PROB_ORIG = "prob_orig"
PROB_RECAL = "prob_recal"
PROB_RESCALED = "prob_rescaled"
PROB_TRAIN = "prob_train"
PROB_FOLD_TMPL = "prob_fold_{k}"

EDIT_DIST_FEATURES = [
    FeatureMapFields.EDIST.value,
    FeatureMapFields.HAMDIST.value,
    FeatureMapFields.HAMDIST_FILT.value,
]

pl.enable_string_cache()


# ───────────────────────── parsers ────────────────────────────
def _parse_interval_list_tabix(path: str) -> tuple[dict[str, int], list[str]]:
    # Parse headers for chrom_sizes (must still scan file start)
    chrom_sizes = {}
    with pysam.TabixFile(path) as tbx:
        for line in tbx.header:
            if line.startswith("@SQ"):
                chrom_name = None
                for field in line.strip().split("\t")[1:]:
                    key, val = field.split(":", 1)
                    if key == "SN":
                        chrom_name = val
                    elif key == "LN" and chrom_name is not None:
                        chrom_sizes[chrom_name] = int(val)
        # chroms_in_data: use tabix index for fast chromosome listing
        chroms_in_data = list(tbx.contigs)
    missing = [c for c in chroms_in_data if c not in chrom_sizes]
    if missing:
        raise ValueError(f"Missing @SQ header for contigs: {missing}")
    return chrom_sizes, chroms_in_data


def _parse_interval_list_manual(path: str) -> tuple[dict[str, int], list[str]]:
    """
    Picard/Broad interval-list:
    header lines: '@SQ\tSN:chr1\tLN:248956422'
    data  lines:  'chr1   100  200  +  region1'

    Supports both plain text and gzipped files (detected by .gz extension).

    Returns
    -------
    chrom_sizes : dict[str, int]
    chroms_in_data : list[str]  # preserve original order of appearance
    """
    chrom_sizes: dict[str, int] = {}
    chroms_in_data: list[str] = []

    # Determine if file is gzipped based on extension
    is_gzipped = path.endswith(".gz")

    if is_gzipped:
        fh = gzip.open(path, "rt", encoding="utf-8")
    else:
        fh = open(path, encoding="utf-8")

    try:
        for line in fh:
            if line.startswith("@SQ"):
                chrom_name = None
                for field in line.strip().split("\t")[1:]:
                    key, val = field.split(":", 1)
                    if key == "SN":
                        chrom_name = val
                    elif key == "LN" and chrom_name is not None:
                        chrom_sizes[chrom_name] = int(val)
            elif not line.startswith("@"):
                chrom = line.split("\t", 1)[0]
                if chrom not in chroms_in_data:
                    chroms_in_data.append(chrom)
    finally:
        fh.close()

    missing = [c for c in chroms_in_data if c not in chrom_sizes]
    if missing:
        raise ValueError(f"Missing @SQ header for contigs: {missing}")
    return chrom_sizes, chroms_in_data


def _parse_interval_list(path: str) -> tuple[dict[str, int], list[str]]:
    """
    Parse a Picard/Broad interval-list file to extract chromosome sizes and order.

    Parameters
    ----------
    path : str
        Path to the interval-list file (supports .gz files).

    Returns
    -------
    tuple[dict[str, int], list[str]]
        chrom_sizes: Dictionary mapping chromosome names to their lengths.
        chroms_in_data: List of chromosomes in the order they appear in the file.
    """
    candidate_tbi = path + ".tbi"
    if os.path.exists(candidate_tbi):
        return _parse_interval_list_tabix(path)
    else:
        return _parse_interval_list_manual(path)


def _parse_model_params(mp: str | None) -> dict[str, Any]:
    """
    Accept either a JSON file or a ':'-separated list of key=value tokens.

    Examples
    --------
    --model-params eta=0.1:max_depth=8
    --model-params /path/to/params.json
    """
    if mp is None:
        return {}
    p = Path(mp)
    if p.is_file() and mp.endswith(".json"):
        with p.open(encoding="utf-8") as fh:
            return json.load(fh)

    params: dict[str, Any] = {}
    for token in filter(None, mp.split(":")):  # skip empty segments
        if "=" not in token:
            raise ValueError(f"Invalid model param token '{token}'. Expected key=value.")
        key, val = token.split("=", 1)
        try:
            params[key] = json.loads(val)  # try numeric / bool / null
        except json.JSONDecodeError:
            params[key] = val
    return params


# ───────────────────────── auxiliary functions ──────────────────────────────


def _probability_rescaling(
    prob: np.ndarray,
    sample_prior: float,
    target_prior: float,
    eps: float = EPS,
) -> np.ndarray:
    """
    Rescale probabilities from the training prior to the real-data prior.

    Formula (odds space, no logs):
        odds_row       =  p / (1-p)
        odds_sample    =  π_s / (1-π_s)
        odds_target    =  π_t / (1-π_t)
        odds_rescaled  =  odds_row * (odds_target / odds_sample)
        p_rescaled     =  odds_rescaled / (1.0 + odds_rescaled)
    """
    sample_prior = np.clip(sample_prior, eps, 1 - eps)
    target_prior = np.clip(target_prior, eps, 1 - eps)

    odds_sample = sample_prior / (1.0 - sample_prior)
    odds_target = target_prior / (1.0 - target_prior)

    p = np.clip(prob, eps, 1 - eps)
    odds_row = p / (1.0 - p)

    odds_rescaled = odds_row * (odds_target / odds_sample)
    p_rescaled = odds_rescaled / (1.0 + odds_rescaled)
    # dividing by 3 to get SNVQ score - we are counting all 3 possible errors per base but we want an SNVQ score
    # per specific substitution error
    p_rescaled_snvq = 1 - ((1 - p_rescaled) / 3)

    return p_rescaled_snvq


def partition_into_folds(series_of_sizes, k_folds, alg="greedy", n_chroms_leave_out=1):
    """Returns a partition of the indices of the series series_of_sizes
    into k_fold groups whose total size is approximately the same.
    Returns a dictionary that maps the indices (keys) of series_of_sizes into
    the corresponding fold number (partition).

    If series_of_sizes is a series, then the list-of-lists partitions below satisfies that:
    [series_of_sizes.loc[partitions[k]].sum() for k in range(k_folds)]
    are approximately equal. Conversely,
    series_of_sizes.groupby(indices_to_folds).sum()
    are approximately equal.

    Arguments:
        - series_of_sizes [pd.Series]: a series of indices and their corresponding sizes.
        - k_folds [int]: the number of folds into which series_of_sizes should be partitioned.
        - alg ['greedy']: the algorithm used. For the time being only the greedy algorithm
          is implemented.
        - n_chroms_leave_out [int]: The n_chroms_leave_out smallest chroms are not assigned to any fold (they are
          excluded from the indices_to_folds dict). These are excluded from training all together, and
          are used for test only.
    Returns:
        - indices_to_folds [dict]: a dictionary that maps indices to the corresponding
          fold numbers.
    """
    if alg != "greedy":
        raise ValueError("Only greedy algorithm implemented at this time")
    series_of_sizes = series_of_sizes.sort_values(ascending=False)
    series_of_sizes = series_of_sizes.iloc[
        : series_of_sizes.shape[0] - n_chroms_leave_out
    ]  # Removing the n_test smallest sizes
    partitions = [[] for _ in range(k_folds)]  # an empty partition
    partition_sums = np.zeros(k_folds)  # The running sum of partitions
    for idx, s in series_of_sizes.items():
        min_fold = partition_sums.argmin()
        partitions[min_fold].append(idx)
        partition_sums[min_fold] += s

    # return partitions
    indices_to_folds = [[i for i, prtn in enumerate(partitions) if idx in prtn][0] for idx in series_of_sizes.index]
    return pd.Series(indices_to_folds, index=series_of_sizes.index).to_dict()


def _probability_recalibration(prob_orig: np.ndarray, y_all: np.ndarray) -> np.ndarray:
    """
    Dummy calibration: identity mapping (y = x).
    Keeps the original probabilities unchanged.
    """
    # Simply return the input probabilities unchanged
    return prob_orig.copy()


# ───────────────────────── core logic ─────────────────────────────────────
class SRSNVTrainer:
    def __init__(self, args: argparse.Namespace):
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
        self.prior_real_error = max(
            self.eps,
            min(1.0 - self.eps, neg_after_filter / (neg_after_filter + pos_after_filter)),
        )

        # Data
        logger.debug(
            "Loading data from positive=%s and negative=%s",
            args.positive,
            args.negative,
        )
        self.data_frame = self._load_data(args.positive, args.negative)
        logger.debug("Data loaded. Shape: %s", self.data_frame.shape)

        # training-set prior
        self.n_neg = self.data_frame.filter(~pl.col(LABEL_COL)).height
        self.prior_train_error = self.n_neg / self.data_frame.height

        self.k_folds = max(1, args.k_folds)

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

    # ─────────────────────── data-loading helpers ───────────────────────
    def _read_positive_df(self, pos_path: str) -> pl.DataFrame:
        """Load and massage the positive parquet."""
        logger.debug("Reading positive examples from %s", pos_path)
        pos_df = pl.read_parquet(pos_path)
        logger.debug("Positive examples shape: %s", pos_df.shape)

        if X_ALT not in pos_df.columns:
            raise ValueError(f"{pos_path} is missing required column 'X_ALT'")

        # Replace REF with X_ALT allele
        ref_enum_dtype = pos_df[REF].dtype
        pos_df = pos_df.with_columns(pl.col(X_ALT).cast(ref_enum_dtype).alias(REF)).drop(X_ALT)

        # Swap homopolymer length features (positive set only)
        if {X_HMER_REF, X_HMER_ALT} <= set(pos_df.columns):
            pos_df = (
                pos_df.with_columns(pl.col(X_HMER_REF).alias("__tmp_hmer_ref__"))
                .with_columns(
                    [
                        pl.col(X_HMER_ALT).alias(X_HMER_REF),
                        pl.col("__tmp_hmer_ref__").alias(X_HMER_ALT),
                    ]
                )
                .drop("__tmp_hmer_ref__")
            )
            logger.debug("Swapped X_HMER_REF and X_HMER_ALT in positive dataframe")

        # Remove rows where EDIST == max(EDIST)
        if FeatureMapFields.EDIST.value in pos_df.columns:
            max_edist = pos_df.select(pl.max(FeatureMapFields.EDIST.value)).item()
            pos_df = pos_df.filter(pl.col(FeatureMapFields.EDIST.value) != max_edist)
            logger.debug(
                "Discarded rows with EDIST == max(EDIST)=%s; new shape: %s",
                max_edist,
                pos_df.shape,
            )

        # Increment edit-distance features
        for feat in EDIT_DIST_FEATURES:
            if feat in pos_df.columns:
                pos_df = pos_df.with_columns((pl.col(feat) + 1).alias(feat))
                logger.debug("Incremented feature '%s' by 1 in positive dataframe", feat)

        # Assign label
        pos_df = pos_df.with_columns(pl.lit(value=True).alias(LABEL_COL))
        return pos_df

    def _read_negative_df(self, neg_path: str) -> pl.DataFrame:
        """Load the negative parquet and attach label column."""
        logger.debug("Reading negative examples from %s", neg_path)
        neg_df = pl.read_parquet(neg_path)
        logger.debug("Negative examples shape: %s", neg_df.shape)
        neg_df = neg_df.with_columns(pl.lit(value=False).alias(LABEL_COL))
        if X_ALT in neg_df.columns:
            # drop X_ALT if it exists in negative set
            neg_df = neg_df.drop(X_ALT)
            logger.debug("Dropped X_ALT column from negative dataframe")
        return neg_df

    # ─────────────────────── data & features ────────────────────────────
    def _load_data(self, pos_path: str, neg_path: str) -> pl.DataFrame:
        """Read, validate and concatenate positive/negative dataframes."""
        pos_df = self._read_positive_df(pos_path)
        neg_df = self._read_negative_df(neg_path)

        # assert that both dataframes have the same columns
        if set(pos_df.columns) != set(neg_df.columns):
            raise ValueError(
                f"Positive and negative dataframes have different columns: {pos_df.columns} vs {neg_df.columns}"
            )
        # assert datatypes are compatible
        if pos_df.dtypes != neg_df.dtypes:
            # raise error on the specific incompatible columns
            incompatible = [c for c in pos_df.columns if pos_df[c].dtype != neg_df[c].dtype]
            dtype_strs = [str(pos_df[c].dtype) for c in incompatible]
            raise ValueError(
                f"Incompatible dtypes between Positive and Negative dataframes for columns: {incompatible}\n"
                f"Dtypes: {dtype_strs}"
            )

        logger.debug("Concatenating positive and negative dataframes")
        combined_df = pl.concat([pos_df, neg_df])
        logger.debug("Combined dataframe shape: %s", combined_df.shape)
        return combined_df

    def _feature_columns(self) -> list[str]:
        exclude = {LABEL_COL, FOLD_COL, CHROM, POS}
        all_feats = [c for c in self.data_frame.columns if c not in exclude]
        logger.debug("Found %d features in dataframe (before filtering)", len(all_feats))

        # if user specified a subset → keep intersection (and sanity-check)
        if self.feature_list:
            missing = [f for f in self.feature_list if f not in all_feats]
            if missing:
                raise ValueError(f"Requested feature(s) absent from data: {missing}")
            features_to_use = [f for f in self.feature_list if f in all_feats]
            logger.debug("Using %d user-specified features", len(features_to_use))
            return features_to_use

        logger.debug("Using all %d features", len(all_feats))
        return all_feats

    def _extract_categorical_encodings(self, pd_df: pd.DataFrame, feat_cols: list[str]) -> None:
        """Extract categorical encodings from pandas DataFrame after conversion to categories."""
        logger.debug("Extracting categorical encodings")
        self.categorical_encodings = {}

        for col in feat_cols:
            if pd_df[col].dtype.name == "category":
                categories = pd_df[col].cat.categories
                # map category string → integer code (code is the index in the categories list)
                encoding = {str(cat): idx for idx, cat in enumerate(categories)}
                self.categorical_encodings[col] = encoding
                logger.debug(
                    "Column '%s' has %d categories: %s",
                    col,
                    len(encoding),
                    list(encoding.keys()),
                )

    def _extract_feature_dtypes(self, pd_df: pd.DataFrame, feat_cols: list[str]) -> None:
        """Extract feature data types from pandas DataFrame."""
        logger.debug("Extracting feature data types")
        self.feature_dtypes = {}

        for col in feat_cols:
            dtype_str = str(pd_df[col].dtype)
            self.feature_dtypes[col] = dtype_str
            logger.debug("Column '%s' has dtype: %s", col, dtype_str)

    def _create_quality_lookup_table(self, eps=None) -> None:
        """
        Build an interpolation table that maps MQUAL → SNVQ.
        """
        if eps is None:
            eps = self.eps
        pd_df = self.data_frame.to_pandas()
        mqual = pd_df[MQUAL]
        prior_real_error = self.prior_real_error
        n_pts = self.args.quality_lut_size

        self.x_lut = np.linspace(0.0, mqual.max(), n_pts)
        mqual_t = pd_df[pd_df[LABEL_COL]][MQUAL]
        mqual_f = pd_df[~pd_df[LABEL_COL]][MQUAL]
        tpr = np.array([(mqual_t >= m_).mean() for m_ in self.x_lut])
        fpr = np.array([(mqual_f >= m_).mean() for m_ in self.x_lut])
        self.y_lut = -10 * np.log10(np.clip((prior_real_error / 3) * (fpr / tpr), eps, 1))

    # ─────────────────────── training / prediction ──────────────────────
    def train(self) -> None:
        feat_cols = self._feature_columns()
        logger.info(f"Training with {len(feat_cols)} features")
        logger.debug("Feature columns: %s", feat_cols)

        # ---------- convert Polars → Pandas with categories -------------
        logger.debug("Converting Polars DataFrame to Pandas for training")
        pd_df = self.data_frame.to_pandas()
        logger.debug("Pandas DataFrame shape: %s", pd_df.shape)
        for col in feat_cols:
            if pd_df[col].dtype == object:
                raise ValueError(f"Feature column '{col}' has dtype 'object', expected categorical or numeric.")

        # Extract metadata after categorical conversion
        self._extract_categorical_encodings(pd_df, feat_cols)
        self._extract_feature_dtypes(pd_df, feat_cols)

        fold_arr = pd_df[FOLD_COL].to_numpy()
        y_all = pd_df[LABEL_COL].to_numpy()
        # ----------------------------------------------------------------
        for fold_idx in range(self.k_folds):
            logger.debug("Starting training for fold %d/%d", fold_idx + 1, self.k_folds)
            val_mask = fold_arr == fold_idx
            train_mask = (~val_mask) & ~np.isnan(fold_arr)

            x_train = pd_df.loc[train_mask, feat_cols]
            y_train = y_all[train_mask]
            x_val = pd_df.loc[val_mask, feat_cols]
            y_val = y_all[val_mask]
            logger.debug("Train size: %d, Validation size: %d", len(x_train), len(x_val))

            self.models[fold_idx].fit(
                x_train,
                y_train,
                eval_set=[
                    (x_train, y_train),
                    (x_val, y_val),
                ],
                verbose=10 if self.args.verbose else False,
            )
            auc_val = safe_roc_auc(y_val, self.models[fold_idx].predict_proba(x_val)[:, 1], logger=logger)
            auc_train = safe_roc_auc(y_train, self.models[fold_idx].predict_proba(x_train)[:, 1], logger=logger)
            logger.debug(
                "Finished training for fold %d, AUC: %.4f (validation) / %.4f (training)", fold_idx, auc_val, auc_train
            )

        # ---------- add calibrated quality columns ----------------------
        self._add_quality_columns(pd_df[feat_cols], fold_arr, y_all)

        # ---------- collect training evaluation results -------------------
        logger.debug("Collecting training evaluation results")
        self._save_training_results()

    def _add_quality_columns(self, x_all, fold_arr: np.ndarray, y_all: np.ndarray) -> None:
        """Attach raw / recalibrated probabilities and quality columns."""
        logger.debug("Adding quality columns")
        prob_orig, _, preds_prob = all_models_predict_proba(
            self.models, x_all, fold_arr, max_phred=self.max_qual, return_val_and_train_preds=True
        )
        mqual = prob_to_phred(prob_orig, max_value=self.max_qual)
        logger.debug("Computed original probabilities and MQUAL scores")

        # ------------------------------------------------------------------
        # quality recalibration
        prob_recal = _probability_recalibration(prob_orig, y_all)
        logger.debug("Applied probability recalibration")

        # ------------------------------------------------------------------
        # global rescaling to the real-data prior
        prob_rescaled = _probability_rescaling(
            prob_recal,
            sample_prior=1 - self.prior_train_error,  # prior of a true call from training data
            target_prior=1 - self.prior_real_error,  # prior of a true call from real data
            eps=self.eps,
        )
        logger.debug("Completed probability rescaling to real-data prior")

        # attach new columns ------------------------------------------------
        logger.debug("Attaching per-fold probabilities and intermediate columns")
        new_cols = [pl.Series(PROB_FOLD_TMPL.format(k=k), preds_prob[k]) for k in range(self.k_folds)] + [
            pl.Series(PROB_ORIG, prob_orig),
            pl.Series(PROB_RECAL, prob_recal),
            pl.Series(PROB_RESCALED, prob_rescaled),
            pl.Series(MQUAL, mqual),
        ]

        self.data_frame = self.data_frame.with_columns(new_cols)

        # final quality (Phred)
        # snvq_raw = prob_to_phred(prob_rescaled, max_value=self.max_qual)
        self._create_quality_lookup_table()
        logger.debug("Created MQUAL→SNVQ lookup table")
        snvq = np.interp(mqual, self.x_lut, self.y_lut)
        logger.debug("Interpolated SNVQ values from lookup table")
        # ------------------------------------------------------------------

        # attach new column ------------------------------------------------
        new_cols = [pl.Series(SNVQ, snvq)]

        self.data_frame = self.data_frame.with_columns(new_cols)
        logger.debug("Finished adding quality columns")

    def _save_training_results(self) -> None:
        """Save training evaluation results for later use in reporting."""
        self.training_results = []
        for fold_idx, model in enumerate(self.models):
            eval_result = model.evals_result()
            self.training_results.append(eval_result)
            logger.debug("Saved training results for fold %d", fold_idx)

    # ───────────────────────── save outputs ─────────────────────────────
    def save(self) -> None:
        logger.debug("Saving outputs to %s", self.out_dir)
        base = (
            (self.args.basename + ".")
            if self.args.basename and not self.args.basename.endswith(".")
            else self.args.basename
        )
        df_path = self.out_dir / f"{base}featuremap_df.parquet"
        logger.debug("Saving dataframe to %s", df_path)
        self.data_frame.to_pandas().to_parquet(df_path)
        logger.info(f"Saved dataframe → {df_path}")

        # models – JSON, one file per fold
        model_paths: dict[int, str] = {}
        for fold_idx, model in enumerate(self.models):
            path = self.out_dir / f"{base}model_fold_{fold_idx}.json"
            model.save_model(path)
            model_paths[fold_idx] = str(path)
            logger.info("Saved model for fold %d → %s", fold_idx, path)

        # map every chromosome to the model-file basename instead of the fold index
        chrom_to_model_file = {chrom: Path(model_paths[fold]).name for chrom, fold in self.chrom_to_fold.items()}

        metadata_path = self.out_dir / f"{base}srsnv_metadata.json"
        logger.debug("Saving metadata to %s", metadata_path)

        # merge dtype + categorical encoding into one list
        features_meta = []
        for feat, dtype in self.feature_dtypes.items():
            if feat in self.categorical_encodings:
                entry_type = "c"
            else:
                dt = dtype.lower()
                if dt.startswith("int"):
                    entry_type = "int"
                elif dt.startswith("float"):
                    entry_type = "float"
                else:
                    entry_type = dt  # keep as-is for other dtypes

            entry = {"name": feat, "type": entry_type}
            if feat in self.categorical_encodings:
                entry["values"] = self.categorical_encodings[feat]
            features_meta.append(entry)

        # quality recalibration table
        quality_recalibration_table = [
            self.x_lut.tolist(),
            self.y_lut.tolist(),
        ]

        # stats and priors
        stats = {
            "positive": self.pos_stats,
            "negative": self.neg_stats,
            "prior_train_error": self.prior_train_error,
            "prior_real_error": self.prior_real_error,
        }

        # Save comprehensive metadata
        metadata = {
            "model_paths": model_paths,
            "training_results": self.training_results,
            "chrom_to_model": chrom_to_model_file,
            "features": features_meta,
            "quality_recalibration_table": quality_recalibration_table,
            "filtering_stats": stats,
            "model_params": self.model_params,
            "training_parameters": {"max_qual": self.max_qual},
            "metadata": self.user_metadata,
        }

        with metadata_path.open("w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info(f"Saved metadata → {metadata_path}")
        logger.info(
            "Metadata includes %d chromosome to model mappings and %d features",
            len(self.chrom_to_fold),
            len(features_meta),
        )

    # ───────────────────────── entry point ──────────────────────────────
    def run(self) -> None:
        logger.debug("Starting SRSNVTrainer.run()")
        self.train()
        self.save()
        logger.debug("Finished SRSNVTrainer.run()")


# ───────────────────────── CLI helpers ────────────────────────────────────
def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train SingleReadSNV classifier", allow_abbrev=True)
    ap.add_argument("--positive", required=True, help="Parquet with label=1 rows")
    ap.add_argument("--negative", required=True, help="Parquet with label=0 rows")
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
    trainer = SRSNVTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
