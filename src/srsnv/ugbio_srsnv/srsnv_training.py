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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_featuremap.filter_dataframe import read_filtering_stats_json

FOLD_COL = "fold_id"
LABEL_COL = "label"
CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value
REF = FeatureMapFields.REF.value
X_ALT = FeatureMapFields.X_ALT.value
MQUAL = FeatureMapFields.MQUAL.value
SNVQ = FeatureMapFields.SNVQ.value
SNVQ_RAW = SNVQ + "_RAW"

PROB_TRAIN_FOLD_TMPL = "prob_train_fold_{k}"
PROB_ORIG = "prob_orig"
PROB_RECAL = "prob_recal"
PROB_RESCALED = "prob_rescaled"
EPS = 1e-10  # small value to avoid division by zero
pl.enable_string_cache()


# ───────────────────────── parsers ────────────────────────────
def _parse_interval_list(path: str) -> tuple[dict[str, int], list[str]]:
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
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Rescale probabilities from the training prior to the real-data prior.

    Formula (odds space, no logs):
        odds_row       =  p / (1-p)
        odds_sample    =  π_s / (1-π_s)
        odds_target    =  π_t / (1-π_t)
        odds_rescaled  =  odds_row * (odds_target / odds_sample)
        p_rescaled     =  odds_rescaled / (1 + odds_rescaled)
    """
    sample_prior = np.clip(sample_prior, eps, 1 - eps)
    target_prior = np.clip(target_prior, eps, 1 - eps)

    odds_sample = sample_prior / (1.0 - sample_prior)
    odds_target = target_prior / (1.0 - target_prior)

    p = np.clip(prob, eps, 1 - eps)
    odds_row = p / (1.0 - p)

    odds_rescaled = odds_row * (odds_target / odds_sample)
    return odds_rescaled / (1.0 + odds_rescaled)


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
        - n_test [int]: The n_test smallest chroms are not assigned to any fold (they are excluded
            from the indices_to_folds dict). These are excluded from training all together, and
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


def prob_to_phred(prob, max_value=None):
    """Transform probabilities to phred scores.
    Arguments:
    - prob [np.ndarray]: array of probabilities
    - max_value [float]: maximum phred score (clips values above this threshold)
    """
    phred_scores = -10 * np.log10(1 - prob)
    if max_value is not None:
        return np.minimum(phred_scores, max_value)
    return phred_scores


def _aggregate_probabilities_from_folds(prob_matrix: np.ndarray, prob_epsilon=1e-10) -> np.ndarray:
    """
    Aggregate probabilities coming from all folds for each data-point.

    Parameters
    ----------
    prob_matrix : np.ndarray
        Shape = (n_folds, n_rows). Each row contains the predicted
        probabilities of one fold for all data-points.
    prob_epsilon : float, optional
        Small value to avoid log(0) issues, by default 1e-10

    Returns
    -------
    np.ndarray
        Aggregated probability per data-point.
    """
    probs = np.clip(prob_matrix, prob_epsilon, 1 - prob_epsilon)

    # logit base-10
    logits = np.log10(probs / (1.0 - probs))

    # Average logits and convert back to probability
    logits_mean = logits.mean(axis=0)
    return 1.0 / (1.0 + 10 ** (-logits_mean))


def _probability_recalibration(
    prob_orig: np.ndarray, y_all: np.ndarray, fold_arr: np.ndarray, k_folds: int
) -> np.ndarray:
    """
    Perform per-fold isotonic calibration.
    Rows with fold_id = k are calibrated using a model trained on all
    other folds. Rows with NaN fold_id are calibrated using a model
    trained on all rows that have a valid fold id.
    """
    n_rows = prob_orig.shape[0]
    prob_recal = np.empty(n_rows, dtype=float)

    valid_mask = ~np.isnan(fold_arr)
    for k in range(k_folds):
        val_mask = valid_mask & (fold_arr == k)
        train_mask = valid_mask & (fold_arr != k)
        if np.any(train_mask) and np.any(val_mask):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(prob_orig[train_mask], y_all[train_mask])
            prob_recal[val_mask] = iso.predict(prob_orig[val_mask])

    # rows lacking a fold_id
    nan_mask = ~valid_mask
    if np.any(nan_mask) and np.any(valid_mask):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob_orig[valid_mask], y_all[valid_mask])
        prob_recal[nan_mask] = iso.predict(prob_orig[nan_mask])

    return prob_recal


def _create_quality_lookup_table(
    mqual: np.ndarray,
    snvq: np.ndarray,
    n_pts: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an interpolation table that maps MQUAL → SNVQ.

    Returns (x, y):
        x – equidistant MQUAL grid (length n_pts)
        y – interpolated SNVQ on that grid
    """
    mask = np.isfinite(mqual) & np.isfinite(snvq)
    if mask.sum() == 0:
        raise ValueError("no finite data to build quality-lookup table")

    m = mqual[mask]
    s = snvq[mask]

    order = np.argsort(m)
    m, s = m[order], s[order]

    x = np.linspace(0.0, m.max(), n_pts)
    y = np.interp(x, m, s)
    return x, y


# ───────────────────────── core logic ─────────────────────────────────────
class SRSNVTrainer:
    def __init__(self, args: argparse.Namespace):
        logger.debug("Initializing SRSNVTrainer")
        self.args = args
        self.out_dir = Path(args.output)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # RNG
        self.seed = args.random_seed or int(datetime.now().timestamp())
        logger.debug("Using random seed: %d", self.seed)
        self.rng = np.random.default_rng(self.seed)

        # ─────────── read filtering-stats JSONs & compute priors ───────────
        self.pos_stats = read_filtering_stats_json(args.stats_positive)
        self.neg_stats = read_filtering_stats_json(args.stats_negative)

        # helper: last entry that is *not* a down-sample operation
        def _last_non_downsample_rows(stats: dict) -> int:
            for f in reversed(stats["filters"]):
                if f.get("type") != "downsample":
                    return f["rows"]
            raise ValueError("stats JSON has no non-downsample filter entry")

        n_random_sample = self.pos_stats["filters"][0]["rows"]
        n_after_all_filter = _last_non_downsample_rows(self.pos_stats)
        self.r_true_calls = n_after_all_filter / n_random_sample

        self.n_error_calls = _last_non_downsample_rows(self.neg_stats)
        self.n_aligned_bases = args.aligned_bases

        # Data
        logger.debug("Loading data from positive=%s and negative=%s", args.positive, args.negative)
        self.data_frame = self._load_data(args.positive, args.negative)
        logger.debug("Data loaded. Shape: %s", self.data_frame.shape)

        # training-set prior
        self.n_neg = self.data_frame.filter(~pl.col(LABEL_COL)).height
        self.prior_train_error = self.n_neg / self.data_frame.height

        # real-data prior
        self.prior_real_error = max(
            EPS,
            min(1.0 - EPS, self.n_error_calls / (self.n_aligned_bases * self.r_true_calls)),
        )

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
        logger.debug("Initializing %d XGBClassifier models with params: %s", self.k_folds, self.model_params)
        self.models = [xgb.XGBClassifier(**self.model_params) for _ in range(self.k_folds)]

        # optional user-supplied feature subset
        self.feature_list: list[str] | None = args.features.split(":") if args.features else None
        logger.debug("Feature list from user: %s", self.feature_list)

        # Initialize containers for metadata
        self.categorical_encodings: dict[str, dict[str, int]] = {}
        self.feature_dtypes: dict[str, str] = {}

    # ─────────────────────── data & features ────────────────────────────
    def _load_data(self, pos_path: str, neg_path: str) -> pl.DataFrame:
        """Read parquet ⇒ polars, massage positive/negative frames and concatenate."""
        # --- positive -------------------------------------------------------
        logger.debug("Reading positive examples from %s", pos_path)
        pos_df = pl.read_parquet(pos_path)
        logger.debug("Positive examples shape: %s", pos_df.shape)

        if X_ALT not in pos_df.columns:
            raise ValueError(f"{pos_path} is missing required column 'X_ALT'")

        # Override REF with X_ALT and drop X_ALT
        pos_df = pos_df.with_columns(pl.col(X_ALT).alias(REF)).drop(X_ALT)
        pos_df = pos_df.with_columns(pl.lit(value=True).alias(LABEL_COL))

        # --- negative -------------------------------------------------------
        logger.debug("Reading negative examples from %s", neg_path)
        neg_df = pl.read_parquet(neg_path)
        logger.debug("Negative examples shape: %s", neg_df.shape)
        neg_df = neg_df.with_columns(pl.lit(value=False).alias(LABEL_COL))

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
        # Concatenate the two dataframes
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
                logger.debug("Column '%s' has %d categories: %s", col, len(encoding), list(encoding.keys()))

    def _extract_feature_dtypes(self, pd_df: pd.DataFrame, feat_cols: list[str]) -> None:
        """Extract feature data types from pandas DataFrame."""
        logger.debug("Extracting feature data types")
        self.feature_dtypes = {}

        for col in feat_cols:
            dtype_str = str(pd_df[col].dtype)
            self.feature_dtypes[col] = dtype_str
            logger.debug("Column '%s' has dtype: %s", col, dtype_str)

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
            logger.debug("Starting training for fold %d/%d", fold_idx, self.k_folds)
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
            )
            logger.debug("Finished training for fold %d", fold_idx)

        logger.debug("Adding quality columns post-training")
        # ---------- add calibrated quality columns ----------------------
        self._add_quality_columns(pd_df[feat_cols], fold_arr, y_all)

    def _add_quality_columns(self, x_all, fold_arr: np.ndarray, y_all: np.ndarray) -> None:
        """Attach raw / recalibrated probabilities and quality columns."""
        n_rows = len(x_all)
        preds_prob = np.empty((self.k_folds, n_rows), dtype=float)
        preds_phred = np.empty_like(preds_prob)

        # predictions from all models
        for k, model in enumerate(self.models):
            prob = model.predict_proba(x_all)[:, 1]
            preds_prob[k] = prob
            preds_phred[k] = prob_to_phred(prob, max_value=self.args.max_qual)

        # ---------- Vectorised prob_orig -------------------
        prob_orig = np.empty(n_rows, dtype=float)

        valid_fold_mask = ~np.isnan(fold_arr)
        idx_valid = np.where(valid_fold_mask)[0]
        idx_nan = np.where(~valid_fold_mask)[0]

        # Per-row probability coming from its own fold
        prob_orig[idx_valid] = preds_prob[fold_arr[idx_valid].astype(int), idx_valid]

        # Aggregated probability for rows without a fold
        if idx_nan.size:
            agg_prob = _aggregate_probabilities_from_folds(preds_prob[:, idx_nan])
            prob_orig[idx_nan] = agg_prob
        mqual = prob_to_phred(prob_orig, max_value=self.args.max_qual)

        # ------------------------------------------------------------------
        # quality recalibration
        prob_recal = _probability_recalibration(prob_orig, y_all, fold_arr, self.k_folds)

        # ------------------------------------------------------------------
        # global rescaling to the real-data prior
        prob_rescaled = _probability_rescaling(
            prob_recal,
            sample_prior=self.prior_train_error,
            target_prior=self.prior_real_error,
        )

        # final quality (Phred)
        snvq_raw = prob_to_phred(prob_rescaled, max_value=self.args.max_qual)
        self.x_lut, self.y_lut = _create_quality_lookup_table(mqual, snvq_raw, self.args.quality_lut_size)
        snvq = np.interp(mqual, self.x_lut, self.y_lut)
        # ------------------------------------------------------------------

        # attach new columns ------------------------------------------------
        new_cols = [
            pl.Series(PROB_ORIG, prob_orig),
            pl.Series(PROB_RECAL, prob_recal),
            pl.Series(PROB_RESCALED, prob_rescaled),
            pl.Series(MQUAL, mqual),
            pl.Series(SNVQ_RAW, snvq_raw),
            pl.Series(SNVQ, snvq),
        ]

        self.data_frame = self.data_frame.with_columns(new_cols)

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
        self.data_frame.write_parquet(df_path)
        logger.info(f"Saved dataframe → {df_path}")

        # models – JSON, one file per fold
        model_paths: dict[int, str] = {}
        for fold_idx, model in enumerate(self.models):
            path = self.out_dir / f"{base}model_fold_{fold_idx}.json"
            model.save_model(path)
            model_paths[fold_idx] = str(path)

        # map every chromosome to the model-file basename instead of the fold index
        chrom_to_model_file = {chrom: Path(model_paths[fold]).name for chrom, fold in self.chrom_to_fold.items()}

        metadata_path = self.out_dir / f"{base}srsnv_metadata.json"
        logger.debug("Saving comprehensive metadata to %s", metadata_path)

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
            "r_true_calls": self.r_true_calls,
            "n_error_calls": self.n_error_calls,
            "n_aligned_bases": self.n_aligned_bases,
            "prior_train_error": self.prior_train_error,
            "prior_real_error": self.prior_real_error,
        }

        # Save comprehensive metadata
        metadata = {
            "chrom_to_model": chrom_to_model_file,
            "features": features_meta,
            "quality_recalibration_table": quality_recalibration_table,
            "filtering_stats": stats,
            "model_params": self.model_params,
        }

        with metadata_path.open("w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info(f"Saved comprehensive metadata → {metadata_path}")
        logger.info(
            "Metadata includes %d chromosome mappings and %d features", len(self.chrom_to_fold), len(features_meta)
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
    ap.add_argument("--training-regions", required=True, help="Picard interval_list file (supports .gz files)")
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
    ap.add_argument("--max-qual", type=float, default=100.0, help="Maximum Phred score for model quality")
    ap.add_argument("--stats-positive", required=True, help="JSON file with filtering stats for positive set")
    ap.add_argument("--stats-negative", required=True, help="JSON file with filtering stats for negative set")
    ap.add_argument("--aligned-bases", type=int, required=True, help="Number of aligned bases for prior calculation")
    ap.add_argument(
        "--quality-lut-size",
        type=int,
        default=1000,
        help="Number of points in the MQUAL→SNVQ lookup table " "(default 1000)",
    )
    return ap.parse_args()


# ───────────────────────── main entry point ──────────────────────────────
def main() -> None:
    args = _cli()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    trainer = SRSNVTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
