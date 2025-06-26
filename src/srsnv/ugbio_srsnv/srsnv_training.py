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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields

FOLD_COL = "fold_id"
LABEL_COL = "label"
CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value
REF = FeatureMapFields.REF.value
X_ALT = FeatureMapFields.X_ALT.value


RAW_QUAL_VAL = "raw_qual_val"
RAW_QUAL_TRAIN_TMPL = "raw_qual_train_{idx}"  # idx = 1 … k-1


# ───────────────────────── parsers ────────────────────────────
def _parse_interval_list(path: str) -> tuple[dict[str, int], list[str]]:
    """
    Picard/Broad interval-list:
    header lines: '@SQ\tSN:chr1\tLN:248956422'
    data  lines:  'chr1   100  200  +  region1'

    Returns
    -------
    chrom_sizes : dict[str, int]
    chroms_in_data : list[str]  # preserve original order of appearance
    """
    chrom_sizes: dict[str, int] = {}
    chroms_in_data: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("@SQ"):
                for field in line.strip().split("\t")[1:]:
                    key, val = field.split(":", 1)
                    if key == "SN":
                        chrom_name = val
                    elif key == "LN":
                        chrom_sizes[chrom_name] = int(val)
            elif not line.startswith("@"):
                chrom = line.split("\t", 1)[0]
                if chrom not in chroms_in_data:
                    chroms_in_data.append(chrom)
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


def partition_into_folds(series_of_sizes, k_folds, alg="greedy", n_test=0):
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
    series_of_sizes = series_of_sizes.iloc[: series_of_sizes.shape[0] - n_test]  # Removing the n_test smallest sizes
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
    - eps [float]: cutoff value (phred values can have maximum value of -10*np.log10(eps)
                   which for the default value 1e-8 means phred of 80)
    """
    if max_value is not None:
        return min(-10 * np.log10(1 - prob), max_value)
    return -10 * np.log10(1 - prob)


# ───────────────────────── core logic ─────────────────────────────────────
class SRSNVTrainer:  # renamed from SRTrainer
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.output)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # RNG
        self.seed = args.random_seed or int(datetime.now().timestamp())
        self.rng = np.random.default_rng(self.seed)

        # Data
        self.data_frame = self._load_data(args.positive, args.negative)
        self.k_folds = max(1, args.k_folds)

        # Folds
        chrom_sizes, chrom_list = _parse_interval_list(args.training_regions)
        # partition_into_folds expects a pandas Series
        self.chrom_to_fold = partition_into_folds(
            pd.Series({c: chrom_sizes[c] for c in chrom_list}),
            self.k_folds,
            n_test=0,
        )
        self.data_frame = self.data_frame.with_columns(
            pl.col(CHROM).map_elements(lambda c: self.chrom_to_fold.get(c, float("nan"))).alias(FOLD_COL)
        )

        # Models
        model_params = _parse_model_params(args.model_params)
        self.models = [xgb.XGBClassifier(**model_params) for _ in range(self.k_folds)]

        # optional user-supplied feature subset
        self.feature_list: list[str] | None = args.features.split(":") if args.features else None

    # ─────────────────────── data & features ────────────────────────────
    def _load_data(self, pos_path: str, neg_path: str) -> pl.DataFrame:
        """Read parquet ⇒ polars, massage positive/negative frames and concatenate."""
        # --- positive -------------------------------------------------------
        pos_df = pl.read_parquet(pos_path)

        if X_ALT not in pos_df.columns:
            raise ValueError(f"{pos_path} is missing required column 'X_ALT'")

        # Override REF with X_ALT and drop X_ALT
        pos_df = pos_df.with_columns(pl.col(X_ALT).alias(REF)).drop(X_ALT)
        pos_df = pos_df.with_columns(pl.lit(value=True).alias(LABEL_COL))

        # --- negative -------------------------------------------------------
        neg_df = pl.read_parquet(neg_path)
        neg_df = neg_df.with_columns(pl.lit(value=False).alias(LABEL_COL))

        combined_df = pl.concat([pos_df, neg_df])
        return combined_df

    def _feature_columns(self) -> list[str]:
        exclude = {LABEL_COL, FOLD_COL, CHROM, POS}
        all_feats = [c for c in self.data_frame.columns if c not in exclude]

        # if user specified a subset → keep intersection (and sanity-check)
        if self.feature_list:
            missing = [f for f in self.feature_list if f not in all_feats]
            if missing:
                raise ValueError(f"Requested feature(s) absent from data: {missing}")
            return [f for f in self.feature_list if f in all_feats]

        return all_feats

    # ─────────────────────── training / prediction ──────────────────────
    def train(self) -> None:
        feat_cols = self._feature_columns()
        logger.info(f"Training with {len(feat_cols)} features")

        # ---------- convert Polars → Pandas with categories -------------
        pd_df = self.data_frame.to_pandas()
        print(pd_df.columns)
        for col in feat_cols:
            if pd_df[col].dtype == object:
                pd_df[col] = pd_df[col].astype("category")

        fold_arr = pd_df[FOLD_COL].to_numpy()
        y_all = pd_df[LABEL_COL].to_numpy()

        # ----------------------------------------------------------------
        for fold_idx in range(self.k_folds):
            val_mask = fold_arr == fold_idx
            train_mask = (~val_mask) & ~np.isnan(fold_arr)

            x_train = pd_df.loc[train_mask, feat_cols]
            y_train = y_all[train_mask]
            x_val = pd_df.loc[val_mask, feat_cols]
            y_val = y_all[val_mask]

            self.models[fold_idx].fit(
                x_train,
                y_train,
                eval_set=[
                    (x_train, y_train),
                    (x_val, y_val),
                ],
            )

        self._add_quality_columns(pd_df[feat_cols], fold_arr)  # pass DataFrame, not ndarray

    def _add_quality_columns(self, x_all, fold_arr: np.ndarray) -> None:
        """x_all is a pandas DataFrame with feature columns."""
        n_rows = len(x_all)
        preds_phred = np.empty((self.k_folds, n_rows), dtype=float)

        for k, model in enumerate(self.models):
            prob = model.predict_proba(x_all)[:, 1]
            preds_phred[k] = prob_to_phred(prob)

        # validation qual
        fold_idx_int = np.where(np.isnan(fold_arr), 0, fold_arr.astype(int))
        raw_qual_val = preds_phred[fold_idx_int, np.arange(n_rows)]

        # training quals
        train_quals = np.full((n_rows, self.k_folds - 1), np.nan, dtype=float)
        for row in range(n_rows):
            col = 0
            for k in range(self.k_folds):
                if k == fold_idx_int[row]:
                    continue
                train_quals[row, col] = preds_phred[k, row]
                col += 1

        # attach new columns
        new_cols = [pl.Series(RAW_QUAL_VAL, raw_qual_val)]
        for idx in range(self.k_folds - 1):
            new_cols.append(pl.Series(RAW_QUAL_TRAIN_TMPL.format(idx=idx + 1), train_quals[:, idx]))
        self.data_frame = self.data_frame.with_columns(new_cols)

    # ───────────────────────── save outputs ─────────────────────────────
    def save(self) -> None:
        base = (
            (self.args.basename + ".")
            if self.args.basename and not self.args.basename.endswith(".")
            else self.args.basename
        )
        df_path = self.out_dir / f"{base}featuremap_df.parquet"
        self.data_frame.write_parquet(df_path)
        logger.info(f"Saved dataframe → {df_path}")

        # models – JSON, one file per fold
        model_paths: dict[int, str] = {}
        for fold_idx, model in enumerate(self.models):
            path = self.out_dir / f"{base}model_fold_{fold_idx}.json"
            model.save_model(path)
            model_paths[fold_idx] = str(path)
        logger.info("Saved %d model JSONs", self.k_folds)

        # chromosome → fold map
        chrom_map_path = self.out_dir / f"{base}chrom_to_model.json"
        with chrom_map_path.open("w") as fh:
            json.dump(self.chrom_to_fold, fh, indent=2)
        logger.info(f"Saved chromosome→model map → {chrom_map_path}")

    # ───────────────────────── entry point ──────────────────────────────
    def run(self) -> None:
        self.train()
        self.save()


# ───────────────────────── CLI helpers ────────────────────────────────────
def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train SR-SNV classifier (refactored)")
    ap.add_argument("--positive", required=True, help="Parquet with label=1 rows")
    ap.add_argument("--negative", required=True, help="Parquet with label=0 rows")
    ap.add_argument("--training-regions", required=True, help="Picard interval_list file")
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
    return ap.parse_args()


# ───────────────────────── main entry point ──────────────────────────────
def main() -> None:
    trainer = SRSNVTrainer(_cli())
    trainer.run()


if __name__ == "__main__":
    main()
