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
import pandas as pd  # only needed to build a Series for partition_into_folds
import polars as pl
import xgboost as xgb
from ugbio_core.logger import logger

# ─── Re-use tiny helpers from the legacy utils ─────────────────────────────
from ugbio_srsnv.srsnv_training_utils import (  # type: ignore
    partition_into_folds,
    prob_to_phred,
)

# ──────────────────────────── NEW CONSTANTS ───────────────────────────────
FOLD_COL = "fold_id"
LABEL_COL = "label"
CHROM_COL = "chrom"
POS_COL = "POS"

RAW_QUAL_VAL = "raw_qual_val"
RAW_QUAL_TRAIN_TMPL = "raw_qual_train_{idx}"  # idx = 1 … k-1


# ───────────────────── interval-list helpers (NEW) ────────────────────────
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
    ap.add_argument("--random-seed", type=int, default=None)
    return ap.parse_args()


# ───────────────────────── model-param parser ─────────────────────────────
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
            pl.col(CHROM_COL).map_elements(lambda c: self.chrom_to_fold.get(c, float("nan"))).alias(FOLD_COL)
        )

        # Models
        model_params = _parse_model_params(args.model_params)
        self.models = [xgb.XGBClassifier(**model_params) for _ in range(self.k_folds)]

    # ─────────────────────── data & features ────────────────────────────
    def _load_data(self, pos_path: str, neg_path: str) -> pl.DataFrame:
        """Read parquet ⇒ polars, add label column and concatenate."""
        pos_df = pl.read_parquet(pos_path).with_columns(pl.lit(value=True).alias(LABEL_COL))
        neg_df = pl.read_parquet(neg_path).with_columns(pl.lit(value=False).alias(LABEL_COL))
        combined_df = pl.concat([pos_df, neg_df])
        if "CHROM" in combined_df.columns and CHROM_COL not in combined_df.columns:
            combined_df = combined_df.rename({"CHROM": CHROM_COL})
        return combined_df

    def _feature_columns(self) -> list[str]:
        exclude = {LABEL_COL, FOLD_COL, CHROM_COL, "CHROM", POS_COL.lower(), POS_COL}
        return [c for c in self.data_frame.columns if c not in exclude]

    # ─────────────────────── training / prediction ──────────────────────
    def train(self) -> None:
        feat_cols = self._feature_columns()
        logger.info(f"Training with {len(feat_cols)} features")

        # Pre-compute numpy matrices once
        x_all = self.data_frame.select(feat_cols).to_numpy()
        y_all = self.data_frame.select(LABEL_COL).to_numpy().ravel()
        fold_arr = self.data_frame.select(FOLD_COL).to_numpy().ravel()

        for fold_idx in range(self.k_folds):
            val_mask = fold_arr == fold_idx
            train_mask = (~val_mask) & ~np.isnan(fold_arr)

            self.models[fold_idx].fit(
                x_all[train_mask],
                y_all[train_mask],
                eval_set=[
                    (x_all[train_mask], y_all[train_mask]),
                    (x_all[val_mask], y_all[val_mask]),
                ],
            )

        self._add_quality_columns(x_all, fold_arr)

    def _add_quality_columns(self, x_all: np.ndarray, fold_arr: np.ndarray) -> None:
        n_rows = x_all.shape[0]
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


def main() -> None:
    trainer = SRSNVTrainer(_cli())
    trainer.run()


if __name__ == "__main__":
    main()
