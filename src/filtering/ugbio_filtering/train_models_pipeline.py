#!/env/python

# Copyright 2022 Ultima Genomics Inc.
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
#    Train ML models to filter callset
# CHANGELOG in reverse chronological order

from __future__ import annotations

import argparse
import logging
import random
import sys

import dill as pickle
import numpy as np
import pandas as pd
from ugbio_core.h5_utils import read_hdf
from ugbio_core.logger import logger

from ugbio_filtering import transformers, variant_filtering_utils
from ugbio_filtering.tprep_constants import GtType, VcfType


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(
        prog="train_models_pipeline.py",
        description="Train filtering models",
    )

    ap_var.add_argument(
        "--train_dfs",
        help="Names of the train h5 files, should be output of prepare_ground_truth",
        type=str,
        nargs="+",
        required=True,
    )

    ap_var.add_argument("--test_dfs", help="Names of the test h5 files", type=str, nargs="+", required=True)

    ap_var.add_argument(
        "--output_file_prefix", help="Output .pkl file with models, .h5 file with results", type=str, required=True
    )

    ap_var.add_argument(
        "--gt_type",
        help='GT type - "exact" or "approximate"',
        type=GtType,
        choices=list(GtType),
        default=GtType.EXACT,
    )

    ap_var.add_argument(
        "--vcf_type",
        help='VCF type - "single_sample"(GATK) or "deep_variant"',
        type=VcfType,
        choices=list(VcfType),
        default=VcfType.SINGLE_SAMPLE,
    )
    ap_var.add_argument(
        "--custom_annotations",
        help="Custom INFO annotations in the training VCF (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
    )

    ap_var.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )

    args = ap_var.parse_args(argv)
    return args


def run(argv: list[str]):
    "Train filtering model"
    np.random.seed(1984)  # noqa: NPY002
    random.seed(1984)
    args = parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))
    logger.debug(args)
    logger.info("Training pipeline: START")
    args.custom_annotations = (
        [x.lower() for x in args.custom_annotations] if args.custom_annotations is not None else []
    )
    try:
        features_to_extract = transformers.get_needed_features(args.vcf_type, args.custom_annotations) + ["label"]
        logger.debug(f"(len(features_to_extract)={len(features_to_extract)}")

        train_df = read_data(args.train_dfs, features_to_extract)
        logger.info("Read training data: success")

        model, transformer = train_model(train_df, args)
        logger.info("Model training: done")

        test_df = read_data(args.test_dfs, features_to_extract)
        logger.info("Read test data: done")

        train_results = evaluate_model(train_df, model, transformer, "training")
        test_results = evaluate_model(test_df, model, transformer, "test")

        save_results(args.output_file_prefix, model, transformer, train_results, test_results)
        logger.info("Model training run: success")

    except Exception as err:
        logger.exception(err)
        logger.error("Model training run: failed")
        raise err


def read_data(file_list: list[str], features_to_extract: list[str]) -> pd.DataFrame:
    dfs = []
    for input_file in file_list:
        dfs.append(read_hdf(input_file, columns_subset=features_to_extract))
    return pd.concat(dfs)


def train_model(train_df: pd.DataFrame, args: argparse.Namespace):
    logger.info("Model training: start")
    model, transformer = variant_filtering_utils.train_model(
        train_df, gt_type=args.gt_type, vtype=args.vcf_type, annots=args.custom_annotations
    )
    return model, transformer


def evaluate_model(df: pd.DataFrame, model, transformer, data_type: str):
    logger.info(f"Evaluate {data_type}: start")
    results = variant_filtering_utils.eval_model(df, model, transformer)
    logger.info(f"Evaluate {data_type}: done")
    return results


def save_results(output_file_prefix: str, model, transformer, train_results, test_results):
    results_dict = {
        "transformer": transformer,
        "xgb": model,
        "xgb_recall_precision": test_results[0],
        "xgb_recall_precision_curve": test_results[1],
        "xgb_train_recall_precision": train_results[0],
        "xgb_train_recall_precision_curve": train_results[1],
    }
    with open(output_file_prefix + ".pkl", "wb") as file:
        pickle.dump(results_dict, file)

    accuracy_dfs = []
    prcdict = {}
    for m_var in ("xgb", "xgb_train"):
        name_optimum = f"{m_var}_recall_precision"
        accuracy_df_per_model = results_dict[name_optimum]
        accuracy_df_per_model["model"] = name_optimum
        accuracy_dfs.append(accuracy_df_per_model)
        prcdict[name_optimum] = results_dict[
            name_optimum.replace("recall_precision", "recall_precision_curve")
        ].set_index("group")

    accuracy_df = pd.concat(accuracy_dfs)
    accuracy_df.to_hdf(output_file_prefix + ".h5", key="optimal_recall_precision")

    results_vals = pd.concat(prcdict, names=["model"])
    results_vals = results_vals[["recall", "precision", "f1"]].reset_index()
    results_vals.to_hdf(output_file_prefix + ".h5", key="recall_precision_curve")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
