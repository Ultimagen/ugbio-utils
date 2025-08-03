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
#    Create single read SNV quality recalibration training report
# CHANGELOG in reverse chronological order


from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields

from ugbio_srsnv.srsnv_plotting_utils import SRSNVReport, create_srsnv_report_html

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
IS_MIXED = "is_mixed"
IS_MIXED_START = "is_mixed_start"
IS_MIXED_END = "is_mixed_end"

PROB_ORIG = "prob_orig"
PROB_RECAL = "prob_recal"
PROB_RESCALED = "prob_rescaled"
PROB_TRAIN = "prob_train"
PROB_FOLD_TMPL = "prob_fold_{k}"

SCORE = "BCSQ"
IS_CYCLE_SKIP = "is_cycle_skip"

EDIT_DIST_FEATURES = ["EDIST", "HAMDIST", "HAMDIST_FILT"]

pl.enable_string_cache()


def add_is_mixed_to_featuremap_df(data_df: pd.DataFrame, adapter_version: str) -> pd.DataFrame:
    """Add is_mixed columns to featuremap_df
    NOTE: THIS FUNCTION IS A PATCH AND SHOULD BE REPLACED
    """
    logger.info("Adding is_mixed columns to featuremap")
    # TODO: use the information from adapter_version instead of this patch

    data_df[IS_MIXED_START] = data_df["st"] == "MIXED"
    data_df[IS_MIXED_END] = data_df["et"] == "MIXED"
    data_df[IS_MIXED] = data_df[IS_MIXED_START] & data_df[IS_MIXED_END]

    return data_df


def prepare_report(  # noqa: C901 PLR0915
    featuremap_df: str,  # Path to the featuremap dataframe parquet file
    srsnv_metadata: str,  # Path to the srsnv_metadata JSON file
    report_path: str,
    basename: str = "",
    models_prefix: str | None = None,
    random_seed: int | None = None,
) -> None:
    """
    Prepare the SNV report based on the provided featuremap dataframe and metadata.

    Args:
        featuremap_df (str): Path to the featuremap dataframe parquet file.
        srsnv_metadata (str): Path to the srsnv_metadata JSON file.
        report_path (str): Path to save the report files.
        basename (str): Basename prefix for output files.
        models_prefix (str | None): Prefix for model JSON files.
        random_seed (int | None): Random seed for reproducibility.
    """
    logger.info("Preparing SNV report...")

    # Read featuremap dataframe
    data_df = pd.read_parquet(featuremap_df)

    # Load srsnv_metadata into a dictionary
    with open(srsnv_metadata) as f:
        metadata = json.load(f)

    # Load models
    models = []
    if models_prefix is not None:
        # Use provided prefix to construct model paths
        # Try to read as many models as possible until file doesn't exist
        fold_idx = 0
        while True:
            model_path = f"{models_prefix}{fold_idx}.json"
            if not os.path.exists(model_path):
                break
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            models.append(model)
            fold_idx += 1
    else:
        # Use model paths from metadata
        for model_path in metadata["model_paths"].values():
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            models.append(model)

    # Load training evaluation results if available
    training_results = None
    if "training_results_path" in metadata:
        training_results_path = metadata["training_results_path"]
        if os.path.exists(training_results_path):
            logger.debug("Loading training evaluation results from %s", training_results_path)
            with open(training_results_path) as f:
                training_results = json.load(f)
            logger.info("Loaded training evaluation results")
        else:
            logger.warning("Training results file not found: %s", training_results_path)

    # Create a wrapper class for models that includes training results
    class ModelWithTrainingResults:
        def __init__(self, model, training_result=None):
            self.model = model
            self._training_result = training_result or {}
            # Delegate all other attributes to the underlying model
            for attr in dir(model):
                if not attr.startswith("_") and attr != "evals_result" and hasattr(model, attr):
                    setattr(self, attr, getattr(model, attr))

        def evals_result(self):
            return self._training_result

        def __getattr__(self, name):
            return getattr(self.model, name)

    # Wrap models with training results
    if training_results is not None:
        models = [
            ModelWithTrainingResults(model, training_results[i] if i < len(training_results) else {})
            for i, model in enumerate(models)
        ]
    else:
        models = [ModelWithTrainingResults(model) for model in models]

    # Create params dictionary
    params = {}

    # Extract categorical and numerical features from metadata
    categorical_features = [feature for feature in metadata["features"] if feature["type"] == "c"]
    numerical_features = [feature for feature in metadata["features"] if feature["type"] != "c"]

    params["categorical_features_names"] = [feature["name"] for feature in categorical_features]
    params["categorical_features_dict"] = {
        feature["name"]: list(feature["values"].keys()) for feature in categorical_features
    }
    params["numerical_features"] = [feature["name"] for feature in numerical_features]
    params["fp_regions_bed_file"] = 1
    params["num_CV_folds"] = len(models)

    # Placeholder for params that will be fixed later TODO: fix these later
    params["adapter_version"] = metadata.get("adapter_version", None)
    params["docker_image"] = metadata.get("docker_image", None)
    params["pipeline_version"] = metadata.get("pipeline_version", None)
    params["pre-filter"] = None
    params["start_tag_col"] = "st"
    params["end_tag_col"] = "et"

    # Add is_mixed columns to featuremap_df
    data_df = add_is_mixed_to_featuremap_df(data_df, params.get("adapter_version", None))
    data_df[IS_CYCLE_SKIP] = data_df[SCORE] == 10  # noqa: PLR2004

    # Handle random seed
    rng = None
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)

    # Create quality interpolating function from quality_recalibration_table
    quality_table = metadata["quality_recalibration_table"]
    x_values = np.array(quality_table[0])
    y_values = np.array(quality_table[1])

    def qual_interpolating_function(x):
        # Use np.interp with left=0 for x < x_values[0] and right=y_values[-1] for x > x_values[-1]
        return np.interp(x, x_values, y_values, left=0, right=y_values[-1])

    # Missing arguments - initialize to None for now
    # These will be fixed later
    lod_filters = None
    lod_label = None
    c_lod = "LoD"
    df_mrd_simulation = None
    statistics_h5_file = None  # str(os.path.join(report_path, f"{basename}.statistics_h5_file.h5"))
    statistics_json_file = None  # str(os.path.join(report_path, f"{basename}.statistics_json_file.json"))

    # Ensure basename has consistent format for the SRSNVReport
    # The create_srsnv_report_html function will add a dot if basename is not empty and doesn't end with one
    report_base_name = basename
    if len(basename) > 0 and not basename.endswith("."):
        report_base_name += "."

    # Initialize SRSNVReport object
    srsnv_report = SRSNVReport(
        models=models,
        data_df=data_df,
        params=params,
        out_path=report_path,
        base_name=report_base_name,
        lod_filters=lod_filters,
        lod_label=lod_label,
        c_lod=c_lod,
        df_mrd_simulation=df_mrd_simulation,
        ml_qual_to_qual_fn=qual_interpolating_function,
        statistics_h5_file=statistics_h5_file,
        statistics_json_file=statistics_json_file,
        srsnv_metadata=srsnv_metadata,
        rng=rng,
    )

    # Run the report generation
    srsnv_report.create_report()

    create_srsnv_report_html(
        out_path=report_path,
        out_basename=basename,
        srsnv_metadata_file=srsnv_metadata,
        simple_pipeline=None,
    )


# ───────────────────────── CLI helpers ────────────────────────────────────
def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create report of SingleReadSNV classifier training", allow_abbrev=True)
    ap.add_argument("--featuremap-df", required=True, help="Path to the training dataframe parquet file")
    ap.add_argument(
        "--srsnv-metadata",
        required=True,
        help="Path to the srsnv_metadata JSON file created by the srsnv training code",
    )
    ap.add_argument("--report-path", required=True, help="Output directory where report files will be saved")
    ap.add_argument("--basename", default="", help="Basename prefix for outputs")
    ap.add_argument(
        "--models-prefix",
        default=None,
        help="Prefix of the models JSON files (e.g., 'path/to/models.model_fold_'). "
        "If not provided, model paths are read from the srsnv_metadata.json file",
    )
    ap.add_argument("--random-seed", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = ap.parse_args()
    return args


# ───────────────────────── main entry point ──────────────────────────────
def main() -> None:
    args = _cli()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    prepare_report(
        featuremap_df=args.featuremap_df,
        srsnv_metadata=args.srsnv_metadata,
        report_path=args.report_path,
        basename=args.basename,
        models_prefix=args.models_prefix,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
