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
from sklearn.dummy import DummyClassifier
from ugbio_core.logger import logger
from ugbio_core.vcfbed.variant_annotation import get_cycle_skip_dataframe
from ugbio_featuremap.featuremap_utils import FeatureMapFields

from ugbio_srsnv.split_scheme import resolve_scheme_and_add_columns
from ugbio_srsnv.srsnv_plotting_utils import SRSNVReport, create_srsnv_report_html
from ugbio_srsnv.srsnv_utils import (
    DEFAULT_FLOW_ORDER_R1,
    ET,
    IS_CYCLE_SKIP_COMBINED,
    PE_DUPLEX_GROUP,
    ST,
    ReportMode,
    dedupe_multi_call_molecules,
    flow_order_per_row,
)

FOLD_COL = "fold_id"
LABEL_COL = "label"
CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value
REF = FeatureMapFields.REF.value
ALT = FeatureMapFields.ALT.value
X_ALT = FeatureMapFields.X_ALT.value
X_HMER_REF = FeatureMapFields.X_HMER_REF.value
X_HMER_ALT = FeatureMapFields.X_HMER_ALT.value
X_PREV1 = FeatureMapFields.X_PREV1.value
X_NEXT1 = FeatureMapFields.X_NEXT1.value
MQUAL = FeatureMapFields.MQUAL.value
SNVQ = FeatureMapFields.SNVQ.value
SNVQ_RAW = SNVQ + "_RAW"
IS_MIXED = "is_mixed"
IS_MIXED_START = "is_mixed_start"
IS_MIXED_END = "is_mixed_end"

PROB_ORIG = "prob_orig"
PROB_RECAL = "prob_recal"
PROB_TRAIN = "prob_train"
PROB_FOLD_TMPL = "prob_fold_{k}"

SCORE = FeatureMapFields.BCSQ.value
IS_CYCLE_SKIP = "is_cycle_skip"

# Joint SNV + hmer-indel report dimension. snvfind tags homopolymer indels with X_IC (ins/del) and X_IL
# (indel length); X_HMER_REF is the reference homopolymer length used to bin hmer-indel metrics.
X_IC = "X_IC"
X_IL = "X_IL"
X_INDEL_CLASS = "X_INDEL_CLASS"  # snvfind: "hmer" | "non_hmer" -- discriminates the two indel classes
VARIANT_TYPE = "variant_type"
VARIANT_TYPE_SNV = "snv"
VARIANT_TYPE_HMER_INDEL = "hmer_indel"
VARIANT_TYPE_NONHMER_INDEL = "non_hmer_indel"

EDIT_DIST_FEATURES = ["EDIST", "HAMDIST", "HAMDIST_FILT"]

pl.enable_string_cache()


def compute_is_cycle_skip_column(
    data_df: pd.DataFrame, flow_order: str = DEFAULT_FLOW_ORDER_R1, flow_order_r2: str | None = None
) -> pd.Series:
    """
    Compute the is_cycle_skip column for a featuremap dataframe.

    This function calculates whether each variant represents a cycle skip
    based on the reference and alternate motifs (prev1 + ref/alt + next1).

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing X_PREV1, REF, ALT, and X_NEXT1 columns (and, for multi-call data, MATE).
    flow_order : str, optional
        Flow order string (default: "TGCA"). For multi-call data this is the R1 flow order.
    flow_order_r2 : str, optional
        Multi-call (ppmSeq+ dual flow) R2 flow order: rows with ``MATE == 2`` are evaluated in this flow order,
        all other rows in ``flow_order``. ``None`` (default) -> every row uses ``flow_order``.

    Returns
    -------
    pd.Series
        Boolean series indicating cycle skip status for each variant.
    """
    logger.info("Computing is_cycle_skip column")

    ref_motif = data_df[X_PREV1].astype(str) + data_df[REF].astype(str) + data_df[X_NEXT1].astype(str)
    alt_motif = data_df[X_PREV1].astype(str) + data_df[ALT].astype(str) + data_df[X_NEXT1].astype(str)

    motif_df = pd.DataFrame(
        {"ref_motif": ref_motif, "alt_motif": alt_motif},
        index=data_df.index,
    )
    row_flow = flow_order_per_row(data_df, flow_order, flow_order_r2)
    flow_orders = list(dict.fromkeys(row_flow.tolist())) or [flow_order]
    if len(flow_orders) == 1:
        cycle_skip_df = get_cycle_skip_dataframe(flow_orders[0])[[IS_CYCLE_SKIP]]
        result = motif_df.merge(
            cycle_skip_df,
            left_on=["ref_motif", "alt_motif"],
            right_index=True,
            how="left",
        )[IS_CYCLE_SKIP]
        return result.loc[data_df.index]

    result = pd.Series(np.nan, index=data_df.index, dtype=object)
    for fo in flow_orders:
        mask = (row_flow == fo).to_numpy()
        cycle_skip_df = get_cycle_skip_dataframe(fo)[[IS_CYCLE_SKIP]]
        sub = motif_df.loc[mask].merge(
            cycle_skip_df,
            left_on=["ref_motif", "alt_motif"],
            right_index=True,
            how="left",
        )[IS_CYCLE_SKIP]
        result.loc[sub.index] = sub.to_numpy()
    return result


def compute_is_cycle_skip_combined_column(
    data_df: pd.DataFrame, flow_order_r1: str = DEFAULT_FLOW_ORDER_R1, flow_order_r2: str | None = None
) -> pd.Series:
    """Multi-call combined cycle skip: the SNV motif is a cycle skip in the R1 OR the R2 flow order.

    A multi-call molecule is sequenced in both flow orders, so it is a cycle-skip observation if either call is.
    Evaluated per motif for every row (independent of MATE). Rows whose motif is not an SNV motif (e.g. indels)
    are NaN in both flow orders and stay NaN. ``flow_order_r2=None`` -> the R1 cycle skip only.
    """
    no_mate = data_df.drop(columns=["MATE"], errors="ignore")
    cs_r1 = compute_is_cycle_skip_column(no_mate, flow_order_r1)
    if flow_order_r2 is None or flow_order_r2 == flow_order_r1:
        return cs_r1
    cs_r2 = compute_is_cycle_skip_column(no_mate, flow_order_r2)
    known = cs_r1.notna() | cs_r2.notna()
    combined = cs_r1.fillna(value=False).astype(bool) | cs_r2.fillna(value=False).astype(bool)
    return combined.astype(object).where(known, np.nan)


def compute_variant_type_column(data_df: pd.DataFrame) -> pd.Series:
    """Label each featuremap row as SNV, homopolymer indel, or non-homopolymer indel (the report's axis).

    In the joint DeepSRSNV mode snvfind emits SNVs plus indels; indels carry ``X_IC`` (``ins``/``del``) and
    an ``X_INDEL_CLASS`` (``hmer``/``non_hmer``) discriminator. Rows without ``X_IC`` (SNVs, or any run
    produced without indel emission) are labeled ``snv``. Indel rows are split by ``X_INDEL_CLASS``:
    ``non_hmer`` -> ``non_hmer_indel``; anything else (including a missing column, for backward
    compatibility with older hmer-only featuremaps) -> ``hmer_indel``. Returns a Series aligned to
    ``data_df.index``.
    """
    if X_IC not in data_df.columns:
        return pd.Series(VARIANT_TYPE_SNV, index=data_df.index)
    ic = data_df[X_IC].astype(str).str.lower()
    is_indel = ic.isin(["ins", "del"])
    if X_INDEL_CLASS in data_df.columns:
        is_non_hmer = is_indel & (data_df[X_INDEL_CLASS].astype(str).str.lower() == "non_hmer")
    else:
        is_non_hmer = pd.Series(np.zeros(len(data_df), dtype=bool), index=data_df.index)
    is_hmer_indel = is_indel & ~is_non_hmer
    labels = np.where(
        is_non_hmer,
        VARIANT_TYPE_NONHMER_INDEL,
        np.where(is_hmer_indel, VARIANT_TYPE_HMER_INDEL, VARIANT_TYPE_SNV),
    )
    return pd.Series(labels, index=data_df.index)


class _ModelWithTrainingResults:
    """Wrapper that delegates to an XGBoost model but overrides evals_result."""

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


def _load_models_from_prefix(models_prefix: str) -> list:
    """Load XGBoost models from numbered JSON files matching *prefix*<N>.json."""
    models = []
    fold_idx = 0
    while True:
        model_path = f"{models_prefix}{fold_idx}.json"
        if not os.path.exists(model_path):
            break
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        models.append(model)
        fold_idx += 1
    return models


def _resolve_model_path(orig_model_path: str) -> str:
    """Resolve a model path, falling back to CWD if the original doesn't exist."""
    if os.path.exists(orig_model_path):
        return orig_model_path
    alt_path = os.path.join(os.getcwd(), os.path.basename(orig_model_path))
    logger.debug(
        "Model file not found at '%s'. Trying CWD path '%s'.",
        orig_model_path,
        alt_path,
    )
    if os.path.exists(alt_path):
        return alt_path
    raise FileNotFoundError(
        "Expected model file not found. Looked for: "
        f"'{orig_model_path}' and '{alt_path}'. Neither location contains the model file."
    )


def _load_models_from_metadata(metadata: dict) -> list:
    """Load XGBoost models using paths stored in metadata."""
    models = []
    for orig_model_path in metadata.get("model_paths", {}).values():
        if not orig_model_path:
            continue
        model = xgb.XGBClassifier()
        path_to_load = _resolve_model_path(orig_model_path)
        model.load_model(path_to_load)
        models.append(model)
    return models


def _fallback_dummy_models(metadata: dict) -> list:
    """Create dummy classifiers when no XGBoost models are available."""
    num_folds = max(len(metadata.get("model_paths", {})), 1)
    models = [DummyClassifier(strategy="constant", constant=0) for _ in range(num_folds)]
    for m in models:
        m.fit([[0], [1]], [0, 1])
    logger.warning("No XGBoost models found; using dummy classifiers. SHAP/training plots will be skipped.")
    return models


def _wrap_models_with_training_results(models: list, training_results) -> list:
    """Wrap models with training evaluation results for report consumption."""
    if training_results is not None:
        return [
            _ModelWithTrainingResults(model, training_results[i] if i < len(training_results) else {})
            for i, model in enumerate(models)
        ]
    return [_ModelWithTrainingResults(model) for model in models]


def _build_params(metadata: dict, user_meta: dict, num_models: int) -> dict:
    """Build the params dictionary from metadata features and user settings."""
    categorical_features = [f for f in metadata["features"] if f["type"] == "c"]
    numerical_features = [f for f in metadata["features"] if f["type"] != "c"]

    return {
        "categorical_features_names": [f["name"] for f in categorical_features],
        "categorical_features_dict": {f["name"]: list(f["values"].keys()) for f in categorical_features},
        "numerical_features": [f["name"] for f in numerical_features],
        "fp_regions_bed_file": 1,
        "num_CV_folds": num_models,
        "adapter_version": user_meta.get("adapter_version", None),
        "docker_image": user_meta.get("docker_image", None),
        "pipeline_version": user_meta.get("pipeline_version", None),
        "pre-filter": None,
        "start_tag_col": ST,
        "end_tag_col": ET,
    }


def _make_qual_interpolating_function(quality_table: list):
    """Create a quality interpolation function from the recalibration table."""
    x_values = np.array(quality_table[0])
    y_values = np.array(quality_table[1])

    def _interpolate(x):
        return np.interp(x, x_values, y_values, left=0, right=y_values[-1])

    return _interpolate


def prepare_report(
    featuremap_df: str,  # Path to the featuremap dataframe parquet file
    srsnv_metadata: str,  # Path to the srsnv_metadata JSON file
    report_path: str,
    basename: str = "",
    models_prefix: str | None = None,
    random_seed: int | None = None,
    *,
    use_gpu_for_shap: bool = False,
    flow_order_r1: str = DEFAULT_FLOW_ORDER_R1,
    flow_order_r2: str | None = None,
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
        use_gpu_for_shap (bool): Whether to use GPU for SHAP calculations if available.
        flow_order_r1 (str): Flow order for cycle skip (multi-call: of the R1 call, MATE != 2). Default TGCA.
        flow_order_r2 (str | None): Multi-call R2 flow order (rows with MATE == 2). None -> same as r1.
    """
    logger.info("Preparing SNV report...")

    # Read featuremap dataframe
    data_df = pd.read_parquet(featuremap_df)

    # Load srsnv_metadata into a dictionary
    with open(srsnv_metadata) as f:
        metadata = json.load(f)
    user_meta = metadata.get("metadata", {})

    # Load models
    if models_prefix is not None:
        models = _load_models_from_prefix(models_prefix)
    else:
        models = _load_models_from_metadata(metadata)

    if not models:
        models = _fallback_dummy_models(metadata)

    # Wrap models with training results
    models = _wrap_models_with_training_results(models, metadata.get("training_results"))

    # Build params dictionary from metadata
    params = _build_params(metadata, user_meta, len(models))

    # Resolve the active split scheme (recipe) from the columns present and add the split columns +
    # ordered read_group it needs. The mode is stored in params so SRSNVReport uses it directly.
    data_df, scheme = resolve_scheme_and_add_columns(
        data_df,
        params["adapter_version"],
        params["categorical_features_names"],
    )
    params["report_mode"] = scheme.mode.value
    data_df[IS_CYCLE_SKIP] = compute_is_cycle_skip_column(data_df, flow_order_r1, flow_order_r2)
    # Flow orders used for cycle skip (the trinuc figure colours ticks per flow order when they differ).
    params["flow_order_r1"] = flow_order_r1
    params["flow_order_r2"] = flow_order_r2
    # Multi-call (ppmSeq+ dual flow; auto-detected from the MATE column): per-molecule report. The combined
    # (R1 or R2) cycle skip is computed on all read rows, then rows are deduplicated to one per molecule (both
    # calls share the score / SNVQ; concordance was already derived by the scheme from the read rows).
    params["has_multi_call"] = scheme.mode is ReportMode.MULTI_CALL
    if params["has_multi_call"]:
        if flow_order_r2 is None or flow_order_r2 == flow_order_r1:
            logger.warning(
                "Multi-call featuremap (MATE) without a distinct --flow-order-r2: cycle skip uses the R1 flow order "
                f"({flow_order_r1}) for both calls"
            )
        data_df[IS_CYCLE_SKIP_COMBINED] = compute_is_cycle_skip_combined_column(data_df, flow_order_r1, flow_order_r2)
        data_df = dedupe_multi_call_molecules(data_df)

    # Joint SNV + hmer-indel report dimension (orthogonal to the read-group split scheme above).
    # Auto-detected from the featuremap: hmer-indel rows carry X_IC. When present, the report can break
    # metrics out by variant type and (via X_HMER_REF) by homopolymer length.
    data_df[VARIANT_TYPE] = compute_variant_type_column(data_df)
    params["has_hmer_indels"] = bool((data_df[VARIANT_TYPE] == VARIANT_TYPE_HMER_INDEL).any())
    params["has_non_hmer_indels"] = bool((data_df[VARIANT_TYPE] == VARIANT_TYPE_NONHMER_INDEL).any())
    # paired-end-duplex use case: the scheme assigned the pe_duplex_group column (CS-family-derived
    # SE/PE split). Gates the duplex-PE CS-concordance table/cell.
    params["has_duplex_pe"] = PE_DUPLEX_GROUP in data_df.columns

    # Handle random seed
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    # Create quality interpolating function from quality_recalibration_table
    qual_interpolating_function = _make_qual_interpolating_function(metadata["quality_recalibration_table"])

    # Ensure basename has consistent format for the SRSNVReport
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
        lod_filters=None,
        lod_label=None,
        c_lod="LoD",
        df_mrd_simulation=None,
        ml_qual_to_qual_fn=qual_interpolating_function,
        statistics_h5_file=None,
        statistics_json_file=None,
        srsnv_metadata=srsnv_metadata,
        rng=rng,
        use_gpu_for_shap=use_gpu_for_shap,
    )

    # Run the report generation
    srsnv_report.create_report()

    create_srsnv_report_html(
        out_path=report_path,
        out_basename=basename,
        srsnv_metadata_file=srsnv_metadata,
        simple_pipeline=None,
        split_mode=scheme.mode.value,
        display_suffix=scheme.display_suffix,
        has_hmer_indels=params["has_hmer_indels"],
        has_non_hmer_indels=params["has_non_hmer_indels"],
        has_duplex_pe=params["has_duplex_pe"],
        has_multi_call=params["has_multi_call"],
        flow_order_r1=flow_order_r1,
        flow_order_r2=flow_order_r2,
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
    ap.add_argument("--use-gpu-for-shap", action="store_true", help="Use GPU for SHAP calculations if available")
    ap.add_argument(
        "--flow-order-r1",
        default=DEFAULT_FLOW_ORDER_R1,
        help="Flow order used for cycle skip (multi-call data: flow order of the R1 call, MATE != 2). Default TGCA",
    )
    ap.add_argument(
        "--flow-order-r2",
        default=None,
        help="Multi-call (ppmSeq+ dual flow) flow order of the R2 call: rows with MATE == 2 are evaluated in this "
        "flow order for cycle skip. Default: same as --flow-order-r1",
    )

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
        use_gpu_for_shap=args.use_gpu_for_shap,
        flow_order_r1=args.flow_order_r1,
        flow_order_r2=args.flow_order_r2,
    )


if __name__ == "__main__":
    main()
