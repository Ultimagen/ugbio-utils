from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost as xgb
from matplotlib import colors
from matplotlib import lines as mlines
from scipy.interpolate import interp1d
from scipy.stats import binom
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from ugbio_core.bed_utils import BedUtils
from ugbio_core.logger import logger
from ugbio_core.plotting_utils import set_pyplot_defaults
from ugbio_core.reports.report_utils import generate_report
from ugbio_core.sorter_utils import read_effective_coverage_from_sorter_json
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_ppmseq.ppmSeq_utils import PpmseqAdapterVersions, PpmseqCategories

from ugbio_srsnv.shap_plotting import SHAPPlotter
from ugbio_srsnv.smoothing_utils import AdaptiveKDEPrecisionEstimator
from ugbio_srsnv.split_scheme import (
    h5_key as split_h5_key,
)
from ugbio_srsnv.split_scheme import (
    resolve_scheme,
    resolve_scheme_and_add_columns,
)
from ugbio_srsnv.srsnv_utils import (
    ET,
    ET_FILLNA,
    FS,
    MAX_PHRED,
    READ_GROUP,
    RS,
    ST,
    ST_FILLNA,
    construct_trinuc_context_with_alt,
    get_base_recall_from_filters,
    phred_to_prob,
    prob_to_logit,
    prob_to_phred,
    safe_roc_auc,
    split_validation_training_preds,
)
from ugbio_srsnv.trinuc_histogram_plotting import calc_and_plot_trinuc_hist

# featuremap_df column names. TODO: make more generic?
ML_PROB_1_TEST = "prob_orig"  # TODO: Maybe prob_recal?
ML_QUAL_1_TEST = "MQUAL"
ML_LOGIT_TEST = "ML_logit_test"
LABEL = "label"
QUAL = FeatureMapFields.SNVQ.value
X_HMER_REF = FeatureMapFields.X_HMER_REF.value
X_IC = "X_IC"  # snvfind indel class: "ins" | "del" (present on homopolymer-indel rows)
X_IL = "X_IL"  # snvfind indel length (|X_IL|)
VARIANT_TYPE = "variant_type"  # per-row snv | hmer_indel (set by prepare_report.compute_variant_type_column)
VARIANT_TYPE_HMER_INDEL = "hmer_indel"
IS_MIXED = "is_mixed"
IS_MIXED_START = "is_mixed_start"
IS_MIXED_END = "is_mixed_end"
FOLD_ID = "fold_id"

EDIST = FeatureMapFields.EDIST.value
SCORE = FeatureMapFields.BCSQ.value
INDEX = FeatureMapFields.INDEX.value
LENGTH = FeatureMapFields.RL.value
READ_COUNT = FeatureMapFields.DP.value
IS_CYCLE_SKIP = "is_cycle_skip"
REV = FeatureMapFields.REV.value
IS_FORWARD = "is_forward"
TRINUC_CONTEXT_WITH_ALT = "trinuc_context_with_alt"

edist_filter = f"{EDIST} <= 5"
HQ_SNV_filter = f"{SCORE} >= 79"
CSKP_SNV_filter = f"{SCORE} >= 100"
read_end_filter = f"{INDEX} > 12 and {INDEX} < ({LENGTH} - 12)"
# LoD filters below reference IS_MIXED; in consensus mode the split column carries the
# consensus flag, so these *_mixed_only filters become consensus-only. The LoD block in
# create_report is currently disabled, so this has no live effect. TODO: use adapter_version.
mixed_read_filter = IS_MIXED
default_LoD_filters = {  # noqa: N816
    "no_filter": f"{SCORE} >= 0",
    "HQ_SNV": f"{HQ_SNV_filter} and {edist_filter}",
    "CSKP": f"{CSKP_SNV_filter} and {edist_filter}",
    "HQ_SNV_trim_ends": f"{HQ_SNV_filter} and {edist_filter} and {read_end_filter}",
    "CSKP_trim_ends": f"{CSKP_SNV_filter} and {edist_filter} and {read_end_filter}",
    "HQ_SNV_mixed_only": f"{HQ_SNV_filter} and {edist_filter} and {mixed_read_filter}",
    "CSKP_mixed_only": f"{CSKP_SNV_filter} and {edist_filter} and {mixed_read_filter}",
    "HQ_SNV_trim_ends_mixed_only": f"{HQ_SNV_filter} and {edist_filter} and {read_end_filter} and {mixed_read_filter}",
    "CSKP_trim_ends_mixed_only": f"{CSKP_SNV_filter} and {edist_filter} and {read_end_filter} and {mixed_read_filter}",
}
TP_READ_RETENTION_RATIO = "tp_read_retention_ratio"
FP_READ_RETENTION_RATIO = "fp_read_retention_ratio"
RESIDUAL_SNV_RATE = "residual_snv_rate"
BASE_PATH = Path(__file__).parent
REPORTS_DIR = "reports"
SIG_DIGITS = 4  # number of significant digits for floating point numbers in report tables


class ExceptionConfig:
    def __init__(self, *, raise_exception=False):
        self.raise_exception = raise_exception


exception_config = ExceptionConfig()


def exception_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception occurred in {func.__name__}: {e}", exc_info=True)
            if exception_config.raise_exception:
                raise e
            return None

    return wrapper


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def list_of_jagged_lists_to_array(list_of_lists, fill=np.nan):
    lens = [len(one_list) for one_list in list_of_lists]
    max_len = max(lens)
    num_lists = len(list_of_lists)
    arr = np.ones((num_lists, max_len)) * fill
    for i, sublist in enumerate(list_of_lists):
        arr[i, : lens[i]] = np.array(sublist)
    return arr


def list_auc_to_qual(auc_list, max_value=MAX_PHRED):
    return list(prob_to_phred(np.array(auc_list), max_value=max_value))


def plot_extended_step(x, y, ax, **kwargs):
    """A plot like ax.step(x, y, where="mid"), but the lines extend beyond the rightmost and leftmost points
    to fill the first and last "bars" (like in sns.histplot(..., element='step')).
    """
    x = list(x)
    y = list(y)
    if len(x) == 0:
        x_ext = []
        y_ext = []
    elif len(x) == 1:
        x_ext = [x[0] - 0.5, x[0], x[0] + 0.5]
        y_ext = [y[0], y[0], y[0]]
    else:
        x_ext = [x[0] - (x[1] - x[0]) / 2] + x + [x[-1] + (x[-1] - x[-2]) / 2]
        y_ext = [y[0]] + y + [y[-1]]
    (line,) = ax.step(x_ext, y_ext, where="mid", **kwargs)
    if len(x) > 0:
        ax.set_xlim([x_ext[0], x_ext[-1]])
    return line


def plot_extended_fill_between(x, y1, y2, ax, **kwargs):
    """A plot like ax.fill_between(x, y1, y2, where="mid"), but the lines extend
    beyond the rightmost and leftmost points to fill the first and last "bars"
    (like in sns.histplot(..., element='step')).
    """
    x = list(x)
    y1 = list(y1)
    y2 = list(y2)
    if len(x) == 0:
        x_ext = []
        y1_ext = []
        y2_ext = []
    elif len(x) == 1:
        x_ext = [x[0] - 0.5, x[0], x[0] + 0.5]
        y1_ext = [y1[0], y1[0], y1[0]]
        y2_ext = [y2[0], y2[0], y2[0]]
    else:
        x_ext = [x[0] - (x[1] - x[0]) / 2] + x + [x[-1] + (x[-1] - x[-2]) / 2]
        y1_ext = [y1[0]] + y1 + [y1[-1]]
        y2_ext = [y2[0]] + y2 + [y2[-1]]
    poly = ax.fill_between(x_ext, y1_ext, y2_ext, step="mid", **kwargs)
    if len(x) > 0:
        ax.set_xlim([x_ext[0], x_ext[-1]])
    return poly


def plot_box_and_line(df, col_x, col_line, col_bottom, col_top, ax, fb_kws=None, step_kws=None):
    fb_kws = fb_kws or {}
    step_kws = step_kws or {}
    if "alpha" not in fb_kws:
        fb_kws["alpha"] = 0.2
    if "alpha" not in step_kws:
        step_kws["alpha"] = 0.7
    poly = plot_extended_fill_between(df[col_x], df[col_bottom], df[col_top], ax=ax, **fb_kws)
    line = plot_extended_step(df[col_x], df[col_line], ax=ax, **step_kws)
    return poly, line


def discretized_bin_edges(a, discretization_size=None, bins=10, full_range=None, weights=None):
    """Wrapper for numpy.histogram_bin_edges() that forces bin
    widths to be a multiple of discretization_size.
    From https://stackoverflow.com/questions/30112420/histogram-for-discrete-values-with-matplotlib
    """
    if discretization_size is None:
        # calculate the minimum distance between values
        discretization_size = np.diff(np.unique(a)).min()

    if full_range is None:
        full_range = (a.min(), a.max())

    # get suggested bin with, and calculate the nearest
    # discretized bin width
    bins = np.histogram_bin_edges(a, bins, full_range, weights)
    bin_width = bins[1] - bins[0]
    discretized_bin_width = discretization_size * max(1, round(bin_width / discretization_size))

    # calculate the discretized bins
    left_of_first_bin = full_range[0] - float(discretization_size) / 2
    right_of_last_bin = full_range[1] + float(discretization_size) / 2
    discretized_bins = np.arange(left_of_first_bin, right_of_last_bin + discretized_bin_width, discretized_bin_width)

    return discretized_bins


# ### Old functions from here


def create_data_for_report(
    classifiers: list[xgb.XGBClassifier],
    df: pd.DataFrame,
):
    """create the data needed for the report plots

    Parameters
    ----------
    classifiers : list[xgb.XGBClassifier]
        A list of the trained ML models
    df : pd.DataFrame
        data dataframe data with predictions

    Returns
    -------
    df_tp : pd.DataFrame
        TP subset of df
    df_fp : pd.DataFrame
        FP subset of df
    max_score : float
        maximal ML model score
    cls_features : list
        list of input features for the ML model
    fprs : dict
        list of false positive rates per ML score per label
    recalls : dict
        list of false positive rates per ML score per label
    """

    cls_features = list(classifiers[0].feature_names_in_)

    labels = np.unique(df[LABEL].astype(int))

    df_tp = df.query(f"{LABEL} == True")
    df_fp = df.query(f"{LABEL} == False")

    fprs = {}
    recalls = {}
    max_score = -1
    for label in labels:
        fprs[label] = []
        recalls[label] = []
        score = f"ML_qual_{label}"
        max_score = np.max((int(np.ceil(df[score].max())), max_score))
        gtr = df[LABEL] == label
        fprs_, recalls_ = precision_recall_curve(df[score], max_score=max_score, y_true=gtr)
        fprs[label].append(fprs_)
        recalls[label].append(recalls_)

    return df, df_tp, df_fp, max_score, cls_features, fprs, recalls


def create_srsnv_report_html(
    out_path,
    out_basename,
    srsnv_metadata_file,
    simple_pipeline=None,
    split_mode: str = "mixed",
    display_suffix: str = "mixed_start",
    *,
    has_hmer_indels: bool = False,
):
    if len(out_basename) > 0 and not out_basename.endswith("."):
        out_basename += "."
    report_html = Path(out_path) / f"{out_basename}report.html"
    # hmer-indel section plot path (reconstructed by the same convention create_report uses). The plot
    # file exists only when has_hmer_indels; the notebook cell is guarded on has_hmer_indels.
    hmer_indel_metrics_plot = os.path.join(out_path, f"{out_basename}hmer_indel_metrics")
    hmer_indel_by_class_plot = os.path.join(out_path, f"{out_basename}hmer_indel_by_class")

    [
        FQ_vs_recall_plot,  # noqa: N806
        qual_vs_ppmseq_tags_table,
        training_progress_plot,
        SHAP_importance_plot,  # noqa: N806
        SHAP_beeswarm_plot,  # noqa: N806
        trinuc_stats_plot,
        output_qual_per_feature,
        qual_histogram,
        logit_histogram,
        calibration_fn_with_hist,
    ] = _get_plot_paths(out_path=out_path, out_basename=out_basename)
    srsnv_qc_h5_filename = os.path.join(out_path, f"{out_basename}single_read_snv.applicationQC.h5")

    template_notebook = BASE_PATH / REPORTS_DIR / "srsnv_report.ipynb"

    parameters = {
        "srsnv_metadata_file": srsnv_metadata_file,
        # "model_file": model_file,
        # "params_file": params_file,
        "srsnv_qc_h5_file": srsnv_qc_h5_filename,
        "FQ_vs_recall_plot": FQ_vs_recall_plot,  # noqa: N806
        "qual_vs_ppmseq_tags_table": qual_vs_ppmseq_tags_table,
        "training_progress_plot": training_progress_plot,
        "SHAP_importance_plot": SHAP_importance_plot,
        "SHAP_beeswarm_plot": SHAP_beeswarm_plot,
        "trinuc_stats_plot": trinuc_stats_plot,
        "output_qual_per_feature": output_qual_per_feature,
        "qual_histogram": qual_histogram,
        "logit_histogram": logit_histogram,
        "calibration_fn_with_hist": calibration_fn_with_hist,
        "split_mode": split_mode,
        "display_suffix": display_suffix,
        "has_hmer_indels": has_hmer_indels,
        "hmer_indel_metrics_plot": hmer_indel_metrics_plot,
        "hmer_indel_by_class_plot": hmer_indel_by_class_plot,
    }

    generate_report(
        template_notebook_path=template_notebook,
        parameters=parameters,
        output_report_html_path=report_html,
    )


def filter_valid_queries(df_test: pd.DataFrame, queries: dict, *, verbose: bool = False) -> dict:
    """
    Test each filter query on the DataFrame and remove any that cause exceptions.

    Parameters:
    ----------
    df_test: pd.DataFrame
        The input DataFrame to test on.
    queries: dict
        A dictionary of filter name to filter query, keys are names and values are query strings.
    verbose: bool
        Whether to print the filter queries that caused an exception.

    Returns
    - A dictionary of valid filters that didn't cause exceptions.
    """

    # Start with an empty dictionary to store filters that don't cause an exception
    valid_filters = {}

    # Test each filter
    for filter_name, filter_query in queries.items():
        try:
            df_test.eval(filter_query)
            valid_filters[filter_name] = filter_query
        except Exception:
            if verbose:
                logger.warning(f"Filter query {filter_query} caused an exception, skipping.")

    return valid_filters


def retention_noise_and_mrd_lod_simulation(  # noqa: PLR0913
    df: pd.DataFrame,
    single_sub_regions: str,
    sorter_json_stats_file: str,
    fp_featuremap_entry_number: int,
    lod_filters: dict,
    sensitivity_at_lod: float = 0.90,
    specificity_at_lod: float = 0.99,
    simulated_signature_size: int = 10_000,
    simulated_coverage: int = 30,
    minimum_number_of_read_for_detection: int = 2,
    output_dataframe_file: str = None,
):
    """Estimate the MRD LoD based on the FeatureMap dataframe and the LoD simulation params

    Parameters
    ----------
    df : pd.DataFrame
        dataset with features, labels, and quals of the model.
    single_sub_regions : str
        a path to the single substitutions (FP) regions bed file.
    sorter_json_stats_file : str
        a path to the cram statistics file.
    fp_featuremap_entry_number : float
        the # of reads in single sub featuremap after intersection with single_sub_regions.
    lod_filters : dict
        filters for LoD simulation.
    sensitivity_at_lod: float
        The sensitivity at which LoD is estimated.
    specificity_at_lod: float
        The specificity at which LoD is estimated.
    simulated_signature_size: int
        The size of the simulated signature.
    simulated_coverage: int
        The simulated coverage.
    minimum_number_of_read_for_detection: int
        The minimum number of reads required for detection in the LoD simulation. Default 2.
    output_dataframe_file: str, optional
        Path to output dataframe file. If None, don't save.

    Returns
    -------
    df_mrd_sim: pd.DataFrame
        The estimated LoD parameters per filter.
    lod_label: str
        Label for the legend in the Lod plot.
    c_lod: str
        lod column name
    """

    # Filter queries in LoD dict that raise an error (e.g. contain non existing fields like "is_mixed")
    lod_filters = filter_valid_queries(df_test=df.head(10), queries=lod_filters)

    # apply filters
    featuremap_df = df.assign(
        **{
            filter_name: df.eval(filter_query)
            for filter_name, filter_query in tqdm(lod_filters.items())
            if filter_query is not None
        }
    )
    df_fp = featuremap_df.query("label == 0")
    df_tp = featuremap_df.query("label == 1")

    # count the # of bases in region
    n_bases_in_region = BedUtils().count_bases_in_bed_file(single_sub_regions)

    # get coverage statistics
    (
        mean_coverage,
        ratio_of_reads_over_mapq,
        ratio_of_bases_in_coverage_range,
        _,
        _,
    ) = read_effective_coverage_from_sorter_json(sorter_json_stats_file)

    # calculate the read filter correction factor (TP reads pre-filtered from the FeatureMap)
    read_filter_correction_factor = 1
    if (f"{FeatureMapFields.FILTERED_COUNT.value}" in df_fp) and (f"{FeatureMapFields.READ_COUNT.value}" in df_fp):
        read_filter_correction_factor = (df_fp[f"{FeatureMapFields.FILTERED_COUNT.value}"] + 1).sum() / df_fp[
            f"{FeatureMapFields.READ_COUNT.value}"
        ].sum()
    else:
        logger.warning(
            f"{FeatureMapFields.FILTERED_COUNT.value} or {FeatureMapFields.READ_COUNT.value} no in dataset, \
                read_filter_correction_factor = 1 in LoD"
        )

    # calculate the error rate (residual SNV rate) normalization factors
    n_noise_reads = df_fp.shape[0]
    n_signal_reads = df_tp.shape[0]
    effective_bases_covered = (
        mean_coverage
        * n_bases_in_region
        * ratio_of_reads_over_mapq
        * ratio_of_bases_in_coverage_range
        * read_filter_correction_factor
    )
    # TODO: next line needs correction when k_folds==1 (test/train split)??
    # Reason: did not take into account that the test set is only specific positions
    residual_snv_rate_no_filter = (
        fp_featuremap_entry_number / effective_bases_covered
    )  # n_noise_reads / effective_bases_covered
    ratio_filtered_prior_to_featuremap = ratio_of_reads_over_mapq * read_filter_correction_factor
    logger.info(f"{n_noise_reads=}, {n_signal_reads=}, {effective_bases_covered=}, {residual_snv_rate_no_filter=}")
    logger.info(
        f"Normalization factors: {mean_coverage=:.1f}, "
        f"{ratio_of_reads_over_mapq=:.3f}, "
        f"{ratio_of_bases_in_coverage_range=:.3f}, "
        f"{read_filter_correction_factor=:.3f}, "
        f"{n_bases_in_region=:.0f}"
    )
    normalization_factor_dict = {
        "n_noise_reads": n_noise_reads,
        "n_signal_reads": n_signal_reads,
        "mean_coverage": mean_coverage,
        "n_bases_in_region": n_bases_in_region,
        "ratio_of_reads_over_mapq": ratio_of_reads_over_mapq,
        "ratio_of_bases_in_coverage_range": ratio_of_bases_in_coverage_range,
        "read_filter_correction_factor": read_filter_correction_factor,
        "effective_bases_covered": effective_bases_covered,
        "residual_snv_rate_no_filter": residual_snv_rate_no_filter,
        "ratio_filtered_prior_to_featuremap": ratio_filtered_prior_to_featuremap,
    }

    # Calculate simulated LoD definitions and correction factors
    effective_signature_bases_covered = int(
        simulated_coverage * simulated_signature_size
    )  # The ratio_filtered_prior_to_featuremap is not taken into account here, but later in the binomial calculation

    # create a dataframe with the LoD parameters per filter
    df_mrd_simulation = pd.concat(
        (
            (df_fp[list(lod_filters.keys())].sum() / n_noise_reads).rename(FP_READ_RETENTION_RATIO),
            (df_tp[list(lod_filters.keys())].sum() / n_signal_reads * ratio_filtered_prior_to_featuremap).rename(
                TP_READ_RETENTION_RATIO
            ),
        ),
        axis=1,
    )

    # remove filters with a low TP read retention ratio
    min_tp_retention = 0.01
    df_mrd_simulation = df_mrd_simulation[df_mrd_simulation[TP_READ_RETENTION_RATIO] > min_tp_retention]

    # Assign the residual SNV rate per filter
    df_mrd_simulation.loc[:, RESIDUAL_SNV_RATE] = (
        df_mrd_simulation[FP_READ_RETENTION_RATIO]
        * residual_snv_rate_no_filter
        / df_mrd_simulation[TP_READ_RETENTION_RATIO]
    )

    # Calculate the minimum number of reads required for detection, per filter, assuming a binomial distribution
    # The probability of success is the product of the read retention ratio and the residual SNV rate
    df_mrd_simulation = df_mrd_simulation.assign(
        min_reads_for_detection=np.ceil(
            binom.ppf(
                n=int(effective_signature_bases_covered),
                p=df_mrd_simulation[TP_READ_RETENTION_RATIO] * df_mrd_simulation[RESIDUAL_SNV_RATE],
                q=specificity_at_lod,
            )
        )
        .clip(min=minimum_number_of_read_for_detection)
        .astype(int),
    )

    # Simulate the LoD per filter, assuming a binomial distribution
    # The simulation is done by drawing the expected number of reads in the Qth percentile, where the percentile is
    # (1-sensitivity), for a range of tumor frequencies (tf_sim). The lowest tumor fraction that passes the prescribed
    # minimum number of reads for detection is the LoD.
    tf_sim = np.logspace(-8, 0, 500)
    c_lod = f"LoD_{sensitivity_at_lod * 100:.0f}"
    df_mrd_simulation = df_mrd_simulation.join(
        df_mrd_simulation.apply(
            lambda row: tf_sim[
                np.argmax(
                    binom.ppf(
                        q=1 - sensitivity_at_lod,
                        n=int(effective_signature_bases_covered),
                        p=row[TP_READ_RETENTION_RATIO] * (tf_sim + row[RESIDUAL_SNV_RATE]),
                    )
                    >= row["min_reads_for_detection"]
                )
            ],
            axis=1,
        ).rename(c_lod)
    )

    lod_label = f"LoD @ {specificity_at_lod * 100:.0f}% specificity, \
    {sensitivity_at_lod * 100:.0f}% sensitivity (estimated)\
    \nsignature size {simulated_signature_size}, \
    {simulated_coverage}x coverage"

    if output_dataframe_file:
        df_mrd_simulation.to_parquet(output_dataframe_file)

    return df_mrd_simulation, lod_filters, lod_label, c_lod, normalization_factor_dict


@exception_handler
def plot_LoD(  # noqa N802
    df_mrd_sim: pd.DataFrame,
    lod_label: str,
    c_lod: str,
    filters: dict,
    adapter_version: str,
    min_LoD_filter: str,  # noqa: N803
    title: str = "",
    output_filename: str = None,
    font_size: int = 24,
    extra_filters: dict = None,
):
    """generates and saves the LoD plot

    Parameters
    ----------
    df_mrd_sim : pd.DataFrame
        the estimated LoD parameters per filter
    lod_label : str
        label for the LoD axis in the Lod plot
    c_lod : str
        lod column name
    filters : dict
        filters applied on data to estimate LoD
    adapter_version : str
        adapter version, indicates if input featuremap is from balanced ePCR data
    min_LoD_filter : str
        the filter which minimizes the LoD
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    # TODO: a less patchy solution, perhaps accept ml_filters as a separate input
    ml_filters = {i: filters[i] for i in filters if i[:2] == "ML" and i in df_mrd_sim.index}
    mixed_ml_filters = {i: filters[i] for i in filters if i[:8] == "mixed_ML" and i in df_mrd_sim.index}

    fig = plt.figure(figsize=(20, 12))

    filters_list = [
        list(ml_filters),
        ["no_filter"],
        ["HQ_SNV"],
    ]
    markers_list = ["o", "*", "D"]
    labels_list = [
        "ML model",
        "No filter",
        "HQ_SNV and Edit Dist <= 5",
    ]
    edgecolors_list = ["r", "r", "r"]
    msize_list = [150, 150, 150]
    if adapter_version in [av.value for av in PpmseqAdapterVersions]:
        filters_list.append(list(mixed_ml_filters))
        markers_list.append("P")
        labels_list.append("ML model, mixed only")
        edgecolors_list.append("r")
        msize_list.append(150)

        filters_list.append(["HQ_SNV_mixed_only"])
        markers_list.append(">")
        labels_list.append("HQ_SNV and Edit Dist <= 5, mixed only")
        edgecolors_list.append("r")
        msize_list.append(150)

        filters_list.append(["CSKP_mixed_only"])
        markers_list.append("s")
        labels_list.append("CSKP and Edit Dist <= 5, mixed only")
        edgecolors_list.append("r")
        msize_list.append(150)

    if extra_filters is not None:
        filters_list += [list(extra_filters.values())]
        markers_list += ["^", "X", "v", "h", "p"][: len(extra_filters)]
        labels_list += list(extra_filters.keys())
        edgecolors_list += ["r"] * len(extra_filters)
        msize_list += [150] * len(extra_filters)

    best_lod = df_mrd_sim.loc[
        [item for sublist in filters_list for item in sublist if item in df_mrd_sim.index.to_numpy()],
        c_lod,
    ].min()  # best LoD across all plotted results

    for f, marker, label, edgecolor, markersize in zip(
        filters_list, markers_list, labels_list, edgecolors_list, msize_list, strict=False
    ):
        df_plot = df_mrd_sim.loc[df_mrd_sim.index.isin(f)]
        plt.plot(
            df_plot[TP_READ_RETENTION_RATIO],
            -10 * np.log10(df_plot[RESIDUAL_SNV_RATE]),
            c="k",
            alpha=0.3,
        )
        best_lod_filter = df_plot[c_lod].min()
        plt.scatter(
            df_plot[TP_READ_RETENTION_RATIO],
            -10 * np.log10(df_plot[RESIDUAL_SNV_RATE]),
            c=df_plot[c_lod],
            marker=marker,
            edgecolor=edgecolor,
            label=f"{label}, best LoD: {best_lod_filter:.1E}".replace("E-0", "E-"),
            s=markersize,
            zorder=markersize,
            norm=colors.LogNorm(
                vmin=best_lod,
                vmax=best_lod * 10,
            ),
            cmap="inferno_r",
        )
    plt.xlabel("Recall (Base retention ratio on HOM SNVs)", fontsize=font_size)
    plt.ylabel("FQ  [ phred( FPR/TPR ) ]", fontsize=font_size)
    title_handle = plt.title(title, fontsize=font_size)
    legend_handle = plt.legend(fontsize=18, fancybox=True, framealpha=0.95)

    cbar = plt.colorbar()
    cbar.set_label(label=lod_label)
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # cbar.ax.yaxis.set_major_formatter(formatter)

    fig.text(
        0.5,
        0.01,
        f"ML qual threshold for min LoD: {min_LoD_filter}",
        ha="center",
        bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5},
    )
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_confusion_matrix(
    df: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    prediction_column_name="ML_prediction_1",
    font_size: int = 18,
):
    """generates and saves confusion matrix

    Parameters
    ----------
    df : pd.DataFrame
        data set with labels and model predictions
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    prediction_column_name : str, optional
        column name of the model predictions, by default "ML_prediction_1"
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    cmat = confusion_matrix(y_true=df[LABEL], y_pred=df[prediction_column_name])
    magic_threshold = 0.01
    cmat_norm = cmat.astype("float") / cmat.sum(axis=1)[:, np.newaxis]  # normalize by rows - true labels
    plt.figure(figsize=(4, 3))
    plt.grid(visible=False)
    ax = sns.heatmap(
        cmat_norm,
        annot=cmat_norm,
        annot_kws={"size": 16},
        cmap="Blues",
        cbar=False,
        fmt=".2%" if cmat_norm.min() < magic_threshold else ".1%",
    )
    ax.set_xticks(ticks=[0.5, 1.5])
    ax.set_xticklabels(labels=["FP", "TP"])
    ax.set_xlabel("Predicted label", fontsize=font_size)
    ax.set_yticks(ticks=[0.5, 1.5])
    ax.set_yticklabels(labels=["FP", "TP"], rotation="horizontal")
    ax.set_ylabel("True label", fontsize=font_size)

    title_handle = plt.title(title)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle],
        )


def plot_observed_vs_measured_qual(
    labels_dict: dict,
    fprs: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate and saves a plot of observed ML qual vs measured FP rates

    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    fprs : dict
        list of false positive rates per label
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    plt.figure(figsize=(8, 6))
    for label, item in labels_dict.items():
        plot_precision_recall(
            fprs[label],
            [f"measured qual {item}"],
            log_scale=False,
            max_score=max_score,
        )
    plt.plot([0, max_score], [0, max_score], "--")
    plt.xlabel("ML qual", fontsize=font_size)
    plt.ylabel("measured qual", fontsize=font_size)
    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_qual_density(
    labels_dict: dict,
    recalls: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate and saves a plot of measured recall rates vs ML qual

    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    recalls : dict
        list of recalls rates per label
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    plt.figure(figsize=(8, 6))

    for label, item in labels_dict.items():
        plot_precision_recall(
            recalls[label],
            [f"density {item}"],
            log_scale=False,
            max_score=max_score,
        )

    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)
    plt.xlabel("ML qual", fontsize=font_size)
    plt.ylabel("qual density", fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_precision_recall_vs_qual_thresh(
    df: pd.DataFrame,
    labels_dict: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate and saves a plot of precision and recall rates vs ML qual threshold

    Parameters
    ----------
    df : pd.DataFrame
        data set with features, labels and quals of the model
    labels_dict : dict
        dict of label values and names
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    plt.figure(figsize=(8, 6))
    for label, item in labels_dict.items():
        cum_avg_precision_recalls = []
        gtr = df[LABEL] == label
        cum_fprs_, cum_recalls_ = precision_recall_curve(
            df[f"ML_qual_{label}"],
            max_score=max_score,
            y_true=gtr,
            cumulative=True,
            apply_log_trans=False,
        )
        cum_avg_precision_recalls.append(
            [(precision + recall) / 2 for precision, recall in zip(cum_fprs_, cum_recalls_, strict=False)]
        )

        plot_precision_recall(
            cum_avg_precision_recalls,
            [f"avg(precision,recall) {item}"],
            log_scale=False,
            max_score=max_score,
        )

    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)
    plt.xlabel("ML qual thresh", fontsize=font_size)
    plt.ylabel("precision/recall average", fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_ml_qual_hist(
    labels_dict: dict,
    df: pd.DataFrame,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate and save histogram of ML qual per label

    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    df : pd.DataFrame
        data set with features, labels and quals of the model
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    score = "ML_qual_1"

    plt.figure(figsize=[8, 6])
    bins = np.arange(0, max_score + 1)
    for label, item in labels_dict.items():
        plt.hist(
            df[df[LABEL] == label][score].clip(upper=max_score),
            bins=bins,
            alpha=0.5,
            label=item,
            density=True,
        )

    plt.xlabel("ML qual", fontsize=font_size)
    plt.ylabel("Density", fontsize=font_size)
    plt.yscale("log")
    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_qual_per_feature(  # noqa: C901, PLR0912 #TODO: too complex and too many branches
    labels_dict: dict,
    cls_features: list,
    df: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate and save distributions of ML qual per input feature

    df : pd.DataFrame
        data set with features, labels and quals of the model
    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    cls_features : list
        list of input features for the ML model
    df : pd.DataFrame
        data set with features, labels and quals of the model
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    for feature in cls_features:
        for j, label in enumerate(labels_dict):
            if df[feature].dtype == bool:
                if j == 0:
                    plt.figure(figsize=(8, 6))
                _ = (
                    df[df[LABEL] == label][feature]
                    .astype(int)
                    .hist(
                        bins=[-0.5, 0.5, 1.5],
                        rwidth=0.8,
                        align="mid",
                        alpha=0.5,
                        label=labels_dict[label],
                        density=True,
                    )
                )
                plt.xticks([0, 1], ["False", "True"])
            elif df[feature].dtype in {
                "category",
                "object",
            }:
                if j == 0:
                    plt.figure(figsize=(8, 6))
                category_counts = df[df[LABEL] == label][feature].value_counts().sort_index()
                plt.bar(
                    category_counts.index,
                    category_counts,
                    alpha=0.5,
                    label=labels_dict[label],
                )
                xticks = plt.gca().get_xticks()
                if len(xticks) > 100:  # noqa: PLR2004
                    plt.xticks(rotation=90, fontsize=6)
                elif len(xticks) > 30:  # noqa: PLR2004
                    plt.xticks(rotation=90, fontsize=9)
                elif len(xticks) > 3:  # noqa: PLR2004
                    plt.xticks(rotation=30, fontsize=12)
            else:
                if j == 0:
                    plt.figure(figsize=(8, 6))
                s_plot = df[df[LABEL] == label][feature]
                if s_plot.dtype.name in {"bool", "category"}:  # category bar plot
                    s_plot.value_counts(normalize=True).plot(kind="bar", label=labels_dict[label])
                else:  # numerical histogram
                    s_plot.hist(bins=min(len(s_plot.unique()), 20), alpha=0.5, label=labels_dict[label], density=True)

        legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
        plt.xlabel(feature, fontsize=font_size)
        title_handle = plt.title(title, fontsize=font_size)
        output_filename_feature = output_filename + feature
        if output_filename_feature is not None:
            if not output_filename_feature.endswith(".png"):
                output_filename_feature += ".png"
        plt.savefig(
            output_filename_feature,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def get_data_subsets(
    df: pd.DataFrame,
    *,
    is_mixed_flag: bool,
):
    """generates subsets of the input df by category: mixed/non mixed * cycle-skip/non cycle-skip

    Parameters
    ----------
    df : pd.DataFrame
        data set with features, labels and quals of the model
    is_mixed_flag : bool
        indicates if the data set contains mixed reads

    Returns
    -------
    df_dict: dict
        dict of data subsets with features, labels and quals of the model
    """

    df_dict = {}

    df[IS_CYCLE_SKIP] = df[IS_CYCLE_SKIP].astype(bool)
    if is_mixed_flag:
        df[IS_MIXED] = df[IS_MIXED].astype(bool)
        df_dict["mixed cycle skip"] = df[(df[IS_MIXED] & df[IS_CYCLE_SKIP])]
        df_dict["mixed non cycle skip"] = df[(df[IS_MIXED] & ~df[IS_CYCLE_SKIP])]
        df_dict["non mixed non cycle skip"] = df[(~df[IS_MIXED] & ~df[IS_CYCLE_SKIP])]
        df_dict["non mixed cycle skip"] = df[(~df[IS_MIXED] & df[IS_CYCLE_SKIP])]
    else:
        df_dict["cycle skip"] = df[df[IS_CYCLE_SKIP]]
        df_dict["non cycle skip"] = df[~df[IS_CYCLE_SKIP]]

    return df_dict


def get_fpr_recalls_subsets(
    df_dict: dict,
    max_score: float,
):
    """get the FP and recall rates for subsamples of the data: mixed/non mixed * cycle skip/non cycle skip

    Parameters
    ----------
    df_dict : dict
        dict of data subsets with features, labels and quals of the model
    max_score : float
        maximal ML model score

    Returns
    -------
    fpr_dict: dict
        dict of false positive rates per data subset
    recall_dict: dict
        dict of recall rates per data subset
    """
    score = "ML_qual_1"
    label = 1

    fpr_dict = {}
    recall_dict = {}

    for key in df_dict:
        gtr = df_dict[key][LABEL] == label
        fpr_dict[key], recall_dict[key] = precision_recall_curve(df_dict[key][score], max_score=max_score, y_true=gtr)

    return fpr_dict, recall_dict


def plot_subsets_hists(
    labels_dict: dict,
    df_dict: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate and save histograms of ML qual, separated by subsets: mixed/non mixed and cycle-skip/non cycle-skip

    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    df_dict : dict
        dict of data subsets with features, labels and quals of the model
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    score = "ML_qual_1"
    bins = np.arange(0, max_score + 1)

    for name, td in df_dict.items():
        plt.figure(figsize=(8, 6))
        for label, item in labels_dict.items():
            h, bin_edges = np.histogram(
                td[td[LABEL] == label][score].clip(upper=max_score),
                bins=bins,
                density=True,
            )
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            plt.bar(
                bin_centers,
                h,
                label=item,
                alpha=0.8,
                width=1,
                align="center",
            )
            if any(h > 0):
                plt.yscale("log")
        plt.xlim([0, max_score])
        legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
        feature_title = title + name
        title_handle = plt.title(feature_title, fontsize=font_size)
        output_filename_feature = output_filename + name.replace(" ", "_")
        if output_filename_feature is not None:
            if not output_filename_feature.endswith(".png"):
                output_filename_feature += ".png"
        plt.xlabel("ML qual", fontsize=font_size)
        plt.ylabel("Density", fontsize=font_size)
        plt.savefig(
            output_filename_feature,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_mixed_fpr(
    fpr_dict: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate and save a plot of FP rates per subset: mixed/non-mixed * cycle-skip/non cycle-skip

    Parameters
    ----------
    fpr_dict : dict
        dict of false positive rates per data subset
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    plt.figure(figsize=(8, 6))
    plot_precision_recall(
        [fpr_dict[item] for item in fpr_dict],
        [item.replace("_", " ") for item in fpr_dict],
        log_scale=False,
        max_score=max_score,
    )
    plt.plot([0, 40], [0, 40], "--")
    plt.xlim([0, max_score])
    plt.xlabel("ML qual", fontsize=font_size)
    plt.ylabel("Observed qual", fontsize=font_size)
    legend_handle = plt.legend(fontsize=font_size - 4, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
    plt.savefig(
        output_filename,
        facecolor="w",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[title_handle, legend_handle],
    )


def plot_mixed_recall(
    recall_dict: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate and save a plot of recall rates per subset: mixed/non-mixed * cycle-skip/non cycle-skip

    Parameters
    ----------
    recall_dict : dict
        dict of recall rates per data subset
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 18
    """
    set_pyplot_defaults()

    plt.figure(figsize=(8, 6))
    plot_precision_recall(
        [recall_dict[item] for item in recall_dict],
        [item.replace("_", " ") for item in recall_dict],
        log_scale=False,
        max_score=max_score,
    )

    plt.xlim([0, max_score])
    plt.ylabel("Recall rate", fontsize=font_size)
    legend_handle = plt.legend(fontsize=font_size - 6, fancybox=True, framealpha=0.95, loc="upper right")
    title_handle = plt.title(title, fontsize=font_size)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
    plt.savefig(
        output_filename,
        facecolor="w",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[title_handle, legend_handle],
    )


def _get_plot_paths(out_path, out_basename):
    outdir = out_path
    basename = out_basename

    FQ_vs_recall_plot = os.path.join(outdir, f"{basename}FQ_vs_recall")  # noqa: N806
    qual_vs_ppmseq_tags_table = os.path.join(outdir, f"{basename}qual_vs_ppmSeq_tags_table")
    training_progerss_plot = os.path.join(outdir, f"{basename}training_progress")
    SHAP_importance_plot = os.path.join(outdir, f"{basename}SHAP_importance")  # noqa: N806
    SHAP_beeswarm_plot = os.path.join(outdir, f"{basename}SHAP_beeswarm")  # noqa: N806
    trinuc_stats_plot = os.path.join(outdir, f"{basename}trinuc_stats")
    qual_histogram = os.path.join(outdir, f"{basename}qual_histogram")
    logit_histogram = os.path.join(outdir, f"{basename}logit_histogram")
    calibration_fn_with_hist = os.path.join(outdir, f"{basename}calibration_fn_with_hist")
    output_qual_per_feature = os.path.join(outdir, f"{basename}qual_per_")

    return [
        FQ_vs_recall_plot,
        qual_vs_ppmseq_tags_table,
        training_progerss_plot,
        SHAP_importance_plot,
        SHAP_beeswarm_plot,
        trinuc_stats_plot,
        output_qual_per_feature,
        qual_histogram,
        logit_histogram,
        calibration_fn_with_hist,
    ]


@exception_handler
def calculate_lod_stats(
    df_mrd_simulation: pd.DataFrame,
    output_h5: str,
    lod_column: str,
    qualities_to_interpolate: tuple | list | np.array = tuple(range(30, 71)),
):
    """Calculate noise and LoD stats from the simulated data

    Parameters
    ----------
    df_mrd_simulation : pd.DataFrame
        simulated data set with noise and LoD stats
    output_h5 : str
        path to output h5 file
    lod_column : str
        name of the LoD column in the data set
    qualities_to_interpolate : tuple
        list of qualities to interpolate what percent of bases surpass, by default all integers in [30,70]

    Returns
    -------
    best_LoD_filter: str
        name of the filter with the best LoD
    """
    df_mrd_simulation_ml = df_mrd_simulation.loc[df_mrd_simulation.index.str.startswith("ML")]
    df_mrd_simulation_mixed_ml = df_mrd_simulation.loc[df_mrd_simulation.index.str.startswith("mixed_ML")]
    df_mrd_simulation_non_ml = df_mrd_simulation.loc[
        ~df_mrd_simulation.index.str.startswith("ML") & ~df_mrd_simulation.index.str.startswith("mixed_ML")
    ]
    # Calculate percent of bases over each given quality threshold
    x_interp = -10 * np.log10(df_mrd_simulation_ml["residual_snv_rate"])  # Phred score
    y_interp = df_mrd_simulation_ml["tp_read_retention_ratio"]
    base_stats = pd.Series(
        index=qualities_to_interpolate,
        data=interp1d(x_interp, y_interp, bounds_error=False, fill_value=(1, 0))(qualities_to_interpolate),
    )  # fill_value=(1, 0) means that values under the minimal Qual will be 1 and values over the maximal will be 0
    base_stats.index.name = "QualThreshold"
    base_stats.name = "BasesOverQualThreshold"
    base_stats.to_hdf(output_h5, key="bases_over_qual_threshold", mode="a")

    # Calculate Optimal filter and LoD result
    best_LoD_filter = df_mrd_simulation[lod_column].idxmin()  # noqa: N806
    best_lod = df_mrd_simulation.loc[best_LoD_filter, lod_column]
    best_ML_LoD_filter = df_mrd_simulation_ml[lod_column].idxmin()  # noqa: N806
    best_ML_LoD = df_mrd_simulation_ml.loc[best_ML_LoD_filter, lod_column]  # noqa: N806
    if df_mrd_simulation_mixed_ml.shape[0] > 0:
        best_mixed_ML_LoD_filter = df_mrd_simulation_mixed_ml[lod_column].idxmin()  # noqa: N806
        best_mixed_ML_LoD = df_mrd_simulation_mixed_ml.loc[best_mixed_ML_LoD_filter, lod_column]  # noqa: N806
    else:
        best_mixed_ML_LoD_filter = pd.NA  # noqa: N806
        best_mixed_ML_LoD = pd.NA  # noqa: N806
    lod_stats_dict = {
        "best_LoD_filter": best_LoD_filter,
        "LoD_best": best_lod,
        "best_ML_LoD_filter": best_ML_LoD_filter,
        "LoD_best_ML": best_ML_LoD,
        "best_mixed_ML_LoD_filter": best_mixed_ML_LoD_filter,
        "LoD_best_mixed_ML": best_mixed_ML_LoD,
    }
    for column, stat_name in zip(
        (lod_column, RESIDUAL_SNV_RATE, TP_READ_RETENTION_RATIO),
        ("LoD", RESIDUAL_SNV_RATE, TP_READ_RETENTION_RATIO),
        strict=False,
    ):
        lod_stats_dict.update(
            {f"{stat_name}-{filter_name}": row[column] for filter_name, row in df_mrd_simulation_non_ml.iterrows()}
        )

    logger.info(f"min_LoD_filter {best_LoD_filter} (LoD={best_lod:.1e})")
    lod_stats = pd.Series(lod_stats_dict)
    lod_stats.to_hdf(output_h5, key="lod_stats", mode="a")

    return best_LoD_filter


class SRSNVReport:
    def __init__(  # noqa: PLR0913 C901 PLR0915 PLR0912
        self,
        models: list[sklearn.base.BaseEstimator],
        data_df: pd.DataFrame,
        params: dict,
        out_path: str,
        srsnv_metadata: dict,
        base_name: str = None,
        lod_filters: dict = None,
        lod_label: dict = None,
        c_lod: str = "LoD",
        # min_LoD_filter: str = None,
        df_mrd_simulation: pd.DataFrame = None,
        ml_qual_to_qual_fn: Callable = None,
        statistics_h5_file: str = None,
        statistics_json_file: str = None,
        rng: Any = None,
        *,
        use_gpu_for_shap: bool = False,
        raise_exceptions: bool = False,
    ):
        """loads model, data, params and generate plots for report. Saves data in hdf5 file

        Parameters
        ----------
        models : list[sklearn.base.BaseEstimator]
            A list of SKlearn models (for all folds)
        df : pd.DataFrame
            Dataframe of all fold data, including labels
        params : str
            params dict
        out_path : str
            path to output directory
        base_name : str, optional
            base name for output files, by default None
        lod_filters : dict, optional
            filters for LoD simulation, by default None (default_LoD_filters)
        df_mrd_simulation : pd.DataFrame, optional
            dataframe created by MRD simulation, by default None
        statistics_h5_file : str, optional
            path to output h5 file with stats, by default None
        statistics_json_file : str, optional
            path to output json file with stats, by default None
        """
        exception_config.raise_exception = raise_exceptions
        self.models = models
        self.num_cv_folds = len(models)
        self.data_df = data_df
        self.params = params
        self.use_gpu_for_shap = use_gpu_for_shap
        self.out_path = out_path
        self.base_name = base_name
        self.lod_filters = lod_filters
        self.lod_label = lod_label
        self.c_lod = c_lod
        # self.min_LoD_filter = min_LoD_filter
        self.df_mrd_simulation = df_mrd_simulation
        self.ML_qual_to_qual_fn = ml_qual_to_qual_fn
        self.statistics_h5_file = statistics_h5_file
        self.statistics_json_file = statistics_json_file
        with open(srsnv_metadata) as f:
            self.srsnv_metadata = json.load(f)
        self.max_qual = self.srsnv_metadata["training_parameters"].get("max_qual", MAX_PHRED)
        self.eps = 10 ** (-self.max_qual / 10)
        if rng is None:
            random_seed = int(datetime.now().timestamp())
            rng = np.random.default_rng(seed=random_seed)
            logger.info(f"SRSNVReport: Initializing random number generator with {random_seed=}")
            self.rng = np.random.default_rng(seed=14)
        else:
            self.rng = rng

        self.all_features = [f["name"] for f in self.srsnv_metadata["features"]]

        if statistics_json_file:  # TODO: Is this needed? Clean this up!
            if not statistics_h5_file:
                raise ValueError("statistics_h5_file is required when statistics_json_file is provided")

        # check model, data and params
        if not isinstance(models, list):
            raise TypeError(f"models should be a list of models, got {type(models)=}")
        for k, model in enumerate(models):
            # Check if it's a classifier (either sklearn or xgboost)
            is_valid_classifier = sklearn.base.is_classifier(model) or (
                hasattr(model, "predict_proba") and hasattr(model, "__class__") and "XGB" in model.__class__.__name__
            )
            if not is_valid_classifier:
                raise ValueError(f"model {model} (fold {k}) is not a classifier, please provide a classifier model")
        if not isinstance(data_df, pd.DataFrame):
            raise TypeError("df is not a DataFrame, please provide a DataFrame")
        expected_keys_in_params = [  # TODO: Clean up!
            #     "fp_featuremap_entry_number",
            #     "fp_test_set_size",
            #     "fp_train_set_size",  # Do I really need this?
            #     "fp_regions_bed_file",
            #     "sorter_json_stats_file",
            #     "adapter_version",
        ]
        for key in expected_keys_in_params:
            if key not in params:
                raise ValueError(f"no {key} in params")
        self.start_tag_col = self.params.get("start_tag_col", None)
        self.end_tag_col = self.params.get("end_tag_col", None)

        # init dir
        os.makedirs(out_path, exist_ok=True)
        self.params["workdir"] = out_path
        if base_name:
            self.params["data_name"] = base_name
        else:
            self.params["data_name"] = ""

        self.output_h5_filename = os.path.join(out_path, f"{base_name}single_read_snv.applicationQC.h5")

        # Add trinuc_context_with_alt column if it doesn't exist
        if TRINUC_CONTEXT_WITH_ALT not in self.data_df.columns:
            logger.info(f"Adding {TRINUC_CONTEXT_WITH_ALT} column to data_df")
            self.data_df.loc[:, TRINUC_CONTEXT_WITH_ALT] = construct_trinuc_context_with_alt(self.data_df)

        # Add is_forward column as negation of REV column
        if IS_FORWARD not in self.data_df.columns and REV in self.data_df.columns:
            logger.info(f"Adding {IS_FORWARD} column to data_df as negation of {REV}")
            self.data_df.loc[:, IS_FORWARD] = self.data_df[REV].astype(int) != 1

        # Resolve the read-split "recipe" (scheme). The scheme may be provided by the caller
        # (params["report_mode"], set by the CLI) or auto-detected from the dataframe columns.
        report_mode_param = self.params.get("report_mode")
        if report_mode_param:
            self.scheme = resolve_scheme(mode=report_mode_param)
        else:
            self.scheme = resolve_scheme(self.data_df)
        self.report_mode = self.scheme.mode  # kept for metadata / backward compat

        # Add the split column(s) if they don't exist yet (e.g. when SRSNVReport is constructed
        # directly, without going through prepare_report which already adds them).
        if READ_GROUP not in self.data_df.columns:
            logger.info(f"Adding split column(s) for split scheme {self.scheme.mode.value} to data_df")
            adapter_version = self.params.get("adapter_version", None)
            categorical_features = [f["name"] for f in self.srsnv_metadata["features"] if f.get("type") == "c"]
            self.data_df, self.scheme = resolve_scheme_and_add_columns(
                self.data_df, adapter_version=adapter_version, categorical_features_names=categorical_features
            )
            self.report_mode = self.scheme.mode

        # add logits to data_df
        self.data_df[ML_LOGIT_TEST] = prob_to_logit(
            self.data_df[ML_PROB_1_TEST], max_value=self.max_qual
        )  # TODO: Should this be prob_recal ??
        # Find training probabilities and logits
        all_model_probs = (
            self.data_df[[f"prob_fold_{i}" for i in range(self.num_cv_folds)]].to_numpy().T
        )  # TODO: change hard coded prob_fold_{i}
        _, preds_train = split_validation_training_preds(all_model_probs, fold_arr=self.data_df[FOLD_ID].to_numpy())
        self.data_df["ML_prob_train"] = preds_train
        self.data_df["ML_logit_train"] = prob_to_logit(preds_train, max_value=self.max_qual)

    # ──────────────────────── split-scheme helpers ────────────────────────
    # These give every figure/table method a uniform, scheme-agnostic interface: iterate
    # `self.scheme.variants`, get group masks / colors / h5 keys from the variant. No method
    # switches on the scheme.

    @property
    def _display_variant(self):
        return self.scheme.display_variant

    @property
    def tag_axis(self):
        return self.scheme.tag_axis

    def _display_groups(self):
        """Ordered list of the display variant's group labels."""
        return list(self._display_variant.groups)

    def _group_masks(self, variant=None, data_df=None):
        """Yield ``(label, boolean_mask)`` per group, in order, for a variant (default: display)."""
        if variant is None:
            variant = self._display_variant
        if data_df is None:
            data_df = self.data_df
        yield from variant.group_masks(data_df).items()

    def _variant_palette(self, variant=None):
        """Stable ``{label: color}`` for a variant: its declared colors, else a seaborn palette."""
        if variant is None:
            variant = self._display_variant
        if variant.colors:
            return dict(variant.colors)
        colors = sns.color_palette(n_colors=len(variant.groups))
        return dict(zip(variant.groups, colors, strict=False))

    def _is_legacy_variant(self, variant):
        """True for the legacy (base-key) variant of a multi-variant scheme (mixed's both-ends)."""
        return len(self.scheme.variants) > 1 and variant is self.scheme.legacy_variant

    def _has_dual_split(self):
        """True when the scheme keeps a distinct legacy + display table (ppmSeq mixed only).

        Single-grouping schemes (consensus/none) render one per-group table; the dual-split path
        additionally emits the historical 3-way (start / both-ends) run-info & quality rows.
        """
        return len(self.scheme.variants) > 1

    def _variant_roc_groups(self, variant):
        """Ordered ``[(name, mask), ...]`` and the ROC-table index for a variant.

        Binary variants (``pos`` set) keep the historical ``[pos, neg]`` order and
        ``["Total", "{pos} only", "{neg} only"]`` index; N-group variants list one row per group.
        """
        masks = variant.group_masks(self.data_df)
        if variant.pos is not None:
            groups = [(variant.pos, masks[variant.pos]), (variant.neg, masks[variant.neg])]
            index = ["Total", f"{variant.pos} only", f"{variant.neg} only"]
        else:
            groups = list(masks.items())
            index = ["Total", *[f"{label} only" for label in variant.groups]]
        return groups, index

    def _write_variant_table(self, base, variant, obj):
        """Write ``obj`` to the variant's h5 key, and also to the bare ``base`` key when the scheme
        has no explicit legacy (``suffix==''``) variant (so keys_to_convert / json conversion works).
        """
        obj.to_hdf(self.output_h5_filename, key=split_h5_key(base, variant), mode="a")
        if variant.is_display and all(v.suffix != "" for v in self.scheme.variants):
            obj.to_hdf(self.output_h5_filename, key=base, mode="a")

    def _save_plt(self, output_filename: str = None, fig=None, *, tight_layout=True, **kwargs):
        if output_filename is not None:
            if not output_filename.endswith(".png"):
                output_filename += ".png"
            if fig is None:
                fig = plt.gcf()
            if tight_layout:
                fig.tight_layout()
            fig.savefig(output_filename, facecolor="w", dpi=300, bbox_inches="tight", **kwargs)

    def _get_snvq_and_recall(
        self,
        labels: np.ndarray,
        ml_quals: np.ndarray,
        base_snvq: float,
        base_recall: float,
        condition: np.ndarray = None,
        ml_qual_max: int = 25,
    ):
        """Get the values of snvq (residual snv rate) and recall (true positive retention rate)
        for different choices of (integer) ML_qual thresholds. ML_qual_max is the maximum value
        of ML_qual that is used as threshold.
        Arguments:
            labels: np.array: true labels
            ML_quals: np.array: ML_qual values
            base_snvq: float: base snvq value
            base_recall: float: base recall value
            condition: np.ndarray: a boolean array of same length as labels, to filter the data
                                 By default, no filtering is done
            ML_qual_max: int: maximum ML_qual threshold to consider
            NOTE: can think about choosing ML_qual_max in a more principled way. Initial thoughts:
                ML_qual_max = np.floor(ML_quals.max()).astype(int) - 5
                ML_qual_max = np.partition(ML_quals, -1000)[-1000]
        """
        if condition is None:
            condition = np.ones_like(labels, dtype=bool)
        snvqs = []
        recalls = []
        for th in range(ml_qual_max + 1):
            tpr = ((ml_quals > th) & condition & labels).sum() / labels.sum()
            fpr = ((ml_quals > th) & condition & ~labels).sum() / (~labels).sum()
            snvqs.append(prob_to_phred(1 - base_snvq * fpr / tpr / 3, max_value=self.max_qual))
            recalls.append(base_recall * tpr)
        return np.array(snvqs), np.array(recalls)

    def _get_recall_at_snvq(
        self,
        snvq: float,
        condition: np.ndarray = None,
        label_col: str = LABEL,
        snvq_col: str = QUAL,
        # ml_qual_col: str = ML_QUAL_1_TEST
    ):
        """Get the recall rate at a given SNVQ value, by interpolating the recall vs SNVQ curve.
        Arguments:
            snvq: float: SNVQ value at which to calculate the recall
            condition: np.ndarray: condition to filter the data frame. Default is no condition
            label_col: str: column name with the labels
            snvq_col: str: column name with the SNVQ values
        """
        # base_snv_rate = self.df_mrd_simulation.loc["no_filter", RESIDUAL_SNV_RATE]
        # base_recall = self.df_mrd_simulation.loc["no_filter", TP_READ_RETENTION_RATIO]
        data_df = self.data_df
        # snvqs, recalls = self._get_snvq_and_recall(
        #     data_df[label_col],
        #     data_df[ml_qual_col],
        #     base_snvq=self.base_error_rate,
        #     base_recall=self.base_recall,
        #     condition=condition,
        # )
        # recall_at_snvq = np.interp(snvq, snvqs, recalls)
        if condition is None:
            condition = np.ones_like(data_df[label_col], dtype=bool)
        # data_df = data_df.loc[condition,:]
        # data_df = data_df.loc[data_df[label_col] == 1, :]
        recall_at_snvq = (
            self.base_recall
            * ((data_df[snvq_col] >= snvq) & (data_df[label_col] == 1) & condition).sum()
            / data_df[label_col].sum()
        )
        return recall_at_snvq

    def _calculate_tpr_fpr_for_thresholds(self, mqual_thresholds, recall0):
        """Calculate TPR and FPR for given MQUAL thresholds.

        Args:
            mqual_thresholds: Array of MQUAL threshold values
            recall0: Base recall value

        Returns:
            tuple: (tprs, fprs) - lists of TPR and FPR values
        """
        tprs = []
        fprs = []
        total_positive = self.data_df[LABEL].sum()
        total_negative = (~self.data_df[LABEL]).sum()

        for mqual_threshold in mqual_thresholds:
            # Count positive cases with MQUAL > threshold (TPR calculation)
            positive_above_threshold = (self.data_df[LABEL] & (self.data_df[ML_QUAL_1_TEST] > mqual_threshold)).sum()

            # Calculate fraction of positive cases above threshold
            fraction_above_threshold = positive_above_threshold / total_positive if total_positive > 0 else 0

            # TPR = recall0 * fraction_above_threshold
            tpr = recall0 * fraction_above_threshold
            tprs.append(tpr)

            # Calculate FPR: fraction of negative cases with MQUAL > threshold
            negative_above_threshold = ((~self.data_df[LABEL]) & (self.data_df[ML_QUAL_1_TEST] > mqual_threshold)).sum()
            fpr = negative_above_threshold / total_negative if total_negative > 0 else 0
            fprs.append(fpr)

        return tprs, fprs

    def _calculate_filter_quality(self, tprs, fprs, base_error_rate):
        """Calculate Filter Quality (FQ) from TPR, FPR, and base error rate.

        Args:
            tprs: List of True Positive Rates
            fprs: List of False Positive Rates
            base_error_rate: Base error rate of low VAF (False) SNVs

        Returns:
            list: Filter Quality values in Phred scale
        """
        fqs = []

        for tpr, fpr in zip(tprs, fprs, strict=True):
            # Calculate precision using: 1 - precision = FPR/TPR * base_error_rate
            if tpr > 0:
                one_minus_precision = (fpr / tpr) * base_error_rate
                precision = 1 - one_minus_precision
                # Ensure precision is between 0 and 1
                precision = max(0, min(1, precision))
            else:
                precision = 0

            # Calculate Filter Quality (FQ) in Phred scale: FQ = -10 * log10(1-precision)
            fq = prob_to_phred(precision, max_value=self.max_qual)
            fqs.append(fq)

        return fqs

    def _calc_threshold_based_fq_recall(self, mquals, condition=None):
        data_df = self.data_df.copy()
        if condition is None:
            condition = np.ones_like(self.data_df[LABEL], dtype=bool)
        data_df["condition"] = condition
        condition_rates_by_label = data_df.groupby(LABEL)["condition"].mean().sort_index()
        cond_true_rate = condition_rates_by_label[True]
        cond_snvq_shift = np.diff(10 * np.log10(condition_rates_by_label.to_numpy()))

        kde_estimator = AdaptiveKDEPrecisionEstimator()
        kde_estimator.fit(data_df.loc[data_df["condition"], :], num_cv_folds=self.num_cv_folds)
        if kde_estimator.config["transform_mode"] == "logit":
            scores = prob_to_logit(phred_to_prob(mquals), max_value=self.max_qual, phred=True)
        else:
            scores = mquals

        tprs = kde_estimator.get_tpr(scores)
        fprs = kde_estimator.get_fpr(scores)
        recall = tprs * self.base_recall * cond_true_rate
        fq = self.base_fq + cond_snvq_shift + 10 * np.log10(tprs / fprs)
        return recall, fq

    @exception_handler
    def plot_fq_recall(  # noqa: C901
        self, output_filename: str = None, *, only_calculate: bool = False, font_size: int = 24
    ):
        """Calculate precision and recall metrics for SRSNV quality thresholds.

        Parameters
        ----------
        output_filename : str, optional
            path to which the plot will be saved, by default None
        only_calculate : bool, optional
            if True, only calculate values without plotting, by default False
        font_size : int, optional
            font size for the plot, by default 24

        Returns:
            pd.DataFrame: DataFrame with MQUAL, SNVQ, recall, FPR, and FQ columns
        """
        set_pyplot_defaults()

        # Create dataframe from quality recalibration table
        pr_df = pd.DataFrame(np.array(self.srsnv_metadata["quality_recalibration_table"]).T, columns=["MQUAL", "SNVQ"])

        # Get filtering statistics
        positive_filters = self.srsnv_metadata["filtering_stats"]["positive"]["filters"]

        # Calculate base values
        base_recall = get_base_recall_from_filters(positive_filters)
        base_fq = pr_df.loc[0, "SNVQ"]

        self.base_recall = base_recall
        self.base_fq = base_fq

        # Legacy variant conditions (mixed: the historical both-ends + start conditions).
        conditions_legacy = self._fq_conditions(self.scheme.legacy_variant)
        pr_df_legacy = pr_df.copy()
        for label, condition in conditions_legacy.items():
            recall, fq = self._calc_threshold_based_fq_recall(pr_df_legacy["MQUAL"].to_numpy(), condition=condition)
            pr_df_legacy[f"recall_{label}"] = recall
            pr_df_legacy[f"FQ_{label}"] = fq
        pr_df_legacy.to_hdf(self.output_h5_filename, key="FQ_recall_LoD", mode="a")

        # Display variant conditions (one curve per read group, or the binary positive group).
        conditions = self._fq_conditions(self._display_variant)

        if only_calculate:
            for label, condition in conditions.items():
                recall, fq = self._calc_threshold_based_fq_recall(pr_df["MQUAL"].to_numpy(), condition=condition)
                pr_df[f"recall_{label}"] = recall
                pr_df[f"FQ_{label}"] = fq
            self.pr_df = pr_df
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            for label, condition in conditions.items():
                recall, fq = self._calc_threshold_based_fq_recall(pr_df["MQUAL"].to_numpy(), condition=condition)
                pr_df[f"recall_{label}"] = recall
                pr_df[f"FQ_{label}"] = fq
                ax.plot(recall, fq, label=label, linewidth=2)
            ax.set_xlabel("Recall", fontsize=font_size)
            ax.set_ylabel("Filter Quality (FQ)", fontsize=font_size)
            ax.legend(fontsize=18, fancybox=True, framealpha=0.95)
            ax.grid(visible=True)
            self._save_plt(output_filename, fig)
        self.pr_df = pr_df
        self.pr_df.to_hdf(self.output_h5_filename, key=split_h5_key("FQ_recall_LoD", self._display_variant), mode="a")

    def _fq_conditions(self, variant):
        """Ordered ``{label: mask|None}`` FQ-recall conditions for a variant.

        Uses the variant's ``fq_conditions_fn`` when set (mixed's historical both-ends/start
        conditions); otherwise ``{"all reads": None}`` plus one condition per non-empty group.
        """
        if variant.fq_conditions_fn is not None:
            return variant.fq_conditions_fn(self.data_df)
        conditions = {"all reads": None}
        for label, mask in variant.group_masks(self.data_df).items():
            if mask.any():
                conditions[label] = pd.Series(mask, index=self.data_df.index)
        return conditions

    def get_dataset_sizes(self):
        """Calculate dataset sizes for different folds and overall."""
        dataset_sizes = {}
        if self.params["num_CV_folds"] >= 2:  # noqa: PLR2004
            dataset_sizes[("All", "Total")] = self.data_df.shape[0]
            dataset_sizes[("TPs", "Total")] = self.data_df[LABEL].sum()
            dataset_sizes[("FPs", "Total")] = (~self.data_df[LABEL]).sum()
            dataset_sizes[("% TPs", "Total")] = signif(100 * self.data_df[LABEL].mean(), SIG_DIGITS)
            for f in range(self.params["num_CV_folds"]):
                dataset_sizes[("All", f"fold {f}")] = (self.data_df[FOLD_ID] == f).sum()
                dataset_sizes[("TPs", f"fold {f}")] = ((self.data_df[FOLD_ID] == f) & (self.data_df[LABEL])).sum()
                dataset_sizes[("FPs", f"fold {f}")] = ((self.data_df[FOLD_ID] == f) & (~self.data_df[LABEL])).sum()
                dataset_sizes[("% TPs", f"fold {f}")] = signif(
                    ((self.data_df[FOLD_ID] == f) & (self.data_df[LABEL])).sum()
                    / ((self.data_df[FOLD_ID] == f).sum())
                    * 100,
                    SIG_DIGITS,
                )
            dataset_sizes[("All", "test only")] = (self.data_df[FOLD_ID].isna()).sum()
            dataset_sizes[("TPs", "test only")] = ((self.data_df[FOLD_ID].isna()) & (self.data_df[LABEL])).sum()
            dataset_sizes[("FPs", "test only")] = ((self.data_df[FOLD_ID].isna()) & (~self.data_df[LABEL])).sum()
            dataset_sizes[("% TPs", "test only")] = signif(
                ((self.data_df[FOLD_ID].isna()) & (self.data_df[LABEL])).sum()
                / ((self.data_df[FOLD_ID].isna()).sum())
                * 100,
                SIG_DIGITS,
            )
            other_fold_ids_cond = np.logical_and(
                ~self.data_df[FOLD_ID].isin(np.arange(self.params["num_CV_folds"])), ~self.data_df[FOLD_ID].isna()
            )
            if other_fold_ids_cond.any():
                dataset_sizes[("All", "other")] = other_fold_ids_cond.sum()
                dataset_sizes[("TPs", "other")] = (other_fold_ids_cond & (self.data_df[LABEL])).sum()
                dataset_sizes[("FPs", "other")] = (other_fold_ids_cond & (~self.data_df[LABEL])).sum()
                dataset_sizes[("% TPs", "other")] = signif(
                    (other_fold_ids_cond & self.data_df[LABEL]).sum() / (other_fold_ids_cond).sum() * 100, SIG_DIGITS
                )
        else:  # Single fold: val (fold_id=0) + test (fold_id=NaN)
            val_cond = self.data_df[FOLD_ID] == 0
            test_cond = self.data_df[FOLD_ID].isna()
            dataset_sizes = {
                ("All", "val"): val_cond.sum(),
                ("TPs", "val"): (val_cond & self.data_df[LABEL]).sum(),
                ("FPs", "val"): (val_cond & ~self.data_df[LABEL]).sum(),
                ("% TPs", "val"): signif(
                    (val_cond & self.data_df[LABEL]).sum() / val_cond.sum() * 100,
                    SIG_DIGITS,
                ),
                ("All", "test"): test_cond.sum(),
                ("TPs", "test"): (test_cond & self.data_df[LABEL]).sum(),
                ("FPs", "test"): (test_cond & ~self.data_df[LABEL]).sum(),
                ("% TPs", "test"): signif(
                    (test_cond & self.data_df[LABEL]).sum() / test_cond.sum() * 100,
                    SIG_DIGITS,
                ),
            }
        return dataset_sizes

    def _calc_run_info_table_by_group(self):
        """Consensus/none: build run info + quality summary with one column per read group.

        Writes the same 5 h5 keys as the binary path; legacy and display tables are identical
        (no start/end distinction). Column/label order follows the ordered read groups.
        """
        version_info = {
            ("Pipeline version", ""): (self.params.get("pipeline_version", None)),
            ("Docker image", ""): self.params.get("docker_image", None),
            ("Adapter version", ""): self.params.get("adapter_version", None),
            ("Report created on", ""): datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        n_tp = self.data_df[LABEL].sum()
        n_fp = (~self.data_df[LABEL]).sum()
        general_info = {
            ("Sample name", ""): self.base_name[:-1],
            ("Median training read length", ""): np.median(self.data_df[LENGTH]),
            ("Median training coverage", ""): np.median(self.data_df[READ_COUNT]),
            ("Training set, % TP reads", ""): signif(self.data_df[LABEL].mean() * 100, 3),
        }
        for label, mask in self._group_masks():
            tp_pct = (mask & self.data_df[LABEL].to_numpy()).sum() / n_tp if n_tp else np.nan
            fp_pct = (mask & (~self.data_df[LABEL]).to_numpy()).sum() / n_fp if n_fp else np.nan
            general_info[(f"{label} training reads", "% of TP")] = f"{signif(100 * tp_pct, 3)}%"
            general_info[(f"{label} training reads", "% of FP")] = f"{signif(100 * fp_pct, 3)}%"

        tp_df = self.data_df[self.data_df[LABEL]]
        median_qual = tp_df[QUAL].median()
        recall_at_0 = self._get_recall_at_snvq(snvq=0)
        roc_auc_phred = prob_to_phred(
            self._safe_roc_auc(self.data_df[LABEL], self.data_df[ML_PROB_1_TEST], name="run info total"),
            max_value=self.max_qual,
        )
        snvq_thresholds = [50, 60, 70]
        # Per-group stats keyed by group label
        grp_median, grp_recall0, grp_recall, grp_roc = {}, {}, {}, {}
        for label, mask in self._group_masks():
            cond = pd.Series(mask, index=self.data_df.index)
            grp_median[label] = tp_df[cond.loc[tp_df.index]][QUAL].median()
            grp_recall0[label] = self._get_recall_at_snvq(snvq=0, condition=cond)
            grp_recall[label] = {s: self._get_recall_at_snvq(snvq=s, condition=cond) for s in snvq_thresholds}
            sub = self.data_df[mask]
            grp_roc[label] = prob_to_phred(
                self._safe_roc_auc(sub[LABEL], sub[ML_PROB_1_TEST], name=f"run info {label}"), max_value=self.max_qual
            )

        groups = list(self._display_variant.groups)
        performance_info = {("Median SNVQ", "All reads"): signif(median_qual, SIG_DIGITS)}
        for label in groups:
            performance_info[("Median SNVQ", label)] = signif(grp_median[label], SIG_DIGITS)
        for snvq in snvq_thresholds:
            performance_info[(f"Recall at SNVQ={snvq}", "All reads")] = signif(
                self._get_recall_at_snvq(snvq=snvq) / recall_at_0, SIG_DIGITS
            )
            for label in groups:
                denom = grp_recall0[label]
                performance_info[(f"Recall at SNVQ={snvq}", label)] = signif(
                    grp_recall[label][snvq] / denom if denom else np.nan, SIG_DIGITS
                )
        performance_info[("Pre-filter Recall", "All reads")] = signif(recall_at_0, SIG_DIGITS)
        for label in groups:
            performance_info[("Pre-filter Recall", label)] = signif(grp_recall0[label], SIG_DIGITS)
        performance_info[("ROC AUC (Phred)", "All reads")] = signif(roc_auc_phred, SIG_DIGITS)
        for label in groups:
            performance_info[("ROC AUC (Phred)", label)] = signif(grp_roc[label], SIG_DIGITS)

        training_info = {("Number of CV folds", ""): self.params["num_CV_folds"]}
        dataset_sizes = self.get_dataset_sizes()
        run_info_table = pd.Series({**general_info, **version_info}, name="")
        run_quality_summary_table = pd.Series({**performance_info}, name="")
        training_info_table = pd.Series({**training_info, **dataset_sizes}, name="")
        # Single display grouping: write to the display variant's suffixed key + bare base key.
        variant = self._display_variant
        self._write_variant_table("run_info_table", variant, run_info_table)
        self._write_variant_table("run_quality_summary_table", variant, run_quality_summary_table)
        training_info_table.to_hdf(self.output_h5_filename, key="training_info_table", mode="a")

    @exception_handler
    def calc_run_info_table(self):  # noqa: PLR0915
        """Calculate run_info_table, a table with general run information."""
        # Generate Run Info table
        logger.info("Generating Run Info table")
        # Schemes with a single display grouping use the per-group path; ppmSeq (mixed) keeps its
        # dual legacy(both-ends)/display(start) tables with the historical 3-way performance rows.
        if not self._has_dual_split():
            self._calc_run_info_table_by_group()
            return
        pos = self._display_variant.pos
        display_col = IS_MIXED_START
        legacy_col = IS_MIXED
        n_tp = self.data_df[LABEL].sum()
        n_fp = (~self.data_df[LABEL]).sum()
        # Legacy positive-group percent using the legacy split column (IS_MIXED both ends for ppmSeq)
        TP_mixed_percent_legacy = (self.data_df[legacy_col] & self.data_df[LABEL]).sum() / n_tp  # noqa: N806
        FP_mixed_percent_legacy = (self.data_df[legacy_col] & ~self.data_df[LABEL]).sum() / n_fp  # noqa: N806
        # New positive-group percent using the display split column
        TP_mixed_percent = (self.data_df[display_col] & self.data_df[LABEL]).sum() / n_tp  # noqa: N806
        FP_mixed_percent = (self.data_df[display_col] & ~self.data_df[LABEL]).sum() / n_fp  # noqa: N806
        general_info_base = {
            ("Sample name", ""): self.base_name[:-1],
            ("Median training read length", ""): np.median(self.data_df[LENGTH]),
            ("Median training coverage", ""): np.median(self.data_df[READ_COUNT]),
            ("Training set, % TP reads", ""): signif(self.data_df[LABEL].mean() * 100, 3),
        }
        general_info = {
            **general_info_base,
            (f"{pos} training reads", "% of TP"): f"{signif(100 * TP_mixed_percent, 3)}%",
            (f"{pos} training reads", "% of FP"): f"{signif(100 * FP_mixed_percent, 3)}%",
        }
        general_info_legacy = {
            **general_info_base,
            (f"{pos} training reads", "% of TP"): f"{signif(100 * TP_mixed_percent_legacy, 3)}%",
            (f"{pos} training reads", "% of FP"): f"{signif(100 * FP_mixed_percent_legacy, 3)}%",
        }
        # Performance info
        mixed_start_df = self.data_df[self.data_df[display_col]]
        mixed_both_df = self.data_df[self.data_df[legacy_col]]
        tp_df = self.data_df[self.data_df[LABEL]]
        tp_mixed_start_df = tp_df[tp_df[display_col]]
        tp_mixed_both_df = tp_df[tp_df[legacy_col]]
        median_qual = tp_df[QUAL].median()
        median_qual_mixed_start = tp_mixed_start_df[QUAL].median()
        median_qual_mixed_both = tp_mixed_both_df[QUAL].median()
        recall_at_0 = self._get_recall_at_snvq(snvq=0)
        recall_at_0_mixed_start = self._get_recall_at_snvq(snvq=0, condition=self.data_df[display_col])
        recall_at_0_mixed_both = self._get_recall_at_snvq(snvq=0, condition=self.data_df[legacy_col])
        snvq_thresholds = [50, 60, 70]
        recalls = {}
        for snvq in snvq_thresholds:
            recalls[snvq] = {
                "all": self._get_recall_at_snvq(snvq=snvq),
                "mixed_start": self._get_recall_at_snvq(snvq=snvq, condition=self.data_df[display_col]),
                "mixed_both": self._get_recall_at_snvq(snvq=snvq, condition=self.data_df[legacy_col]),
            }
        roc_auc_phred = prob_to_phred(
            self._safe_roc_auc(self.data_df[LABEL], self.data_df[ML_PROB_1_TEST], name="run info total"),
            max_value=self.max_qual,
        )
        roc_auc_phred_mixed_start = prob_to_phred(
            self._safe_roc_auc(mixed_start_df[LABEL], mixed_start_df[ML_PROB_1_TEST], name="run info mixed start"),
            max_value=self.max_qual,
        )
        roc_auc_phred_mixed_both = prob_to_phred(
            self._safe_roc_auc(mixed_both_df[LABEL], mixed_both_df[ML_PROB_1_TEST], name="run info mixed both"),
            max_value=self.max_qual,
        )
        # Note: entry order matters — unstack(sort=False) in the report notebook relies on
        # each metric having all its categories together before the next metric appears.
        performance_info = {
            ("Median SNVQ", "All reads"): signif(median_qual, SIG_DIGITS),
            ("Median SNVQ", pos): signif(median_qual_mixed_start, SIG_DIGITS),
        }
        for snvq in snvq_thresholds:
            performance_info[(f"Recall at SNVQ={snvq}", "All reads")] = signif(
                recalls[snvq]["all"] / recall_at_0, SIG_DIGITS
            )
            performance_info[(f"Recall at SNVQ={snvq}", pos)] = signif(
                recalls[snvq]["mixed_start"] / recall_at_0_mixed_start, SIG_DIGITS
            )
        performance_info[("Pre-filter Recall", "All reads")] = signif(recall_at_0, SIG_DIGITS)
        performance_info[("Pre-filter Recall", pos)] = signif(recall_at_0_mixed_start, SIG_DIGITS)
        performance_info[("ROC AUC (Phred)", "All reads")] = signif(roc_auc_phred, SIG_DIGITS)
        performance_info[("ROC AUC (Phred)", pos)] = signif(roc_auc_phred_mixed_start, SIG_DIGITS)

        # Legacy performance info with 3-way split for backward-compatible h5 storage.
        # In non-mixed modes the "start" and "both ends" sub-splits are identical.
        performance_info_legacy = {
            ("Median SNVQ", "All reads"): signif(median_qual, SIG_DIGITS),
            ("Median SNVQ", f"{pos}, start"): signif(median_qual_mixed_start, SIG_DIGITS),
            ("Median SNVQ", f"{pos}, both ends"): signif(median_qual_mixed_both, SIG_DIGITS),
        }
        for snvq in snvq_thresholds:
            performance_info_legacy[(f"Recall at SNVQ={snvq}", "All reads")] = signif(
                recalls[snvq]["all"] / recall_at_0, SIG_DIGITS
            )
            performance_info_legacy[(f"Recall at SNVQ={snvq}", f"{pos}, start")] = signif(
                recalls[snvq]["mixed_start"] / recall_at_0_mixed_start, SIG_DIGITS
            )
            performance_info_legacy[(f"Recall at SNVQ={snvq}", f"{pos}, both ends")] = signif(
                recalls[snvq]["mixed_both"] / recall_at_0_mixed_both, SIG_DIGITS
            )
        performance_info_legacy[("Pre-filter Recall", "All reads")] = signif(recall_at_0, SIG_DIGITS)
        performance_info_legacy[("Pre-filter Recall", f"{pos}, start")] = signif(recall_at_0_mixed_start, SIG_DIGITS)
        performance_info_legacy[("Pre-filter Recall", f"{pos}, both ends")] = signif(recall_at_0_mixed_both, SIG_DIGITS)
        performance_info_legacy[("ROC AUC (Phred)", "All reads")] = signif(roc_auc_phred, SIG_DIGITS)
        performance_info_legacy[("ROC AUC (Phred)", f"{pos}, start")] = signif(roc_auc_phred_mixed_start, SIG_DIGITS)
        performance_info_legacy[("ROC AUC (Phred)", f"{pos}, both ends")] = signif(roc_auc_phred_mixed_both, SIG_DIGITS)
        # Info about versions
        version_info = {
            ("Pipeline version", ""): (self.params.get("pipeline_version", None)),
            ("Docker image", ""): self.params.get("docker_image", None),
            ("Adapter version", ""): self.params.get("adapter_version", None),
            ("Report created on", ""): datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Training info
        training_info = {
            # ("Pre-filter", ""): self.params.get("pre_filter", None),
            # ("Columns for balancing", ""): self.params.get("balanced_sampling_info_fields", None),
            ("Number of CV folds", ""): self.params["num_CV_folds"],
        }
        # Info about training set size
        dataset_sizes = self.get_dataset_sizes()
        # Generate tables and save to h5
        run_info_table_legacy = pd.Series({**general_info_legacy, **version_info}, name="")
        run_info_table = pd.Series({**general_info, **version_info}, name="")
        run_quality_summary_table = pd.Series({**performance_info}, name="")
        run_quality_summary_table_legacy = pd.Series({**performance_info_legacy}, name="")
        training_info_table = pd.Series({**training_info, **dataset_sizes}, name="")
        run_info_table_legacy.to_hdf(self.output_h5_filename, key="run_info_table", mode="a")
        run_info_table.to_hdf(self.output_h5_filename, key="run_info_table_mixed_start", mode="a")
        run_quality_summary_table_legacy.to_hdf(self.output_h5_filename, key="run_quality_summary_table", mode="a")
        run_quality_summary_table.to_hdf(self.output_h5_filename, key="run_quality_summary_table_mixed_start", mode="a")
        training_info_table.to_hdf(self.output_h5_filename, key="training_info_table", mode="a")

    @exception_handler
    def plot_interpolating_function_with_histograms(self, output_filename: str = None):
        """Plot the interpolating function with histograms of the data.
        Arguments:
            output_filename: str: path to output file
        """
        logger.info("Generating plot of ML_qual -> SNVQ link function (with histograms)")
        ml_qual_max_int = np.floor(self.data_df[ML_QUAL_1_TEST].max())
        x_ml_qual = np.arange(self.eps, ml_qual_max_int, 0.1)

        xs = self.ML_qual_to_qual_fn((x_ml_qual[1:] + x_ml_qual[:-1]) / 2).reshape(
            -1,
        )
        ys = 0.1 / (self.ML_qual_to_qual_fn(x_ml_qual)[1:] - self.ML_qual_to_qual_fn(x_ml_qual)[:-1]).reshape(
            -1,
        )

        plot_df = self.data_df  # [self.data_df[LABEL]]

        # Set the axis limits
        xmin, xmax = 0.99 * xs.min(), xs.max() * 1.01
        ymin, ymax = 0, ml_qual_max_int + 1

        # Create a figure with a gridspec layout to control the positions of the marginals
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(8, 4, hspace=0.0, wspace=0.0)

        # Joint scatterplot
        ax_joint = fig.add_subplot(gs[3:8, 0:3])
        sns.scatterplot(data=plot_df, x=QUAL, y=ML_QUAL_1_TEST, color="black", s=1, alpha=1, ax=ax_joint)
        ax_joint.grid(visible=True)
        ax_joint.set_xlim(xmin, xmax)
        ax_joint.set_ylim(ymin, ymax)
        ax_joint.set_ylabel("ML_qual")

        # X-axis marginal histplot
        ax_marg_x = fig.add_subplot(gs[0:2, 0:3], sharex=ax_joint)
        sns.histplot(
            data=plot_df,
            x=QUAL,
            hue=LABEL,  # IS_MIXED,
            hue_order=[False, True],
            element="step",
            stat="density",
            common_norm=True,  # False,
            linewidth=1,
            ax=ax_marg_x,
            # palette={False: "red", True: "green"},
        )
        ax_marg_x.grid(visible=True)
        ax_marg_x.set_yscale("log")
        ax_marg_x.set_xlabel("")
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        sns.move_legend(
            ax_marg_x,
            "upper left",
        )

        # Extra plot (in the middle-left position)
        ax_extra = fig.add_subplot(gs[2, 0:3], sharex=ax_joint)
        ax_extra.plot(xs, ys, "k.-")
        ax_extra.grid(visible=True)
        ax_extra.set_xlabel("")
        ax_extra.set_ylabel("deriv")
        plt.setp(ax_extra.get_xticklabels(), visible=False)

        # Y-axis marginal histplot
        ax_marg_y = fig.add_subplot(gs[3:8, 3], sharey=ax_joint)
        sns.histplot(
            data=plot_df,
            y=ML_QUAL_1_TEST,
            hue=LABEL,  # IS_MIXED,
            hue_order=[False, True],
            element="step",
            stat="density",
            common_norm=True,  # False,
            linewidth=1,
            ax=ax_marg_y,
            # palette={False: "red", True: "green"},
        )
        ax_marg_y.grid(visible=True)
        ax_marg_y.set_ylabel("")
        ax_marg_y.set_xscale("log")
        ax_marg_y.get_legend().set_visible(False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # Adjust the layout to ensure no space between the plots
        plt.tight_layout()
        gs.update(hspace=0.0, wspace=0.0)
        self._save_plt(output_filename, fig)

    def _safe_roc_auc(self, y_true, y_pred, name=None):
        """
        Calculate ROC AUC score with checks for cases where the calculation is not possible.

        Parameters:
        y_true: array-like, true labels
        y_pred: array-like, predicted probabilities
        name: str, name of dataset (for logging). If None, ignore dataset name.

        Returns:
        ROC AUC score if calculable, otherwise np.nan
        """
        return safe_roc_auc(y_true, y_pred, name=name, logger=logger)

    @exception_handler
    def calc_roc_auc_table(self, holdout_fold_size_thresh: int = 100):  # noqa: C901, PLR0915 #TODO: Refactor
        """Calculte statistics of ROC AUC
        Arguments:
            holdout_fold_size_thresh: int: holdout set (fold_id is NaN) must be at least
                this size to calculate statistics on it
        """
        logger.info("Generating ROC AUC table")

        num_cv_folds = self.params["num_CV_folds"]
        holdout_fold_cond = self.data_df[FOLD_ID].isna()

        def _calc_roc_auc_dict(groups, label_suffix=""):  # noqa: C901
            """Compute ROC-AUC stats for the whole set plus each group in ``groups``.

            ``groups`` is an ordered list of ``(name, boolean_mask)``. The returned dict maps each
            statistic to a list ``[total, *per_group]`` aligned with ``["Total", *group_names]``.
            """
            data_df = self.data_df

            def _auc(mask, name):
                sub = data_df.loc[mask]
                return prob_to_phred(
                    self._safe_roc_auc(sub[LABEL], sub[ML_PROB_1_TEST], name=name), max_value=self.max_qual
                )

            all_mask = np.ones(len(data_df), dtype=bool)
            auc_total = _auc(all_mask, f"total{label_suffix}")
            auc_per_group = [_auc(mask, f"{gname}{label_suffix}") for gname, mask in groups]

            auc_per_fold_total = []
            auc_per_fold_groups = [[] for _ in groups]
            auc_on_holdout_total = []
            auc_on_holdout_groups = [[] for _ in groups]
            # Track which series (by name) already produced a NaN on holdout, to stop recomputing.
            error_on_holdout = set()
            for k in range(num_cv_folds):
                fold_cond = (data_df[FOLD_ID] == k).to_numpy()
                auc_per_fold_total.append(_auc(fold_cond, f"fold {k} total{label_suffix}"))
                for gi, (gname, mask) in enumerate(groups):
                    auc_per_fold_groups[gi].append(_auc(fold_cond & mask, f"fold {k} {gname}{label_suffix}"))
                if holdout_fold_cond.sum() > holdout_fold_size_thresh:
                    ho = holdout_fold_cond.to_numpy()
                    preds_ho = data_df.loc[ho, f"prob_fold_{k}"]
                    if "total" not in error_on_holdout:
                        val = prob_to_phred(
                            self._safe_roc_auc(data_df.loc[ho, LABEL], preds_ho, name=f"holdout total{label_suffix}")
                        )
                        auc_on_holdout_total.append(val)
                        if np.isnan(val):
                            error_on_holdout.add("total")
                    else:
                        auc_on_holdout_total.append(np.nan)
                    for gi, (gname, mask) in enumerate(groups):
                        if gname not in error_on_holdout:
                            gho = ho & mask
                            val = prob_to_phred(
                                self._safe_roc_auc(
                                    data_df.loc[gho, LABEL],
                                    data_df.loc[gho, f"prob_fold_{k}"],
                                    name=f"holdout {gname}{label_suffix}",
                                ),
                                max_value=self.max_qual,
                            )
                            auc_on_holdout_groups[gi].append(val)
                            if np.isnan(val):
                                error_on_holdout.add(gname)
                        else:
                            auc_on_holdout_groups[gi].append(np.nan)

            result = {
                "ROC AUC": [auc_total, *auc_per_group],
                "ROC AUC per fold mean": [
                    np.array(auc_per_fold_total).mean(),
                    *[np.array(g).mean() for g in auc_per_fold_groups],
                ],
                "ROC AUC per fold std": [
                    np.array(auc_per_fold_total).std(),
                    *[np.array(g).std() for g in auc_per_fold_groups],
                ],
            }
            if holdout_fold_cond.sum() > holdout_fold_size_thresh:
                result["ROC AUC on holdout mean"] = [
                    np.array(auc_on_holdout_total).mean(),
                    *[np.array(g).mean() for g in auc_on_holdout_groups],
                ]
                result["ROC AUC on holdout std"] = [
                    np.array(auc_on_holdout_total).std(),
                    *[np.array(g).std() for g in auc_on_holdout_groups],
                ]
            return result

        # One table per variant. Binary variants (pos/neg set) keep the historical
        # "{pos} only"/"{neg} only" row order; N-group variants list one row per group.
        for variant in self.scheme.variants:
            groups, auc_index = self._variant_roc_groups(variant)
            label_suffix = " legacy" if self._is_legacy_variant(variant) else ""
            auc_dict = _calc_roc_auc_dict(groups, label_suffix=label_suffix)
            self._write_variant_table("roc_auc_table", variant, pd.DataFrame(auc_dict, index=auc_index).T)

    @exception_handler
    def calc_run_quality_table(
        self,
        qual_stat_ps=None,
        cols_for_stats=None,
        display_columns=None,
        # col_order = [
        #     'qual_FP', 'qual_TP', 'qual_TP_mixed', 'qual_TP_non-mixed',
        #     'ML_qual_FP', 'ML_qual_TP', 'ML_qual_TP_mixed', 'ML_qual_TP_non-mixed'
        # ]
    ):
        """Calculate table with quality metrics for the run."""
        # Default values
        if qual_stat_ps is None:
            qual_stat_ps = [0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95]
        if cols_for_stats is None:
            cols_for_stats = {QUAL: QUAL, ML_QUAL_1_TEST: "ML_qual", ML_LOGIT_TEST: "ML_logit"}
        logger.info("Generating Run Quality table")

        def _quality_table(conds_dict):
            """Describe QUAL/ML_qual/ML_logit per condition and stack into a table (drop 'count')."""
            stats = pd.concat(
                [
                    self.data_df.loc[cond, list(cols_for_stats.keys())]
                    .describe(percentiles=qual_stat_ps)
                    .rename(columns={stat_key: stat_name + key for stat_key, stat_name in cols_for_stats.items()})
                    for key, cond in conds_dict.items()
                ],
                axis=1,
            )
            return pd.DataFrame(signif(stats.values, SIG_DIGITS), index=stats.index, columns=stats.columns).drop(
                index="count"
            )

        base_conds = {
            "": np.ones(self.data_df[LABEL].shape, dtype=bool),
            "_TP": self.data_df[LABEL].to_numpy(),
            "_FP": (~self.data_df[LABEL]).to_numpy(),
        }

        if not self._has_dual_split():
            # Single-grouping scheme (consensus / none): one TP column per read group. The internal
            # join keys use the group label directly (kept literal for col_order); display uses the
            # same labels.
            display_variant = self._display_variant
            tp = self.data_df[LABEL].to_numpy()
            conds = dict(base_conds)
            for label, mask in self._group_masks():
                conds[f"_TP_{label}"] = mask & tp
            run_quality_table = _quality_table(conds)
            display_columns = []
            for qcol in (QUAL, "ML_qual"):
                display_columns.append((qcol, "FP", ""))
                display_columns.append((qcol, "TP", "overall"))
                for label in display_variant.groups:
                    display_columns.append((qcol, "TP", label))
            col_order = [
                "_".join(cols) if cols[2] not in ["", "overall"] else "_".join(cols[:2]) for cols in display_columns
            ]
            run_quality_table_display = run_quality_table.loc[:, col_order].copy()
            run_quality_table_display.columns = pd.MultiIndex.from_tuples(display_columns)
            # No legacy/both-ends distinction: write the same table to the display key + bare base.
            self._write_variant_table("run_quality_table", display_variant, run_quality_table.T)
            run_quality_table_display.to_hdf(self.output_h5_filename, key="run_quality_table_display", mode="a")
            return

        # Dual-split scheme (ppmSeq mixed) — unchanged historical output.
        legacy_variant = self.scheme.legacy_variant
        display_variant = self._display_variant
        if display_columns is None:
            display_columns = [
                (QUAL, "FP", ""),
                (QUAL, "TP", "overall"),
                (QUAL, "TP", "mixed"),
                (QUAL, "TP", "non-mixed"),
                ("ML_qual", "FP", ""),
                ("ML_qual", "TP", "overall"),
                ("ML_qual", "TP", "mixed"),
                ("ML_qual", "TP", "non-mixed"),
            ]
        # Note: the "_TP_mixed"/"_TP_non-mixed" keys below are internal join keys (matched by
        # col_order); they stay literal regardless of scheme. Only the displayed MultiIndex labels
        # are swapped to the variant's split labels further down.
        label = self.data_df[LABEL].to_numpy()
        legacy_masks = legacy_variant.group_masks(self.data_df)
        display_masks = display_variant.group_masks(self.data_df)
        # Legacy table using the legacy variant's positive group (IS_MIXED both ends for ppmSeq)
        conds_dict_legacy = {
            **base_conds,
            "_TP_mixed": legacy_masks[legacy_variant.pos] & label,
            "_TP_non-mixed": legacy_masks[legacy_variant.neg] & label,
        }
        _quality_table(conds_dict_legacy).T.to_hdf(self.output_h5_filename, key="run_quality_table", mode="a")

        # New table using the display variant's positive group for display and new h5 key
        conds_dict = {
            **base_conds,
            "_TP_mixed": display_masks[display_variant.pos] & label,
            "_TP_non-mixed": display_masks[display_variant.neg] & label,
        }
        run_quality_table = _quality_table(conds_dict)

        col_order = [
            "_".join(cols) if cols[2] not in ["", "overall"] else "_".join(cols[:2]) for cols in display_columns
        ]
        run_quality_table_display = run_quality_table.loc[:, col_order].copy()
        # Swap the literal "mixed"/"non-mixed" display labels for this variant's split labels
        # (the join keys used for col_order above stay literal).
        label_remap = {"mixed": display_variant.pos_lc, "non-mixed": display_variant.neg_lc}
        display_columns_relabeled = [(a, b, label_remap.get(c, c)) for a, b, c in display_columns]
        run_quality_table_display.columns = pd.MultiIndex.from_tuples(display_columns_relabeled)
        # Log in hdf5
        run_quality_table.T.to_hdf(self.output_h5_filename, key="run_quality_table_mixed_start", mode="a")
        run_quality_table_display.to_hdf(self.output_h5_filename, key="run_quality_table_display", mode="a")

    def _render_summary_table_figure(self, summary_df: pd.DataFrame, output_filename: str = None):
        """Render a 2-column (Median SNVQ, % of reads) summary table as a matplotlib figure."""
        fig, ax = plt.subplots(figsize=(6, max(2, 0.6 * len(summary_df) + 1)))
        ax.axis("off")
        table = ax.table(
            cellText=summary_df.values,
            rowLabels=summary_df.index,
            colLabels=summary_df.columns,
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        _disable_auto_fontsize = False
        table.auto_set_font_size(_disable_auto_fontsize)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        # Style header row
        for j in range(len(summary_df.columns)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # Style row labels
        for i in range(len(summary_df)):
            table[i + 1, -1].set_facecolor("#D9E2F3")
        self._save_plt(output_filename, fig=fig)

    def _quality_per_split_group(self, output_filename: str = None):
        """Generic per-read-group median SNVQ + % of reads table (used when a scheme has no
        ppmSeq start x end cross-tab hook).

        Groups TP reads by the ordered ``read_group`` categorical (e.g. single read / one strand /
        duplex) and writes the summary to the display variant's suffixed key plus the bare
        ``ppmseq_category_quality_table`` / ``ppmseq_category_quantity_table`` keys (so
        keys_to_convert / h5->json conversion finds them).
        """
        variant = self._display_variant
        data_df_tp = self.data_df[self.data_df[LABEL]].copy()
        total_count = len(data_df_tp)
        # Group by the ordered read_group categorical; reindex to keep group order and include
        # any empty groups as 0 rows.
        grouped = data_df_tp.groupby(READ_GROUP, dropna=False, observed=False)[QUAL]
        median_qual = grouped.median().reindex(list(variant.groups))
        pct_reads = (grouped.count().reindex(list(variant.groups)).fillna(0) / total_count) * 100
        summary_df = pd.DataFrame({"Median SNVQ": median_qual, "% of reads": pct_reads})
        summary_df.index = summary_df.index.astype(str)
        summary_df.index.name = self.tag_axis
        summary_df["Median SNVQ"] = summary_df["Median SNVQ"].apply(lambda x: signif(x, 3))
        summary_df["% of reads"] = summary_df["% of reads"].apply(lambda x: signif(x, 2))

        # Display table under the variant's suffixed key.
        summary_df.to_hdf(self.output_h5_filename, key=split_h5_key("ppmseq_category_quality_table", variant), mode="a")
        # Bare base keys so downstream keys_to_convert / h5->json conversion finds them.
        summary_df["Median SNVQ"].to_hdf(self.output_h5_filename, key="ppmseq_category_quality_table", mode="a")
        summary_df["% of reads"].to_hdf(self.output_h5_filename, key="ppmseq_category_quantity_table", mode="a")

        self._render_summary_table_figure(summary_df, output_filename=output_filename)

    @exception_handler
    def quality_per_ppmseq_tags(self, output_filename: str = None):  # noqa: C901, PLR0915
        """Generate table of median quality and data quantity per read-split group.

        Schemes that provide a start x end ppmSeq cross-tab (``scheme.ppmseq_crosstab``) render the
        legacy 2D tables plus the 1D start-tag display table. Other schemes fall back to a generic
        per-read-group summary with the same h5 keys.
        """
        if not self.scheme.ppmseq_crosstab:
            self._quality_per_split_group(output_filename=output_filename)
            return

        data_df_tp = self.data_df[self.data_df[LABEL]].copy()
        # Determine start and end tag columns
        ppmseq_fillna_tags_in_data = ST_FILLNA in data_df_tp.columns and ET_FILLNA in data_df_tp.columns
        if ppmseq_fillna_tags_in_data:
            start_tag_col, end_tag_col = (ST_FILLNA, ET_FILLNA)
        else:
            if self.start_tag_col is None or self.end_tag_col is None:
                logger.warning("ppmSeq tag columns not specified.")
                start_tag_col, end_tag_col = (ST, ET)
            else:
                start_tag_col, end_tag_col = (self.start_tag_col, self.end_tag_col)
            ppmseq_tags_in_data = start_tag_col in data_df_tp.columns and end_tag_col in data_df_tp.columns
            if not ppmseq_tags_in_data:
                data_df_tp[start_tag_col] = np.nan
                data_df_tp[end_tag_col] = np.nan
        # If there are NaNs in the tags, convert to string to avoid issues with groupby
        if data_df_tp[start_tag_col].isna().any() or data_df_tp[end_tag_col].isna().any():
            data_df_tp = data_df_tp.astype({start_tag_col: str, end_tag_col: str})

        # Legacy 2D tables (start x end) for backward-compatible h5 storage
        ppmseq_category_quality_table = (
            data_df_tp.groupby([start_tag_col, end_tag_col], dropna=False)[QUAL].median().unstack()  # noqa: PD010
        )
        if PpmseqCategories.END_UNREACHED.value in ppmseq_category_quality_table.index:
            ppmseq_category_quality_table = ppmseq_category_quality_table.drop(
                index=PpmseqCategories.END_UNREACHED.value
            )
        ppmseq_category_quantity_table = (
            data_df_tp.groupby([start_tag_col, end_tag_col], dropna=False)[QUAL].count().unstack()  # noqa: PD010
        )
        if PpmseqCategories.END_UNREACHED.value in ppmseq_category_quantity_table.index:
            ppmseq_category_quantity_table = ppmseq_category_quantity_table.drop(
                index=PpmseqCategories.END_UNREACHED.value
            )
        ppmseq_category_quantity_table = (
            ppmseq_category_quantity_table / ppmseq_category_quantity_table.to_numpy().sum()
        ) * 100
        ppmseq_category_quality_table.index = ppmseq_category_quality_table.index.astype(str)
        ppmseq_category_quality_table.columns = ppmseq_category_quality_table.columns.astype(str)
        ppmseq_category_quantity_table.index = ppmseq_category_quantity_table.index.astype(str)
        ppmseq_category_quantity_table.columns = ppmseq_category_quantity_table.columns.astype(str)
        # Save legacy 2D tables to hdf5
        ppmseq_category_quality_table_for_h5 = ppmseq_category_quality_table.T.unstack(level=0)  # noqa: PD010
        ppmseq_category_quantity_table_for_h5 = ppmseq_category_quantity_table.T.unstack(level=0)  # noqa: PD010
        ppmseq_category_quality_table_for_h5.to_hdf(
            self.output_h5_filename, key="ppmseq_category_quality_table", mode="a"
        )
        ppmseq_category_quantity_table_for_h5.to_hdf(
            self.output_h5_filename, key="ppmseq_category_quantity_table", mode="a"
        )

        # New simplified 1D table (by start tag only) for display and new h5 key
        if data_df_tp[start_tag_col].isna().any():
            data_df_tp[start_tag_col] = data_df_tp[start_tag_col].astype(str)
        total_count = len(data_df_tp)
        grouped = data_df_tp.groupby(start_tag_col, dropna=False)[QUAL]
        median_qual = grouped.median()
        counts = grouped.count()
        pct_reads = (counts / total_count) * 100
        if PpmseqCategories.END_UNREACHED.value in median_qual.index:
            median_qual = median_qual.drop(index=PpmseqCategories.END_UNREACHED.value)
            pct_reads = pct_reads.drop(index=PpmseqCategories.END_UNREACHED.value)
        summary_df = pd.DataFrame({"Median SNVQ": median_qual, "% of reads": pct_reads})
        summary_df.index = summary_df.index.astype(str)
        summary_df.index.name = self.tag_axis
        summary_df["Median SNVQ"] = summary_df["Median SNVQ"].apply(lambda x: signif(x, 3))
        summary_df["% of reads"] = summary_df["% of reads"].apply(lambda x: signif(x, 2))

        # Save new simplified table to hdf5
        summary_df.to_hdf(self.output_h5_filename, key="ppmseq_category_quality_table_mixed_start", mode="a")

        # Render as a table figure
        self._render_summary_table_figure(summary_df, output_filename=output_filename)

    @exception_handler
    def training_progress_plot(self, output_filename: str = None, ylims=None):
        """Generate plot of training progress plot, i.e., logloss and roc auc
        as a function of training steps.
        """
        # Default values
        if ylims is None:
            ylims = [None, None]

        set_pyplot_defaults()

        training_results = [clf.evals_result() for clf in self.models]
        num_folds = len(training_results)

        # Skip if no training results available (e.g. DNN dummy models with no metrics)
        if not training_results or not training_results[0] or "validation_0" not in training_results[0]:
            logger.info("No training results available — skipping training progress plot")
            return

        # Determine which logloss metric is available (logloss for binary, mlogloss for multiclass)
        # Check the first model's results to determine the metric name
        logloss_key = None
        if "logloss" in training_results[0]["validation_0"]:
            logloss_key = "logloss"
        elif "mlogloss" in training_results[0]["validation_0"]:
            logloss_key = "mlogloss"
        else:
            raise ValueError(
                "Neither 'logloss' nor 'mlogloss' found in training results. "
                f"Available metrics: {list(training_results[0]['validation_0'].keys())}"
            )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        train_logloss = list_of_jagged_lists_to_array(
            [result["validation_0"][logloss_key] for result in training_results]
        )
        val_logloss = list_of_jagged_lists_to_array(
            [result["validation_1"][logloss_key] for result in training_results]
        )
        for ri, result in enumerate(training_results):
            label = "Individual folds" if ri == 1 else None  # will reach ri==1 iff when using CV
            ax.plot(result["validation_0"][logloss_key], c="grey", alpha=0.7, label=label)
            ax.plot(result["validation_1"][logloss_key], c="grey", alpha=0.7)

        kfolds_label = f" (mean of {num_folds} folds)" if num_folds >= 2 else ""  # noqa: PLR2004
        ax.plot(np.nanmean(train_logloss, axis=0), label="Train" + kfolds_label)
        ax.plot(np.nanmean(val_logloss, axis=0), label="Val" + kfolds_label)
        # Get handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 2, 0] if num_folds >= 2 else [0, 1]  # The order in whichthe labels are displayed  # noqa: PLR2004
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

        ax.set_ylabel("Log Loss")
        ax.set_xlabel("Training step")
        if ylims[0] is not None:
            ax.set_ylim(ylims[0])
        # Plot AUC
        ax = axes[1]
        train_auc = list_of_jagged_lists_to_array(
            [list_auc_to_qual(result["validation_0"]["auc"]) for result in training_results]
        )
        val_auc = list_of_jagged_lists_to_array(
            [list_auc_to_qual(result["validation_1"]["auc"]) for result in training_results]
        )
        for ri, result in enumerate(training_results):
            label = "Individual folds" if ri == 1 else None  # will reach ri==1 iff when using CV
            ax.plot(list_auc_to_qual(result["validation_0"]["auc"]), c="grey", alpha=0.5, label=label)
            ax.plot(list_auc_to_qual(result["validation_1"]["auc"]), c="grey", alpha=0.5)
        ax.plot(np.nanmean(train_auc, axis=0), label="Train" + kfolds_label)
        ax.plot(np.nanmean(val_auc, axis=0), label="Val" + kfolds_label)
        # Get handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 2, 0] if num_folds >= 2 else [0, 1]  # The order in whichthe labels are displayed  # noqa: PLR2004
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        ax.set_ylabel("ROC AUC (phred)")
        ax.set_xlabel("Training step")
        if ylims[1] is not None:
            ax.set_ylim(ylims[1])
        self._save_plt(output_filename, fig=fig)
        pd.concat(
            [
                pd.DataFrame(
                    train_logloss.T,
                    columns=pd.MultiIndex.from_tuples([("logloss", "train", f"fold{i}") for i in range(num_folds)]),
                ),
                pd.DataFrame(
                    val_logloss.T,
                    columns=pd.MultiIndex.from_tuples([("logloss", "val", f"fold{i}") for i in range(num_folds)]),
                ),
                pd.DataFrame(
                    train_auc.T,
                    columns=pd.MultiIndex.from_tuples([("auc", "train", f"fold{i}") for i in range(num_folds)]),
                ),
                pd.DataFrame(
                    val_auc.T, columns=pd.MultiIndex.from_tuples([("auc", "val", f"fold{i}") for i in range(num_folds)])
                ),
            ],
            axis=1,
        ).T.to_hdf(self.output_h5_filename, key="training_progress", mode="a")

    def _create_shap_plotter(self):
        """Create a SHAPPlotter instance for this SRSNVReport."""
        return SHAPPlotter(
            models=self.models,
            data=self.data_df,
            fold_id_col=FOLD_ID,
            features_metadata=self.srsnv_metadata["features"],
            label_col=LABEL,
            random_state=42,  # Use a fixed seed for reproducible plots
            logger=logger,
            use_gpu=self.use_gpu_for_shap,
        )

    @exception_handler
    def calc_and_plot_shap_values(
        self,
        output_filename_importance: str = None,
        output_filename_beeswarm: str = None,
        n_sample: int = 10_000,
        feature_importance_kws: dict = None,
        beeswarm_kws: dict = None,
    ):
        """Calculate and plot SHAP values for the model."""
        # Skip SHAP for dummy classifiers (no real model available)
        model_to_check = self.models[0]
        underlying = getattr(model_to_check, "model", model_to_check)
        if isinstance(underlying, DummyClassifier):
            logger.info("Skipping SHAP: no real model available (DummyClassifier)")
            return

        feature_importance_kws = feature_importance_kws or {}
        beeswarm_kws = beeswarm_kws or {}

        # Create SHAPPlotter instance
        shap_plotter = self._create_shap_plotter()

        # Define fold to use
        k = 0  # fold_id
        data_subset = self.data_df[self.data_df[FOLD_ID] == k]

        logger.info("Calculating SHAP values")

        # Use plot_both for efficient calculation and plotting
        fig_importance, fig_beeswarm, shap_values, x_sample = shap_plotter.plot_both(
            data_subset=data_subset,
            fold_id=k,
            n_sample=n_sample,
            n_features_importance=feature_importance_kws.get("n_features", 15),
            n_features_beeswarm=beeswarm_kws.get("n_features", 10),
            nplot_sample=beeswarm_kws.get("nplot_sample", None),
            cmap=beeswarm_kws.get("cmap", "brg"),
            xlims_importance=feature_importance_kws.get("xlims", None),
            xlims_beeswarm=beeswarm_kws.get("xlims", None),
            output_filename_importance=None,  # We'll save using our own method
            output_filename_beeswarm=None,  # We'll save using our own method
            figsize_importance=(20, 10),
            figsize_beeswarm=(20, 10),
            show_colorbar=beeswarm_kws.get("show_colorbar", True),
            show_other_features=beeswarm_kws.get("show_other_features", True),
        )

        logger.info("Done calculating SHAP values")

        # Calculate and save SHAP feature importance scores
        mean_abs_SHAP_scores = pd.Series(  # noqa: N806
            np.abs(shap_values[:, :-1]).mean(axis=0), index=x_sample.columns
        ).sort_values(ascending=False)
        mean_abs_SHAP_scores.to_hdf(self.output_h5_filename, key="mean_abs_SHAP_scores", mode="a")

        # Save plots using the existing _save_plt method to maintain consistency
        self._save_plt(output_filename=output_filename_importance, fig=fig_importance)
        self._save_plt(output_filename=output_filename_beeswarm, fig=fig_beeswarm)

    @exception_handler
    def calc_and_plot_trinuc_plot(  # noqa: PLR0915 #TODO: refactor
        self,
        output_filename: str = None,
        order: str = "symmetric",
        motif_orientation: str = "seq_dir",
    ):
        logger.info("Calculating trinuc context statistics")

        # The display variant declares how to split the trinuc quality panel. Mixed uses the raw
        # is_mixed bool (trinuc_split_col set, group_specs=None -> the plotter auto-detects, keeping
        # trinuc_stats byte-identical). N-group schemes leave trinuc_split_col=None: we materialize
        # the read_group column and pass explicit group_specs.
        variant = self._display_variant
        split_col = variant.trinuc_split_col
        group_specs = variant.trinuc_group_specs
        if split_col is None:
            split_col = READ_GROUP
            palette = self._variant_palette(variant)
            # calc_trinuc_stats names group columns "mixed=<group value>"; match that prefix.
            group_specs = [(f"mixed={label}", label, palette[label]) for label in variant.groups]

        # Call the new plotting function
        fig, stats_df = calc_and_plot_trinuc_hist(
            self.data_df,
            trinuc_col=TRINUC_CONTEXT_WITH_ALT,
            label_col=LABEL,
            labels=[True, False],
            is_forward_col=IS_FORWARD,
            qual_col=QUAL,
            order=order,
            # figsize=(16, 9),
            collapsed=False,  # Use non-collapsed mode to show both forward and reverse
            include_quality=True,  # Include quality panels like the original
            motif_orientation=motif_orientation,
            q1=0.1,
            q2=0.9,
            split_col=split_col,
            group_specs=group_specs,
        )
        stats_df.to_hdf(self.output_h5_filename, key="trinuc_stats", mode="a")

        self._save_plt(output_filename=output_filename, fig=fig)

    def _feature_group_series(self):
        """Ordered list of (group_label, color) for the per-feature quality curves.

        Colors come from the display variant (mixed declares Non-mixed=tab:red, Mixed=tab:green;
        N-group schemes fall back to a seaborn palette), in the variant's group order.
        """
        palette = self._variant_palette()
        return [(label, palette[label]) for label in self._display_variant.groups]

    def _plot_feature_qual_stats(
        self, stats_for_plot, ax, c_true="tab:green", c_false="tab:red", min_count=100, fb_kws=None, step_kws=None
    ):
        """Generate a plot of quality median + 10-90 percentile range, per read group."""
        fb_kws = fb_kws or {}
        step_kws = step_kws or {}
        col = stats_for_plot.columns[0]
        polys, lines = [], []
        for label, color in self._feature_group_series():
            group_mask = stats_for_plot[READ_GROUP].astype(str) == str(label)
            qual_df = stats_for_plot.loc[group_mask & stats_for_plot[LABEL] & (stats_for_plot["count"] > min_count), :]
            fb_kws["color"] = color
            step_kws["color"] = color
            if qual_df.shape[0] > 0:
                poly, line = plot_box_and_line(
                    qual_df,
                    col,
                    "median_qual",
                    "quantile1_qual",
                    "quantile3_qual",
                    ax=ax,
                    fb_kws=fb_kws,
                    step_kws=step_kws,
                )
            else:
                poly, line = None, None
            polys.append(poly)
            lines.append(line)
        return polys, lines

    def _get_stats_for_feature_plot(self, col, q1=0.1, q2=0.9, bin_edges=None):
        """Get stats (median, quantiles) for a feature, for plotting.
        Args:
            data_df [pd.DataFrame]: DataFrame with the data
            col [str]: name of column for which to get stats
            q1 [float]: lower quantile for interquartile range
            q2 [float]: upper quantile for interquartile range
            bin_edges [list]: bin edges for discretization. If it is None, use discrete (integer) values
        """
        required_cols = [col, LABEL, READ_GROUP, QUAL]

        # If binning is needed, create binned column in a view/subset
        if bin_edges is not None:
            # Work with a subset of data to avoid full copy
            subset = self.data_df[required_cols].copy()  # Only copy required columns
            subset[col] = pd.cut(subset[col], bin_edges, labels=(bin_edges[1:] + bin_edges[:-1]) / 2)
        else:
            # Use a view of the original dataframe (no copy needed)
            subset = self.data_df[required_cols]

        stats_for_plot = subset.groupby([col, LABEL, READ_GROUP], observed=True).agg(
            median_qual=(QUAL, "median"),
            quantile1_qual=(QUAL, lambda x: x.quantile(q1)),
            quantile3_qual=(QUAL, lambda x: x.quantile(q2)),
            count=(QUAL, "size"),
        )
        total_count = len(self.data_df)
        stats_for_plot["fraction"] = stats_for_plot["count"] / total_count
        stats_for_plot = stats_for_plot.reset_index()
        return stats_for_plot

    @exception_handler
    def plot_numerical_feature_hist_and_qual(
        self, col: str, q1: float = 0.1, q2: float = 0.9, nbins: int = 50, output_filename: str = None
    ):
        """Plot histogram and quality stats for a numerical feature."""
        logger.info(f"Plotting quality and histogram for feature {col}")
        if self.data_df[col].isna().all():
            logger.warning(f"Column {col} contains only NaN values. Skipping plot.")
            return
        if hasattr(self.data_df[col], "cat") or self.data_df[col].dtype.name in ("category", "object"):
            is_discrete = True
        else:
            is_discrete = (
                (self.data_df[col] - np.round(self.data_df[col])).abs().max() < 0.05  # noqa: PLR2004
                or (
                    self.data_df.loc[self.data_df[col].notna(), col]
                    == self.data_df.loc[self.data_df[col].notna(), col].iloc[0]
                ).all()  # A constant column
            )
        if is_discrete:
            bin_edges = None
        else:
            bin_edges = discretized_bin_edges(self.data_df[col].values, bins=nbins)
        stats_for_plot = self._get_stats_for_feature_plot(col, q1=q1, q2=q2, bin_edges=bin_edges)

        # Plot paramters
        yticks_fontsize = 12
        label_fontsize = 14
        legend_fontsize = 14
        fig, (ax2, ax) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 3], "hspace": 0}
        )

        # Bottom figure: histogram
        sns.histplot(
            data=self.data_df,
            x=col,
            hue=LABEL,
            discrete=is_discrete,
            bins=bin_edges,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
            hue_order=[False, True],
        )
        ax.set_xlabel(col, fontsize=label_fontsize)
        ax.set_ylabel("Density", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=yticks_fontsize)
        ax.grid(visible=True)
        legend = ax.get_legend()
        hist_handles = legend.legend_handles
        legend.remove()

        # Top figure: quality
        polys, lines = self._plot_feature_qual_stats(stats_for_plot, ax=ax2, min_count=50)
        ax2.grid(visible=True)
        ax2.tick_params(axis="y", labelsize=yticks_fontsize)
        ax2.set_ylabel("SNVQ on\nTP reads", fontsize=label_fontsize)
        plt.tight_layout()

        # plot legends
        plt.subplots_adjust(bottom=0.2)
        empty_handle = mlines.Line2D([], [], color="none")
        # Create a custom legend
        fig.legend(
            hist_handles,
            ["FP", "TP"],
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=(0.25, -0.15),
            ncol=1,
            frameon=False,
        )
        # Replace missing handles (e.g. an empty group) so the legend still renders.
        lines = [line if line is not None else empty_handle for line in lines]
        polys = [poly if poly is not None else empty_handle for poly in polys]
        # One legend block per read group, in group order.
        group_labels = [label for label, _ in self._feature_group_series()]
        legend_handles = []
        legend_texts = []
        for gi, label in enumerate(group_labels):
            legend_handles += [empty_handle, lines[gi], polys[gi]]
            legend_texts += [label, "median", "10%-90% range"]
        fig.legend(
            legend_handles,
            legend_texts,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=(0.7, -0.15),
            ncol=len(group_labels),
            frameon=False,
        )
        self._save_plt(output_filename=output_filename, fig=fig)

    def _get_histogram_data(self, ax, col_name=""):
        """Get histogram data from a matplotlib axis that was created by a call
        for seaborn's sns.histplot(..., element='step').
        Arguments:
            ax [matplotlib.Axes]: matplotlib axis
            col_name [str]: column name to assign to the histogram data
        """
        hist_dict = {}
        for poly_collection, label in zip(ax.collections, ax.get_legend().get_texts(), strict=False):
            vertices = poly_collection.get_paths()[0].vertices
            hist_dict[(col_name + " " + label.get_text(), "bin_edges")] = vertices[:, 0]
            hist_dict[(col_name + " " + label.get_text(), "density")] = vertices[:, 1]
        return pd.DataFrame(hist_dict)

    @exception_handler
    def plot_quality_histogram(  # noqa: PLR0915
        self, *, plot_interpolating_function: bool = False, output_filename: str = None
    ):
        """Plot a histogram of qual values for TP reads, split by the mode's read groups."""
        # label_fontsize = 12
        # ticklabelsfontsize = 12
        logger.info("Plotting SNVQ histogram")
        if plot_interpolating_function:
            fig, axes = plt.subplots(
                3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [5, 2, 3], "hspace": 0}
            )
            ax = axes[0]
        else:
            fig, ax = plt.subplots(figsize=(12, 7))
            axes = [ax]

        plot_df = self.data_df[self.data_df[LABEL]].copy()
        hue_col = self.tag_axis
        legacy_variant = self.scheme.legacy_variant
        display_variant = self._display_variant

        # Legacy histogram for backward-compatible h5 storage. ppmSeq (mixed) uses a 3-category
        # split (non-mixed / exactly one end / both ends); other schemes group by the legacy
        # variant's read groups.
        plot_df_legacy = plot_df.copy()
        neg_lc, pos_lc = display_variant.neg_lc, display_variant.pos_lc
        if self.scheme.ppmseq_legacy_hist:
            legacy_one_end = f"{pos_lc}, exactly one end"
            legacy_both_ends = f"{pos_lc}, both ends"
            plot_df_legacy[hue_col] = neg_lc
            one_end_mask = plot_df_legacy[IS_MIXED_START] ^ plot_df_legacy[IS_MIXED_END]
            both_ends_mask = plot_df_legacy[IS_MIXED_START] & plot_df_legacy[IS_MIXED_END]
            plot_df_legacy.loc[one_end_mask, hue_col] = legacy_one_end
            plot_df_legacy.loc[both_ends_mask, hue_col] = legacy_both_ends
            legacy_hue_order = [neg_lc, legacy_one_end, legacy_both_ends]
            legacy_palette = {neg_lc: "red", legacy_one_end: "blue", legacy_both_ends: "green"}
            display_hue_order = [neg_lc, pos_lc]
            display_palette = {neg_lc: "red", pos_lc: "blue"}
            # Map the display groups to their lowercase labels (these become the h5 column names,
            # so they must stay the literal "non-mixed"/"mixed" strings for byte-identical output).
            lc_map = {display_variant.neg: neg_lc, display_variant.pos: pos_lc}
            plot_df[hue_col] = [lc_map[g] for g in display_variant.group_fn(plot_df)]
        else:
            # N-group schemes: legacy plot equals the display plot (no start/end distinction).
            plot_df_legacy[hue_col] = np.asarray(legacy_variant.group_fn(plot_df_legacy)).astype(str)
            plot_df[hue_col] = np.asarray(display_variant.group_fn(plot_df)).astype(str)
            legacy_hue_order = list(legacy_variant.groups)
            display_hue_order = list(display_variant.groups)
            legacy_palette = {str(k): v for k, v in self._variant_palette(legacy_variant).items()}
            display_palette = {str(k): v for k, v in self._variant_palette(display_variant).items()}
        legacy_col_name = legacy_variant.hist_col_name or self.tag_axis
        display_col_name = display_variant.hist_col_name or self.tag_axis
        fig_legacy, ax_legacy = plt.subplots(figsize=(12, 7))
        g_legacy = sns.histplot(
            data=plot_df_legacy,
            x=QUAL,
            hue=hue_col,
            hue_order=legacy_hue_order,
            element="step",
            stat="density",
            common_norm=False,
            linewidth=1,
            ax=ax_legacy,
            palette=legacy_palette,
        )
        hist_data_df_legacy = self._get_histogram_data(g_legacy, col_name=legacy_col_name)
        hist_data_df_legacy.to_hdf(self.output_h5_filename, key="quality_histogram", mode="a")
        plt.close(fig_legacy)

        # New histogram (display groups)
        g = sns.histplot(
            data=plot_df,
            x=QUAL,
            hue=hue_col,
            hue_order=display_hue_order,
            element="step",
            stat="density",
            common_norm=False,
            kde=True,
            kde_kws={"bw_adjust": 3},
            linewidth=1,
            ax=ax,
            palette=display_palette,
        )
        sns.move_legend(
            ax,
            "upper left",
        )
        ax.set_ylabel("Density")  # , fontsize=label_fontsize)
        ax.grid(visible=True)
        xlims = [0.99 * plot_df[QUAL].min(), plot_df[QUAL].max() * 1.01]
        hist_data_df = self._get_histogram_data(g, col_name=display_col_name)
        hist_data_df.to_hdf(self.output_h5_filename, key=split_h5_key("quality_histogram", display_variant), mode="a")
        if plot_interpolating_function:
            xs = np.arange(self.eps, np.floor(self.data_df[ML_QUAL_1_TEST].max()), 0.1)
            ax = axes[1]
            ax.plot(
                self.ML_qual_to_qual_fn((xs[1:] + xs[:-1]) / 2),
                0.1 / (self.ML_qual_to_qual_fn(xs)[1:] - self.ML_qual_to_qual_fn(xs)[:-1]),
                ".-",
            )
            ax.set_ylabel("ML_qual'")  # , fontsize=label_fontsize)
            ax.grid(visible=True)

            ax = axes[2]
            ax.plot(self.ML_qual_to_qual_fn(xs), xs, ".-")
            ax.set_ylabel("ML_qual")  # , fontsize=label_fontsize)
            ax.grid(visible=True)
            # If plotting the interpolating function, might need to widen the xlims
            xlims = [0.99 * self.ML_qual_to_qual_fn(xs).min(), self.ML_qual_to_qual_fn(xs).max() * 1.01]

        ax.set_xlabel("SNVQ")  # , fontsize=label_fontsize)
        ax.set_xlim(xlims)
        # ticklabels = ax.get_xmajorticklabels()
        # ax.set_xticklabels(ticklabels, fontsize=ticklabelsfontsize)
        # for ax in axes:
        #     ticklabels = ax.get_ymajorticklabels()
        #     ax.set_yticklabels(ticklabels, fontsize=ticklabelsfontsize)

        fig.tight_layout()
        self._save_plt(output_filename=output_filename, fig=fig)

    def _logit_extra_cols(self):
        """Columns (besides read_group) that the active scheme's variant group_fns read, so the
        logit-histogram slice includes them. Only columns present in data_df are returned.

        Covers every scheme's raw group inputs (mixed: is_mixed/_start; consensus: fs/rs); a new
        scheme should add its raw group columns here so its group_fn can run on the sliced frame."""
        candidates = [IS_MIXED, IS_MIXED_START, FS, RS]
        return [c for c in candidates if c in self.data_df.columns]

    def _plot_logit_variant(self, plot_df, ax, variant, alpha=0.4):
        """Plot one logit histogram for a variant: FP + one TP series per group.

        Binary variants (``variant.pos`` set) reproduce the historical FP / TP {pos_lc} / TP {neg_lc}
        series and palette (FP blue, neg red, pos green). N-group variants get one TP series per
        group with a seaborn palette.
        """
        grp = np.asarray(variant.group_fn(plot_df))
        if variant.pos is not None:
            tp_pos = f"TP {variant.pos_lc}"
            tp_neg = f"TP {variant.neg_lc}"
            plot_df[""] = plot_df[LABEL].astype(str)
            plot_df.loc[~plot_df[LABEL], ""] = "FP"
            plot_df.loc[plot_df[LABEL] & (grp == variant.pos), ""] = tp_pos
            plot_df.loc[plot_df[LABEL] & (grp == variant.neg), ""] = tp_neg
            hue_order = ["FP", tp_pos, tp_neg]
            palette = {"FP": "tab:blue", tp_neg: "red", tp_pos: "green"}
        else:
            tp_labels = [f"TP {label}" for label in variant.groups]
            plot_df[""] = "FP"
            for label in variant.groups:
                plot_df.loc[plot_df[LABEL] & (grp == label), ""] = f"TP {label}"
            group_colors = list(sns.color_palette(n_colors=len(variant.groups)))
            palette = {"FP": "tab:blue", **dict(zip(tp_labels, group_colors, strict=False))}
            hue_order = ["FP", *tp_labels]
        sns.histplot(
            data=plot_df,
            x=ML_LOGIT_TEST,
            hue="",
            hue_order=hue_order,
            palette=palette,
            element="step",
            stat="density",
            common_norm=False,
            alpha=alpha,
            ax=ax,
        )

    @exception_handler
    def plot_logit_histograms(self, *, plot_by_fold: bool = True, output_filename: str = None):
        """Plot a histogram of logit values, by: FP + one TP series per group.
        If plot_by_fold is True, overlay histograms for each fold.
        """
        # label_fontsize = 12
        # ticklabelsfontsize = 12
        logger.info("Plotting logit histogram")

        plot_df_all = self.data_df[[ML_LOGIT_TEST, LABEL, READ_GROUP, FOLD_ID, *self._logit_extra_cols()]].copy()
        legacy_variant = self.scheme.legacy_variant
        display_variant = self._display_variant

        # Legacy histogram (legacy variant) for backward-compatible h5 storage.
        fig_legacy, ax_legacy = plt.subplots(figsize=(12, 7))
        self._plot_logit_variant(plot_df_all.copy(), ax_legacy, legacy_variant)
        hist_data_df_legacy = self._get_histogram_data(ax_legacy, col_name="")
        hist_data_df_legacy.to_hdf(self.output_h5_filename, key="logit_histogram", mode="a")
        plt.close(fig_legacy)

        # New histogram (display variant) for display and new h5 key
        fig, ax = plt.subplots(figsize=(12, 7))
        self._plot_logit_variant(plot_df_all.copy(), ax, display_variant)
        hist_data_df = self._get_histogram_data(ax, col_name="")
        hist_data_df.to_hdf(self.output_h5_filename, key=split_h5_key("logit_histogram", display_variant), mode="a")

        if plot_by_fold:
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(12, 7))
            plot_dfs = [plot_df_all[plot_df_all[FOLD_ID] == k] for k in range(len(self.models))]
            for plot_df in plot_dfs:
                self._plot_logit_variant(plot_df.copy(), ax, display_variant, alpha=0.15)

        sns.move_legend(
            ax,
            "upper left",
        )
        ax.set_ylabel("Density")  # , fontsize=label_fontsize)
        ax.grid(visible=True)
        xmin, xmax = [0.99 * self.data_df[ML_LOGIT_TEST].min(), self.data_df[ML_LOGIT_TEST].max() * 1.01]
        ax.set_xlabel("ML logit")  # , fontsize=label_fontsize)
        ax.set_xlim([xmin, xmax])
        fig.tight_layout()
        # plt.show()
        self._save_plt(output_filename=output_filename, fig=fig)

    @exception_handler
    def plot_hmer_indel_metrics(self, output_filename: str = None):
        """Joint SNV vs hmer-indel figure: model quality and TP count by reference homopolymer length.

        Only produced when the featuremap carries homopolymer indels (``variant_type`` == "hmer_indel",
        set by ``prepare_report``); for SNV-only runs the method returns without writing a file and the
        report notebook skips the section. Metrics are computed on TP rows and split by variant type so
        SNV and hmer-indel behaviour can be compared across homopolymer lengths (0..12). Also writes the
        per-(variant_type, hmer_length) table to the QC h5 under key ``hmer_indel_stats``.
        """
        if VARIANT_TYPE not in self.data_df.columns or X_HMER_REF not in self.data_df.columns:
            logger.info("plot_hmer_indel_metrics: variant_type / X_HMER_REF missing; skipping")
            return
        data = self.data_df
        if not (data[VARIANT_TYPE] == VARIANT_TYPE_HMER_INDEL).any():
            logger.info("plot_hmer_indel_metrics: no hmer-indel rows; skipping")
            return

        tp = data[data[LABEL].astype(bool)]
        hmer_len = pd.to_numeric(tp[X_HMER_REF], errors="coerce").clip(lower=0, upper=12)
        stats = (
            pd.DataFrame({"hmer_length": hmer_len, VARIANT_TYPE: tp[VARIANT_TYPE], "mqual": tp[ML_QUAL_1_TEST]})
            .dropna(subset=["hmer_length"])
            .groupby([VARIANT_TYPE, "hmer_length"])
            .agg(median_mqual=("mqual", "median"), count=("mqual", "size"))
            .reset_index()
        )
        stats.to_hdf(self.output_h5_filename, key="hmer_indel_stats", mode="a")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for grp, grp_rows in stats.groupby(VARIANT_TYPE):
            grp_sorted = grp_rows.sort_values("hmer_length")
            axes[0].plot(grp_sorted["hmer_length"], grp_sorted["median_mqual"], marker="o", label=grp)
            axes[1].plot(grp_sorted["hmer_length"], grp_sorted["count"], marker="o", label=grp)
        axes[0].set_xlabel("reference homopolymer length")
        axes[0].set_ylabel("median MQUAL (TP)")
        axes[0].set_title("Model quality vs homopolymer length")
        axes[1].set_xlabel("reference homopolymer length")
        axes[1].set_ylabel("TP count")
        axes[1].set_title("TP count vs homopolymer length")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.legend(title="variant type")
            ax.grid(visible=True, alpha=0.3)
        self._save_plt(output_filename=output_filename, fig=fig)
        plt.close(fig)

    @exception_handler
    def plot_hmer_indel_by_class(self, output_filename: str = None):
        """Homopolymer-indel TP metrics split by indel class (insertion vs deletion) and hmer length.

        Complements :func:`plot_hmer_indel_metrics` (which splits SNV vs hmer-indel) by breaking the
        hmer-indel TPs into insertions vs deletions. Writes the per-(indel_class, hmer_length) table to
        the QC h5 under ``hmer_indel_class_stats``. Self-skips when there are no hmer-indel TP rows.
        """
        if VARIANT_TYPE not in self.data_df.columns or X_IC not in self.data_df.columns:
            logger.info("plot_hmer_indel_by_class: variant_type / X_IC missing; skipping")
            return
        data = self.data_df
        ind = data[(data[VARIANT_TYPE] == VARIANT_TYPE_HMER_INDEL) & data[LABEL].astype(bool)]
        if ind.empty:
            logger.info("plot_hmer_indel_by_class: no hmer-indel TP rows; skipping")
            return
        hmer_len = pd.to_numeric(ind[X_HMER_REF], errors="coerce").clip(lower=0, upper=12)
        stats = (
            pd.DataFrame(
                {
                    "hmer_length": hmer_len,
                    "indel_class": ind[X_IC].astype(str).str.lower(),
                    "mqual": ind[ML_QUAL_1_TEST],
                }
            )
            .dropna(subset=["hmer_length"])
            .groupby(["indel_class", "hmer_length"])
            .agg(median_mqual=("mqual", "median"), count=("mqual", "size"))
            .reset_index()
        )
        stats.to_hdf(self.output_h5_filename, key="hmer_indel_class_stats", mode="a")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for cls_name, cls_rows in stats.groupby("indel_class"):
            cls_sorted = cls_rows.sort_values("hmer_length")
            axes[0].plot(cls_sorted["hmer_length"], cls_sorted["median_mqual"], marker="o", label=cls_name)
            axes[1].plot(cls_sorted["hmer_length"], cls_sorted["count"], marker="o", label=cls_name)
        axes[0].set_xlabel("reference homopolymer length")
        axes[0].set_ylabel("median MQUAL (TP)")
        axes[0].set_title("Homopolymer-indel model quality by class")
        axes[1].set_xlabel("reference homopolymer length")
        axes[1].set_ylabel("TP count")
        axes[1].set_title("Homopolymer-indel TP count by class")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.legend(title="indel class")
            ax.grid(visible=True, alpha=0.3)
        self._save_plt(output_filename=output_filename, fig=fig)
        plt.close(fig)

    @exception_handler
    def calc_hmer_indel_auc_table(self):
        """ROC-AUC of MQUAL discriminating TP vs FP, split by group: SNV, hmer-indel, and ins / del.

        The existing ``roc_auc_table`` splits by read-group (duplex/ppmSeq); this adds the orthogonal
        variant-type / indel-class split. Written to the QC h5 under ``hmer_indel_auc_table`` (rendered as
        a table by the report notebook). AUC is reported both raw and phred (``-10*log10(1-AUC)``).
        """
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415  (optional [ml] dependency)

        if (
            VARIANT_TYPE not in self.data_df.columns
            or not (self.data_df[VARIANT_TYPE] == VARIANT_TYPE_HMER_INDEL).any()
        ):
            logger.info("calc_hmer_indel_auc_table: no hmer-indel rows; skipping")
            return
        data = self.data_df
        y = data[LABEL].astype(int).to_numpy()
        q = pd.to_numeric(data[ML_QUAL_1_TEST], errors="coerce").to_numpy()
        cls = data[X_IC].astype(str).str.lower() if X_IC in data.columns else pd.Series("", index=data.index)
        is_indel = data[VARIANT_TYPE] == VARIANT_TYPE_HMER_INDEL
        groups = {
            "snv": (data[VARIANT_TYPE] == "snv").to_numpy(),
            "hmer_indel": is_indel.to_numpy(),
            "hmer_indel:ins": (is_indel & (cls == "ins")).to_numpy(),
            "hmer_indel:del": (is_indel & (cls == "del")).to_numpy(),
        }
        rows = []
        for name, mask in groups.items():
            m = mask & np.isfinite(q)
            yy, qq = y[m], q[m]
            n_tp, n_fp = int((yy == 1).sum()), int((yy == 0).sum())
            if n_tp > 0 and n_fp > 0:
                auc = float(roc_auc_score(yy, qq))
                auc_phred = float(-10 * np.log10(max(1 - auc, 1e-10)))
            else:
                auc, auc_phred = np.nan, np.nan
            rows.append({"group": name, "roc_auc": auc, "roc_auc_phred": auc_phred, "n_TP": n_tp, "n_FP": n_fp})
        pd.DataFrame(rows).to_hdf(self.output_h5_filename, key="hmer_indel_auc_table", mode="a")

    def create_report(self):
        """Generate plots for report and save data in hdf5 file."""
        logger.info("Creating report")
        # Get filenames for plots
        [
            FQ_vs_recall_plot,  # noqa: N806
            qual_vs_ppmseq_tags_table,
            training_progerss_plot,
            SHAP_importance_plot,  # noqa: N806
            SHAP_beeswarm_plot,  # noqa: N806
            trinuc_stats_plot,
            output_qual_per_feature,
            qual_histogram,
            logit_histogram,
            calibration_fn_with_hist,
        ] = _get_plot_paths(out_path=self.params["workdir"], out_basename=self.params["data_name"])
        # General info
        self.plot_fq_recall(output_filename=FQ_vs_recall_plot)
        self.calc_run_info_table()
        # Quality stats
        self.calc_run_quality_table()
        self.quality_per_ppmseq_tags(output_filename=qual_vs_ppmseq_tags_table)

        # ROC AUC stats
        self.calc_roc_auc_table()

        # ML_qual histograms
        self.plot_logit_histograms(output_filename=logit_histogram)
        self.plot_interpolating_function_with_histograms(output_filename=calibration_fn_with_hist)

        # Training progress
        self.training_progress_plot(output_filename=training_progerss_plot)

        # SHAP
        self.calc_and_plot_shap_values(
            output_filename_importance=SHAP_importance_plot, output_filename_beeswarm=SHAP_beeswarm_plot
        )

        # Quality histogram
        self.plot_quality_histogram(output_filename=qual_histogram)

        # Trinuc stats plot
        self.calc_and_plot_trinuc_plot(output_filename=trinuc_stats_plot, order="symmetric")

        # Quality and histogram for numerical features
        for col in self.params["numerical_features"]:
            self.plot_numerical_feature_hist_and_qual(col, output_filename=output_qual_per_feature + col)

        # Joint SNV + hmer-indel metrics (only writes a file when hmer indels are present). Path follows
        # the same naming convention create_srsnv_report_html reconstructs, so _get_plot_paths is untouched.
        hmer_indel_metrics_plot = os.path.join(self.params["workdir"], f"{self.params['data_name']}hmer_indel_metrics")
        self.plot_hmer_indel_metrics(output_filename=hmer_indel_metrics_plot)
        hmer_indel_by_class_plot = os.path.join(
            self.params["workdir"], f"{self.params['data_name']}hmer_indel_by_class"
        )
        self.plot_hmer_indel_by_class(output_filename=hmer_indel_by_class_plot)
        self.calc_hmer_indel_auc_table()

        # # Create LoD plot
        # # TODO: Update the following to new conform with new report logic
        # if self.params.get("fp_regions_bed_file", None) is not None:  # TODO: Check why this if statement?
        #     logger.info("Calculating LoD statistics")
        #     min_LoD_filter = calculate_lod_stats(  # noqa: N806
        #         df_mrd_simulation=self.df_mrd_simulation,
        #         output_h5=self.statistics_h5_file,
        #         lod_column=self.c_lod,
        #     )
        #     logger.info("Creating SNVQ-recall-LoD plot")
        #     self.df_mrd_simulation.unstack().to_hdf(self.output_h5_filename, key="SNV_recall_LoD", mode="a")  # noqa PD010
        #     plot_LoD(
        #         self.df_mrd_simulation,
        #         self.lod_label,
        #         self.c_lod,
        #         self.lod_filters,
        #         self.params["adapter_version"],
        #         min_LoD_filter,
        #         output_filename=output_LoD_plot,
        #     )

        # Adding keys_to_convert to h5
        keys_to_convert = pd.Series(
            [
                "mean_abs_SHAP_scores",
                "ppmseq_category_quality_table",
                "ppmseq_category_quantity_table",
                "roc_auc_table",
                "run_quality_summary_table",
                "run_quality_table",
                "training_info_table",
                "training_progress",
                "trinuc_stats",
                "FQ_recall_LoD",
            ]
        )
        keys_to_convert.to_hdf(self.output_h5_filename, key="keys_to_convert", mode="a")

        # TODO: Check what following lines do and uncomment if needed
        # # convert statistics to json
        # convert_h5_to_json(
        #     input_h5_filename=self.statistics_h5_file,
        #     root_element="metrics",
        #     ignored_h5_key_substring=None,
        #     output_json=self.statistics_json_file,
        # )


def precision_score_with_mask(y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray):
    """calculates the precision score for the predictions and labels with mask applied

    Parameters
    ----------
    y_pred : np.ndarray
        model predictions
    y_true : np.ndarray
        labels
    mask : np.ndarray
        mask for the data

    Returns
    -------
    float
        sklearn precision score for the predictions and labels with mask applied
    """
    if y_pred[mask].sum() == 0:
        return 1
    return precision_score(y_true[mask], y_pred[mask])


def recall_score_with_mask(y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray):
    """calculates the recall score for the predictions and labels with mask applied

    Parameters
    ----------
    y_pred : np.ndarray
        model predictions
    y_true : np.ndarray
        labels
    mask : np.ndarray
        mask for the data

    Returns
    -------
    float
        sklearn recall score for the predictions and labels with mask applied
    """
    if y_true[mask].sum() == 0:
        return 1
    return recall_score(y_true[mask], y_pred[mask])


def precision_recall_curve(score, max_score, y_true: np.ndarray, *, cumulative=False, apply_log_trans=True):
    """apply threshold on score and calculate precision and recall rates

    Parameters
    ----------
    score : pd.dataframe
        model score
    max_score : _type_
        maximal ML model score for threshold
    y_true : np.ndarray
        labels
    cumulative : bool, optional
        whether to calculate cumulative scores, by default False
    apply_log_trans : bool, optional
        whether to use precision for false positive rate, or apply log transformation to get qual, by default True

    Returns
    -------
    fprs : list
        list of false positive rates per threshold
    recalls : list
        list of recall rates per threshold
    """
    precisions = []
    recalls = []
    fprs = []
    for i in range(max_score):
        if cumulative:
            mask = score.apply(np.floor) >= i
        else:
            mask = score.apply(np.floor) == i
        prediction = mask
        no_mask = (score + 1).astype(bool)
        precisions.append(precision_score_with_mask(prediction, y_true, mask))
        recalls.append(recall_score_with_mask(prediction, y_true, no_mask))
        if precisions[-1] == np.nan:
            fprs.append(np.none)
        elif apply_log_trans:
            if precisions[-1] == 1:
                qual = max_score
            else:
                qual = -10 * np.log10(1 - precisions[-1])
            qual = min(qual, max_score)
            fprs.append(qual)
        else:
            fprs.append(precisions[-1])
    return fprs, recalls


def plot_precision_recall(lists, labels, max_score, *, log_scale=False, font_size=18):
    """generate a plot of precision and recall rates per threshold

    Parameters
    ----------
    lists : list[list]
        lists of precision and recall rates per threshold
    labels : list[str]
        list of labels for the lists
    max_score : float
        maximal model score for plot
    log_scale : bool, optional
        whether to plot in log scale, by default False
    font_size : int, optional
        font size, by default 18
    """
    set_pyplot_defaults()

    for lst, label in zip(lists, labels, strict=False):
        plt.plot(lst[0:max_score], ".-", label=label)
        plt.xlabel("ML qual", fontsize=font_size)
        if log_scale:
            plt.yscale("log")


def plot_LoD_vs_qual(  # noqa: N802
    df_mrd_sim: pd.DataFrame,
    c_lod: str,
    title: str = "",
    output_filename: str = None,
    font_size: int = 18,
):
    """generate a plot of LoD vs ML qual

    Parameters:
    ----------
    df_mrd_sim : pd.DataFrame,
        simulated data set with noise and LoD stats
    c_lod : str
        name of the LoD column in the data set
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size, by default 18
    """

    set_pyplot_defaults()

    x = np.array(
        [
            [
                int(item.split("_")[-1]),
                df_mrd_sim[c_lod].loc[item],
                df_mrd_sim["tp_read_retention_ratio"].loc[item],
            ]
            for item in df_mrd_sim.index
            if item[:2] == "ML"
        ]
    )

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ln1 = ax1.plot(x[:, 0], x[:, 1], marker="o", color="blue", label="LoD")
    ln2 = ax2.plot(
        x[:, 0],
        x[:, 2],
        marker="o",
        color="red",
        label="Base retention ratio \non HOM SNVs (TP)",
    )
    lns = ln1 + ln2
    labs = [ln.get_label() for ln in lns]

    ax1.invert_xaxis()
    ax1.set_yscale("log")
    ax2.yaxis.grid(visible=True)
    ax1.xaxis.grid(visible=True)
    ax1.yaxis.grid(visible=False)
    ax1.set_ylabel("LoD", fontsize=font_size)
    ax1.set_xlabel("ML qual", fontsize=font_size)
    ax2.set_ylabel("Base retention ratio \non HOM SNVs (TP)", fontsize=font_size)
    legend_handle = ax1.legend(
        lns,
        labs,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=font_size,
    )
    title_handle = plt.title(title, fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )
