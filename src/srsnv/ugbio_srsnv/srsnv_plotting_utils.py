from __future__ import annotations

import functools
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
import xgboost as xgb
from matplotlib import cm, colors
from matplotlib import lines as mlines
from scipy.interpolate import interp1d
from scipy.stats import binom
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from ugbio_core.filter_bed import count_bases_in_bed_file
from ugbio_core.h5_utils import convert_h5_to_json
from ugbio_core.logger import logger
from ugbio_core.plotting_utils import set_pyplot_defaults
from ugbio_core.reports.report_utils import generate_report
from ugbio_core.sorter_utils import read_effective_coverage_from_sorter_json
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_ppmseq.ppmSeq_utils import PpmseqAdapterVersions, PpmseqCategories

# featuremap_df column names. TODO: make more generic?
ML_PROB_1_TEST = "ML_prob_1_test"
ML_QUAL_1_TEST = "ML_qual_1_test"
ML_LOGIT_TEST = "ML_logit_test"
LABEL = "label"
QUAL = "qual"
IS_MIXED = "is_mixed"
IS_MIXED_START = "is_mixed_start"
IS_MIXED_END = "is_mixed_end"
FOLD_ID = "fold_id"

edist_filter = f"{FeatureMapFields.X_EDIST.value} <= 5"
HQ_SNV_filter = f"{FeatureMapFields.X_SCORE.value} >= 7.9"
CSKP_SNV_filter = f"{FeatureMapFields.X_SCORE.value} >= 10"
read_end_filter = (
    f"{FeatureMapFields.X_INDEX.value} > 12 and "
    f"{FeatureMapFields.X_INDEX.value} < ({FeatureMapFields.X_LENGTH.value} - 12)"
)
mixed_read_filter = IS_MIXED  # TODO use adapter_version
default_LoD_filters = {  # noqa: N816
    "no_filter": f"{FeatureMapFields.X_SCORE.value} >= 0",
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


def prob_to_phred(arr, eps=1e-10):
    return -10 * np.log(1 - arr + eps) / np.log(10)


def prob_to_logit(arr, eps=1e-10):
    return 10 * np.log((arr + eps) / (1 - arr + eps)) / np.log(10)


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


def list_auc_to_qual(auc_list, eps=1e-10):
    return list(prob_to_phred(np.array(auc_list), eps=eps))


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

    labels = np.unique(df["label"].astype(int))

    df_tp = df.query("label == True")
    df_fp = df.query("label == False")

    fprs = {}
    recalls = {}
    max_score = -1
    for label in labels:
        fprs[label] = []
        recalls[label] = []
        score = f"ML_qual_{label}"
        max_score = np.max((int(np.ceil(df[score].max())), max_score))
        gtr = df["label"] == label
        fprs_, recalls_ = precision_recall_curve(df[score], max_score=max_score, y_true=gtr)
        fprs[label].append(fprs_)
        recalls[label].append(recalls_)

    return df, df_tp, df_fp, max_score, cls_features, fprs, recalls


def srsnv_report(
    out_path,
    out_basename,
    report_name,
    model_file,
    params_file,
    simple_pipeline=None,
):
    if len(out_basename) > 0 and not out_basename.endswith("."):
        out_basename += "."
    report_html = Path(out_path) / f"{out_basename}{report_name}_report.html"

    [
        output_LoD_plot,  # noqa: N806
        qual_vs_ppmseq_tags_table,
        training_progerss_plot,
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
        "model_file": model_file,
        "params_file": params_file,
        "srsnv_qc_h5_file": srsnv_qc_h5_filename,
        "output_LoD_plot": output_LoD_plot,
        "qual_vs_ppmseq_tags_table": qual_vs_ppmseq_tags_table,
        "training_progerss_plot": training_progerss_plot,
        "SHAP_importance_plot": SHAP_importance_plot,
        "SHAP_beeswarm_plot": SHAP_beeswarm_plot,
        "trinuc_stats_plot": trinuc_stats_plot,
        "output_qual_per_feature": output_qual_per_feature,
        "qual_histogram": qual_histogram,
        "logit_histogram": logit_histogram,
        "calibration_fn_with_hist": calibration_fn_with_hist,
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
    n_bases_in_region = count_bases_in_bed_file(single_sub_regions)

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
    c_lod = f"LoD_{sensitivity_at_lod*100:.0f}"
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

    lod_label = f"LoD @ {specificity_at_lod*100:.0f}% specificity, \
    {sensitivity_at_lod*100:.0f}% sensitivity (estimated)\
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

    cmat = confusion_matrix(y_true=df["label"], y_pred=df[prediction_column_name])
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
        gtr = df["label"] == label
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
            df[df["label"] == label][score].clip(upper=max_score),
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
                    df[df["label"] == label][feature]
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
                category_counts = df[df["label"] == label][feature].value_counts().sort_index()
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
                s_plot = df[df["label"] == label][feature]
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

    df["is_cycle_skip"] = df["is_cycle_skip"].astype(bool)
    if is_mixed_flag:
        df["is_mixed"] = df["is_mixed"].astype(bool)
        df_dict["mixed cycle skip"] = df[(df["is_mixed"] & df["is_cycle_skip"])]
        df_dict["mixed non cycle skip"] = df[(df["is_mixed"] & ~df["is_cycle_skip"])]
        df_dict["non mixed non cycle skip"] = df[(~df["is_mixed"] & ~df["is_cycle_skip"])]
        df_dict["non mixed cycle skip"] = df[(~df["is_mixed"] & df["is_cycle_skip"])]
    else:
        df_dict["cycle skip"] = df[df["is_cycle_skip"]]
        df_dict["non cycle skip"] = df[~df["is_cycle_skip"]]

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
        gtr = df_dict[key]["label"] == label
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
                td[td["label"] == label][score].clip(upper=max_score),
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

    output_LoD_plot = os.path.join(outdir, f"{basename}LoD_curve")  # noqa: N806
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
        output_LoD_plot,
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
    def __init__(  # noqa: PLR0913 C901
        self,
        models: list[sklearn.base.BaseEstimator],
        data_df: pd.DataFrame,
        params: dict,
        out_path: str,
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
        self.data_df = data_df
        self.params = params
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
        if rng is None:
            random_seed = int(datetime.now().timestamp())
            rng = np.random.default_rng(seed=random_seed)
            logger.info(f"SRSNVReport: Initializing random number generator with {random_seed=}")
            self.rng = np.random.default_rng(seed=14)
        else:
            self.rng = rng

        if statistics_json_file:
            if not statistics_h5_file:
                raise ValueError("statistics_h5_file is required when statistics_json_file is provided")

        # check model, data and params
        if not isinstance(models, list):
            raise TypeError(f"models should be a list of models, got {type(models)=}")
        for k, model in enumerate(models):
            if not sklearn.base.is_classifier(model):
                raise ValueError(f"model {model} (fold {k}) is not a classifier, please provide a classifier model")
        if not isinstance(data_df, pd.DataFrame):
            raise TypeError("df is not a DataFrame, please provide a DataFrame")
        expected_keys_in_params = [
            "fp_featuremap_entry_number",
            "fp_test_set_size",
            "fp_train_set_size",  # Do I really need this?
            "fp_regions_bed_file",
            "sorter_json_stats_file",
            "adapter_version",
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
        # add logits to data_df
        self.data_df["ML_logit_test"] = prob_to_logit(self.data_df["ML_prob_1_test"])
        self.data_df["ML_logit_train"] = prob_to_logit(self.data_df["ML_prob_1_train"])

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
            snvqs.append(prob_to_phred(1 - base_snvq * fpr / tpr))
            recalls.append(base_recall * tpr)
        return np.array(snvqs), np.array(recalls)

    def _get_recall_at_snvq(
        self, snvq: float, condition: np.ndarray = None, label_col: str = LABEL, ml_qual_col: str = ML_QUAL_1_TEST
    ):
        """Get the recall rate at a given SNVQ value, by interpolating the recall vs SNVQ curve.
        Arguments:
            snvq: float: SNVQ value at which to calculate the recall
            condition: np.ndarray: condition to filter the data frame. Default is no condition
            label_col: str: column name with the labels
            ML_qual_col: str: column name with the ML_qual values
        """
        base_snv_rate = self.df_mrd_simulation.loc["no_filter", RESIDUAL_SNV_RATE]
        base_recall = self.df_mrd_simulation.loc["no_filter", TP_READ_RETENTION_RATIO]
        data_df = self.data_df
        snvqs, recalls = self._get_snvq_and_recall(
            data_df[label_col],
            data_df[ml_qual_col],
            base_snvq=base_snv_rate,
            base_recall=base_recall,
            condition=condition,
        )
        recall_at_snvq = np.interp(snvq, snvqs, recalls)
        return recall_at_snvq

    @exception_handler
    def calc_run_info_table(self):
        """Calculate run_info_table, a table with general run information."""
        # Generate Run Info table
        logger.info("Generating Run Info table")
        TP_mixed_percent = (self.data_df[IS_MIXED] & self.data_df[LABEL]).sum() / (self.data_df[LABEL].sum())  # noqa: N806
        FP_mixed_percent = (self.data_df[IS_MIXED] & ~self.data_df[LABEL]).sum() / ((~self.data_df[LABEL]).sum())  # noqa: N806
        general_info = {
            ("Sample name", ""): self.base_name[:-1],
            ("Median training read length", ""): np.median(self.data_df["X_LENGTH"]),
            ("Median training coverage", ""): np.median(self.data_df["X_READ_COUNT"]),
            ("Training set, % TP reads", ""): signif(self.data_df[LABEL].mean() * 100, 3),
            (
                "Mixed training reads",
                "% of TP",
            ): f"{signif(100*TP_mixed_percent, 3)}%",
            (
                "Mixed training reads",
                "% of FP",
            ): f"{signif(100*FP_mixed_percent, 3)}%",
        }
        # Performance info
        mixed_df = self.data_df[self.data_df[IS_MIXED]]
        mixed_start_df = self.data_df[self.data_df[IS_MIXED_START]]
        tp_df = self.data_df[self.data_df[LABEL]]
        tp_mixed_df = tp_df[tp_df[IS_MIXED]]
        tp_mixed_start_df = tp_df[tp_df[IS_MIXED_START]]
        median_qual = tp_df[QUAL].median()
        median_qual_mixed = tp_mixed_df[QUAL].median()
        median_qual_mixed_start = tp_mixed_start_df[QUAL].median()
        recall_at_0 = self._get_recall_at_snvq(snvq=0)
        recall_at_0_mixed = self._get_recall_at_snvq(snvq=0, condition=self.data_df[IS_MIXED])
        recall_at_0_mixed_start = self._get_recall_at_snvq(snvq=0, condition=self.data_df[IS_MIXED_START])
        recall_at_50 = self._get_recall_at_snvq(snvq=50)
        recall_at_50_mixed = self._get_recall_at_snvq(snvq=50, condition=self.data_df[IS_MIXED])
        recall_at_50_mixed_start = self._get_recall_at_snvq(snvq=50, condition=self.data_df[IS_MIXED_START])
        recall_at_60 = self._get_recall_at_snvq(snvq=60)
        recall_at_60_mixed = self._get_recall_at_snvq(snvq=60, condition=self.data_df[IS_MIXED])
        recall_at_60_mixed_start = self._get_recall_at_snvq(snvq=60, condition=self.data_df[IS_MIXED_START])
        roc_auc_phred = prob_to_phred(
            self._safe_roc_auc(self.data_df[LABEL], self.data_df[ML_PROB_1_TEST], name="run info total")
        )
        roc_auc_phred_mixed = prob_to_phred(
            self._safe_roc_auc(mixed_df[LABEL], mixed_df[ML_PROB_1_TEST], name="run info mixed")
        )
        roc_auc_phred_mixed_start = prob_to_phred(
            self._safe_roc_auc(mixed_start_df[LABEL], mixed_start_df[ML_PROB_1_TEST], name="run info mixed")
        )
        performance_info = {
            ("Median SNVQ", "All reads"): signif(median_qual, 3),
            ("Median SNVQ", "Mixed, start"): signif(median_qual_mixed_start, 3),
            ("Median SNVQ", "Mixed, both ends"): signif(median_qual_mixed, 3),
            ("Recall at SNVQ=50", "All reads"): signif(recall_at_50 / recall_at_0, 3),
            ("Recall at SNVQ=50", "Mixed, start"): signif(recall_at_50_mixed_start / recall_at_0_mixed_start, 3),
            ("Recall at SNVQ=50", "Mixed, both ends"): signif(recall_at_50_mixed / recall_at_0_mixed, 3),
            ("Recall at SNVQ=60", "All reads"): signif(recall_at_60 / recall_at_0, 3),
            ("Recall at SNVQ=60", "Mixed, start"): signif(recall_at_60_mixed_start / recall_at_0_mixed_start, 3),
            ("Recall at SNVQ=60", "Mixed, both ends"): signif(recall_at_60_mixed / recall_at_0_mixed, 3),
            ("Pre-filter Recall", "All reads"): signif(recall_at_0, 3),
            ("Pre-filter Recall", "Mixed, start"): signif(recall_at_0_mixed_start, 3),
            ("Pre-filter Recall", "Mixed, both ends"): signif(recall_at_0_mixed, 3),
            ("ROC AUC (Phred)", "All reads"): signif(roc_auc_phred, 3),
            ("ROC AUC (Phred)", "Mixed, start"): signif(roc_auc_phred_mixed_start, 3),
            ("ROC AUC (Phred)", "Mixed, both ends"): signif(roc_auc_phred_mixed, 3),
        }
        # Info about versions
        version_info = {
            ("Pipeline version", ""): (self.params["pipeline_version"]),
            ("Docker image", ""): self.params["docker_image"],
            ("Adapter version", ""): self.params["adapter_version"],
            ("Report created on", ""): datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Training info
        training_info = {
            ("Pre-filter", ""): self.params["pre_filter"],
            ("Columns for balancing", ""): self.params["balanced_sampling_info_fields"],
            ("Number of CV folds", ""): self.params["num_CV_folds"],
        }
        # Info about training set size
        if self.params["num_CV_folds"] >= 2:  # noqa: PLR2004
            dataset_sizes = {
                ("dataset size", f"fold {f}"): (self.data_df["fold_id"] == f).sum()
                for f in range(self.params["num_CV_folds"])
            }
            dataset_sizes[("dataset size", "test only")] = (self.data_df["fold_id"].isna()).sum()
            other_fold_ids = np.logical_and(
                ~self.data_df["fold_id"].isin(np.arange(self.params["num_CV_folds"])), ~self.data_df["fold_id"].isna()
            ).sum()
            if other_fold_ids > 0:
                dataset_sizes[("dataset size", "other")] = other_fold_ids
        else:  # Train/test split
            dataset_sizes = {
                ("dataset size", "train"): (self.data_df["fold_id"] == -1).sum(),
                ("dataset size", "test"): (self.data_df["fold_id"] == 0).sum(),
            }
            other_fold_ids = ~self.data_df["fold_id"].isin([-1, 0]).sum()
            if other_fold_ids > 0:
                dataset_sizes[("dataset size", "other")] = other_fold_ids
        run_info_table = pd.Series({**general_info, **version_info}, name="")
        run_quality_summary_table = pd.Series({**performance_info}, name="")
        training_info_table = pd.Series({**training_info, **dataset_sizes}, name="")
        run_info_table.to_hdf(self.output_h5_filename, key="run_info_table", mode="a")
        run_quality_summary_table.to_hdf(self.output_h5_filename, key="run_quality_summary_table", mode="a")
        training_info_table.to_hdf(self.output_h5_filename, key="training_info_table", mode="a")

    @exception_handler
    def plot_interpolating_function_with_histograms(self, output_filename: str = None):
        """Plot the interpolating function with histograms of the data.
        Arguments:
            output_filename: str: path to output file
        """
        logger.info("Generating plot of ML_qual -> SNVQ link function (with histograms)")
        ml_qual_max_int = np.floor(self.data_df[ML_QUAL_1_TEST].max())
        x_ml_qual = np.arange(1e-10, ml_qual_max_int, 0.1)

        xs = self.ML_qual_to_qual_fn((x_ml_qual[1:] + x_ml_qual[:-1]) / 2).reshape(
            -1,
        )
        ys = 0.1 / (self.ML_qual_to_qual_fn(x_ml_qual)[1:] - self.ML_qual_to_qual_fn(x_ml_qual)[:-1]).reshape(
            -1,
        )

        plot_df = self.data_df[self.data_df[LABEL]]

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
            hue=IS_MIXED,
            hue_order=[False, True],
            element="step",
            stat="density",
            common_norm=False,
            linewidth=1,
            ax=ax_marg_x,
            palette={False: "red", True: "green"},
        )
        ax_marg_x.grid(visible=True)
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
            hue=IS_MIXED,
            hue_order=[False, True],
            element="step",
            stat="density",
            common_norm=False,
            linewidth=1,
            ax=ax_marg_y,
            palette={False: "red", True: "green"},
        )
        ax_marg_y.grid(visible=True)
        ax_marg_y.set_ylabel("")
        ax_marg_y.get_legend().set_visible(False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # Adjust the layout to ensure no space between the plots
        plt.tight_layout()
        gs.update(hspace=0.0, wspace=0.0)
        self._save_plt(output_filename, fig)
        plt.show()

    def _safe_roc_auc(self, y_true, y_pred, name=None):
        """
        Calculate ROC AUC score with checks for cases where the calculation is not possible.

        Parameters:
        y_true: array-like, true labels
        y_pred: array-like, predicted probabilities
        name: str, name of dataset (for logging). If None, ignore datset name.

        Returns:
        ROC AUC score if calculable, otherwise np.nan
        """
        y_true = np.array(y_true)  # Convert to numpy array
        name_for_log = f" ({name=})" if name is not None else ""
        # Check if the array is empty
        if y_true.size == 0:
            logger.warning(f"ROC AUC cannot be calculated: dataset is empty {name_for_log}")
            return np.nan

        # Check if all labels are the same
        if len(np.unique(y_true)) == 1:
            logger.warning(f"ROC AUC cannot be calculated: dataset has only one class {name_for_log}")
            return np.nan

        # Calculate the ROC AUC score
        try:
            return roc_auc_score(y_true, y_pred)
        except Exception as e:
            logger.error(f"An error occurred while calculating ROC AUC{name_for_log}: {e}")
            return np.nan

    @exception_handler
    def calc_roc_auc_table(self, holdout_fold_size_thresh: int = 100):  # noqa: C901, PLR0915 #TODO: Refactor
        """Calculte statistics of ROC AUC
        Arguments:
            holdout_fold_size_thresh: int: holdout set (fold_id is NaN) must be at least
                this size to calculate statistics on it
        """
        logger.info("Generating ROC AUC table")

        num_cv_folds = self.params["num_CV_folds"]
        auc_total = prob_to_phred(self._safe_roc_auc(self.data_df[LABEL], self.data_df[ML_PROB_1_TEST], name="total"))
        mix_cond = self.data_df[IS_MIXED]
        auc_mixed = prob_to_phred(
            self._safe_roc_auc(
                self.data_df.loc[mix_cond, LABEL], self.data_df.loc[mix_cond, ML_PROB_1_TEST], name="mixed"
            )
        )
        auc_non_mixed = prob_to_phred(
            self._safe_roc_auc(
                self.data_df.loc[~mix_cond, LABEL], self.data_df.loc[~mix_cond, ML_PROB_1_TEST], name="non-mixed"
            )
        )

        auc_per_fold = []
        auc_per_fold_mixed = []
        auc_per_fold_non_mixed = []
        auc_on_holdout = []
        auc_on_holdout_mixed = []
        auc_on_holdout_non_mixed = []
        error_on_holdout = []  # log cases of ROC AUC error on holdout,. Values in ["total", "mixed", "nonmixed"]
        holdout_fold_cond = self.data_df[FOLD_ID].isna()
        if holdout_fold_cond.sum() > holdout_fold_size_thresh:
            mix_holdout_fold_cond = self.data_df[FOLD_ID].isna() & (self.data_df[IS_MIXED])
            nonmix_holdout_fold_cond = self.data_df[FOLD_ID].isna() & (~self.data_df[IS_MIXED])
        all_features = self.params["numerical_features"] + self.params["categorical_features_names"]
        for k in range(num_cv_folds):
            fold_cond = self.data_df[FOLD_ID] == k
            mix_fold_cond = (self.data_df[FOLD_ID] == k) & (self.data_df[IS_MIXED])
            nonmix_fold_cond = (self.data_df[FOLD_ID] == k) & (~self.data_df[IS_MIXED])
            auc_per_fold.append(
                prob_to_phred(
                    self._safe_roc_auc(
                        self.data_df.loc[fold_cond, LABEL],
                        self.data_df.loc[fold_cond, "ML_prob_1_test"],
                        name=f"fold {k} total",
                    )
                )
            )
            auc_per_fold_mixed.append(
                prob_to_phred(
                    self._safe_roc_auc(
                        self.data_df.loc[mix_fold_cond, LABEL],
                        self.data_df.loc[mix_fold_cond, "ML_prob_1_test"],
                        name=f"fold {k} mixed",
                    )
                )
            )
            auc_per_fold_non_mixed.append(
                prob_to_phred(
                    self._safe_roc_auc(
                        self.data_df.loc[nonmix_fold_cond, LABEL],
                        self.data_df.loc[nonmix_fold_cond, "ML_prob_1_test"],
                        name=f"fold {k} non-mixed",
                    )
                )
            )
            # Calculate ROC AUC on holdout set
            if holdout_fold_cond.sum() > holdout_fold_size_thresh:
                if "total" not in error_on_holdout:  # Checked to supress multiple error messages
                    preds = self.models[k].predict_proba(self.data_df.loc[holdout_fold_cond, all_features])
                    preds = preds[:, 1] if preds.shape[1] == 2 else preds  # noqa: PLR2004
                    auc_on_holdout.append(
                        prob_to_phred(
                            self._safe_roc_auc(
                                self.data_df.loc[holdout_fold_cond, LABEL],
                                preds,
                                name="holdout total",
                            )
                        )
                    )
                    if np.isnan(auc_on_holdout[-1]):
                        error_on_holdout.append("total")
                else:
                    auc_on_holdout.append(np.nan)
                if "mixed" not in error_on_holdout:
                    preds = self.models[k].predict_proba(self.data_df.loc[mix_holdout_fold_cond, all_features])
                    preds = preds[:, 1] if preds.shape[1] == 2 else preds  # noqa: PLR2004
                    auc_on_holdout_mixed.append(
                        prob_to_phred(
                            self._safe_roc_auc(
                                self.data_df.loc[mix_holdout_fold_cond, LABEL],
                                preds,
                                name="holdout mixed",
                            )
                        )
                    )
                    if np.isnan(auc_on_holdout_mixed[-1]):
                        error_on_holdout.append("mixed")
                else:
                    auc_on_holdout_mixed.append(np.nan)
                if "nonmixed" not in error_on_holdout:
                    preds = self.models[k].predict_proba(self.data_df.loc[nonmix_holdout_fold_cond, all_features])
                    preds = preds[:, 1] if preds.shape[1] == 2 else preds  # noqa: PLR2004
                    auc_on_holdout_non_mixed.append(
                        prob_to_phred(
                            self._safe_roc_auc(
                                self.data_df.loc[nonmix_holdout_fold_cond, LABEL],
                                preds,
                                name="holdout non-mixed",
                            )
                        )
                    )
                    if np.isnan(auc_on_holdout_non_mixed[-1]):
                        error_on_holdout.append("nonmixed")

        auc_table_dict = {
            "ROC AUC": [auc_total, auc_mixed, auc_non_mixed],
            "ROC AUC per fold mean": [
                np.array(auc_per_fold).mean(),
                np.array(auc_per_fold_mixed).mean(),
                np.array(auc_per_fold_non_mixed).mean(),
            ],
            "ROC AUC per fold std": [
                np.array(auc_per_fold).std(),
                np.array(auc_per_fold_mixed).std(),
                np.array(auc_per_fold_non_mixed).std(),
            ],
        }

        if holdout_fold_cond.sum() > holdout_fold_size_thresh:
            auc_table_dict["ROC AUC on holdout mean"] = [
                np.array(auc_on_holdout).mean(),
                np.array(auc_on_holdout_mixed).mean(),
                np.array(auc_on_holdout_non_mixed).mean(),
            ]
            auc_table_dict["ROC AUC on holdout std"] = [
                np.array(auc_on_holdout).std(),
                np.array(auc_on_holdout_mixed).std(),
                np.array(auc_on_holdout_non_mixed).std(),
            ]

        pd.DataFrame(auc_table_dict, index=["Total", "Mixed only", "Non-mixed only"]).T.to_hdf(
            self.output_h5_filename, key="roc_auc_table", mode="a"
        )

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
            cols_for_stats = {"qual": "qual", "ML_qual_1_test": "ML_qual", "ML_logit_test": "ML_logit"}
        if display_columns is None:
            display_columns = [
                ("qual", "FP", ""),
                ("qual", "TP", "overall"),
                ("qual", "TP", "mixed"),
                ("qual", "TP", "non-mixed"),
                ("ML_qual", "FP", ""),
                ("ML_qual", "TP", "overall"),
                ("ML_qual", "TP", "mixed"),
                ("ML_qual", "TP", "non-mixed"),
            ]

        logger.info("Generating Run Quality table")
        conds_dict = {
            "": np.ones(self.data_df["label"].shape, dtype=bool),
            "_TP": self.data_df["label"],
            "_FP": ~self.data_df["label"],
            "_TP_mixed": np.logical_and(self.data_df["is_mixed"], self.data_df["label"]),
            "_TP_non-mixed": np.logical_and(~self.data_df["is_mixed"], self.data_df["label"]),
        }
        qual_stats_description = {
            key: self.data_df.loc[cond, list(cols_for_stats.keys())]
            .describe(percentiles=qual_stat_ps)
            .rename(columns={stat_key: stat_name + key for stat_key, stat_name in cols_for_stats.items()})
            for key, cond in conds_dict.items()
        }

        qual_stats_description = pd.concat(list(qual_stats_description.values()), axis=1)
        run_quality_table = pd.DataFrame(
            signif(qual_stats_description.values, 3),
            index=qual_stats_description.index,
            columns=qual_stats_description.columns,
        ).drop(index="count")

        col_order = [
            "_".join(cols) if cols[2] not in ["", "overall"] else "_".join(cols[:2]) for cols in display_columns
        ]
        run_quality_table_display = run_quality_table.loc[:, col_order].copy()
        run_quality_table_display.columns = pd.MultiIndex.from_tuples(display_columns)
        # Log in hdf5
        run_quality_table.T.to_hdf(self.output_h5_filename, key="run_quality_table", mode="a")
        run_quality_table_display.to_hdf(self.output_h5_filename, key="run_quality_table_display", mode="a")

    @exception_handler
    def quality_per_ppmseq_tags(self, output_filename: str = None):
        """Generate tables of median quality and data quantity per start and end ppmseq tags."""
        data_df_tp = self.data_df[self.data_df["label"]].copy()
        ppmseq_tags_in_data = self.start_tag_col is not None and self.end_tag_col is not None
        start_tag_col, end_tag_col = (self.start_tag_col, self.end_tag_col) if ppmseq_tags_in_data else ("st", "et")
        if not ppmseq_tags_in_data:
            data_df_tp[start_tag_col] = np.nan
            data_df_tp[end_tag_col] = np.nan
        if data_df_tp[start_tag_col].isna().any() or data_df_tp[end_tag_col].isna().any():
            data_df_tp = data_df_tp.astype({start_tag_col: str, end_tag_col: str})
        ppmseq_category_quality_table = (
            data_df_tp.groupby([start_tag_col, end_tag_col], dropna=False)["qual"].median().unstack()  # noqa PD010
        )
        if PpmseqCategories.END_UNREACHED.value in ppmseq_category_quality_table.index:
            ppmseq_category_quality_table = ppmseq_category_quality_table.drop(
                index=PpmseqCategories.END_UNREACHED.value
            )
        ppmseq_category_quantity_table = (
            data_df_tp.groupby([start_tag_col, end_tag_col], dropna=False)["qual"].count().unstack()  # noqa PD010
        )
        if PpmseqCategories.END_UNREACHED.value in ppmseq_category_quantity_table.index:
            ppmseq_category_quantity_table = ppmseq_category_quantity_table.drop(
                index=PpmseqCategories.END_UNREACHED.value
            )
        ppmseq_category_quantity_table = (
            ppmseq_category_quantity_table / ppmseq_category_quantity_table.to_numpy().sum()
        ) * 100
        # Convert index and columns from categorical to string
        ppmseq_category_quality_table.index = ppmseq_category_quality_table.index.astype(str)
        ppmseq_category_quality_table.columns = ppmseq_category_quality_table.columns.astype(str)
        ppmseq_category_quantity_table.index = ppmseq_category_quantity_table.index.astype(str)
        ppmseq_category_quantity_table.columns = ppmseq_category_quantity_table.columns.astype(str)

        # Save to hdf5
        ppmseq_category_quality_table_for_h5 = ppmseq_category_quality_table.T.unstack(level=0)  # noqa PD010
        ppmseq_category_quantity_table_for_h5 = ppmseq_category_quantity_table.T.unstack(level=0)  # noqa PD010
        ppmseq_category_quality_table_for_h5.to_hdf(
            self.output_h5_filename, key="ppmseq_category_quality_table", mode="a"
        )
        ppmseq_category_quantity_table_for_h5.to_hdf(
            self.output_h5_filename, key="ppmseq_category_quantity_table", mode="a"
        )
        # Generate heatmap
        ppmseq_category_combined_table = (
            ppmseq_category_quality_table.apply(lambda x: signif(x, 3)).astype(str)
            + "\n["
            + ppmseq_category_quantity_table.apply(lambda x: signif(x, 2)).astype(str)
            + "%]"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            ppmseq_category_quality_table,
            annot=ppmseq_category_combined_table,
            fmt="",
            cmap="inferno",
            cbar=False,
            linewidths=1,
            linecolor="black",
            annot_kws={"size": 12},
            square=True,
            ax=ax,
        )

        # Customization to make it more readable
        plt.yticks(rotation=0, fontsize=12)
        plt.xticks(rotation=45, fontsize=12)
        plt.xlabel("ppmSeq tag end", fontsize=14)
        plt.ylabel("ppmSeq tag start", fontsize=14)
        self._save_plt(output_filename, fig=fig)

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
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        train_logloss = list_of_jagged_lists_to_array(
            [result["validation_0"]["mlogloss"] for result in training_results]
        )
        val_logloss = list_of_jagged_lists_to_array([result["validation_1"]["mlogloss"] for result in training_results])
        for ri, result in enumerate(training_results):
            label = "Individual folds" if ri == 1 else None  # will reach ri==1 iff when using CV
            ax.plot(result["validation_0"]["mlogloss"], c="grey", alpha=0.7, label=label)
            ax.plot(result["validation_1"]["mlogloss"], c="grey", alpha=0.7)

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

    def _X_to_display(self, X_val: pd.DataFrame, cat_features_dict: dict = None):  # noqa: N802, N803
        """Rename categorical feature values as numbers, for SHAP plots."""
        if cat_features_dict is None:
            cat_features_dict = self.params["categorical_features_dict"]
        X_val_display = X_val.copy()  # noqa: N806
        for col, cat_vals in cat_features_dict.items():
            X_val_display[col] = X_val_display[col].map({cv: i for i, cv in enumerate(cat_vals)}).astype(int)

        return X_val_display

    def _shap_on_sample(self, model, data_df, features=None, label_col="label", n_sample=10_000):
        """Calculate shap value on a sample of the data from data_df."""
        if features is None:
            features = self.params["numerical_features"] + self.params["categorical_features_names"]
        if not hasattr(model, "best_ntree_limit"):
            model.best_ntree_limit = model.best_iteration + 1
        X_val = data_df[features].sample(n=n_sample, random_state=self.rng)  # noqa: N806
        y_val = data_df.loc[X_val.index, label_col]
        X_val_dm = xgb.DMatrix(data=X_val, label=y_val, enable_categorical=True)  # noqa: N806
        shap_values = model.get_booster().predict(
            X_val_dm, pred_contribs=True, iteration_range=(0, model.best_ntree_limit)
        )
        return shap_values, X_val, y_val

    def plot_SHAP_feature_importance(  # noqa: N802
        self,
        shap_values: np.ndarray,
        X_val: pd.DataFrame,  # noqa: N803
        output_filename: str = None,
        n_features: int = 15,
        xlims=None,
    ):
        """Plot a SHAP feature importance plot."""
        set_pyplot_defaults()

        fig, ax = plt.subplots(figsize=(20, 10))
        X_val_plot = self._X_to_display(X_val)  # noqa: N806

        base_values_plot = shap_values[0, 1, -1] - shap_values[0, 0, -1]
        shap_values_plot = shap_values[:, 1, :-1] - shap_values[:, 0, :-1]
        explanation = shap.Explanation(
            values=shap_values_plot, base_values=base_values_plot, feature_names=X_val_plot.columns, data=X_val_plot
        )
        plt.sca(ax)
        shap.plots.bar(
            explanation,
            max_display=n_features,
            show=False,
        )
        ax.set_xlabel("")
        ax.set_xlim(xlims)
        ticklabels = ax.get_ymajorticklabels()
        ax.set_yticklabels(ticklabels)
        ax.grid(visible=True, axis="x", linestyle=":", linewidth=1)
        # ax.grid(visible=False, axis="y")
        self._save_plt(output_filename=output_filename, fig=fig)

    def plot_SHAP_beeswarm(  # noqa: N802
        self,
        shap_values: np.ndarray,
        X_val: pd.DataFrame,  # noqa: N803
        output_filename: str = None,
        n_features: int = 10,
        nplot_sample: int = None,
        cmap="brg",
        xlims=None,
    ):
        """Plot a SHAP beeswarm plot."""
        set_pyplot_defaults()

        # Prepare data df
        grouped_features = self._group_categorical_features(self.params["categorical_features_dict"])
        X_val_plot = self._X_to_display(X_val)  # noqa: N806
        X_val_plot = X_val_plot.rename(  # noqa: N806
            columns={
                feature: f"{feature} [{group_idx}]"
                for group_idx, features in enumerate(grouped_features.values(), start=1)
                for feature in features
            }
        )
        # Get SHAP data
        top_features = np.abs(shap_values[:, 1, :-1] - shap_values[:, 0, :-1]).mean(axis=0).argsort()[::-1][:n_features]
        inds_for_plot = np.arange(X_val_plot.shape[0])
        if nplot_sample is not None:
            if nplot_sample < X_val_plot.shape[0]:
                inds_for_plot = self.rng.choice(X_val_plot.shape[0], nplot_sample, replace=False)
        base_values_plot = shap_values[0, 1, -1] - shap_values[0, 0, -1]
        shap_values_plot = (
            shap_values[inds_for_plot.reshape((-1, 1)), 1, top_features]
            - shap_values[inds_for_plot.reshape((-1, 1)), 0, top_features]
        )
        explanation = shap.Explanation(
            values=shap_values_plot,
            base_values=base_values_plot,
            # feature_names=X_val_display.columns[top_features],
            data=X_val_plot.iloc[
                inds_for_plot, top_features
            ],  # data=X_val_plot_display.iloc[inds_for_plot,top_features]
        )
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 10))
        if xlims is None:
            xlims = [
                np.floor(shap_values_plot.min() - base_values_plot),
                np.ceil(shap_values_plot.max() - base_values_plot),
            ]
        plt.sca(ax)
        shap.plots.beeswarm(
            explanation,
            color=plt.get_cmap(cmap),
            max_display=30,
            alpha=0.2,
            show=False,
            plot_size=0.4,
            color_bar=False,  # (k==2)
        )
        ax.set_xlabel("SHAP value", fontsize=12)
        ax.set_xlim(xlims)
        fig.tight_layout()

        self._add_colorbars(fig, grouped_features)

        self._save_plt(output_filename=output_filename, fig=fig)

    def _group_categorical_features(self, cat_dict):
        """Group all categorical features by the category values
        (e.g, boolean features, nucleotide features, etc.)
        """
        # Create a defaultdict to hold lists of feature names for each tuple of values
        grouped_features = defaultdict(list)

        for feature, values in cat_dict.items():
            # Convert the list of values to a tuple (so it can be used as a dictionary key)
            value_tuple = tuple(values)
            # Append the feature name to the list corresponding to this tuple of values
            grouped_features[value_tuple].append(feature)

        # Convert the defaultdict to a regular dict before returning
        return dict(grouped_features)

    def _add_colorbars(
        self,
        fig,
        grouped_features,
        cmap="brg",
        total_width: float = 0.3,  # Total width of all colorbars
        vertical_padding: float = 0.2,  # Padding between colorbars
        start_y=0.9,  # Start near the top of the figure
        rotated_padding_factor: float = 8.0,  # controls how much extra padding when ticklabels are rotated
    ):
        """Add colorbars to SHAP beeswarm plot"""
        group_labels = []
        for i, (cat_values, features) in enumerate(grouped_features.items()):
            n_cat = len(cat_values)
            cbar_length = total_width  # * (n_cat / max_ncat)
            max_cat_val_length = max(len(f"{val}") for val in cat_values)
            rotation = 0 if max_cat_val_length <= 5 else 1  # rotate the labels if they are too long  # noqa: PLR2004
            ha = "center" if rotation == 0 else "right"  # horizontal alignment for rotated labels

            group_label = f"[{i + 1}]"
            group_labels.append(f"{group_label}: {', '.join(features)}")

            # Calculate the rectangle for the colorbar axis
            cbar_ax = fig.add_axes([1.1, start_y - 0.03, cbar_length, 0.03])

            # Create the colorbar
            fig_cbar = fig.colorbar(
                cm.ScalarMappable(norm=None, cmap=plt.get_cmap(cmap, n_cat)), cax=cbar_ax, orientation="horizontal"
            )
            fig_cbar.set_label(group_label, fontsize=12)  # , labelpad=-50*(1+1.5*rotation), loc='center')
            fig_cbar.set_ticks(np.arange(0, 1, 1 / n_cat) + 1 / (2 * n_cat))
            fig_cbar.set_ticklabels(cat_values, fontsize=12, rotation=rotation * 25, ha=ha)
            fig_cbar.outline.set_visible(False)

            # Update the start position for the next colorbar
            start_y -= vertical_padding + rotation / (
                rotated_padding_factor
            )  # Adjust this value to control vertical spacing between colorbars

        # Add an extra colorbar for numerical features below the others
        cbar_num_ax = fig.add_axes([1.1, start_y - 0.03, total_width, 0.03])
        fig_num_cbar = fig.colorbar(
            cm.ScalarMappable(norm=None, cmap=plt.get_cmap(cmap)), cax=cbar_num_ax, orientation="horizontal"
        )
        fig_num_cbar.set_label("Numerical features", fontsize=12)
        fig_num_cbar.set_ticks([0, 1])
        fig_num_cbar.set_ticklabels(["Low value", "High value"], fontsize=12)
        fig_num_cbar.outline.set_visible(False)

    @exception_handler
    def calc_and_plot_shap_values(
        self,
        output_filename_importance: str = None,
        output_filename_beeswarm: str = None,
        n_sample: int = 10_000,
        feature_importance_kws: dict = None,
        beeswarm_kws: dict = None,
        *,
        plot_feature_importance: bool = True,
        plot_beeswarm: bool = True,
    ):
        """Calculate and plot SHAP values for the model."""
        feature_importance_kws = feature_importance_kws or {}
        beeswarm_kws = beeswarm_kws or {}
        # Define model, data
        k = 0  # fold_id
        model = self.models[k]
        X_val = self.data_df[self.data_df["fold_id"] == k]  # noqa: N806
        # Get SHAP values
        logger.info("Calculating SHAP values")
        shap_values, X_val, _ = self._shap_on_sample(  # noqa: N806
            model, X_val, n_sample=n_sample
        )
        logger.info("Done calculating SHAP values")
        # SHAP feature importance
        mean_abs_SHAP_scores = pd.Series(  # noqa: N806
            np.abs(shap_values[:, 1, :-1] - shap_values[:, 0, :-1]).mean(axis=0), index=X_val.columns
        ).sort_values(ascending=False)
        mean_abs_SHAP_scores.to_hdf(self.output_h5_filename, key="mean_abs_SHAP_scores", mode="a")
        # Plot shap scores
        if plot_feature_importance:
            self.plot_SHAP_feature_importance(
                shap_values, X_val, output_filename=output_filename_importance, **feature_importance_kws
            )
        if plot_beeswarm:
            self.plot_SHAP_beeswarm(shap_values, X_val, output_filename=output_filename_beeswarm, **beeswarm_kws)

    def _get_trinuc_stats(self, q1: float = 0.1, q2: float = 0.9):
        data_df = self.data_df.copy()
        data_df["is_cycle_skip"] = data_df["is_cycle_skip"].astype(int)
        trinuc_stats = data_df.groupby(["trinuc_context_with_alt", "label", "is_forward", "is_mixed"]).agg(
            median_qual=("qual", "median"),
            quantile1_qual=("qual", lambda x: x.quantile(q1)),
            quantile3_qual=("qual", lambda x: x.quantile(q2)),
            is_cycle_skip=("is_cycle_skip", "mean"),
            count=("qual", "size"),
        )
        trinuc_stats["fraction"] = trinuc_stats["count"] / self.data_df.shape[0]
        trinuc_stats = trinuc_stats.reset_index()
        trinuc_stats["is_forward"] = trinuc_stats["is_forward"].astype(bool)
        return trinuc_stats

    def _get_trinuc_with_alt_in_order(self, order: str = "symmetric"):
        """Get trinuc_with_context in right order, so that each SNV in position i
        has the opposite SNV in position i+96.
        E.g., in position 0 we have A[A>C]A and in posiiton 96 we have A[C>A]A
        """
        if order not in {"symmetric", "reverse"}:
            raise ValueError(f'order must be either "symmetric" or "reverse". Got {order}')
        if order == "symmetric":
            trinuc_ref_alt = [
                c1 + r + c2 + a
                for r, a in (("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"))
                for c1 in ("A", "C", "G", "T")
                for c2 in ("A", "C", "G", "T")
                if r != a
            ] + [
                c1 + r + c2 + a
                for a, r in (("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"))
                for c1 in ("A", "C", "G", "T")
                for c2 in ("A", "C", "G", "T")
                if r != a
            ]
        elif order == "reverse":
            trinuc_ref_alt = [
                c1 + r + c2 + a
                for r, a in (("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"))
                for c1 in ("A", "C", "G", "T")
                for c2 in ("A", "C", "G", "T")
                if r != a
            ] + [
                c1 + r + c2 + a
                for r, a in (("T", "G"), ("T", "C"), ("T", "A"), ("G", "C"), ("G", "A"), ("C", "A"))
                for c2 in ("T", "G", "C", "A")
                for c1 in ("T", "G", "C", "A")
                if r != a
            ]
        trinuc_index = np.array([f"{t[0]}[{t[1]}>{t[3]}]{t[2]}" for t in trinuc_ref_alt])
        snv_labels = [" ".join(trinuc_snv[2:5]) for trinuc_snv in trinuc_index[np.arange(0, 16 * 12, 16)]]
        return trinuc_ref_alt, trinuc_index, snv_labels

    @exception_handler
    def calc_and_plot_trinuc_plot(  # noqa: PLR0915 #TODO: refactor
        self,
        output_filename: str = None,
        order: str = "symmetric",
        *,
        filter_on_is_forward: bool = True,  # Filter out reverse trinucs
    ):
        logger.info("Calculating trinuc context statistics")
        trinuc_stats = self._get_trinuc_stats(q1=0.1, q2=0.9)
        trinuc_stats.set_index(["trinuc_context_with_alt", "label", "is_forward", "is_mixed"]).to_hdf(
            self.output_h5_filename, key="trinuc_stats", mode="a"
        )
        # get trinuc_with_context in right order
        trinuc_symmetric_ref_alt, symmetric_index, snv_labels = self._get_trinuc_with_alt_in_order(order=order)
        snv_positions = [8, 24, 40, 56, 72, 88]  # Midpoint for each SNV titles in plot
        trinuc_is_cycle_skip = (
            trinuc_stats.groupby("trinuc_context_with_alt")["is_cycle_skip"]
            .mean()
            .astype(bool)
            .loc[trinuc_symmetric_ref_alt]
            .to_numpy()
        )

        # Configurable boolean column name
        boolean_column_frac = "label"  # Change this to any other boolean column name as needed
        boolean_column_qual = "is_mixed"  # Change this to any other boolean column name as needed
        cond_for_frac_plot = True
        cond_for_qual_plot = trinuc_stats["label"]
        if filter_on_is_forward:
            cond_for_frac_plot = trinuc_stats["is_forward"]
            cond_for_qual_plot = cond_for_qual_plot & trinuc_stats["is_forward"]

        # Plot parameters
        label_fontsize = 14
        yticks_fontsize = 12
        xticks_fontsize = 10
        fcolor_frac = "tab:blue"
        tcolor_frac = "tab:orange"
        fcolor_qual = "tab:red"
        tcolor_qual = "tab:green"
        # First Plot: Fractions
        plot_df_true_frac = (
            trinuc_stats[trinuc_stats[boolean_column_frac] & cond_for_frac_plot]
            .groupby("trinuc_context_with_alt")["fraction"]
            .sum()
            .reindex(trinuc_symmetric_ref_alt)
        )
        plot_df_false_frac = (
            trinuc_stats[~trinuc_stats[boolean_column_frac] & cond_for_frac_plot]
            .groupby("trinuc_context_with_alt")["fraction"]
            .sum()
            .reindex(trinuc_symmetric_ref_alt)
        )
        plot_df_true_frac.index = symmetric_index
        plot_df_false_frac.index = symmetric_index

        # Second Plot: Median Qual
        plot_df_true_qual = (
            trinuc_stats[(trinuc_stats[boolean_column_qual]) & (cond_for_qual_plot)]
            .groupby("trinuc_context_with_alt")
            .agg(
                median_qual=("median_qual", "mean"),
                quantile1_qual=("quantile1_qual", "mean"),
                quantile3_qual=("quantile3_qual", "mean"),
            )
            .reindex(trinuc_symmetric_ref_alt)
        )
        plot_df_false_qual = (
            trinuc_stats[(~trinuc_stats[boolean_column_qual]) & (cond_for_qual_plot)]
            .groupby("trinuc_context_with_alt")
            .agg(
                median_qual=("median_qual", "mean"),
                quantile1_qual=("quantile1_qual", "mean"),
                quantile3_qual=("quantile3_qual", "mean"),
            )
            .reindex(trinuc_symmetric_ref_alt)
        )
        plot_df_true_qual.index = symmetric_index
        plot_df_false_qual.index = symmetric_index

        # Plotting
        fig, axes = plt.subplots(2, 1, figsize=(16, 9))
        x_values = list(range(96))
        x_values_ext = (
            [x_values[0] - (x_values[1] - x_values[0]) / 2]
            + x_values
            + [x_values[-1] + (x_values[-1] - x_values[-2]) / 2]
        )
        # for legend
        handles_frac, labels_frac = [], []
        handles_qual, labels_qual = [], []

        for i in range(2):  # Loop through both subplots (original SNV and reverse SNV)
            inds = np.array(x_values) + 96 * i
            inds_ext = [inds[0]] + list(inds) + [inds[-1]]
            ax = axes[i]
            ax2 = ax.twinx()  # Create a secondary y-axis

            # First Plot (Fractions) on the primary y-axis
            ylim = 1.05 * max(plot_df_true_frac.max(), plot_df_false_frac.max())
            for j in range(5):
                ax.plot([(j + 1) * 16 - 0.5] * 2, [0, ylim], "k--")
            bars_false = plot_df_false_frac.iloc[inds].plot.bar(
                ax=ax, color=fcolor_frac, width=1.0, alpha=0.5, legend=False
            )
            bars_true = plot_df_true_frac.iloc[inds].plot.bar(
                ax=ax, color=tcolor_frac, width=1.0, alpha=0.5, legend=False
            )
            x_tick_labels = symmetric_index[inds]
            ax_is_cycle_skip = trinuc_is_cycle_skip[inds]
            for j in range(len(x_tick_labels)):
                ax.get_xticklabels()[j].set_color("green" if ax_is_cycle_skip[j] else "red")
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_tick_labels, rotation=90, fontsize=xticks_fontsize)
            ax.tick_params(axis="x", pad=-2)

            ax.set_xlim(-1, 96)
            ax.set_ylim(0, ylim)
            ax.set_ylabel("Fraction", fontsize=label_fontsize)
            ax.tick_params(axis="y", labelsize=yticks_fontsize)  # Set y-axis tick label size
            ax.grid(visible=True, axis="both", alpha=0.75, linestyle=":")

            # Add labels for each ref>alt pair
            for label, pos in zip(snv_labels[6 * i : 6 * i + 6], snv_positions, strict=False):
                ax.annotate(
                    label,
                    xy=(pos, ylim),  # Position at the top of the plot
                    xytext=(-2, 6),  # Offset from the top of the plot
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )

            # Second Plot (Median Qual) on the secondary y-axis
            ylims_qual = [
                0 * np.nanmin([plot_df_true_qual.to_numpy().min(), plot_df_false_qual.to_numpy().min()]),
                1.05 * np.nanmax([plot_df_true_qual.to_numpy().max(), plot_df_false_qual.to_numpy().max()]),
            ]
            (line_false,) = ax2.step(
                x_values_ext,
                plot_df_false_qual.iloc[inds_ext, 0],
                where="mid",
                color=fcolor_qual,
                alpha=0.7,
                label=f"{boolean_column_qual} = False",
            )
            (line_true,) = ax2.step(
                x_values_ext,
                plot_df_true_qual.iloc[inds_ext, 0],
                where="mid",
                color=tcolor_qual,
                alpha=0.7,
                label=f"{boolean_column_qual} = True",
            )
            ax2.fill_between(
                x_values_ext,
                plot_df_false_qual.iloc[inds_ext, 1],
                plot_df_false_qual.iloc[inds_ext, 2],
                step="mid",
                color=fcolor_qual,
                alpha=0.2,
            )
            ax2.fill_between(
                x_values_ext,
                plot_df_true_qual.iloc[inds_ext, 1],
                plot_df_true_qual.iloc[inds_ext, 2],
                step="mid",
                color=tcolor_qual,
                alpha=0.2,
            )
            ax2.set_ylim(ylims_qual)
            ax2.set_ylabel("SNVQ on TP reads", fontsize=label_fontsize)
            ax2.tick_params(axis="y", labelsize=yticks_fontsize)  # Set y-axis tick label size
            ax2.grid(visible=False)

        # legend
        handles_frac.append(bars_false.patches[0])
        labels_frac.append("FP")
        handles_frac.append(bars_true.patches[96])
        labels_frac.append("TP")
        handles_qual.append(line_false)
        labels_qual.append("Non-mixed reads")
        handles_qual.append(line_true)
        labels_qual.append("Mixed reads")

        fig.tight_layout(rect=[0, 0.1, 1, 1])

        plt.subplots_adjust(bottom=0.2)
        # Create a custom legend
        fig.legend(
            handles_frac,
            labels_frac,
            title="Fraction (left axis)",
            fontsize=14,
            title_fontsize=14,
            loc="lower center",
            bbox_to_anchor=(0.2, -0.08),
            ncol=2,
            frameon=False,
        )
        fig.legend(
            handles_qual,
            labels_qual,
            title="SNVQ on TP reads (right axis)",
            fontsize=14,
            title_fontsize=14,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=False,
        )
        # Add explanatory text about x-tick label colors
        is_forward_text = "(fwd only read)" if filter_on_is_forward else "(fwd and rev reads)"
        fig.text(
            0.75,  # x-position (slightly to the right of the second legend)
            -0.06 + 2 * 0.025,  # y-position (adjust as necessary)
            f"Trinuc-SNV colors {is_forward_text}:",
            ha="left",  # Horizontal alignment
            fontsize=14,  # Font size
            color="black",  # Text color
        )
        fig.text(
            0.75,  # x-position (slightly to the right of the second legend)
            -0.06 + 0.025,  # y-position (adjust as necessary)
            "Green: Cycle skip",
            ha="left",  # Horizontal alignment
            fontsize=14,  # Font size
            color="green",  # Text color
        )
        fig.text(
            0.75,  # x-position (slightly to the right of the second legend)
            -0.06,  # y-position (adjust as necessary)
            "Red: No cycle skip",
            ha="left",  # Horizontal alignment
            fontsize=14,  # Font size
            color="red",  # Text color
        )
        self._save_plt(output_filename=output_filename, fig=fig)

    def _plot_feature_qual_stats(
        self, stats_for_plot, ax, c_true="tab:green", c_false="tab:red", min_count=100, fb_kws=None, step_kws=None
    ):
        """Generate a plot of quality median + 10-90 percentile range, for mixed and non-mixed reads."""
        fb_kws = fb_kws or {}
        step_kws = step_kws or {}
        col = stats_for_plot.columns[0]
        polys, lines = [], []
        for is_mixed, color in zip(
            [~stats_for_plot["is_mixed"], stats_for_plot["is_mixed"]], [c_false, c_true], strict=False
        ):
            qual_df = stats_for_plot.loc[is_mixed & stats_for_plot["label"] & (stats_for_plot["count"] > min_count), :]
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
        data_df = self.data_df.copy()
        if bin_edges is not None:
            data_df[col] = pd.cut(data_df[col], bin_edges, labels=(bin_edges[1:] + bin_edges[:-1]) / 2)
        stats_for_plot = (
            data_df.sample(frac=1)
            .groupby([col, "label", "is_mixed"])
            .agg(
                median_qual=("qual", "median"),
                quantile1_qual=("qual", lambda x: x.quantile(q1)),
                quantile3_qual=("qual", lambda x: x.quantile(q2)),
                count=("qual", "size"),
            )
        )
        stats_for_plot["fraction"] = stats_for_plot["count"] / data_df.shape[0]
        stats_for_plot = stats_for_plot.reset_index()
        return stats_for_plot

    @exception_handler
    def plot_numerical_feature_hist_and_qual(
        self, col: str, q1: float = 0.1, q2: float = 0.9, nbins: int = 50, output_filename: str = None
    ):
        """Plot histogram and quality stats for a numerical feature."""
        logger.info(f"Plotting quality and histogram for feature {col}")
        is_discrete = (self.data_df[col] - np.round(self.data_df[col])).abs().max() < 0.05  # noqa: PLR2004
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
            hue="label",
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
        # Next two lines are to display correctly the legend if there are no mixed reads
        lines = [line if line is not None else empty_handle for line in lines]
        polys = [poly if poly is not None else empty_handle for poly in polys]
        fig.legend(
            [empty_handle, lines[0], polys[0], empty_handle, lines[1], polys[1]],
            ["Non-mixed", "median", "10%-90% range", "Mixed", "median", "10%-90% range"],
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=(0.7, -0.15),
            ncol=2,
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
    def plot_quality_histogram(self, *, plot_interpolating_function: bool = False, output_filename: str = None):
        """Plot a histogram of qual values for TP reads, both mixed and non-mixed."""
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

        plot_df = self.data_df[self.data_df[LABEL]]
        plot_df["ppmSeq tags"] = "non-mixed"
        plot_df.loc[plot_df[IS_MIXED_START] ^ plot_df[IS_MIXED_END], "ppmSeq tags"] = "mixed, exactly one end"
        plot_df.loc[plot_df[IS_MIXED_START] & plot_df[IS_MIXED_END], "ppmSeq tags"] = "mixed, both ends"
        g = sns.histplot(
            data=plot_df,
            x=QUAL,
            # bins=50,
            hue="ppmSeq tags",
            hue_order=["non-mixed", "mixed, exactly one end", "mixed, both ends"],
            element="step",
            stat="density",
            common_norm=False,
            kde=True,
            kde_kws={"bw_adjust": 3},
            linewidth=1,
            ax=ax,
            palette={"non-mixed": "red", "mixed, exactly one end": "blue", "mixed, both ends": "green"},
        )
        sns.move_legend(
            ax,
            "upper left",
        )
        ax.set_ylabel("Density")  # , fontsize=label_fontsize)
        ax.grid(visible=True)
        xlims = [0.99 * plot_df[QUAL].min(), plot_df[QUAL].max() * 1.01]
        hist_data_df = self._get_histogram_data(g, col_name=IS_MIXED)
        hist_data_df.to_hdf(self.output_h5_filename, key="quality_histogram", mode="a")
        if plot_interpolating_function:
            xs = np.arange(1e-10, np.floor(self.data_df[ML_QUAL_1_TEST].max()), 0.1)
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
        plt.show()
        self._save_plt(output_filename, fig=fig)

    def _plot_logit_histogram(self, plot_df, ax, alpha=0.4):
        """Plot a single histogram of logit values, by: FP, TP mixed, TP non-mixed."""
        plot_df[""] = plot_df[LABEL].astype(str)
        plot_df.loc[~plot_df[LABEL], ""] = "FP"
        plot_df.loc[plot_df[LABEL] & plot_df[IS_MIXED], ""] = "TP mixed"
        plot_df.loc[plot_df[LABEL] & ~plot_df[IS_MIXED], ""] = "TP non-mixed"
        sns.histplot(
            data=plot_df,
            x=ML_LOGIT_TEST,
            hue="",
            hue_order=["FP", "TP mixed", "TP non-mixed"],
            palette={"FP": "tab:blue", "TP non-mixed": "red", "TP mixed": "green"},
            element="step",
            stat="density",
            common_norm=False,
            alpha=alpha,
            ax=ax,
        )

    @exception_handler
    def plot_logit_histograms(self, *, plot_by_fold: bool = True, output_filename: str = None):
        """Plot a histogram of logit values, by: FP, TP mixed, TP non-mixed.
        If plot_by_fold is True, overlay histograms for each fold.
        """
        # label_fontsize = 12
        # ticklabelsfontsize = 12
        logger.info("Plotting logit histogram")
        fig, ax = plt.subplots(figsize=(12, 7))

        plot_df = self.data_df[[ML_LOGIT_TEST, LABEL, IS_MIXED, FOLD_ID]].copy()
        self._plot_logit_histogram(plot_df, ax)
        hist_data_df = self._get_histogram_data(ax, col_name="")
        hist_data_df.to_hdf(self.output_h5_filename, key="logit_histogram", mode="a")

        if plot_by_fold:
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(12, 7))
            plot_dfs = [plot_df[plot_df[FOLD_ID] == k] for k in range(len(self.models))]
            for plot_df in plot_dfs:
                self._plot_logit_histogram(plot_df, ax, alpha=0.15)

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
        plt.show()
        self._save_plt(output_filename, fig=fig)

    def create_report(self):
        """Generate plots for report and save data in hdf5 file."""
        logger.info("Creating report")
        # Get filenames for plots
        [
            output_LoD_plot,  # noqa: N806
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

        # Create LoD plot
        # TODO: Update the following to new conform with new report logic
        if self.params["fp_regions_bed_file"] is not None:
            logger.info("Calculating LoD statistics")
            min_LoD_filter = calculate_lod_stats(  # noqa: N806
                df_mrd_simulation=self.df_mrd_simulation,
                output_h5=self.statistics_h5_file,
                lod_column=self.c_lod,
            )
            logger.info("Creating SNVQ-recall-LoD plot")
            self.df_mrd_simulation.unstack().to_hdf(self.output_h5_filename, key="SNV_recall_LoD", mode="a")  # noqa PD010
            plot_LoD(
                self.df_mrd_simulation,
                self.lod_label,
                self.c_lod,
                self.lod_filters,
                self.params["adapter_version"],
                min_LoD_filter,
                output_filename=output_LoD_plot,
            )

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
                "SNV_recall_LoD",
            ]
        )
        keys_to_convert.to_hdf(self.output_h5_filename, key="keys_to_convert", mode="a")

        # convert statistics to json
        convert_h5_to_json(
            input_h5_filename=self.statistics_h5_file,
            root_element="metrics",
            ignored_h5_key_substring=None,
            output_json=self.statistics_json_file,
        )


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
