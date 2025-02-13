from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame
from ugbio_core import stats_utils
from ugbio_core.logger import logger


def calc_accuracy_metrics(
    concordance_df: DataFrame,
    classify_column_name: str,
    ignored_filters: Iterable[str] | None = None,
    group_testing_column_name: str | None = None,
) -> DataFrame:
    """
    Parameters
    ----------
    concordance_df: DataFrame
        concordance dataframe
    classify_column_name: str
        column name which contains tp,fp,fn status before applying filter
    ignored_filters: Iterable[str]
        list of filters to ignore (the ignored filters will not be applied before calculating accuracy)
    group_testing_column_name: str
        column name to be used as grouping column (to output statistics on each group)

    Returns
    -------
    data-frame with variant types and their scores

    Raises
    ------
    RuntimeError
        if the output of get_concordance_metrics is not a DataFrame, should not happen
    """
    if ignored_filters is None:
        ignored_filters = {"PASS"}

    concordance_df = validate_preprocess_concordance(concordance_df, group_testing_column_name)
    concordance_df["vc_call"] = concordance_df["filter"].apply(
        lambda x: convert_filter2call(x, ignored_filters=set(ignored_filters) | {"PASS"})
    )

    # calc recall,precision, f1 per variant category
    if group_testing_column_name is None:
        concordance_df = add_grouping_column(concordance_df, get_selection_functions(), "group_testing")
        group_testing_column_name = "group_testing"
    accuracy_df = init_metrics_df()
    groups = list(get_selection_functions().keys())
    for g_val in groups:
        dfselect = concordance_df[concordance_df[group_testing_column_name] == g_val]
        acc = get_concordance_metrics(
            dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
            dfselect["tree_score"].to_numpy(),
            dfselect[classify_column_name].replace({"tp": 1, "fn": 1, "fp": 0, "tn": 0}).to_numpy(),
            (dfselect[classify_column_name] == "fn").to_numpy(),
            return_curves=False,
        )
        if isinstance(acc, pd.DataFrame):
            acc["group"] = g_val
        else:
            raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
        accuracy_df = pd.concat((accuracy_df, acc), ignore_index=True)

    # Add summary for indels
    df_indels = concordance_df.copy()
    df_indels["group_testing"] = np.where(df_indels["indel"], "INDELS", "SNP")
    g_val = "INDELS"
    dfselect = df_indels[df_indels["group_testing"] == g_val]
    acc = get_concordance_metrics(
        dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
        dfselect["tree_score"].to_numpy(),
        dfselect[classify_column_name].replace({"tp": 1, "fn": 1, "fp": 0, "tn": 0}).to_numpy(),
        (dfselect[classify_column_name] == "fn").to_numpy(),
        return_curves=False,
    )
    if isinstance(acc, pd.DataFrame):
        acc["group"] = g_val
    else:
        raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
    accuracy_df = pd.concat((accuracy_df, acc), ignore_index=True)

    # Add summary for h-indels
    df_indels = concordance_df.copy()
    df_indels["group_testing"] = np.where(df_indels["hmer_indel_length"] > 0, "H-INDELS", "SNP")

    g_val = "H-INDELS"
    dfselect = df_indels[df_indels["group_testing"] == g_val]
    acc = get_concordance_metrics(
        dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
        dfselect["tree_score"].to_numpy(),
        dfselect[classify_column_name].replace({"tp": 1, "fn": 1, "fp": 0, "tn": 0}).to_numpy(),
        (dfselect[classify_column_name] == "fn").to_numpy(),
        return_curves=False,
    )
    if isinstance(acc, pd.DataFrame):
        acc["group"] = g_val
    else:
        raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
    accuracy_df = pd.concat((accuracy_df, acc), ignore_index=True)

    accuracy_df = accuracy_df.round(5)

    return accuracy_df


def calc_recall_precision_curve(
    concordance_df: DataFrame,
    classify_column_name: str,
    ignored_filters: Iterable[str] | None = None,
    group_testing_column_name: str | None = None,
) -> DataFrame:
    """
    calc recall/precision curve

    Parameters
    ----------
    concordance_df: DataFrame
        concordance dataframe
    classify_column_name: str
        column name which contains tp,fp,fn status before applying filter
    ignored_filters: Iterable[str]
        list of filters to ignore (the ignored filters will not be applied before calculating accuracy)
    group_testing_column_name: str
        column name to be used as grouping column (to output statistics on each group)

    Returns
    -------
    data-frame with variant types and their recall-precision curves

    Raises
    ------
    RuntimeError
        if the output of get_concordance_metrics is not a DataFrame, should not happen
    """

    if ignored_filters is None:
        ignored_filters = {"PASS"}

    concordance_df = validate_preprocess_concordance(concordance_df, group_testing_column_name)
    concordance_df["vc_call"] = concordance_df["filter"].apply(
        lambda x: convert_filter2call(x, ignored_filters=set(ignored_filters) | {"PASS"})
    )

    # calc recall,precision, f1 per variant category
    if group_testing_column_name is None:
        concordance_df = add_grouping_column(concordance_df, get_selection_functions(), "group_testing")
        group_testing_column_name = "group_testing"

    recall_precision_curve_df = pd.DataFrame(columns=["group", "precision", "recall", "f1", "threshold"])

    groups = list(get_selection_functions().keys())
    for g_val in groups:
        dfselect = concordance_df[concordance_df[group_testing_column_name] == g_val]
        curve = get_concordance_metrics(
            dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
            dfselect["tree_score"].to_numpy(),
            dfselect[classify_column_name].replace({"tp": 1, "fn": 1, "fp": 0, "tn": 0}).to_numpy(),
            (dfselect[classify_column_name] == "fn").to_numpy(),
            return_metrics=False,
        )
        if isinstance(curve, pd.DataFrame):
            curve["group"] = g_val
        else:
            raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
        recall_precision_curve_df = pd.concat((recall_precision_curve_df, curve), ignore_index=True)

    # Add summary for indels
    df_indels = concordance_df.copy()
    df_indels["group_testing"] = np.where(df_indels["indel"], "INDELS", "SNP")
    g_val = "INDELS"
    dfselect = df_indels[df_indels["group_testing"] == g_val]
    curve = get_concordance_metrics(
        dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
        dfselect["tree_score"].to_numpy(),
        dfselect[classify_column_name].replace({"tp": 1, "fn": 1, "fp": 0, "tn": 0}).to_numpy(),
        (dfselect[classify_column_name] == "fn").to_numpy(),
        return_metrics=False,
    )
    if isinstance(curve, pd.DataFrame):
        curve["group"] = g_val
    else:
        raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
    recall_precision_curve_df = pd.concat((recall_precision_curve_df, curve), ignore_index=True)

    return recall_precision_curve_df


def validate_preprocess_concordance(
    concordance_df: DataFrame, group_testing_column_name: str | None = None
) -> DataFrame:
    """
    prepare concordance data-frame for accuracy assessment or fail if it's not possible to do

    Parameters
    ----------
    concordance_df: DataFrame
        concordance data-frame
    group_testing_column_name:
        name of column to use later for group_testing
    """
    assert "tree_score" in concordance_df.columns, "Input concordance file should be after applying a model"  # noqa S101
    concordance_df.loc[pd.isna(concordance_df["hmer_indel_nuc"]), "hmer_indel_nuc"] = "N"
    if np.any(pd.isna(concordance_df["filter"])):
        logger.warning(
            "Null values in filter column (n=%i). Setting them as PASS, but it is suspicious",
            pd.isna(concordance_df["filter"]).sum(),
        )
        concordance_df.loc[pd.isna(concordance_df["filter"]), "filter"] = "PASS"
    if np.any(pd.isna(concordance_df["tree_score"])):
        logger.warning(
            "Null values in concordance dataframe tree_score (n=%i). Setting them as zero, but it is suspicious",
            pd.isna(concordance_df["tree_score"]).sum(),
        )
        concordance_df.loc[pd.isna(concordance_df["tree_score"]), "tree_score"] = 0

    # add non-default group_testing column
    if group_testing_column_name is not None:
        concordance_df["group_testing"] = concordance_df[group_testing_column_name]
        removed = pd.isna(concordance_df["group_testing"])
        logger.info("Removing %i/%i variants with no type", removed.sum(), concordance_df.shape[0])
        concordance_df = concordance_df[~removed]
    return concordance_df


def convert_filter2call(filter_str: str, ignored_filters: set | None = None) -> str:
    """Converts the filter value of the variant into tp (PASS or ignored_filters) or fp (other filters)
    Parameters
    ----------
    filter_str : str
        filter value of the variant
    ignored_filters : set, optional
        list of filters to ignore (call will be considered tp), default "PASS"

    Returns
    -------
    str:
        tp or fp
    """
    ignored_filters = {"PASS"}
    return "tp" if all(_filter in ignored_filters for _filter in filter_str.split(";")) else "fp"


def apply_filter(pre_filtering_classification: pd.Series, is_filtered: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    pre_filtering_classification : pd.Series
        classification to 'tp', 'fp', 'fn' before applying filter
    is_filtered : pd.Series
        boolean series denoting which rows where filtered

    Returns
    -------
    pd.Series
        classification to 'tp', 'fp', 'fn', 'tn' after applying filter
    """
    post_filtering_classification = pre_filtering_classification.copy()
    post_filtering_classification.loc[is_filtered & (post_filtering_classification == "fp")] = "tn"
    post_filtering_classification.loc[is_filtered & (post_filtering_classification == "tp")] = "fn"
    return post_filtering_classification


def get_selection_functions() -> OrderedDict:
    sfs = OrderedDict()
    sfs["SNP"] = lambda x: np.logical_not(x.indel)
    sfs["Non-hmer INDEL"] = lambda x: x.indel & (x.hmer_indel_length == 0)
    sfs["HMER indel <= 4"] = lambda x: x.indel & (x.hmer_indel_length > 0) & (x.hmer_indel_length < 5)  # noqa PLR2004
    sfs["HMER indel (4,8)"] = lambda x: x.indel & (x.hmer_indel_length >= 5) & (x.hmer_indel_length < 8)  # noqa PLR2004
    sfs["HMER indel [8,10]"] = lambda x: x.indel & (x.hmer_indel_length >= 8) & (x.hmer_indel_length <= 10)  # noqa PLR2004
    sfs["HMER indel 11,12"] = lambda x: x.indel & (x.hmer_indel_length >= 11) & (x.hmer_indel_length <= 12)  # noqa PLR2004
    sfs["HMER indel > 12"] = lambda x: x.indel & (x.hmer_indel_length > 12)  # noqa PLR2004
    return sfs


def add_grouping_column(concordance_df: pd.DataFrame, selection_functions: dict, column_name: str) -> pd.DataFrame:
    """
    Add a column for grouping according to the values of selection functions

    Parameters
    ----------
    concordance_df: pd.DataFrame
        concordance dataframe
    selection_functions: dict
        Dictionary of selection functions to be applied on the df, keys - are the name of the group
    column_name: str
        Name of the column to contain grouping

    Returns
    -------
    pd.DataFrame
        df with column_name added to it that is filled with the group name according
        to the selection function
    """
    concordance_df[column_name] = None
    for k, v in selection_functions.items():
        concordance_df.loc[v(concordance_df), column_name] = k
    return concordance_df


def init_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "group",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "initial_tp",
            "initial_fp",
            "initial_fn",
            "initial_precision",
            "initial_recall",
            "initial_f1",
        ]
    )


def _get_empty_recall_precision() -> dict:
    """Return empty recall precision dictionary for category given"""
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "initial_tp": 0,
        "initial_fp": 0,
        "initial_fn": 0,
        "initial_precision": 1.0,
        "initial_recall": 1.0,
        "initial_f1": 1.0,
    }


def _get_empty_recall_precision_curve() -> dict:
    """Return empty recall precision curve dictionary for category given"""
    return {"threshold": 0, "predictions": [], "precision": [], "recall": [], "f1": []}


def get_concordance_metrics(
    predictions: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    fn_mask: np.ndarray,
    *,
    return_metrics: bool = True,
    return_curves: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """Calculate concordance metrics. The input of predictions is assumed to be numbers,
    with zeros be negative calls. fn_mask denotes the locations that were not called in
    predictions and that are called in the truth (false negatives).
    The scores are the scores of the predictions.

    Parameters
    ----------
    predictions: np.ndarray
        Predictions (number array)
    scores: np.ndarray
        Scores (float array of scores for predictions)
    truth: np.ndarray
        Truth (number array)
    fn_mask: np.ndarray
        False negative mask (boolean array of the length of truth, predictions and scores that
        contains True for false negatives and False for the rest of the values)
    return_metrics: bool
        Convenience, should the function return metrics (True) or only precision-recall curves (False)
    return_curves: bool
        Convenience, should the function return precision-recall curves (True) or only metrics (False)

    Returns
    -------
    tuple or pd.DataFrame
        Concordance metrics and precision recall curves or one of them dependent on the return_metrics and return_curves

    Raises
    ------
    AssertionError
        At least one of return_curves or return_metrics should be True
    """

    truth_curve = truth > 0
    truth_curve[fn_mask] = True
    min_example_count = 20
    precisions_curve, recalls_curve, f1_curve, thresholds_curve = stats_utils.precision_recall_curve(
        truth, scores, fn_mask, min_class_counts_to_output=min_example_count
    )
    if len(f1_curve) > 0:
        threshold_loc = np.argmax(f1_curve)
        threshold = thresholds_curve[threshold_loc]
    else:
        threshold = 0

    curve_df = pd.DataFrame(
        pd.Series(
            {
                "predictions": thresholds_curve,
                "precision": precisions_curve,
                "recall": recalls_curve,
                "f1": f1_curve,
                "threshold": threshold,
            }
        )
    ).T

    fn = fn_mask.sum()
    predictions = predictions.copy()[~fn_mask]
    scores = scores.copy()[~fn_mask]
    truth = truth.copy()[~fn_mask]

    if len(predictions) == 0:
        result = (
            pd.DataFrame(_get_empty_recall_precision(), index=[0]),
            pd.DataFrame(pd.Series(_get_empty_recall_precision_curve())).T,
        )
    else:
        tp = ((truth > 0) & (predictions > 0) & (truth == predictions)).sum()
        fp = (predictions > truth).sum()
        fn = fn + (predictions < truth).sum()
        precision = stats_utils.get_precision(fp, tp)
        recall = stats_utils.get_recall(fn, tp)
        f1 = stats_utils.get_f1(precision, recall)
        initial_tp = (truth > 0).sum()
        initial_fp = len(truth) - initial_tp
        initial_fn = fn_mask.sum()
        initial_precision = stats_utils.get_precision(initial_fp, initial_tp)
        initial_recall = stats_utils.get_recall(initial_fn, initial_tp)
        initial_f1 = stats_utils.get_f1(initial_precision, initial_recall)
        metrics_df = pd.DataFrame(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "initial_tp": initial_tp,
                "initial_fp": initial_fp,
                "initial_fn": initial_fn,
                "initial_precision": initial_precision,
                "initial_recall": initial_recall,
                "initial_f1": initial_f1,
            },
            index=[0],
        )
        result = metrics_df, curve_df
    metrics_df, curve_df = result
    assert return_curves or return_metrics, "At least one of return_curves or return_metrics should be True"  # noqa S101
    if return_curves and return_metrics:
        return metrics_df, curve_df
    if return_curves:
        return curve_df
    return metrics_df
