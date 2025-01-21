from collections import OrderedDict
from enum import Enum

import numpy as np
import pandas as pd
import xgboost
from pandas.core.groupby import DataFrameGroupBy
from sklearn import compose
from ugbio_core import math_utils
from ugbio_core.concordance.concordance_utils import add_grouping_column, get_concordance_metrics, init_metrics_df
from ugbio_core.logger import logger

from ugbio_filtering import multiallelics as mu
from ugbio_filtering import transformers
from ugbio_filtering.tprep_constants import GtType, VcfType

MAX_CHUNK_SIZE = 1000000


def train_model(
    concordance: pd.DataFrame,
    gt_type: GtType,
    vtype: VcfType,
    annots: list | None = None,
) -> tuple[compose.ColumnTransformer, xgboost.XGBRFClassifier]:
    """Trains model xgboost model on a subset of dataframe

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    gt_type: GtType
        Is the ground truth approximate or exact (in the first case the model will predict 0/1, in the second: 0/1/2)
    vtype: string
        The type of the input vcf. Either "single_sample" or "joint"
    annots: list, optional
        Optional list of annotations present in the dataframe

    Returns
    -------
    tuple:
        Trained transformer and classifier model

    Raises
    ------
    ValueError
        If the gt_type is not recognized
    """
    transformer = transformers.get_transformer(vtype, annots=annots)
    if gt_type == GtType.APPROXIMATE:
        select_train = concordance["label"].apply(lambda x: x in {0, 1})
    elif gt_type == GtType.EXACT:
        select_train = concordance["label"].apply(lambda x: x in {(0, 0), (0, 1), (1, 1), (1, 0)})
    else:
        raise ValueError("Unknown gt_type")
    df_train = concordance.loc[select_train]
    if gt_type == GtType.APPROXIMATE:
        labels_train = df_train["label"].to_numpy()
    elif gt_type == GtType.EXACT:
        labels_train = transformers.label_encode.transform(list(df_train["label"].values))
    else:
        raise ValueError("Unknown gt_type")

    logger.info(f"Training model on {len(df_train)} samples: start")
    x_train_df = pd.DataFrame(transformer.fit_transform(df_train))
    _validate_data(x_train_df)
    logger.info("Transform: done")
    clf = xgboost.XGBClassifier(
        n_estimators=100,
        learning_rate=0.15,
        subsample=0.4,
        max_depth=6,
        random_state=0,
        colsample_bytree=0.4,
        n_jobs=14,
    )
    clf.fit(x_train_df, labels_train)
    return clf, transformer


def apply_model(
    input_df: pd.DataFrame, model: xgboost.XGBClassifier, transformer: compose.ColumnTransformer
) -> tuple[np.ndarray, np.ndarray]:
    """Applies model to the input dataframe

    Parameters
    ----------
    input_df: pd.DataFrame
        Input dataframe
    model: xgboost.XGBClassifier
        Model
    transformer: compose.ColumnTransformer
        Transformer

    Returns
    -------
    tuple:
        Predictions and probabilities
    """

    # We apply models on batches of the dataframe to avoid memory issues
    chunks = np.arange(0, input_df.shape[0], MAX_CHUNK_SIZE, dtype=int)
    chunks = np.concatenate((chunks, [input_df.shape[0]]))
    transformed_chunks = [
        transformer.transform(input_df.iloc[chunks[i] : chunks[i + 1]]) for i in range(len(chunks) - 1)
    ]
    x_test_df = pd.concat(transformed_chunks)
    _validate_data(x_test_df)
    predictions = model.predict(x_test_df)
    probabilities = model.predict_proba(x_test_df)
    return predictions, probabilities


def _validate_data(data: np.ndarray | pd.Series | pd.DataFrame) -> None:
    """Validates that the data does not contain nulls"""

    if isinstance(data, np.ndarray):
        test_data = data
    else:
        test_data = pd.DataFrame(data).to_numpy()
    try:
        if len(test_data.shape) == 1 or test_data.shape[1] <= 1:
            assert pd.isna(test_data).sum() == 0, "data vector contains null"  # noqa S101
        else:
            for c_val in range(test_data.shape[1]):
                assert pd.isna(test_data[:, c_val]).sum() == 0, f"Data matrix contains null in column {c_val}"  # noqa S101
    except AssertionError as af_val:
        logger.error(str(af_val))
        raise af_val


def eval_model(
    df: pd.DataFrame,  # noqa PD901
    model: xgboost.XGBClassifier,
    transformer: compose.ColumnTransformer,
    *,
    add_testing_group_column: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate precision/recall for the decision tree classifier

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe
    model: xgboost.XGBClassifier
        Model
    transformer: compose.ColumnTransformer
        Data prep transformer
    add_testing_group_column: bool
        Should default testing grouping be added (default: True),
        if False will look for grouping in `group_testing`

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]:
        recall/precision for each category and recall precision curve for each category in dataframe

    Raises
    ------
    RuntimeError
        If the group_testing column is not present in the dataframe
    """
    df = df.copy()  # noqa PD901
    predictions, probs = apply_model(df, model, transformer)
    phred_pls = -math_utils.phred(probs)
    sorted_pls = np.sort(phred_pls, axis=1)
    gqs = sorted_pls[:, -1] - sorted_pls[:, -2]
    quals = phred_pls[:, 1:].max(axis=1) - phred_pls[:, 0]
    df["ml_gq"] = gqs
    df["ml_qual"] = quals
    df["predict"] = predictions

    labels = df["label"]
    if probs.shape[1] == 2:  # noqa PLR2004
        gt_type = GtType.APPROXIMATE
    elif probs.shape[1] == 3:  # noqa PLR2004
        gt_type = GtType.EXACT
    else:
        raise RuntimeError("Unknown gt_type")

    if gt_type == GtType.APPROXIMATE:
        select = df["label"].apply(lambda x: x in {0, 1})
    elif gt_type == GtType.EXACT:
        select = df["label"].apply(lambda x: x in {(0, 0), (0, 1), (1, 1), (1, 0)})
    labels = labels[select]
    if gt_type == GtType.EXACT:
        labels = np.array(transformers.label_encode.transform(list(labels))) > 0
        df["predict"] = df["predict"] > 0

    df = df.loc[select]  # noqa PD901
    result = evaluate_results(
        df, pd.Series(list(labels), index=df.index), add_testing_group_column=add_testing_group_column
    )
    if isinstance(result, tuple):
        return result
    raise RuntimeError("Unexpected result")


def evaluate_results(
    df: pd.DataFrame,  # noqa PD901
    labels: pd.Series,
    *,
    add_testing_group_column: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate concordance results for the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    labels : pd.Series
        _description_
    add_testing_group_column : bool, optional
        _description_, by default True

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Returns summary metrics and precision/recall curves in two dataframes
    """
    if add_testing_group_column:
        df = add_grouping_column(df, get_selection_functions(), "group_testing")  # noqa PD901
        groups = list(get_selection_functions().keys())

    else:
        assert "group_testing" in df.columns, "group_testing column should be given"  # noqa S101
        groups = list(set(df["group_testing"]))

    accuracy_df = init_metrics_df()
    curve_df = pd.DataFrame(columns=["group", "precision", "recall", "f1", "threshold"])
    for g_val in groups:
        select = df["group_testing"] == g_val
        group_df = df[select]
        group_labels = labels[select]
        acc, curve = get_concordance_metrics(
            group_df["predict"],
            group_df["ml_qual"],
            np.array(group_labels) > 0,
            np.zeros(group_df.shape[0], dtype=bool),
        )
        acc["group"] = g_val
        curve["group"] = g_val
        accuracy_df = pd.concat((accuracy_df, acc), ignore_index=True)
        curve_df = pd.concat((curve_df, curve), ignore_index=True)
    return accuracy_df, curve_df


def get_selection_functions() -> OrderedDict:
    sfs = OrderedDict()
    sfs["SNP"] = lambda x: np.logical_not(x.indel)
    sfs["Non-hmer INDEL"] = lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) == 0)
    sfs["HMER indel <= 4"] = (
        lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) > 0) & (x["x_hil"].apply(lambda y: y[0]) < 5)  # noqa PLR2004
    )
    sfs["HMER indel (4,8)"] = (
        lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) >= 5) & (x["x_hil"].apply(lambda y: y[0]) < 8)  # noqa PLR2004
    )
    sfs["HMER indel [8,10]"] = (
        lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) >= 8) & (x["x_hil"].apply(lambda y: y[0]) <= 10)  # noqa PLR2004
    )
    sfs["HMER indel 11,12"] = (
        lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) >= 11) & (x["x_hil"].apply(lambda y: y[0]) <= 12)  # noqa PLR2004
    )
    sfs["HMER indel > 12"] = lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) > 12)  # noqa PLR2004
    return sfs


class VariantSelectionFunctions(Enum):
    """Collecton of variant selection functions - all get DF as input and return boolean np.array"""

    @staticmethod
    def ALL(df: pd.DataFrame) -> np.ndarray:  # noqa N802
        return np.ones(df.shape[0], dtype=bool)

    @staticmethod
    def HMER_INDEL(df: pd.DataFrame) -> np.ndarray:  # noqa N802
        return np.array(df.hmer_indel_length > 0)

    @staticmethod
    def ALL_except_HMER_INDEL_greater_than_or_equal_5(df: pd.DataFrame) -> np.ndarray:  # noqa N802
        return np.array(~(df.hmer_indel_length >= 5))  # noqa PLR2004


def combine_multiallelic_spandel(df: pd.DataFrame, df_unsplit: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Combine predictions and scores for multiallelic variants and variants with spanning deletions. Since the
    genotyping model is trained to spit likelihoods of only three genotypes, we split the multiallelic VCF records
    into multiple rows with two alleles in each (see `multiallelics.split_multiallelic_variants` for the details).
    In this function we combine the multiple rows into a single one.

    Note that since the function receives the dataframe before the splitting and the split dataframe rows point
    to the original rows we only need to combine the PLs of the split rows.

    Parameters
    ----------
    df: pd.DataFrame
        Multiallelic spandel dataframe
    df_unsplit: pd.DataFrame
        Unsplit dataframe (the original read from the VCF)
    scores: np.ndarray
        Scores, in the order of the split dataframe

    Returns
    -------
    pd.DataFrame:
        df_unsplit with ml_lik column added representing the merged likelihoods
    """

    multiallelics = df[~pd.isna(df["multiallelic_group"])].groupby("multiallelic_group")
    multiallelic_scores = scores[~pd.isna(df["multiallelic_group"]), :]
    multiallelic_scores = pd.Series(
        [list(x) for x in multiallelic_scores], index=df[~pd.isna(df["multiallelic_group"])].index
    )
    df_unsplit = merge_and_assign_pls(df_unsplit, multiallelics, multiallelic_scores)
    spandels = df[~pd.isna(df["spanning_deletion"])].groupby(["chrom", "pos"])
    spandel_scores = scores[~pd.isna(df["spanning_deletion"]), :]
    spandel_scores = pd.Series([list(x) for x in spandel_scores], index=df[~pd.isna(df["spanning_deletion"])].index)
    df_unsplit = merge_and_assign_pls(df_unsplit, spandels, spandel_scores)
    return df_unsplit


def merge_and_assign_pls(
    original_df: pd.DataFrame, grouped_split_df: DataFrameGroupBy, split_scores: pd.Series
) -> pd.DataFrame:
    """Merges the PL scores of multiallelic and spanning deletion calls. The asprocess is like this:
    In the step of splitting the multiallelics, we have generated the rows as follows:
    Pick the "strongest" allele (e.g. A1),
    Combine the support for A1 and A2 into a single allele A
    Genotype (R,A): the result is (PL'(R,R), PL'(R,A), PL'(A,A))
    Genotype (A1,A2), where A1 is the reference, A2 is the "alt".
    In this function we are merging the split genotyping rows back as follows:
    Merge the support in the following way to generate PL(R,R),PL(R,A1),PL(A1,A1),PL(R,A2),PL(A1,A2),PL(A2,A2) tuple
    PL(R,R) = PL'(R,R)
    PL(R,A1) = PL'(R,A)
    PL(A1,A1) = PL'(A,A) * PL'(A1,A1)
    PL(A1,A2) = PL'(A,A) * PL(A1,A2)
    PL(A2,A2) = PL(A,A) * PL(A2,A2)

    Note that PL(R,A2) is zero here.

    Parameters
    ----------
    original_df: pd.DataFrame
        Original dataframe
    grouped_split_df: DataFrameGroupBy
        Dataframe of calls, where the multiallelics are split between the records,
        the dataframe is grouped by the multiallelics original locations
    split_scores: pd.Series
        Scores of the multiallelic variants

    Returns
    -------
    pd.DataFrame:
        The dataframe with multiallelics merged into a single record

    See also
    --------
    multiallelics.split_multiallelic_variants
    """
    result = []
    for g in grouped_split_df.groups:
        k = g
        rg = grouped_split_df.get_group(k)
        if rg.shape[0] == 1:
            pls = split_scores[grouped_split_df.groups[g][0]]
        else:
            orig_alleles = original_df.at[k, "alleles"]  # noqa PD008
            n_alleles = len(orig_alleles)
            n_pls = n_alleles * (n_alleles + 1) / 2
            pls = np.zeros(int(n_pls))
            idx1 = orig_alleles.index(rg.iloc[1]["alleles"][0])
            idx2 = orig_alleles.index(rg.iloc[1]["alleles"][1])
            tmp = split_scores[grouped_split_df.groups[g][0]][2] * np.array(split_scores[grouped_split_df.groups[g][1]])
            tmp = np.insert(tmp, 1, 0)
            merge_pls = np.concatenate((split_scores[grouped_split_df.groups[g][0]][:2], tmp))
            set_idx = [
                mu.get_pl_idx(x) for x in ((0, 0), (0, idx1), (idx1, idx1), (0, idx2), (idx1, idx2), (idx2, idx2))
            ]
            pls[set_idx] = merge_pls
        result.append(pls)

    set_dest = list(grouped_split_df.groups.keys())
    original_df.loc[set_dest, "ml_lik"] = pd.Series(result, index=original_df.loc[set_dest].index)
    return original_df
