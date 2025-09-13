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
#    Utility functions for single read SNV quality recalibration
# CHANGELOG in reverse chronological order

from __future__ import annotations

import warnings
from functools import partial
from itertools import cycle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from ugbio_ppmseq.ppmSeq_utils import (
    MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
    MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
    STRAND_RATIO_LOWER_THRESH,
    STRAND_RATIO_UPPER_THRESH,
    PpmseqCategories,
)

# Column name constants
PREV1 = "X_PREV1"
REF = "REF"
NEXT1 = "X_NEXT1"
ALT = "ALT"
REV = "REV"

IS_MIXED = "is_mixed"
IS_MIXED_START = "is_mixed_start"
IS_MIXED_END = "is_mixed_end"
ST = "st"
ET = "et"
AS = "as"
AE = "ae"
TS = "ts"
TE = "te"
TM = "tm"  # Trimmer tags column
ST_FILLNA = "st_fillna"  # start tag with NAs filled in
ET_FILLNA = "et_fillna"  # end tag with NAs filled in

FLOW_ORDER = ["T", "G", "C", "A"]

MAX_PHRED = 100  # Default maximum Phred score for clipping
EPS = 10 ** (-MAX_PHRED / 10)  # Small epsilon value for numerical stability in log calculations


def seq2key_common(Seq, start=0, *, flowOrder=FLOW_ORDER, iterative=True):  # noqa: N803
    if iterative:
        flow_iter = cycle(flowOrder)
    else:
        flow_iter = flowOrder
        m = 0
    for _ in range(start):
        next(flow_iter)
    key = []
    j = 0
    while j < len(Seq):
        if iterative:
            current_flow = next(flow_iter)
        else:
            if m >= len(flow_iter):
                break
            current_flow = flow_iter[m]
            m += 1
        if np.any(Seq[j] in current_flow):
            h = 0
            while np.any(Seq[j] in current_flow):
                j += 1
                h += 1
                if j == len(Seq):
                    break
            key.append(h)
        else:
            key.append(0)
    return np.array(key)


def seq2key(seq, start=0, flow_order=FLOW_ORDER, *, iterative=True, pad_zeros=False):
    """Given a base-space sequence seq (string), return the corresponding keys in flow space.
    If pad_zeros is True, append zeros at end to return an integer number of cycles."""
    key = seq2key_common(seq, start=start, flowOrder=flow_order, iterative=iterative)
    if pad_zeros:
        key = np.array(key.tolist() + [0] * (4 - key.shape[0] % 4))
    return key


def key2seq(key, flow_order=FLOW_ORDER):
    """Translate the flow-space sequence key (a numpy array) to a base-space sequence (string)"""
    seq = ""
    for i, f in enumerate(key):
        seq += flow_order[i % 4] * f
    return seq


def is_possible_cycle_skip(tcwa):
    W, X, Z, Y = (tcwa[i] for i in range(4))  # noqa: N806
    pcsk = True
    if W == Z:
        if X != W and Y != W:  # noqa: PLR1714
            pcsk = False
    elif (X == W and Y == Z) or (X == Z and Y == W):
        pcsk = False
    return pcsk


def is_cycle_skip(tcwa, flow_order=FLOW_ORDER):
    W, X, Z, Y = (tcwa[i] for i in range(4))  # noqa: N806
    csk = len(seq2key(f"{W}{X}{Z}", flow_order=flow_order)) != len(seq2key(f"{W}{Y}{Z}", flow_order=flow_order))
    return csk


def prob_to_phred(prob_correct, max_value=MAX_PHRED):
    """Transform probabilities to Phred scores.

    The probability of being correct is translated to a Phred quality score using the formula:
        Q = -10 * log10(1 - p)

    Arguments:
    - prob_correct [np.ndarray]: array of probabilities (probability of being correct)
    - max_value [float]: maximum Phred score (clips values above this threshold)

    Returns:
    - np.ndarray: Phred scores corresponding to the input probabilities
    """
    prob_error = 1 - prob_correct
    if max_value is not None:
        prob_error = np.maximum(prob_error, 10 ** (-max_value / 10))
    phred_scores = -10 * np.log10(prob_error)
    return phred_scores


def prob_to_logit(prob: np.ndarray, max_value: float = MAX_PHRED, *, phred: bool = True) -> np.ndarray:
    """
    Convert probabilities to logit space (base 10).
    """
    logit = prob_to_phred(prob, max_value=max_value) - prob_to_phred(1 - prob, max_value=max_value)
    if not phred:
        logit = logit / 10.0  # convert to logit space
    return logit


def phred_to_prob(phred: np.ndarray) -> np.ndarray:
    """
    Convert Phred scores to probabilities.
    """
    return 1.0 - 10.0 ** (-phred / 10)


def logit_to_prob(logit: np.ndarray, *, phred: bool = True) -> np.ndarray:
    """
    Convert logit scores (base 10) to probabilities.
    """
    if phred:
        logit = logit / 10.0  # convert from Phred to logit space
    return 1.0 / (1.0 + 10 ** (-logit))


def _aggregate_probabilities_from_folds(
    prob_matrix: np.ndarray, transform: str = "logit", max_phred: float = MAX_PHRED
) -> np.ndarray:
    """
    Aggregate probabilities coming from all folds for each data-point.

    Parameters
    ----------
    prob_matrix : np.ndarray
        Shape = (n_folds, n_rows). Each row contains the predicted
        probabilities of one fold for all data-points.
    transform : str, optional
        The transformation to apply to the probabilities. Can have 3 values: 'phred', 'logit', 'prob'.
        By default 'phred'.
    max_phred : float, optional
        The largest Phred score to clip the probabilities to, by default 100.

    Returns
    -------
    np.ndarray
        Aggregated probability per data-point.
    """
    if transform not in {"phred", "logit", "prob"}:
        raise ValueError(f"Invalid transform '{transform}'. Expected one of: 'phred', 'logit', 'prob'.")

    if transform == "phred":
        transform_fn = partial(prob_to_phred, max_value=max_phred)
        inverse_transform_fn = phred_to_prob
    elif transform == "logit":
        transform_fn = partial(prob_to_logit, max_value=max_phred, phred=True)
        inverse_transform_fn = partial(logit_to_prob, phred=True)
    else:  # transform == 'prob'

        def transform_fn(x):
            return x

        def inverse_transform_fn(x):
            return x

    transformed_probs = transform_fn(prob_matrix)
    # Average transformed probabilities and convert back to probability
    # Use nanmean to allow for NaNs (e.g., for in-fold exclusion)
    transformed_mean = np.nanmean(transformed_probs, axis=0)
    return inverse_transform_fn(transformed_mean)


def k_fold_predict_proba(
    models: list[xgb.XGBClassifier],
    x_all: pd.DataFrame,
    fold_arr: np.ndarray,
    max_phred: float = MAX_PHRED,
    **kwargs,
):
    """
    Predict probability using k-folds CV.

    Returns a 1-d numpy array of out-of-fold "validation" predictions for rows with valid fold assignment,
    and "test" predictions (aggregated across all models) for rows with fold_arr == nan.
    For rows with invalid fold assignment (not nan, but <0 or >=num_folds), returns np.nan.

    Clarification about train/val/test:
        - "validation": all SNVs where fold_arr is in [0, 1, ..., k_folds-1] and fold_arr == k (the current fold).
          These are the held-out data for each fold, predicted only by the model not trained on them.
        - "test": all SNVs where fold_arr is np.nan. These are not assigned to any fold and are predicted by aggregating
          the predictions from all k models.
        - "train": all SNVs where fold_arr is in [0, 1, ..., k_folds-1] and fold_arr != k (i.e., used for training
          each model). If fold_arr is not np.nan and also not in [0, 1, ..., k_folds-1] (e.g., it is -1), then the
          read is considered "train".
        This function does not return "train" predictions; see k_fold_predict_proba_train for those.
    """
    num_folds = len(models)
    n_rows = x_all.shape[0]
    fold_arr = np.asarray(fold_arr)
    preds = np.full(n_rows, np.nan, dtype=float)

    # Validation: for rows with valid fold assignment (0 <= fold < num_folds)
    is_valid_fold = (fold_arr >= 0) & (fold_arr < num_folds)
    valid_idx = np.where(is_valid_fold)[0]
    if valid_idx.size > 0:
        # For each fold, predict only for its own validation rows
        for k in range(num_folds):
            idx_k = valid_idx[fold_arr[valid_idx] == k]
            if idx_k.size > 0:
                preds[idx_k] = models[k].predict_proba(x_all.iloc[idx_k], **kwargs)[:, 1]

    # Test: for rows with fold_arr nan
    is_test = np.isnan(fold_arr)
    test_idx = np.where(is_test)[0]
    if test_idx.size > 0:
        # For test rows, need predictions from all models
        all_model_probs = np.empty((num_folds, test_idx.size), dtype=float)
        for k, model in enumerate(models):
            all_model_probs[k] = model.predict_proba(x_all.iloc[test_idx], **kwargs)[:, 1]
        preds[test_idx] = _aggregate_probabilities_from_folds(all_model_probs, max_phred=max_phred)

    # For rows with invalid fold assignment (not nan, but <0 or >=num_folds), leave as np.nan
    return preds


def all_models_predict_proba(
    models: list[xgb.XGBClassifier],
    x_all: pd.DataFrame,
    fold_arr: np.ndarray,
    max_phred: float = MAX_PHRED,
    *,
    return_val_and_train_preds: bool = False,
    **kwargs,
):
    """
    Return a np.ndarray of shape (n_rows, k_folds) with predictions from all models for each row.

    If return_val_and_train_preds is True, returns "validation" and "test" predictions for each row, as well as
    aggregated "train" predictions for each row (for each row, aggregates predictions from all models except the
    out-of-fold model).

    Clarification about train/val/test:
        - "train": all SNVs where fold_arr is in [0, 1, ..., k_folds-1] and fold_arr != k (i.e., used for training
          each model). If fold_arr is not np.nan and also not in [0, 1, ..., k_folds-1] (e.g., it is -1), then the
          read is considered "train".
        - "validation": all SNVs where fold_arr is in [0, 1, ..., k_folds-1] and fold_arr == k (the current fold).
          These are the held-out data for each fold, predicted only by the model not trained on them.
        - "test": all SNVs where fold_arr is np.nan. These are not assigned to any fold and are predicted by aggregating
          the predictions from all k models.
        This function only returns "train" predictions; see k_fold_predict_proba for "validation" and "test"
        predictions.
    """
    num_folds = len(models)
    n_rows = x_all.shape[0]
    fold_arr = np.asarray(fold_arr)
    all_model_probs = np.empty((num_folds, n_rows), dtype=float)
    for k, model in enumerate(models):
        all_model_probs[k] = model.predict_proba(x_all, **kwargs)[:, 1]

    if not return_val_and_train_preds:
        return all_model_probs
    else:
        preds_val, preds_train = split_validation_training_preds(all_model_probs, fold_arr, max_phred=max_phred)
        return preds_val, preds_train, all_model_probs


def split_validation_training_preds(
    all_model_probs: np.ndarray,
    fold_arr: np.ndarray,
    max_phred: float = 100,
):
    """
    Convert k-fold model predictions into validation and training predictions.

    Given predictions from all k models for all data points, this function separates:
    - Out-of-fold (validation) predictions: for each row, the prediction from the model
      that was NOT trained on that row
    - In-fold (training) predictions: for each row, the aggregated predictions from all
      models that WERE trained on that row

    Parameters
    ----------
    all_model_probs : np.ndarray
        Array of shape (n_folds, n_rows) containing predictions from all k models
        for all data points. Each row corresponds to predictions from one fold's model.
    fold_arr : np.ndarray
        Array of length n_rows indicating which fold each data point belongs to.
        Values should be in [0, 1, ..., k_folds-1] for assigned folds, np.nan for
        test data, or other values for training-only data.
    max_phred : float, optional
        Maximum Phred score for probability aggregation, by default 100.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        preds_val : np.ndarray
            Out-of-fold (validation) predictions. For rows with valid fold assignment
            (0 <= fold < n_folds), contains the prediction from the model that excluded
            this row during training. For test rows (fold_arr == nan), contains
            aggregated predictions from all models.
        preds_train : np.ndarray
            In-fold (training) predictions. For rows with valid fold assignment,
            contains aggregated predictions from all models that included this row
            during training (excluding the out-of-fold model). For training-only
            rows, contains aggregated predictions from all models.
    """
    num_folds, n_rows = all_model_probs.shape
    preds_val = np.full(n_rows, np.nan, dtype=float)
    preds_train = np.full(n_rows, np.nan, dtype=float)
    is_val_fold = (fold_arr >= 0) & (fold_arr < num_folds)
    idx_val = np.where(is_val_fold)[0]
    idx_nan = np.where(np.isnan(fold_arr))[0]
    idx_train_only = np.where((~np.isnan(fold_arr)) & (~is_val_fold))[0]
    if idx_val.size > 0:
        preds_val[idx_val] = all_model_probs[fold_arr[idx_val].astype(int), idx_val]
        train_probs = all_model_probs[:, idx_val].copy()
        train_probs[fold_arr[idx_val].astype(int), np.arange(len(idx_val))] = np.nan
        preds_train[idx_val] = _aggregate_probabilities_from_folds(train_probs, max_phred=max_phred)
    if idx_nan.size > 0:
        preds_val[idx_nan] = _aggregate_probabilities_from_folds(all_model_probs[:, idx_nan], max_phred=max_phred)
    if idx_train_only.size > 0:
        preds_train[idx_train_only] = _aggregate_probabilities_from_folds(
            all_model_probs[:, idx_train_only], max_phred=max_phred
        )
    # For rows with invalid fold assignment (not nan, but <0 or >=num_folds), leave as np.nan
    return preds_val, preds_train


def get_base_recall_from_filters(filters):
    """Calculate base recall from filtering statistics.

    Base recall is calculated as the number of 'rows' in the last filter before 'ref_eq_alt'
    divided by the number of 'rows' in the last filter before the first 'quality' filter.

    Returns:
        float: Base recall value
    """
    # Find last filter before ref_eq_alt
    ref_eq_alt_index = None
    for i, filter_info in enumerate(filters):
        if filter_info["name"] == "ref_eq_alt":
            ref_eq_alt_index = i
            break

    if ref_eq_alt_index is None or ref_eq_alt_index == 0:
        raise ValueError("ref_eq_alt filter not found or is the first filter")

    last_filter_before_ref_eq_alt = filters[ref_eq_alt_index - 1]

    # Find the first filter with type "quality"
    first_quality_filter_index = None
    for i, filter_info in enumerate(filters):
        if filter_info.get("type") == "quality":
            first_quality_filter_index = i
            break

    # Find the last filter before the first quality filter
    if first_quality_filter_index is not None and first_quality_filter_index > 0:
        last_filter_before_quality = filters[first_quality_filter_index - 1]
    elif first_quality_filter_index is None:
        # If no quality filter found, use the last filter in the list
        last_filter_before_quality = filters[-1] if filters else None
    else:
        # If quality filter is the first filter, use first filter
        last_filter_before_quality = filters[0]

    if last_filter_before_quality is None:
        raise ValueError("Could not find appropriate filter for denominator")

    return last_filter_before_ref_eq_alt["rows"] / last_filter_before_quality["rows"]


def get_base_error_rate_from_filters(filters):
    """Calculate base error rate (b_f) from negative filtering statistics.

    This represents the base rate of low VAF (False) SNVs, calculated as the ratio
    of rows at the last filter before downsample to the total raw rows in the
    negative filtering statistics.

    Returns:
        float: Base error rate of low VAF SNVs (rows before downsample / total raw rows)
    """
    downsample_index = None
    for i, filter_info in enumerate(filters):
        if filter_info["name"] == "downsample":
            downsample_index = i
            break

    # b_f = rows at last filter before downsample / rows at first filter (raw)
    if downsample_index is not None:
        last_filter_before_downsample = filters[downsample_index - 1]
    else:
        # If no downsample filter, use the last filter
        last_filter_before_downsample = filters[-1]

    first_filter = filters[0]  # TODO: Should this be the first filter, or the last filter before 'ref_ne_alt'?
    return last_filter_before_downsample["rows"] / first_filter["rows"]


def construct_trinuc_context_with_alt(
    df: pd.DataFrame, *, prev1: str = PREV1, ref: str = REF, next1: str = NEXT1, alt: str = ALT
) -> pd.Series:
    """Construct trinuc_context_with_alt column from prev1, ref, next1, alt columns.

    Args:
        df: DataFrame containing the required columns
        prev1: Name of the previous nucleotide column, default 'X_PREV1'
        ref: Name of the reference nucleotide column, default 'REF'
        next1: Name of the next nucleotide column, default 'X_NEXT1'
        alt: Name of the alternate nucleotide column, default 'ALT'

    Returns:
        pd.Series: The trinuc_context_with_alt column constructed by concatenating the four columns

    TODO: Add tcwa_fwd!
    """
    required_cols = [prev1, ref, next1, alt]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert categorical columns to string to enable concatenation
    return df[prev1].astype(str) + df[ref].astype(str) + df[next1].astype(str) + df[alt].astype(str)


def get_trinuc_context_with_alt_fwd_vectorized(tcwa, is_forward):
    """
    Vectorized function to compute forward-oriented trinuc context + alt base.

    Parameters:
        is_forward (pd.Series): Series indicating strand direction.
        tcwa (pd.Series): Series with trinuc context + alt base (4 letters).

    Returns:
        pd.Series: A Series with the computed forward-oriented (i.e., in sequence direction)
        trinuc context + alt.
    """
    # Extract individual bases
    split_chars = tcwa.str.extract(r"(.)(.)(.)(.)")
    split_chars.columns = ["prv", "ref", "nxt", "alt"]

    # Define complement map and apply it
    complement = str.maketrans("ACGT", "TGCA")
    split_chars_comp = split_chars.apply(lambda col: col.str.translate(complement))

    # Create reverse complement version
    rev_comp = split_chars_comp["nxt"] + split_chars_comp["ref"] + split_chars_comp["prv"] + split_chars_comp["alt"]

    # Choose original or reverse-complement depending on is_forward
    return np.where(is_forward, tcwa, rev_comp)


def safe_roc_auc(y_true, y_pred, name=None, logger=None):
    """
    Calculate ROC AUC score with checks for cases where the calculation is not possible.

    Parameters:
    y_true: array-like, true labels
    y_pred: array-like, predicted probabilities
    name: str, name of dataset (for logging). If None, ignore dataset name.
    logger: logging.Logger, optional logger instance. If None, uses warnings module.

    Returns:
    ROC AUC score if calculable, otherwise np.nan
    """
    y_true = np.array(y_true)  # Convert to numpy array
    name_for_log = f" ({name=})" if name is not None else ""
    # Check if the array is empty
    if y_true.size == 0:
        msg = f"ROC AUC cannot be calculated: dataset is empty {name_for_log}"
        if logger:
            logger.warning(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)
        return np.nan

    # Check if all labels are the same
    if len(np.unique(y_true)) == 1:
        msg = f"ROC AUC cannot be calculated: dataset has only one class {name_for_log}"
        if logger:
            logger.warning(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)
        return np.nan

    # Calculate the ROC AUC score
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception as e:
        msg = f"An error occurred while calculating ROC AUC{name_for_log}: {e}"
        if logger:
            logger.error(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)
        return np.nan


class HandlePPMSeqTagsInFeatureMapDataFrame:
    """Class to handle ppmSeq tags in the featuremap dataframe.

    This class is responsible for determining the correct column names for the ppmSeq start and end tags
    based on the provided featuremap dataframe and categorical features.
    It handles different versions of the ppmSeq adapter and ensures that the tags are consistent.

    TODO: This whole chunk of code is a mix of legacy and new logic, needs to be cleaned up.
    """

    def __init__(
        self,
        featuremap_df: pd.DataFrame,
        categorical_features_names: list[str],
        ppmseq_adapter_version: str,
        start_tag_col: str | None = None,
        end_tag_col: str | None = None,
        logger=None,
    ):
        self.featuremap_df = featuremap_df.copy()
        self.categorical_features_names = categorical_features_names
        self.ppmseq_adapter_version = ppmseq_adapter_version
        self.start_tag_col = start_tag_col
        self.end_tag_col = end_tag_col
        self.start_tag_fillna_col = None
        self.end_tag_fillna_col = None
        self.logger = logger
        self.setup_ppmseq_tags()  # This adds ST and ET columns if they are not already present

    def log(self, msg, level="info"):
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
            elif level == "debug":
                self.logger.debug(msg)
            else:
                self.logger.info(msg)
        elif level == "warning":
            warnings.warn(msg, UserWarning, stacklevel=2)
        elif level == "error":
            raise RuntimeError(msg)
        else:
            print(msg)

    def setup_ppmseq_tags(self):
        """Set up ppmSeq start and end tag columns in the featuremap dataframe.

        This method handles the complete workflow for ppmSeq tag setup:
        - If tag columns are provided as arguments/parameters, use them.
        - Otherwise, infer and potentially create them using the following logic:
          - If ST/ET columns exist in featuremap, use them directly
          - If AS/AE/TS/TE columns exist but ST/ET don't, calculate and add ST/ET columns
          - If neither set exists, set tag columns to None
        - Use categorical features dict and ppmseq_adapter_version for disambiguation
        - Raise warnings if any values are inconsistent

        Note: This method may modify self.featuremap_df by adding ST/ET columns.
        """
        # TODO: check this huge change
        self._initialize_tag_columns()
        self._infer_tag_columns()
        self._log_final_tag_columns()

    def _initialize_tag_columns(self):
        """Initialize tag columns based on adapter version.
        This function is a placeholder in case more complicated logic is needed in the future.
        """
        if self.ppmseq_adapter_version is None:
            self.tag_cols_from_adapter = [None, None]
        else:
            self.tag_cols_from_adapter = [ST, ET]

    def _infer_tag_columns(self):
        """Infer and set tag columns based on featuremap content and categorical features.

        This method:
        1. Checks if ST/ET columns already exist in featuremap (CRAM tags)
        2. If not, checks if AS/AE/TS/TE exist (v5 tags) and calculates ST/ET
        3. Sets start_tag_col and end_tag_col accordingly, or None if no data available
        """
        if self.start_tag_col is None and self.end_tag_col is None:
            if self._cram_tags_in_featuremap():
                # Check if ST, ET in featuremap_df, applies to both v1 and v5_legacy
                self._set_cram_tags()
            elif self._v5_tags_in_featuremap():
                # ST, ET no in featermap_df, AS, AE, TS, TE are. Adds ST, ET to featuremap_df
                self._add_ppmseq_tags_to_featuremap()
                self._set_cram_tags()
            else:
                self.start_tag_col = None
                self.end_tag_col = None

    def _v5_tags_in_featuremap(self):
        return (
            AS in self.featuremap_df
            and AE in self.featuremap_df
            and TS in self.featuremap_df
            and TE in self.featuremap_df
        )

    def _cram_tags_in_featuremap(self):
        return ST in self.featuremap_df and ET in self.featuremap_df

    def _both_tags_in_categorical(self):
        return (
            "strand_ratio_category_end" in self.categorical_features_names
            and "strand_ratio_category_start" in self.categorical_features_names
            and ST in self.categorical_features_names
            and ET in self.categorical_features_names
        )

    def _neither_tags_in_categorical(self):
        return (
            "strand_ratio_category_end" not in self.categorical_features_names
            and "strand_ratio_category_start" not in self.categorical_features_names
            and ST not in self.categorical_features_names
            and ET not in self.categorical_features_names
        )

    def _strand_ratio_in_categorical(self):
        return (
            "strand_ratio_category_end" in self.categorical_features_names
            and "strand_ratio_category_start" in self.categorical_features_names
        )

    def _cram_tags_in_categorical(self):
        return ST in self.categorical_features_names and ET in self.categorical_features_names

    def _add_ppmseq_tags_to_featuremap(
        self,
        sr_lower: float = STRAND_RATIO_LOWER_THRESH,
        sr_upper: float = STRAND_RATIO_UPPER_THRESH,
        min_total_hmer: int = MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
        max_total_hmer: int = MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
    ):
        """Calculate and add ppmSeq tags to featuremap_df from the start and end annotations.

        Parameters
        ----------
        sr_lower : float, optional
            Lower strand ratio threshold for determining MIXED category, by default 0.27
        sr_upper : float, optional
            Upper strand ratio threshold for determining MIXED category, by default 0.73
        min_total_hmer : int, optional
            Minimum total hmer lengths for valid range, by default 4
        max_total_hmer : int, optional
            Maximum total hmer lengths for valid range, by default 8
        """

        # Helper function to compute strand ratio category vectorized
        def compute_strand_ratio_category(as_col, ts_col, sr_lower, sr_upper, min_total, max_total):
            """
            Vectorized computation of strand ratio category.

            Parameters
            ----------
            as_col : pd.Series
                A's count column (AS or AE)
            ts_col : pd.Series
                T's count column (TS or TE)
            sr_lower : float
                Lower threshold for MIXED category
            sr_upper : float
                Upper threshold for MIXED category
            min_total : int
                Minimum total count for valid range
            max_total : int
                Maximum total count for valid range

            Returns
            -------
            pd.Series
                Series with strand ratio categories
            """
            # Initialize result with UNDETERMINED as default
            result = pd.Series([PpmseqCategories.UNDETERMINED.value] * len(as_col), index=as_col.index)

            # Where either AS/AE or TS/TE is NaN, set result to NaN
            nan_mask = as_col.isna() | ts_col.isna()
            result[nan_mask] = np.nan

            # Valid data mask (not NaN)
            valid_mask = ~nan_mask

            # Sum of T's and A's for valid data
            total_count = as_col + ts_col

            # Filter for valid range among valid data
            valid_range_mask = valid_mask & (total_count >= min_total) & (total_count <= max_total)

            if valid_range_mask.any():
                # TS==0 -> PLUS
                plus_indices = valid_range_mask & (ts_col == 0)
                result[plus_indices] = PpmseqCategories.PLUS.value

                # AS==0 -> MINUS
                minus_indices = valid_range_mask & (as_col == 0)
                result[minus_indices] = PpmseqCategories.MINUS.value

                # Calculate strand ratio for remaining cases (neither AS nor TS is 0)
                remaining_indices = valid_range_mask & (as_col != 0) & (ts_col != 0)
                if remaining_indices.any():
                    strand_ratio = ts_col[remaining_indices] / total_count[remaining_indices]

                    # sr_lower <= ratio <= sr_upper -> MIXED
                    mixed_indices = remaining_indices & (strand_ratio >= sr_lower) & (strand_ratio <= sr_upper)
                    result[mixed_indices] = PpmseqCategories.MIXED.value

                    # All other ratios remain UNDETERMINED (already set as default)

            return result

        # Calculate ST (start tag) from AS and TS
        self.featuremap_df[ST] = compute_strand_ratio_category(
            self.featuremap_df[AS],
            self.featuremap_df[TS],
            sr_lower,
            sr_upper,
            min_total_hmer,
            max_total_hmer,
        )

        # Calculate ET (end tag) from AE and TE
        self.featuremap_df[ET] = compute_strand_ratio_category(
            self.featuremap_df[AE],
            self.featuremap_df[TE],
            sr_lower,
            sr_upper,
            min_total_hmer,
            max_total_hmer,
        )

    def _set_cram_tags(self):
        self.start_tag_col = ST
        self.end_tag_col = ET
        if not self._cram_tags_in_categorical():
            self.log("ppmSeq tags not in categorical_features_dict", level="warning")

    def _log_final_tag_columns(self):
        self.log(f"Using [start_tag, end_tag] = {[self.start_tag_col, self.end_tag_col]}")
        if self.start_tag_col != self.tag_cols_from_adapter[0] or self.end_tag_col != self.tag_cols_from_adapter[1]:
            self.log(f"ppmSeq tags are not consistent with respect to {self.tag_cols_from_adapter=}", level="warning")

    def fill_nan_tags(self, st_fillna: str = ST_FILLNA, et_fillna: str = ET_FILLNA):
        """Fill NaN values in start and end tag columns.

        For ST (start tag): all NaN values are replaced with UNDETERMINED.
        For ET (end tag):
            - If TM column exists and contains 'A', replace NaN with UNDETERMINED
            - If TM column exists and doesn't contain 'A', replace NaN with END_UNREACHED
            - If TM column doesn't exist, log warning and replace all NaN with UNDETERMINED

        NOTE: The categorical values of the ST and ET columns cannot be changed because they are
        used in inference and must keep the same categories. Their values are duplicated in the
        columns ST_FILLNA and ET_FILLNA. The categories of the latter are taken from PpmseqCategories.

        Parameters
        ----------
        st_fillna : str, optional
            Name of the column to fill NaN values in start tag, by default 'st_fillna'
        et_fillna : str, optional
            Name of the column to fill NaN values in end tag, by default 'et_fillna'
        """
        # Ensure we have tag columns set up
        if self.start_tag_col is None or self.end_tag_col is None:
            self.log("Tag columns not set up, cannot fill NaN values", level="warning")
            return

        self.start_tag_fillna_col = st_fillna
        self.end_tag_fillna_col = et_fillna

        # Convert to string first, then to categorical with PpmseqCategories
        ppmseq_categories = [category.value for category in PpmseqCategories]

        # Fill NaN values in start tag column with UNDETERMINED
        if self.start_tag_col in self.featuremap_df.columns:
            self.featuremap_df[st_fillna] = pd.Categorical(
                self.featuremap_df[self.start_tag_col].astype(str), categories=ppmseq_categories
            )
            self.featuremap_df[st_fillna] = self.featuremap_df[st_fillna].fillna(PpmseqCategories.UNDETERMINED.value)

        # Fill NaN values in end tag column based on TM column
        if self.end_tag_col in self.featuremap_df.columns:
            self.featuremap_df[et_fillna] = pd.Categorical(
                self.featuremap_df[self.end_tag_col].astype(str), categories=ppmseq_categories
            )
            if TM in self.featuremap_df.columns:
                # Check if adapter was reached (TM column contains 'A')
                adapter_reached = self.featuremap_df[TM].str.contains("A", na=False).to_numpy()

                # Use pandas operations to preserve categorical dtype
                nan_mask = (
                    self.featuremap_df[et_fillna].isna()
                    | (self.featuremap_df[et_fillna] == PpmseqCategories.UNDETERMINED.value)  # for legacy_v5 adapters
                )
                self.featuremap_df.loc[nan_mask & adapter_reached, et_fillna] = PpmseqCategories.UNDETERMINED.value
                self.featuremap_df.loc[nan_mask & ~adapter_reached, et_fillna] = PpmseqCategories.END_UNREACHED.value
            else:
                # TM column doesn't exist, log warning and set all NaN to UNDETERMINED
                self.log(
                    f"TM column '{TM}' not found in featuremap. Setting all NaN end tag values to UNDETERMINED",
                    level="warning",
                )
                self.featuremap_df[et_fillna] = self.featuremap_df[et_fillna].fillna(
                    PpmseqCategories.UNDETERMINED.value
                )

    def add_is_mixed_to_featuremap_df(self):
        """Add is_mixed column to self.featuremap_df"""
        # TODO: use the information from adapter_version instead of this patch
        # Get start tag
        if self.start_tag_col is not None:
            self.featuremap_df[IS_MIXED_START] = self.featuremap_df[self.start_tag_col] == PpmseqCategories.MIXED.value
        else:  # If no strand ratio information is available, set is_mixed to False
            self.featuremap_df[IS_MIXED_START] = False
            self.log("No start ppmSeq tags in data, setting is_mixed_start to False", level="warning")
        # Get end tag
        if self.end_tag_col is not None:
            self.featuremap_df[IS_MIXED_END] = self.featuremap_df[self.end_tag_col] == PpmseqCategories.MIXED.value
        else:  # If no strand ratio information is available, set is_mixed to False
            self.featuremap_df[IS_MIXED_END] = False
            self.log("No end ppmSeq tags in data, setting is_mixed_end to False", level="warning")
        # Combine start and end tags
        self.featuremap_df[IS_MIXED] = np.logical_and(
            self.featuremap_df[IS_MIXED_START], self.featuremap_df[IS_MIXED_END]
        )
