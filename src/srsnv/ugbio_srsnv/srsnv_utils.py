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

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Column name constants
PREV1 = "X_PREV1"
REF = "REF"
NEXT1 = "X_NEXT1"
ALT = "ALT"

MAX_PHRED = 100  # Default maximum Phred score for clipping
EPS = 10 ** (-MAX_PHRED / 10)  # Small epsilon value for numerical stability in log calculations


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


def get_trinuc_context_with_alt_fwd_vectorized(df, is_forward_col="is_forward", tcwa_col="trinuc_context_with_alt"):
    """
    Vectorized function to compute forward-oriented trinuc context + alt base.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        is_forward_col (str): Name of the column indicating strand direction.
        tcwa_col (str): Name of the column with trinuc context + alt base (4 letters).

    Returns:
        pd.Series: A Series with the computed forward-oriented trinuc context + alt.
    """
    # Extract individual bases
    split_chars = df[tcwa_col].str.extract(r"(.)(.)(.)(.)")
    split_chars.columns = ["prv", "ref", "nxt", "alt"]

    # Define complement map and apply it
    complement = str.maketrans("ACGT", "TGCA")
    split_chars_comp = split_chars.apply(lambda col: col.str.translate(complement))

    # Create reverse complement version
    rev_comp = split_chars_comp["nxt"] + split_chars_comp["ref"] + split_chars_comp["prv"] + split_chars_comp["alt"]

    # Choose original or reverse-complement depending on is_forward
    return np.where(df[is_forward_col], df[tcwa_col], rev_comp)


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
