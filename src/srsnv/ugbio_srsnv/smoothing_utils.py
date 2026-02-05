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
#    Utilities for smoothing precision estimation in SRSNV quality score mapping
# CHANGELOG in reverse chronological order

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
from statsmodels.nonparametric import smoothers_lowess
from ugbio_core.logger import logger
from ugbio_featuremap.featuremap_utils import FeatureMapFields

from ugbio_srsnv.srsnv_utils import EPS, prob_to_logit, prob_to_phred

# Dataframe column names
FOLD_COL = "fold_id"
LABEL_COL = "label"
PROB_ORIG = "prob_orig"
PROB_FOLD_TMPL = "prob_fold_{k}"
MQUAL = FeatureMapFields.MQUAL.value

# KDE smoothing default parameters
DEFAULT_KDE_GRID_SIZE = 8192
DEFAULT_KDE_NUM_BANDWIDTH_LEVELS = 1  # For mqual mode, 5 might be good
DEFAULT_KDE_LOWESS_FRAC = 0.3
DEFAULT_KDE_ENFORCE_MONOTONIC = False
DEFAULT_KDE_TRUNCATION_MODE = "auto_detect"
DEFAULT_KDE_TRANSFORM_MODE = "logit"  # "mqual" or "logit"
FFT_THRESHOLD = 1000  # Use FFT for convolutions larger than this.

# Constants
MIN_FOLDS_FOR_STD = 2
MIN_BINS_DEFAULT = 10
MIN_INTERPOLATION_POINTS = 2
MIN_POINTS_FOR_LOWESS = 10
MIN_SAVGOL_WINDOW = 11
SIGMA_MIN = 1e-2
SIGMA_MAX = 100.0


def extract_validation_subset(
    pd_df: pd.DataFrame, fold_col: str, label_col: str, num_cv_folds: int, min_val_size: int = 5000
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Extract validation subset and metadata for uncertainty model estimation.

    Step 1.1: Extract validation subset and metadata
    - Role in architecture: Data ingress for the uncertainty model; ensures leakage-free estimation.

    Parameters
    ----------
    pd_df : pd.DataFrame
        Full dataframe with fold information and per-fold predictions
    fold_col : str
        Column name for fold assignments (NaN indicates validation set)
    label_col : str
        Column name for true labels
    num_cv_folds : int
        Number of cross-validation folds
    min_val_size : int, optional
        Minimum required validation set size, by default 5000

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - val: DataFrame restricted to held-out rows with intact per-fold probabilities
        - metadata: Dict with validation info including available folds, counts, etc.

    Raises
    ------
    ValueError
        If validation set is too small or missing required columns
    """
    # Filter validation set (rows with NaN fold_id)
    val = pd_df[pd_df[fold_col].isna()].copy()

    # Check minimum size
    if len(val) < min_val_size:
        logger.warning(f"Validation set size ({len(val)}) is smaller than minimum recommended ({min_val_size})")

    # Check for required prob_fold columns
    prob_fold_cols = [f"prob_fold_{k}" for k in range(num_cv_folds)]
    missing_cols = [col for col in prob_fold_cols if col not in val.columns]
    if missing_cols:
        raise ValueError(f"Missing probability columns: {missing_cols}")

    # Check for all-NaN columns
    all_nan_cols = []
    available_folds = []
    for k in range(num_cv_folds):
        col = f"prob_fold_{k}"
        if val[col].isna().all():
            all_nan_cols.append(col)
        else:
            available_folds.append(k)

    if all_nan_cols:
        logger.warning(f"Columns with all NaN values: {all_nan_cols}")

    if not available_folds:
        raise ValueError("No valid probability columns found (all are NaN)")

    # Basic distribution statistics for diagnostics
    prob_stats = {}
    for k in available_folds:
        col = f"prob_fold_{k}"
        prob_values = val[col].dropna()
        prob_stats[f"fold_{k}"] = {
            "count": len(prob_values),
            "mean": float(prob_values.mean()),
            "std": float(prob_values.std()),
            "min": float(prob_values.min()),
            "max": float(prob_values.max()),
            "q25": float(prob_values.quantile(0.25)),
            "q75": float(prob_values.quantile(0.75)),
        }

    metadata = {
        "validation_size": len(val),
        "available_folds": available_folds,
        "missing_prob_cols": missing_cols,
        "all_nan_cols": all_nan_cols,
        "prob_statistics": prob_stats,
        "label_distribution": {
            "positive": int((val[label_col] == 1).sum()),
            "negative": int((val[label_col] == 0).sum()),
        },
    }

    logger.info(f"Extracted validation subset: {len(val)} rows, {len(available_folds)} valid folds")

    return val, metadata


def compute_per_fold_scores(  # noqa: C901
    val_df: pd.DataFrame,
    num_cv_folds: int,
    prob_to_phred_fn: Callable = prob_to_phred,
    prob_to_logit_fn: Callable = prob_to_logit,
    transform_mode: str = "mqual",
    p_min: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Compute per-fold scores and summary statistics.

    Step 1.2: Compute per-fold scores and summary statistics
    - Role in architecture: Produces the raw signal for smoothing (mean and std in mqual or logit space).

    Parameters
    ----------
    val_df : pd.DataFrame
        Validation dataframe from extract_validation_subset
    num_cv_folds : int
        Number of cross-validation folds
    prob_to_phred_fn : Callable
        Function to convert probabilities to Phred scores
    prob_to_logit_fn : Callable
        Function to convert probabilities to logit scores
    transform_mode : str, optional
        Transform mode: "mqual" or "logit", by default "mqual"
    p_min : float, optional
        Minimum probability for clipping, by default 1e-12

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
        - score_mean: Mean scores across folds
        - score_std: Standard deviation of scores across folds
        - diagnostics: Dictionary with diagnostic information

    Raises
    ------
    ValueError
        If transform_mode is not "mqual" or "logit"
    """
    if transform_mode not in ["mqual", "logit"]:
        raise ValueError(f"transform_mode must be 'mqual' or 'logit', got {transform_mode}")

    # Select appropriate transform function
    if transform_mode == "mqual":
        transform_fn = prob_to_phred_fn
        score_prefix = "mqual"
    else:
        transform_fn = prob_to_logit_fn
        score_prefix = "logit"

    val = val_df.copy()

    # Identify available folds
    available_folds = []
    for k in range(num_cv_folds):
        col = f"prob_fold_{k}"
        if col in val.columns and not val[col].isna().all():
            available_folds.append(k)

    if len(available_folds) < MIN_FOLDS_FOR_STD:
        raise ValueError(
            f"Need at least {MIN_FOLDS_FOR_STD} valid folds for std calculation, " f"got {len(available_folds)}"
        )

    # Apply numerical guardrails: clip probabilities to [p_min, 1-p_min]
    score_cols = []
    for k in available_folds:
        prob_col = f"prob_fold_{k}"
        score_col = f"{score_prefix}_{k}"

        # Clip probabilities
        probs_clipped = np.clip(val[prob_col], p_min, 1 - p_min)

        # Transform to scores
        val[score_col] = transform_fn(probs_clipped)
        score_cols.append(score_col)

    # Compute mean and std across folds
    score_mean_col = f"{score_prefix}_mean"
    score_std_col = f"{score_prefix}_std"

    val[score_mean_col] = val[score_cols].mean(axis=1)
    val[score_std_col] = val[score_cols].std(axis=1)

    score_mean = val[score_mean_col].to_numpy()
    score_std = val[score_std_col].to_numpy()

    # Generate diagnostics: quantiles of score_std per score_mean bins
    try:
        # Create bins for score_mean
        n_bins = min(50, len(score_mean) // 100)  # Adaptive number of bins
        n_bins = max(n_bins, MIN_BINS_DEFAULT)

        score_mean_bins = np.linspace(np.nanmin(score_mean), np.nanmax(score_mean), n_bins + 1)

        # Compute quantiles of std within each mean bin
        bin_diagnostics = {}
        for i in range(n_bins):
            mask = (score_mean >= score_mean_bins[i]) & (score_mean < score_mean_bins[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (score_mean >= score_mean_bins[i]) & (score_mean <= score_mean_bins[i + 1])

            if mask.sum() > 0:
                std_in_bin = score_std[mask]
                bin_diagnostics[f"bin_{i}"] = {
                    "mean_range": [float(score_mean_bins[i]), float(score_mean_bins[i + 1])],
                    "count": int(mask.sum()),
                    "std_q25": float(np.nanquantile(std_in_bin, 0.25)),
                    "std_median": float(np.nanquantile(std_in_bin, 0.5)),
                    "std_q75": float(np.nanquantile(std_in_bin, 0.75)),
                    "std_mean": float(np.nanmean(std_in_bin)),
                }
    except Exception as e:
        logger.warning(f"Failed to compute bin diagnostics: {e}")
        bin_diagnostics = {}

    diagnostics = {
        "transform_mode": transform_mode,
        "available_folds": available_folds,
        "num_valid_points": int((~np.isnan(score_mean)).sum()),
        "score_mean_stats": {
            "min": float(np.nanmin(score_mean)),
            "max": float(np.nanmax(score_mean)),
            "mean": float(np.nanmean(score_mean)),
            "std": float(np.nanstd(score_mean)),
        },
        "score_std_stats": {
            "min": float(np.nanmin(score_std)),
            "max": float(np.nanmax(score_std)),
            "mean": float(np.nanmean(score_std)),
            "std": float(np.nanstd(score_std)),
        },
        "bin_diagnostics": bin_diagnostics,
        "p_min_used": p_min,
    }

    logger.info(
        f"Computed {transform_mode} scores: mean range [{diagnostics['score_mean_stats']['min']:.3f}, "
        f"{diagnostics['score_mean_stats']['max']:.3f}], "
        f"std range [{diagnostics['score_std_stats']['min']:.3f}, "
        f"{diagnostics['score_std_stats']['max']:.3f}]"
    )

    return score_mean, score_std, diagnostics


def fit_score_std_lowess(  # noqa: C901, PLR0912, PLR0915
    score_mean: np.ndarray,
    score_std: np.ndarray,
    frac: float = 0.3,
    it: int = 3,
    delta: float = 0.0,
    tiny: float = 1e-10,
    *,
    use_log_transform: bool = True,
    use_density_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Fit smoother for std vs mean using LOWESS.

    Step 1.3: Fit smoother for std vs mean (LOWESS or alternative)
    - Role in architecture: Converts noisy per-point std into a smooth function usable by KDE/Bootstrap.

    Parameters
    ----------
    score_mean : np.ndarray
        X values (score means)
    score_std : np.ndarray
        Y values (score standard deviations)
    frac : float, optional
        Fraction of data to use for smoothing, by default 0.3
    it : int, optional
        Number of robust iterations, by default 3
    use_log_transform : bool, optional
        Whether to fit on log(std + tiny), by default True
    use_density_weights : bool, optional
        Whether to use density-based weights, by default True
    delta : float, optional
        Delta parameter for LOWESS, by default 0.0
    tiny : float, optional
        Small value for log transform stability, by default 1e-10

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
        - score_grid: Grid points for evaluation
        - std_smooth: Smoothed standard deviation values
        - quality_metrics: Dictionary with fit quality information
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(score_mean) | np.isnan(score_std))
    x = score_mean[valid_mask]
    y = score_std[valid_mask]

    if len(x) < MIN_POINTS_FOR_LOWESS:
        raise ValueError(f"Insufficient valid data points for LOWESS: {len(x)}")

    # Ensure positive std values
    y = np.maximum(y, tiny)

    # Optional log transform
    if use_log_transform:
        y_fit = np.log(y + tiny)
    else:
        y_fit = y.copy()

    # Compute optional density weights
    if use_density_weights:
        try:
            # Use kernel density estimation for weights
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(x)
            weights = kde(x)
            weights = weights / np.sum(weights) * len(weights)  # Normalize to sum to n
        except Exception as e:
            logger.warning(f"Failed to compute density weights, using uniform: {e}")
            weights = None
    else:
        weights = None

    # Sort data for LOWESS
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y_fit[sorted_indices]
    if weights is not None:
        weights_sorted = weights[sorted_indices]
    else:
        weights_sorted = None

    # Apply LOWESS
    try:
        if weights_sorted is not None:
            # statsmodels lowess doesn't directly support weights, so we'll use a workaround
            # by duplicating points according to weights (approximate)
            weights_int = np.round(weights_sorted * 10).astype(int) + 1
            x_weighted = np.repeat(x_sorted, weights_int)
            y_weighted = np.repeat(y_sorted, weights_int)
            lowess_result = smoothers_lowess.lowess(
                y_weighted, x_weighted, frac=frac, it=it, delta=delta, return_sorted=True
            )
        else:
            lowess_result = smoothers_lowess.lowess(
                y_sorted, x_sorted, frac=frac, it=it, delta=delta, return_sorted=True
            )

        score_grid = lowess_result[:, 0]
        std_smooth = lowess_result[:, 1]

    except Exception as e:
        logger.error(f"LOWESS fitting failed: {e}")
        # Fallback to simple interpolation
        score_grid = x_sorted
        std_smooth = y_sorted

    # Inverse transform if log was used
    if use_log_transform:
        std_smooth = np.exp(std_smooth) - tiny

    # Ensure positivity
    std_smooth = np.maximum(std_smooth, tiny)

    # Optional post-processing for smoothness (Savitzky-Golay filter)
    if len(std_smooth) > MIN_SAVGOL_WINDOW:  # Need at least window_length points
        try:
            std_smooth = savgol_filter(std_smooth, window_length=MIN_SAVGOL_WINDOW, polyorder=2)
            std_smooth = np.maximum(std_smooth, tiny)  # Ensure positivity after filtering
        except Exception as e:
            logger.warning(f"Savitzky-Golay filtering failed: {e}")

    # Compute quality metrics
    try:
        # Compute residuals on original scale
        if use_log_transform:
            y_pred_orig = np.interp(x, score_grid, std_smooth)
        else:
            y_pred_orig = np.interp(x, score_grid, std_smooth)

        residuals = y - y_pred_orig

        # R²-like metric
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_like = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Tail smoothness check: slope changes in the last 20% of the curve
        tail_start_idx = int(0.8 * len(std_smooth))
        if tail_start_idx < len(std_smooth) - 2:
            tail_diffs = np.diff(std_smooth[tail_start_idx:])
            tail_smoothness = np.std(tail_diffs) if len(tail_diffs) > 0 else 0
        else:
            tail_smoothness = 0

        quality_metrics = {
            "num_points": len(x),
            "frac_used": frac,
            "it_used": it,
            "use_log_transform": use_log_transform,
            "use_density_weights": use_density_weights,
            "residual_stats": {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
            },
            "r2_like": float(r2_like),
            "tail_smoothness": float(tail_smoothness),
            "grid_size": len(score_grid),
            "std_range": [float(np.min(std_smooth)), float(np.max(std_smooth))],
        }

    except Exception as e:
        logger.warning(f"Failed to compute quality metrics: {e}")
        quality_metrics = {
            "num_points": len(x),
            "grid_size": len(score_grid),
            "std_range": [float(np.min(std_smooth)), float(np.max(std_smooth))],
        }

    logger.info(
        f"LOWESS fit complete: {len(score_grid)} grid points, "
        f"std range [{quality_metrics['std_range'][0]:.4f}, {quality_metrics['std_range'][1]:.4f}]"
    )

    return score_grid, std_smooth, quality_metrics


def build_score_std_interpolator(
    score_grid: np.ndarray,
    std_smooth: np.ndarray,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
    fill_value: str = "extrapolate",
    *,
    bounds_error: bool = False,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """
    Build and persist the interpolation function.

    Step 1.4: Build and persist the interpolation function
    - Role in architecture: Provides a fast, vectorized callable (get_score_std) used by KDE and Bootstrap steps.

    Parameters
    ----------
    score_grid : np.ndarray
        Grid points from LOWESS fit
    std_smooth : np.ndarray
        Smoothed standard deviation values
    sigma_min : float, optional
        Minimum allowed standard deviation, by default 1e-6
    sigma_max : float, optional
        Maximum allowed standard deviation, by default 100.0
    bounds_error : bool, optional
        Whether to raise error for out-of-bounds values, by default False
    fill_value : str, optional
        How to handle out-of-bounds values, by default "extrapolate"

    Returns
    -------
    Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]
        - get_score_std: Interpolation function that takes score values and returns std
        - metadata: Dictionary with interpolator metadata
    """
    # Ensure inputs are sorted by score_grid
    sort_indices = np.argsort(score_grid)
    score_grid_sorted = score_grid[sort_indices]
    std_smooth_sorted = std_smooth[sort_indices]

    # Remove any duplicate x values (can cause interpolation issues)
    unique_mask = np.concatenate([[True], np.diff(score_grid_sorted) > 0])
    score_grid_unique = score_grid_sorted[unique_mask]
    std_smooth_unique = std_smooth_sorted[unique_mask]

    if len(score_grid_unique) < MIN_INTERPOLATION_POINTS:
        raise ValueError(f"Need at least {MIN_INTERPOLATION_POINTS} unique grid points for interpolation")

    # Handle fill_value for edge cases
    if fill_value == "extrapolate":
        fill_value_tuple = "extrapolate"
    else:
        # Use edge values for out-of-bounds
        fill_value_tuple = (std_smooth_unique[0], std_smooth_unique[-1])

    # Create interpolator
    try:
        interpolator = interpolate.interp1d(
            score_grid_unique,
            std_smooth_unique,
            kind="linear",
            bounds_error=bounds_error,
            fill_value=fill_value_tuple,
            assume_sorted=True,
        )
    except Exception as e:
        logger.error(f"Failed to create interpolator: {e}")
        raise

    def get_score_std(score: float | np.ndarray) -> float | np.ndarray:
        """
        Vectorized function to get standard deviation for given score(s).

        Parameters
        ----------
        score : Union[float, np.ndarray]
            Score value(s) to evaluate

        Returns
        -------
        Union[float, np.ndarray]
            Standard deviation value(s), clamped to [sigma_min, sigma_max]
        """
        # Ensure input is numpy array for vectorized operations
        score_arr = np.asarray(score)
        is_scalar = score_arr.ndim == 0

        if is_scalar:
            score_arr = score_arr.reshape(1)

        # Interpolate
        std_values = interpolator(score_arr)

        # Clamp to bounds
        std_values = np.clip(std_values, sigma_min, sigma_max)

        # Return scalar if input was scalar
        if is_scalar:
            return float(std_values[0])
        else:
            return std_values

    # Metadata for provenance
    metadata = {
        "grid_size": len(score_grid_unique),
        "score_range": [float(score_grid_unique[0]), float(score_grid_unique[-1])],
        "std_range": [float(std_smooth_unique.min()), float(std_smooth_unique.max())],
        "sigma_bounds": [sigma_min, sigma_max],
        "bounds_error": bounds_error,
        "fill_value": str(fill_value),
        "interpolation_kind": "linear",
    }

    logger.info(
        f"Built interpolator: score range [{metadata['score_range'][0]:.3f}, "
        f"{metadata['score_range'][1]:.3f}], std range [{metadata['std_range'][0]:.4f}, "
        f"{metadata['std_range'][1]:.4f}]"
    )

    return get_score_std, metadata


def create_uncertainty_function_pipeline_fast(  # noqa: PLR0913, PLR0915, C901
    pd_df: pd.DataFrame,
    fold_col: str,
    label_col: str,
    num_cv_folds: int,
    prob_to_phred_fn: Callable = prob_to_phred,
    prob_to_logit_fn: Callable = prob_to_logit,
    transform_mode: str = "mqual",
    lowess_frac: float = 0.3,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
    min_val_size: int = 5000,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """
    Fast bare-bones pipeline to create uncertainty function (optimized for production).

    This is a streamlined version of create_uncertainty_function_pipeline that skips
    most diagnostic computations to run 5-10x faster. Suitable for production use
    when detailed diagnostics are not needed.

    Parameters
    ----------
    pd_df : pd.DataFrame
        Full dataframe with fold information and per-fold predictions
    fold_col : str
        Column name for fold assignments
    label_col : str
        Column name for true labels
    num_cv_folds : int
        Number of cross-validation folds
    prob_to_phred_fn : Callable
        Function to convert probabilities to Phred scores
    prob_to_logit_fn : Callable
        Function to convert probabilities to logit scores
    transform_mode : str, optional
        Transform mode: "mqual" or "logit", by default "mqual"
    lowess_frac : float, optional
        LOWESS smoothing fraction, by default 0.3
    sigma_min : float, optional
        Minimum allowed standard deviation, by default 1e-6
    sigma_max : float, optional
        Maximum allowed standard deviation, by default 100.0
    min_val_size : int, optional
        Minimum validation set size, by default 5000

    Returns
    -------
    Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]
        - get_score_std: Function that maps score values to uncertainty estimates
        - minimal_metadata: Essential metadata only
    """
    # Step 1: Extract validation data (minimal)
    val_mask = pd_df[fold_col].isna()
    if val_mask.sum() < min_val_size:
        logger.warning(
            f"Insufficient validation data: {val_mask.sum()} < {min_val_size}. "
            f"Using constant std = 1.0 as fallback."
        )

        # Return constant function that always returns 1.0
        def get_score_std(score: float | np.ndarray) -> float | np.ndarray:
            score_arr = np.asarray(score)
            is_scalar = score_arr.ndim == 0

            if is_scalar:
                return 1.0
            else:
                return np.ones_like(score_arr, dtype=float)

        # Fallback metadata
        fallback_metadata = {
            "pipeline_version": "1.0-fast",
            "transform_mode": transform_mode,
            "validation_size": val_mask.sum(),
            "folds_used": 0,
            "score_range": [0.0, 0.0],
            "std_range": [1.0, 1.0],
            "grid_size": 0,
            "fallback_used": True,
            "fallback_reason": f"Insufficient validation data: {val_mask.sum()} < {min_val_size}",
            "constant_std_value": 1.0,
        }

        return get_score_std, fallback_metadata

    val_df = pd_df.loc[val_mask].copy()

    # Step 2: Compute scores quickly (no extensive diagnostics)
    prob_cols = [f"prob_fold_{k}" for k in range(num_cv_folds)]
    available_cols = [col for col in prob_cols if col in val_df.columns]

    if len(available_cols) == 0:
        raise ValueError("No probability columns found")

    # Convert probabilities to scores efficiently
    if transform_mode == "mqual":
        score_fn = prob_to_phred_fn
    else:  # logit
        score_fn = prob_to_logit_fn

    # Compute mean and std across folds
    score_arrays = []
    for col in available_cols:
        scores = score_fn(val_df[col].values)
        score_arrays.append(scores)

    score_matrix = np.column_stack(score_arrays)
    score_mean = np.nanmean(score_matrix, axis=1)
    score_std = np.nanstd(score_matrix, axis=1, ddof=1)

    # Remove NaN values
    valid_mask = ~(np.isnan(score_mean) | np.isnan(score_std))
    score_mean = score_mean[valid_mask]
    score_std = score_std[valid_mask]

    # Step 3: Fast LOWESS fit (minimal options)
    try:
        # Sort for LOWESS
        sort_idx = np.argsort(score_mean)
        x_sorted = score_mean[sort_idx]
        y_sorted = score_std[sort_idx]

        # Apply LOWESS without density weights or log transforms
        lowess_result = smoothers_lowess.lowess(y_sorted, x_sorted, frac=lowess_frac, it=0, return_sorted=True)

        score_grid = lowess_result[:, 0]
        std_smooth = lowess_result[:, 1]

        # Ensure positivity
        std_smooth = np.maximum(std_smooth, sigma_min)

    except Exception as e:
        logger.warning(f"LOWESS failed, using linear interpolation: {e}")
        sort_idx = np.argsort(score_mean)
        score_grid = score_mean[sort_idx]
        std_smooth = score_std[sort_idx]

    # Step 4: Fast interpolator
    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(score_grid) > 0])
    score_grid = score_grid[unique_mask]
    std_smooth = std_smooth[unique_mask]

    if len(score_grid) < MIN_INTERPOLATION_POINTS:
        raise ValueError("Insufficient unique grid points for interpolation")

    # Create interpolator
    interpolator = interpolate.interp1d(
        score_grid, std_smooth, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True
    )

    def get_score_std(score: float | np.ndarray) -> float | np.ndarray:
        score_arr = np.asarray(score)
        is_scalar = score_arr.ndim == 0

        if is_scalar:
            score_arr = score_arr.reshape(1)

        std_values = np.clip(interpolator(score_arr), sigma_min, sigma_max)

        return float(std_values[0]) if is_scalar else std_values

    # Minimal metadata
    minimal_metadata = {
        "pipeline_version": "1.0-fast",
        "transform_mode": transform_mode,
        "validation_size": len(score_mean),
        "folds_used": len(available_cols),
        "score_range": [float(score_grid[0]), float(score_grid[-1])],
        "std_range": [float(std_smooth.min()), float(std_smooth.max())],
        "grid_size": len(score_grid),
        "fallback_used": False,
    }

    return get_score_std, minimal_metadata


def create_uncertainty_function_pipeline(  # noqa: PLR0913
    pd_df: pd.DataFrame,
    fold_col: str,
    label_col: str,
    num_cv_folds: int,
    prob_to_phred_fn: Callable = prob_to_phred,
    prob_to_logit_fn: Callable = prob_to_logit,
    transform_mode: str = "mqual",
    lowess_frac: float = 0.3,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
    min_val_size: int = 5000,
    **lowess_kwargs,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """
    Complete pipeline to create uncertainty function from cross-validation predictions.

    Combines Steps 1.1-1.4 into a single convenient function.

    Parameters
    ----------
    pd_df : pd.DataFrame
        Full dataframe with fold information and per-fold predictions
    fold_col : str
        Column name for fold assignments
    label_col : str
        Column name for true labels
    num_cv_folds : int
        Number of cross-validation folds
    prob_to_phred_fn : Callable
        Function to convert probabilities to Phred scores
    prob_to_logit_fn : Callable
        Function to convert probabilities to logit scores
    transform_mode : str, optional
        Transform mode: "mqual" or "logit", by default "mqual"
    lowess_frac : float, optional
        LOWESS smoothing fraction, by default 0.3
    sigma_min : float, optional
        Minimum allowed standard deviation, by default SIGMA_MIN
    sigma_max : float, optional
        Maximum allowed standard deviation, by default SIGMA_MAX
    min_val_size : int, optional
        Minimum validation set size, by default 5000
    **lowess_kwargs
        Additional arguments for LOWESS fitting

    Returns
    -------
    Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]
        - get_score_std: Function that maps score values to uncertainty estimates
        - full_metadata: Complete metadata from all pipeline steps
    """
    logger.info(f"Starting uncertainty function pipeline in {transform_mode} mode")

    # Step 1.1: Extract validation subset
    val_df, val_metadata = extract_validation_subset(pd_df, fold_col, label_col, num_cv_folds, min_val_size)

    # Step 1.2: Compute per-fold scores
    score_mean, score_std, score_diagnostics = compute_per_fold_scores(
        val_df, num_cv_folds, prob_to_phred_fn, prob_to_logit_fn, transform_mode
    )

    # Step 1.3: Fit LOWESS smoother
    score_grid, std_smooth, lowess_quality = fit_score_std_lowess(
        score_mean, score_std, frac=lowess_frac, **lowess_kwargs
    )

    # Step 1.4: Build interpolator
    get_score_std, interp_metadata = build_score_std_interpolator(score_grid, std_smooth, sigma_min, sigma_max)

    # Combine all metadata
    full_metadata = {
        "pipeline_version": "1.0",
        "transform_mode": transform_mode,
        "validation_metadata": val_metadata,
        "score_diagnostics": score_diagnostics,
        "lowess_quality": lowess_quality,
        "interpolator_metadata": interp_metadata,
        "pipeline_config": {
            "lowess_frac": lowess_frac,
            "sigma_bounds": [sigma_min, sigma_max],
            "min_val_size": min_val_size,
        },
    }

    logger.info("Uncertainty function pipeline completed successfully")

    return get_score_std, full_metadata


def make_grid_and_transform(  # noqa: C901, PLR0912, PLR0915
    data_values: np.ndarray,
    grid_size: int = 8192,
    transform_mode: str = "mqual",
    padding_factor: float = 0.1,
) -> tuple[np.ndarray, float, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """
    Step 2.1: Grid definition and coordinate transforms.

    Establish a common numerical mesh used by KDE and Bootstrap to make all heavy ops O(G log G) after first binning.

    Parameters
    ----------
    data_values : np.ndarray
        Input data values to determine domain. Assumed to be in the correct scale (phred for mqual, logits for logit).
    grid_size : int, optional
        Number of grid points, by default 8192
    mqual_max_lut : float, optional
        Maximum mqual value for lookup table. If None, inferred from data
    transform_mode : str, optional
        Coordinate system: "mqual" (default) or "logit", by default "mqual"
    padding_factor : float, optional
        Fraction of range to add as padding on each side, by default 0.1

    Returns
    -------
    tuple
        - grid: 1D array of grid points in grid space
        - dx: grid spacing
        - to_grid_space: function to convert data to grid coordinates
        - from_grid_space: function to convert grid coordinates back to data space
        - metadata: dict with grid information

    Raises
    ------
    ValueError
        If transform_mode is not supported, input_type is invalid, or data is invalid
    """
    if len(data_values) == 0:
        raise ValueError("Cannot create grid from empty data")

    data_values = np.asarray(data_values)
    data_values = data_values[np.isfinite(data_values)]

    if len(data_values) == 0:
        raise ValueError("No finite values in data")

    # Validate transform_mode parameter
    if transform_mode not in {"mqual", "logit"}:
        raise ValueError(f"transform_mode must be 'mqual' or 'logit', got '{transform_mode}'")

    # Validate data type
    if np.all((data_values >= -EPS) & (data_values <= 1 + EPS)):
        # Data looks like probabilities
        logger.warning(f"Data values look like probabilities in [0,1]; while {transform_mode=}")
    elif np.all(data_values >= -EPS) and (transform_mode != "mqual"):
        # Data looks like MQUAL/Phred scores
        logger.warning(f"Data values look like MQUAL/Phred scores >= 0; while {transform_mode=}")
    elif np.any(data_values < -EPS) and (transform_mode != "logit"):
        # Data looks like logits
        logger.warning(f"Data values look like logit scores (can be negative); while {transform_mode=}")

    # Determine data range and add padding
    data_min = 0.0 if transform_mode == "mqual" else np.min(data_values)  # mqual >= 0 always
    data_max = np.max(data_values)
    range_size = data_max - data_min
    grid_min = data_min
    grid_max = data_max + padding_factor * range_size

    if transform_mode == "mqual":
        # mqual-mode: grid over [0, mqual_max_lut], reflection at 0, no upper boundary. Padding only on the upper side
        boundary_policy = "reflect_at_zero"
    else:  # logit
        # logit-mode: unbounded both sides, symmetric padding
        grid_min = grid_min - padding_factor * range_size
        boundary_policy = "none"

    # Create the grid
    grid = np.arange(grid_size, dtype=float)
    dx = (grid_max - grid_min) / (grid_size - 1)

    def to_grid_space(x):
        """Convert mqual values to grid coordinates [0, grid_size-1]"""
        x = np.asarray(x)
        return (x - grid_min) / dx

    def from_grid_space(u):
        """Convert grid coordinates back to mqual values"""
        u = np.asarray(u)
        return u * dx + grid_min

    metadata = {
        "grid_size": grid_size,
        "dx": dx,
        "domain": [grid_min, grid_max],
        "data_range": [np.min(data_values), np.max(data_values)],
        "transform_mode": transform_mode,
        "boundary_policy": boundary_policy,
        "padding_factor": padding_factor,
    }

    logger.debug(
        f"Created {transform_mode} grid: size={grid_size}, dx={dx:.6f}, domain=[{grid_min:.3f}, {grid_max:.3f}]"
    )

    return grid, dx, to_grid_space, from_grid_space, metadata


def bin_data_to_grid(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    grid_size: int = 8192,
    to_grid_space: Callable[[np.ndarray], np.ndarray] | None = None,
    boundary_policy: str = "clamp",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Step 2.2: Fast linear binning (like smooth_round_to_grid).

    Convert large datasets to grid counts efficiently; cornerstone for all subsequent convolutions.
    Generalizes relplot.kernels.smooth_round_to_grid logic.

    ⚠️  IMPORTANT: grid_size must match the grid size used in make_grid_and_transform()
    to avoid broadcasting errors in downstream KDE functions!

    Parameters
    ----------
    values : np.ndarray
        Values to bin (in original data space)
    weights : np.ndarray, optional
        Weights for each value. If None, uniform weights of 1.0 are used
    grid_size : int, optional
        Number of grid bins, by default 8192
        ⚠️  This MUST match the grid_size used in make_grid_and_transform()!
    to_grid_space : Callable, optional
        Function to convert values to grid coordinates. If None, assumes values are already in [0, 1]
    boundary_policy : str, optional
        How to handle out-of-range values: "clamp" or "drop", by default "clamp"

    Returns
    -------
    tuple
        - counts: array of length grid_size with binned weights
        - metadata: dict with binning statistics

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    values = np.asarray(values, dtype=float)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")

    if len(values) == 0:
        return np.zeros(grid_size), {"total_weight": 0.0, "n_values": 0, "n_dropped": 0}

    # Convert to grid coordinates
    if to_grid_space is not None:
        grid_coords = to_grid_space(values)
    else:
        # Assume values are in [0, 1] and scale to grid
        grid_coords = values * (grid_size - 1)

    # Handle boundary conditions
    original_length = len(grid_coords)
    if boundary_policy == "clamp":
        grid_coords = np.clip(grid_coords, 0, grid_size - 1 - 1e-10)  # Small epsilon to avoid out-of-bounds
        valid_mask = np.ones(len(grid_coords), dtype=bool)
    elif boundary_policy == "drop":
        valid_mask = (grid_coords >= 0) & (grid_coords < grid_size - 1e-10)
        grid_coords = grid_coords[valid_mask]
        weights = weights[valid_mask]
    else:
        raise ValueError(f"Unknown boundary_policy: {boundary_policy}")

    # Fast linear binning using np.add.at
    counts = np.zeros(grid_size)

    if len(grid_coords) > 0:
        # Compute left bin indices and fractional parts
        left_bins = np.floor(grid_coords).astype(int)
        fractions = grid_coords - left_bins

        # Ensure indices are in bounds
        left_bins = np.clip(left_bins, 0, grid_size - 2)
        right_bins = left_bins + 1

        # Distribute weights linearly between adjacent bins
        np.add.at(counts, left_bins, (1 - fractions) * weights)
        np.add.at(counts, right_bins, fractions * weights)

    # Compute statistics
    n_dropped = original_length - len(grid_coords) if boundary_policy == "drop" else 0
    total_weight_in = np.sum(weights) if len(weights) > 0 else 0.0
    total_weight_out = np.sum(counts)

    metadata = {
        "total_weight_in": total_weight_in,
        "total_weight_out": total_weight_out,
        "weight_conservation_error": abs(total_weight_in - total_weight_out),
        "n_values": len(values),
        "n_dropped": n_dropped,
        "boundary_policy": boundary_policy,
    }

    logger.debug(
        f"Binned {len(values)} values to grid, weight conservation error: {metadata['weight_conservation_error']:.2e}"
    )

    return counts, metadata


def create_gaussian_kernel(
    sigma: float,
    dx: float,
    radius_k: float = 4.0,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Step 2.3: Kernel construction and boundary handling.

    Build a truncated Gaussian kernel for convolution operations.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian in data units
    dx : float
        Grid spacing in data units
    radius_k : float, optional
        Truncation radius in units of sigma, by default 4.0
    normalize : bool, optional
        Whether to normalize kernel to unit sum, by default True

    Returns
    -------
    tuple
        - kernel: 1D array with Gaussian samples centered at middle index
        - metadata: dict with kernel information
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    if dx <= 0:
        raise ValueError("dx must be positive")

    # Convert sigma to grid units
    sigma_grid = sigma / dx

    # Determine kernel support
    radius_grid = radius_k * sigma_grid
    half_support = int(np.ceil(radius_grid))

    # Build kernel centered at zero
    kernel_indices = np.arange(-half_support, half_support + 1)
    kernel = np.exp(-0.5 * (kernel_indices / sigma_grid) ** 2)

    # Normalize to unit sum if requested
    if normalize:
        kernel = kernel / np.sum(kernel)

    metadata = {
        "sigma": sigma,
        "sigma_grid": sigma_grid,
        "dx": dx,
        "radius_k": radius_k,
        "half_support": half_support,
        "kernel_length": len(kernel),
        "kernel_sum": np.sum(kernel),
    }

    logger.debug(f"Created Gaussian kernel: sigma={sigma:.3f}, length={len(kernel)}, sum={np.sum(kernel):.6f}")

    return kernel, metadata


def create_exponential_kernel(
    sigma: float,
    dx: float,
    radius_k: float = 8.0,
    gcos: float = 1.0,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build a truncated exponential (Laplace) kernel for convolution operations.

    The exponential kernel has heavier tails than Gaussian and is useful for data with
    outliers or when less aggressive smoothing in the tails is desired. The kernel
    follows the form: K(x) = exp( -x**2 / (sigma(gcos*sigma + np.abs(x))) ).

    Parameters
    ----------
    sigma : float
        Scale parameter of the exponential distribution in data units.
        For a Laplace distribution, this is the scale parameter b where the
        standard deviation is sqrt(2) * b.
    dx : float
        Grid spacing in data units
    radius_k : float, optional
        Truncation radius in units of scale, by default 6.0.
        Note: exponential tails decay slower than Gaussian, so a larger
        radius is typically needed for good approximation.
    gcos : float, optional
        Gaussian crossover scale, measured in units of sigma. For x << gcos*sigma
        the kernel behaves like a Gaussian, while for x >> gcos it behaves like an exponential.
    normalize : bool, optional
        Whether to normalize kernel to unit sum, by default True

    Returns
    -------
    tuple
        - kernel: 1D array with exponential samples centered at middle index
        - metadata: dict with kernel information

    Notes
    -----
    When gcos=0, the exponential kernel has the form K(x) = exp(-|x| / scale) and corresponds
    to a Laplace distribution. The relationship to standard deviation is:
    std_dev = sqrt(2) * scale, so scale = std_dev / sqrt(2).

    Compared to Gaussian kernels, exponential kernels:
    - Have heavier tails (slower decay)
    - Are less smooth (not infinitely differentiable)
    - Preserve sharp features better
    - Are more robust to outliers
    """
    if sigma <= 0:
        raise ValueError("Scale must be positive")
    if dx <= 0:
        raise ValueError("dx must be positive")
    if gcos < 0:
        raise ValueError("gcos must be non-negative")

    # Convert scale to grid units
    scale_grid = sigma / dx

    # Determine kernel support
    radius_grid = radius_k * scale_grid
    half_support = int(np.ceil(radius_grid))

    # Build kernel centered at zero
    kernel_indices = np.arange(-half_support, half_support + 1)
    kernel = np.exp(-(kernel_indices**2) / (scale_grid * (gcos * scale_grid + np.abs(kernel_indices))))

    # Normalize to unit sum if requested
    if normalize:
        kernel = kernel / np.sum(kernel)

    # Calculate equivalent standard deviation for comparison with Gaussian kernels
    equivalent_sigma = sigma * np.sqrt(2)

    metadata = {
        "sigma": sigma,
        "scale_grid": scale_grid,
        "equivalent_sigma": equivalent_sigma,
        "dx": dx,
        "radius_k": radius_k,
        "gcos": gcos,
        "half_support": half_support,
        "kernel_length": len(kernel),
        "kernel_sum": np.sum(kernel),
        "kernel_type": "exponential",
    }

    logger.debug(
        f"Created exponential kernel: scale={sigma:.3f}, equiv_sigma={equivalent_sigma:.3f}, "
        f"length={len(kernel)}, sum={np.sum(kernel):.6f}"
    )

    return kernel, metadata


def create_kernel_bank(
    sigma_values: list[float],
    dx: float,
    radius_k: float = 4.0,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Create a bank of Gaussian kernels for different sigma values.

    Parameters
    ----------
    sigma_values : list[float]
        List of standard deviations
    dx : float
        Grid spacing
    radius_k : float, optional
        Truncation radius in units of sigma, by default 4.0

    Returns
    -------
    tuple
        - kernels: list of kernel arrays
        - metadata: dict with bank information
    """
    kernels = []
    kernel_metadata = []

    for sigma in sigma_values:
        kernel, meta = create_gaussian_kernel(sigma, dx, radius_k)
        kernels.append(kernel)
        kernel_metadata.append(meta)

    metadata = {
        "n_kernels": len(kernels),
        "sigma_values": sigma_values,
        "dx": dx,
        "radius_k": radius_k,
        "kernel_metadata": kernel_metadata,
    }

    return kernels, metadata


def fft_convolve(
    signal: np.ndarray,
    kernel: np.ndarray,
    boundary_policy: str = "none",
    mode: str = "same",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Step 2.4: FFT-based convolution utilities.

    Perform fast smoothing at scale using FFT. Supports boundary reflection for mqual mode.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to convolve
    kernel : np.ndarray
        Convolution kernel (should be centered)
    boundary_policy : str, optional
        Boundary handling: "none", "reflect_at_zero", by default "none"
    mode : str, optional
        Convolution mode: "same", "full", by default "same"

    Returns
    -------
    tuple
        - result: convolved signal
        - metadata: dict with convolution information
    """
    signal = np.asarray(signal)
    kernel = np.asarray(kernel)

    if len(signal) == 0 or len(kernel) == 0:
        return np.zeros_like(signal), {"method": "empty"}

    # For very small arrays, use direct convolution
    if len(signal) * len(kernel) < FFT_THRESHOLD:
        if boundary_policy == "reflect_at_zero":
            result = _convolve_reflect_at_zero_direct(signal, kernel, mode)
            method = "direct_with_reflection"
        else:
            result = np.convolve(signal, kernel, mode)
            method = "direct"
    # Use FFT convolution
    elif boundary_policy == "reflect_at_zero":
        result = _fft_convolve_reflect_at_zero(signal, kernel, mode)
        method = "fft_with_reflection"
    else:
        result = _fft_convolve_plain(signal, kernel, mode)
        method = "fft"

    metadata = {
        "method": method,
        "signal_length": len(signal),
        "kernel_length": len(kernel),
        "boundary_policy": boundary_policy,
        "mode": mode,
    }

    return result, metadata


def _fft_convolve_plain(signal: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """FFT convolution without boundary handling."""
    # Determine output size
    if mode == "same":
        output_size = len(signal)
    elif mode == "full":
        output_size = len(signal) + len(kernel) - 1
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Pad to avoid circular convolution
    padded_size = len(signal) + len(kernel) - 1

    # FFT convolution
    signal_fft = np.fft.rfft(signal, n=padded_size)
    kernel_fft = np.fft.rfft(kernel, n=padded_size)
    result_fft = signal_fft * kernel_fft
    result = np.fft.irfft(result_fft, n=padded_size)

    # Trim to desired output size
    if mode == "same":
        # Center the output
        start = (len(result) - output_size) // 2
        result = result[start : start + output_size]

    return result


def _fft_convolve_reflect_at_zero(signal: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """FFT convolution with reflection at zero boundary (for mqual mode)."""
    # Create reflected signal: [signal[1:] reversed, signal, signal[:-1] reversed]
    if len(signal) > 1:
        left_reflect = np.flip(signal[1:])  # Don't duplicate the zero point
        right_reflect = np.flip(signal[:-1])  # Don't duplicate the last point
        extended_signal = np.concatenate([left_reflect, signal, right_reflect])
    else:
        extended_signal = signal

    # Convolve extended signal
    extended_result = _fft_convolve_plain(extended_signal, kernel, "same")

    # Extract the central part corresponding to original signal
    start = len(signal) - 1 if len(signal) > 1 else 0
    end = start + len(signal)
    result = extended_result[start:end]

    return result


def _convolve_reflect_at_zero_direct(signal: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """Direct convolution with reflection at zero boundary."""
    # Create reflected signal similar to FFT version
    if len(signal) > 1:
        left_reflect = np.flip(signal[1:])
        right_reflect = np.flip(signal[:-1])
        extended_signal = np.concatenate([left_reflect, signal, right_reflect])
    else:
        extended_signal = signal

    # Direct convolution
    extended_result = np.convolve(extended_signal, kernel, "same")

    # Extract central part
    start = len(signal) - 1 if len(signal) > 1 else 0
    end = start + len(signal)
    result = extended_result[start:end]

    return result


def cumulative_sum_from_grid(
    counts: np.ndarray,
    *,
    reverse: bool = True,
    normalize_to: float | None = None,
    eps: float = 1e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Step 2.5: Cumulative sums and survival/rate utilities.

    Turn smoothed densities into survival functions and rates (TPR/FPR) efficiently.

    Parameters
    ----------
    counts : np.ndarray
        Density or count values on grid
    reverse : bool, optional
        If True, compute reverse cumulative sum (for ≥-threshold survival), by default True
    normalize_to : float, optional
        If provided, normalize the result so that the total sums to this value
    eps : float, optional
        Small value to add for numerical stability, by default 1e-10

    Returns
    -------
    tuple
        - survival: cumulative sum array
        - metadata: dict with statistics
    """
    counts = np.asarray(counts)

    if len(counts) == 0:
        return np.array([]), {"total": 0.0, "reverse": reverse}

    # Compute cumulative sum
    if reverse:
        # Reverse cumulative sum: survival[i] = sum(counts[i:])
        survival = np.cumsum(counts[::-1])[::-1]
    else:
        # Forward cumulative sum: survival[i] = sum(counts[:i+1])
        survival = np.cumsum(counts)

    # Add small epsilon for numerical stability
    survival = survival + eps

    # Normalize if requested
    if normalize_to is not None:
        total = survival[0] if reverse else survival[-1]
        if total > eps:
            survival = survival * (normalize_to / total)

    metadata = {
        "total": float(survival[0] if reverse else survival[-1]),
        "reverse": reverse,
        "normalized_to": normalize_to,
        "eps_added": eps,
    }

    return survival, metadata


def interpolate_grid_to_points(
    grid_values: np.ndarray,
    grid: np.ndarray,
    target_points: np.ndarray,
    from_grid_space: Callable[[np.ndarray], np.ndarray],
    to_grid_space: Callable[[np.ndarray], np.ndarray],
    fill_value: str | float = "extrapolate",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Interpolate values from grid to arbitrary target points.

    Parameters
    ----------
    grid_values : np.ndarray
        Values defined on the grid
    grid : np.ndarray
        Grid coordinates
    target_points : np.ndarray
        Points where to evaluate (in data space)
    from_grid_space : Callable
        Function to convert grid coordinates to data space
    to_grid_space : Callable
        Function to convert data coordinates to grid space
    fill_value : str or float, optional
        How to handle extrapolation, by default "extrapolate"

    Returns
    -------
    tuple
        - interpolated_values: values at target points
        - metadata: dict with interpolation info
    """
    # Convert grid to data space for interpolation
    grid_data_coords = from_grid_space(grid)

    # Create interpolator
    interpolator = interpolate.interp1d(
        grid_data_coords,
        grid_values,
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,
    )

    # Interpolate to target points
    result = interpolator(target_points)

    metadata = {
        "n_grid_points": len(grid),
        "n_target_points": len(target_points),
        "fill_value": fill_value,
        "grid_range": [float(np.min(grid_data_coords)), float(np.max(grid_data_coords))],
        "target_range": [float(np.min(target_points)), float(np.max(target_points))],
    }

    return result, metadata


# ─────────────────────────────── Step 3: Adaptive KDE ───────────────────────────────


def weighted_quantile(
    values: np.ndarray,
    quantiles: float | np.ndarray,
    weights: np.ndarray | None = None,
    *,
    values_sorted: bool = False,
) -> float | np.ndarray:
    """
    Calculate weighted quantiles of an array.

    For the qth weighted quantile, finds x such that:
    weights[values < x].sum() / weights.sum() = q

    Parameters
    ----------
    values : np.ndarray
        Input array of values
    quantiles : float or np.ndarray
        Quantile(s) to compute (in range [0, 1])
    weights : np.ndarray, optional
        Weights for each value. If None, uses uniform weights (equivalent
        to np.quantile)
    values_sorted : bool, optional
        If True, assumes values are already sorted (with matching weights).
        By default False

    Returns
    -------
    Union[float, np.ndarray]
        Weighted quantile value(s)

    Examples
    --------
    >>> values = np.array([1, 2, 3, 4, 5])
    >>> weights = np.array([1, 1, 1, 1, 1])
    >>> weighted_quantile(values, 0.5, weights)  # Median
    3.0
    >>> weights = np.array([0, 0, 1, 0, 0])
    >>> weighted_quantile(values, 0.5, weights)  # All weight on value 3
    3.0
    """
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    is_scalar = quantiles.ndim == 0

    if is_scalar:
        quantiles = quantiles.reshape(1)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights)

    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")

    if len(values) == 0:
        return np.nan if is_scalar else np.full_like(quantiles, np.nan)

    # Remove NaN values
    valid_mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[valid_mask]
    weights = weights[valid_mask]

    if len(values) == 0:
        return np.nan if is_scalar else np.full_like(quantiles, np.nan)

    # Sort values and weights together
    if not values_sorted:
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        weights = weights[sort_idx]

    # Compute cumulative sum of weights
    weighted_cumsum = np.cumsum(weights)
    total_weight = weighted_cumsum[-1]

    if total_weight <= 0:
        return np.nan if is_scalar else np.full_like(quantiles, np.nan)

    # Normalize to [0, 1]
    weighted_cumsum_norm = weighted_cumsum / total_weight

    # Interpolate to find quantile values
    result = np.interp(quantiles, weighted_cumsum_norm, values)

    return float(result[0]) if is_scalar else result


def truncate_density_tails(  # noqa: PLR0912
    density: np.ndarray,
    grid: np.ndarray | None = None,
    from_grid_space: Callable[[np.ndarray], np.ndarray] | None = None,
    truncation_mode: str = "auto_detect",
    machine_precision_threshold: float = 1e-14,
    counts_array: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply tail truncation to a density array.

    Parameters
    ----------
    density : np.ndarray
        Input density values on grid
    grid : np.ndarray, optional
        Grid coordinates (in grid space), needed for some truncation modes
    from_grid_space : Callable, optional
        Function to convert from grid space to original space
    truncation_mode : str, optional
        How to handle tail truncation:
        - "none": no truncation (current behavior)
        - "data_max": truncate at maximum of original data (requires counts_array)
        - "auto_detect": truncate where density drops to machine precision (default)
    machine_precision_threshold : float, optional
        Threshold for detecting machine precision drops
    counts_array : np.ndarray, optional
        Original counts array (before smoothing) needed for data_max mode

    Returns
    -------
    tuple
        - truncated_density: density array (possibly truncated)
        - metadata: truncation information including truncation points
    """
    # Step 1: Determine truncation points for metadata (always calculated)
    truncation_info = {}

    if grid is not None and from_grid_space is not None:
        # Convert grid to original score space for truncation analysis
        score_values = from_grid_space(grid)

        # For data_max mode, find the maximum from original counts
        if truncation_mode == "data_max" and counts_array is not None:
            # Find the index of the last non-zero count
            nonzero_indices = np.where(counts_array > 0)[0]
            if len(nonzero_indices) > 0:
                data_max_idx = nonzero_indices[-1]
                data_max_score = float(score_values[data_max_idx])
            else:
                # Fallback to grid maximum if no non-zero counts
                data_max_score = float(np.max(score_values))
        else:
            # For other modes, use the grid maximum
            data_max_score = float(np.max(score_values))

        # Find auto-detection point where density drops to machine precision
        valid_mask = density > machine_precision_threshold

        if np.any(valid_mask):
            last_valid_idx = np.where(valid_mask)[0][-1]
            auto_detect_score = float(score_values[last_valid_idx])
        else:
            # If no valid points, use the maximum
            auto_detect_score = data_max_score

        truncation_info = {
            "data_max_score": data_max_score,
            "auto_detect_score": auto_detect_score,
            "machine_precision_threshold": machine_precision_threshold,
        }
    else:
        # No grid information available, set to None
        truncation_info = {
            "data_max_score": None,
            "auto_detect_score": None,
            "machine_precision_threshold": machine_precision_threshold,
        }

    # Step 2: Apply truncation based on mode (by zeroing out, not changing array length)
    truncated_density = density.copy()  # Start with full array

    if truncation_mode == "none":
        # No truncation - use full array as-is
        truncation_applied = False
        truncation_idx = len(density)

    elif (
        truncation_mode == "data_max" and grid is not None and from_grid_space is not None and counts_array is not None
    ):
        # Truncate at maximum of original data (from counts)
        score_values = from_grid_space(grid)
        nonzero_indices = np.where(counts_array > 0)[0]

        if len(nonzero_indices) > 0:
            data_max_idx = nonzero_indices[-1]
            truncation_idx = data_max_idx + 1  # Include the data_max point
            # Zero out everything beyond the truncation point
            truncated_density[truncation_idx:] = 0.0
            truncation_applied = True
        else:
            # No non-zero counts, don't truncate
            truncation_applied = False
            truncation_idx = len(density)

    elif truncation_mode == "auto_detect":
        # Truncate where density drops to machine precision
        valid_mask = density > machine_precision_threshold

        if np.any(valid_mask):
            truncation_idx = np.where(valid_mask)[0][-1] + 1  # Include the last valid point
            # Zero out everything beyond the truncation point
            truncated_density[truncation_idx:] = 0.0
            truncation_applied = True
        else:
            # If no valid points found, don't truncate
            truncation_applied = False
            truncation_idx = len(density)
    else:
        # Fallback to no truncation
        truncation_applied = False
        truncation_idx = len(density)

    metadata = {
        "truncation_mode": truncation_mode,
        "truncation_applied": truncation_applied,
        "truncation_idx": truncation_idx,
        "original_length": len(density),
        **truncation_info,
    }

    return truncated_density, metadata


def calculate_tpr_fpr_from_densities(
    density_true: np.ndarray,
    density_false: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Calculate TPR and FPR curves from smoothed densities.

    Parameters
    ----------
    density_true : np.ndarray
        Smoothed probability density for true positives
    density_false : np.ndarray
        Smoothed probability density for false positives
    eps : float, optional
        Numerical floor to avoid division by zero

    Returns
    -------
    tuple
        - tpr: True positive rate array
        - fpr: False positive rate array
        - metadata: calculation information
    """
    # Calculate survival functions (reverse cumulative sums)
    s_true, _ = cumulative_sum_from_grid(density_true, reverse=True)
    s_false, _ = cumulative_sum_from_grid(density_false, reverse=True)

    # Calculate rates by normalizing to total density
    tpr = s_true / max(density_true.sum(), eps)  # True positive rate
    fpr = s_false / max(density_false.sum(), eps)  # False positive rate

    # Apply numerical floor
    tpr = np.maximum(tpr, eps)
    fpr = np.maximum(fpr, eps)

    metadata = {
        "tpr_range": [float(np.min(tpr)), float(np.max(tpr))],
        "fpr_range": [float(np.min(fpr)), float(np.max(fpr))],
        "eps_floor": eps,
        "density_true_sum": float(density_true.sum()),
        "density_false_sum": float(density_false.sum()),
    }

    return tpr, fpr, metadata


def calculate_smoothed_precision_kde(  # noqa: PLR0913
    counts_true: np.ndarray,
    counts_false: np.ndarray,
    grid: np.ndarray,
    dx: float,
    get_score_std: Callable[[np.ndarray], np.ndarray],
    from_grid_space: Callable[[np.ndarray], np.ndarray],
    n_true: int,
    n_false: int,
    *,
    num_bandwidth_levels: int = 5,
    enforce_monotonic: bool = True,
    boundary_policy: str = "reflect_at_zero",
    truncation_mode: str = "auto_detect",
    machine_precision_threshold: float = 1e-14,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Main utility for adaptive KDE-based precision estimation.

    Step 3.2: Main Utility for calculate_smoothed_precision_kde
    - Role: Primary, self-contained, reusable function for adaptive KDE method
    - Knows nothing about SNVQ, only about smoothing densities to produce precision curve

    Parameters
    ----------
    counts_true : np.ndarray
        Binned counts for true positives on grid
    counts_false : np.ndarray
        Binned counts for false positives on grid
    grid : np.ndarray
        Grid coordinates (in grid space)
    dx : float
        Grid spacing
    get_score_std : Callable
        Function mapping grid coordinates to uncertainty estimates
    from_grid_space : Callable
        Function to convert grid coordinates to data space
    n_true : int
        Total number of true positives
    n_false : int
        Total number of false positives
    num_bandwidth_levels : int, optional
        Number of bandwidth levels for kernel mixing, by default 5
    enforce_monotonic : bool, optional
        Whether to apply isotonic regression, by default True
    boundary_policy : str, optional
        Boundary handling policy, by default "reflect_at_zero"
    truncation_mode : str, optional
        How to handle tail truncation:
        - "none": no truncation (current behavior)
        - "data_max": truncate at maximum of original data
        - "auto_detect": truncate where density drops to machine precision (default)
    machine_precision_threshold : float, optional
        Threshold for detecting machine precision drops, by default 1e-14

    Returns
    -------
    tuple
        - precision_grid: 1D array of precision values on input grid
        - metadata: dict with processing info and diagnostics
    """
    # Step 3.3: Variable bandwidth approximation via kernel mixing
    density_true, density_false, mixing_metadata = _variable_bandwidth_kernel_mixing(
        counts_true,
        counts_false,
        grid,
        dx,
        get_score_std,
        from_grid_space,
        num_bandwidth_levels,
        boundary_policy,
        truncation_mode=truncation_mode,
        machine_precision_threshold=machine_precision_threshold,
    )

    # Step 3.4: Rate and precision calculation
    ratio_false_to_true = n_false / max(n_true, 1)
    tpr, fpr, precision_grid, rates_metadata = _calculate_rates_and_precision(
        density_true, density_false, ratio_false_to_true, mixing_metadata
    )

    # Step 3.5: Monotonic enforcement
    if enforce_monotonic:
        precision_grid, mono_metadata = _enforce_monotonic_precision(precision_grid, grid)
    else:
        mono_metadata = {"monotonic_enforcement": False}

    # Combine metadata
    metadata = {
        "method": "adaptive_kde",
        "num_bandwidth_levels": num_bandwidth_levels,
        "enforce_monotonic": enforce_monotonic,
        "boundary_policy": boundary_policy,
        "mixing": mixing_metadata,
        "rates": rates_metadata,
        "monotonic": mono_metadata,
    }

    return tpr, fpr, density_true, density_false, precision_grid, metadata


def _variable_bandwidth_kernel_mixing(
    counts_true: np.ndarray,
    counts_false: np.ndarray,
    grid: np.ndarray,
    dx: float,
    get_score_std: Callable[[np.ndarray], np.ndarray],
    from_grid_space: Callable[[np.ndarray], np.ndarray],
    num_bandwidth_levels: int,
    boundary_policy: str,
    *,
    truncation_mode: str = "auto_detect",
    machine_precision_threshold: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Step 3.3: Variable Bandwidth Approximation via Kernel Mixing.

    Approximates variable bandwidth KDE by mixing K fixed-bandwidth convolutions.
    This makes the computation tractable with FFT while still adapting to local uncertainty.

    Parameters
    ----------
    counts_true, counts_false : np.ndarray
        Binned counts for each class
    grid : np.ndarray
        Grid coordinates
    dx : float
        Grid spacing
    get_score_std : Callable
        Function providing uncertainty estimates
    from_grid_space : Callable
        Grid to data space transform
    num_bandwidth_levels : int
        Number of bandwidth levels
    boundary_policy : str
        Boundary handling policy
    truncation_mode : str, optional
        Truncation mode to use if apply_truncation is True, by default "auto_detect"
    machine_precision_threshold : float, optional
        Threshold for machine precision detection, by default 1e-14

    Returns
    -------
    tuple
        - density_true: smoothed probability densities for true positives (normalized to integrate to 1)
        - density_false: smoothed probability densities for false positives (normalized to integrate to 1)
        - metadata: processing information including truncation metadata if applied
    """
    # Convert grid to data space and get uncertainty estimates
    grid_data_coords = from_grid_space(grid)
    sigma_data = get_score_std(grid_data_coords)

    # Convert to grid units
    sigma_grid = sigma_data / dx

    # Quantize sigma_grid into K levels.
    # TODO: check, is this necessary? When sigma goes down to zero this wouldn't I want this to go down to zero?
    sigma_min = np.percentile(sigma_grid, 5)  # Avoid extreme outliers
    sigma_max = np.percentile(sigma_grid, 95)

    # Handle case where all sigma values are the same
    if sigma_max - sigma_min < EPS:
        # All uncertainties are essentially the same, use single level
        sigma_levels = np.array([np.mean(sigma_grid)])
    else:
        sigma_levels = np.linspace(sigma_min, sigma_max, num_bandwidth_levels)

    # Build kernel bank for each level
    create_kernel_fn = create_exponential_kernel  # Can substitute create_gaussian_kernel or create_exponential_kernel
    kernels = []
    for sigma_k in sigma_levels:
        kernel, _ = create_kernel_fn(
            sigma=sigma_k,
            dx=1.0,
            # radius_k=4,  # dx=1 since sigma already in grid units
        )
        kernels.append(kernel)

    # Initialize smoothed densities
    density_true = np.zeros_like(counts_true, dtype=float)
    density_false = np.zeros_like(counts_false, dtype=float)

    # For each bandwidth level, apply masked convolution
    for k, (_sigma_k, kernel) in enumerate(zip(sigma_levels, kernels, strict=True)):
        # Create mask for grid points using this bandwidth level
        if len(sigma_levels) == 1:
            # Single level - use all points equally
            weight_k = np.ones_like(sigma_grid)
        else:
            # Use soft assignment based on distance to nearest level
            distances = np.abs(sigma_grid[:, None] - sigma_levels[None, :])
            bandwidth = (sigma_levels[1] - sigma_levels[0]) if len(sigma_levels) > 1 else 1.0
            weights = np.exp(-(distances**2) / (2 * bandwidth**2))
            weights = weights / np.sum(weights, axis=1, keepdims=True)  # Normalize
            weight_k = weights[:, k]

        # Apply kernel to masked counts
        masked_counts_true = counts_true * weight_k
        masked_counts_false = counts_false * weight_k

        # Convolve with FFT
        smoothed_true, _ = fft_convolve(masked_counts_true, kernel, boundary_policy)
        smoothed_false, _ = fft_convolve(masked_counts_false, kernel, boundary_policy)

        # Accumulate
        density_true += smoothed_true
        density_false += smoothed_false

    # Store counts before normalization for conservation check
    counts_true_sum = np.sum(counts_true)
    counts_false_sum = np.sum(counts_false)
    density_true_sum_before_norm = np.sum(density_true)
    density_false_sum_before_norm = np.sum(density_false)

    # Normalize to proper densities: normalize to sum=1, then divide by dx
    if np.sum(density_true) > 0:
        density_true = density_true / np.sum(density_true) / dx
    if np.sum(density_false) > 0:
        density_false = density_false / np.sum(density_false) / dx

    # Apply truncation separately to each density
    truncation_metadata_true = {}
    truncation_metadata_false = {}

    # Apply truncation to density_true (now pads with zeros, maintains length)
    density_true, truncation_metadata_true = truncate_density_tails(
        density_true, grid, from_grid_space, truncation_mode, machine_precision_threshold, counts_true
    )

    # Apply truncation to density_false (now pads with zeros, maintains length)
    density_false, truncation_metadata_false = truncate_density_tails(
        density_false, grid, from_grid_space, truncation_mode, machine_precision_threshold, counts_false
    )

    metadata = {
        "sigma_levels": sigma_levels.tolist(),
        "sigma_range": [float(np.min(sigma_grid)), float(np.max(sigma_grid))],
        "total_counts_true_before": float(counts_true_sum),
        "total_counts_false_before": float(counts_false_sum),
        "total_density_true_after": float(np.sum(density_true) * dx),  # Should be ~1
        "total_density_false_after": float(np.sum(density_false) * dx),  # Should be ~1
        "conservation_error_true": float(abs(density_true_sum_before_norm - counts_true_sum) / counts_true_sum),
        "conservation_error_false": float(abs(density_false_sum_before_norm - counts_false_sum) / counts_false_sum),
        "truncation_metadata_true": truncation_metadata_true,
        "truncation_metadata_false": truncation_metadata_false,
    }

    return density_true, density_false, metadata


def _calculate_rates_and_precision(
    density_true: np.ndarray,
    density_false: np.ndarray,
    ratio_false_to_true: float,
    mixing_metadata: dict[str, Any],
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Step 3.4: Rate and Precision Calculation.

    Converts smoothed densities into TPR/FPR curves and then precision curve.
    Works with any normalization of density arrays by normalizing to proper rates.
    Sets precision to NaN beyond the truncation point of density_false.

    Parameters
    ----------
    density_true, density_false : np.ndarray
        Smoothed densities (any normalization - counts, densities, etc.)
    ratio_false_to_true : float
        Ratio of false to true population sizes (n_false / n_true)
    mixing_metadata : dict
        Metadata from kernel mixing containing truncation information
    eps : float, optional
        Numerical floor to avoid division by zero

    Returns
    -------
    tuple
        - tpr: True positive rate array
        - fpr: False positive rate array
        - precision_grid: precision values on grid (NaN beyond false density truncation)
        - metadata: calculation information
    """
    # Calculate TPR and FPR from densities
    tpr, fpr, tpr_fpr_metadata = calculate_tpr_fpr_from_densities(density_true, density_false, eps)

    # Calculate precision from TPR and FPR
    precision_grid = tpr / (tpr + ratio_false_to_true * fpr)
    precision_grid = np.clip(precision_grid, eps, 1.0)

    # Apply NaN masking beyond the truncation point of density_false
    truncation_metadata_false = mixing_metadata.get("truncation_metadata_false", {})
    false_truncation_idx = truncation_metadata_false.get("truncation_idx", len(precision_grid))

    if false_truncation_idx < len(precision_grid):
        # Set precision to NaN beyond the false density truncation point
        precision_grid[false_truncation_idx:] = np.nan

    # Calculate precision range excluding NaN values
    valid_precision = precision_grid[~np.isnan(precision_grid)]
    if len(valid_precision) > 0:
        precision_range = [float(np.min(valid_precision)), float(np.max(valid_precision))]
    else:
        precision_range = [np.nan, np.nan]

    # Combine metadata
    metadata = {
        "ratio_false_to_true": float(ratio_false_to_true),
        "precision_range": precision_range,
        "eps_floor": eps,
        "density_true_sum": float(density_true.sum()),
        "density_false_sum": float(density_false.sum()),
        "false_truncation_idx": false_truncation_idx,
        "precision_valid_length": len(valid_precision),
        **tpr_fpr_metadata,
    }

    return tpr, fpr, precision_grid, metadata


def _enforce_monotonic_precision(
    precision_grid: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Step 3.5: Monotonic Enforcement.

    Optional denoising step using isotonic regression to ensure sensible quality mapping.
    Handles NaN values by only applying isotonic regression to the valid (non-NaN) portion.

    Parameters
    ----------
    precision_grid : np.ndarray
        Input precision values (may contain NaN values)
    grid : np.ndarray
        Grid coordinates (monotonic increasing)

    Returns
    -------
    tuple
        - monotonic_precision: monotonically constrained precision (preserves NaN locations)
        - metadata: enforcement information
    """
    try:
        from sklearn.isotonic import IsotonicRegression

        # Find valid (non-NaN) values
        valid_mask = ~np.isnan(precision_grid)

        if np.sum(valid_mask) == 0:
            # All values are NaN, nothing to do
            metadata = {
                "monotonic_enforcement": False,
                "method": "no_valid_values",
                "all_nan": True,
            }
            return precision_grid.copy(), metadata

        # Extract valid portions
        valid_grid = grid[valid_mask]
        valid_precision = precision_grid[valid_mask]

        # Fit isotonic regression on valid data only
        iso_reg = IsotonicRegression(increasing=True, out_of_bounds="clip")
        valid_monotonic = iso_reg.fit_transform(valid_grid, valid_precision)

        # Reconstruct full array with NaN preserved
        monotonic_precision = precision_grid.copy()
        monotonic_precision[valid_mask] = valid_monotonic

        # Calculate metrics on valid portion only
        mse_before_after = float(np.mean((valid_precision - valid_monotonic) ** 2))
        max_violation_fixed = float(np.max(np.diff(valid_monotonic)))  # Should be >= 0

        metadata = {
            "monotonic_enforcement": True,
            "method": "isotonic_regression",
            "mse_change": mse_before_after,
            "max_positive_slope": max_violation_fixed,
            "points_changed": int(np.sum(np.abs(valid_precision - valid_monotonic) > EPS)),
            "valid_points": int(np.sum(valid_mask)),
            "total_points": len(precision_grid),
        }

        return monotonic_precision, metadata

    except ImportError:
        logger.warning("sklearn not available, skipping monotonic enforcement")
        metadata = {"monotonic_enforcement": False, "reason": "sklearn_not_available"}
        return precision_grid, metadata


# ─────────────────────────────── Precision Estimation Class ───────────────────────────────


class AdaptiveKDEPrecisionEstimator:
    """
    A class that encapsulates the adaptive KDE precision estimation pipeline.

    This class performs steps 1-4 from the _create_quality_lookup_table_kde method:
    1. Creates uncertainty function from cross-validation data
    2. Extracts scores and sets up computational grid
    3. Bins data to grid
    4. Calculates smoothed precision using adaptive KDE

    Attributes after fitting:
    - to_grid: Function to transform from original to grid space
    - from_grid: Function to transform from grid to original space
    - counts_true: Binned true positive counts on grid
    - counts_false: Binned false positive counts on grid
    - grid: Computational grid points
    - dx: Grid spacing
    - tpr: True positive rate as function of threshold
    - fpr: False positive rate as function of threshold
    - precision_grid: Precision values on grid
    - metadata: Dictionary containing processing metadata
    """

    def __init__(
        self,
        grid_size: int = DEFAULT_KDE_GRID_SIZE,
        num_bandwidth_levels: int = DEFAULT_KDE_NUM_BANDWIDTH_LEVELS,
        lowess_frac: float = DEFAULT_KDE_LOWESS_FRAC,
        truncation_mode: str = DEFAULT_KDE_TRUNCATION_MODE,
        transform_mode: str = DEFAULT_KDE_TRANSFORM_MODE,
        random_seed: int | None = None,
        *,
        enforce_monotonic: bool = DEFAULT_KDE_ENFORCE_MONOTONIC,
    ):
        """Initialize the estimator.

        Parameters
        ----------
        grid_size : int, optional
            Fine grid for high resolution, by default 8192
        num_bandwidth_levels : int, optional
            Number of adaptive bandwidth levels, by default 5
        lowess_frac : float, optional
            LOWESS smoothing fraction for uncertainty, by default 0.3
        enforce_monotonic : bool, optional
            Whether to enforce monotonicity, by default True
        truncation_mode : str, optional
            Tail truncation mode, by default "auto_detect"
        transform_mode : str, optional
            Transform scale "mqual" or "logit", by default "mqual"
        """
        # Configuration
        self.config = {
            "grid_size": grid_size,
            "num_bandwidth_levels": num_bandwidth_levels,
            "lowess_frac": lowess_frac,
            "enforce_monotonic": enforce_monotonic,
            "truncation_mode": truncation_mode,
            "transform_mode": transform_mode,
        }

        # RNG
        if random_seed is None:
            # pick seed from clock state
            self.random_seed = np.random.SeedSequence().entropy
        else:
            self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

        # Grid and transforms
        self.grid = None
        self.dx = None
        self.to_grid = None
        self.from_grid = None
        self.grid_metadata = None

        # Binned data
        self.counts_true = None
        self.counts_false = None
        self.bin_metadata_true = None
        self.bin_metadata_false = None

        # Results
        self.tpr = None
        self.fpr = None
        self.precision_grid = None
        self.kde_metadata = None

        # Uncertainty function
        self.uncertainty_fn = None
        self.uncertainty_metadata = None

        # Store data for later steps
        self.scores_t = None
        self.scores_f = None

    def create_uncertainty_function(
        self,
        pd_df: pd.DataFrame,
        num_cv_folds: int,
        fold_col: str = FOLD_COL,
        label_col: str = LABEL_COL,
    ) -> None:
        """
        Step 1a: Create uncertainty function from cross-validation data.

        Parameters
        ----------
        pd_df : pd.DataFrame
            DataFrame with validation data including fold assignments and per-fold predictions
        fold_col : str, optional
            Column name for fold assignments (NaN indicates validation set), by default "fold_id"
        label_col : str, optional
            Column name for true labels, by default "label"
        num_cv_folds : int, optional
            Number of cross-validation folds, by default 5
        """
        self.uncertainty_fn, self.uncertainty_metadata = create_uncertainty_function_pipeline_fast(
            pd_df=pd_df,
            fold_col=fold_col,
            label_col=label_col,
            num_cv_folds=num_cv_folds,
            transform_mode=self.config["transform_mode"],
            lowess_frac=self.config["lowess_frac"],
        )

        logger.debug(
            "Uncertainty function created from %d validation points",
            self.uncertainty_metadata.get("validation_size", 0),
        )

    def extract_scores(
        self,
        pd_df: pd.DataFrame,
        label_col: str = LABEL_COL,
        mqual_col: str = MQUAL,
        prob_orig_col: str = PROB_ORIG,
    ) -> None:
        """
        Step 1b: Extract scores for true positives and false positives.

        Parameters
        ----------
        pd_df : pd.DataFrame
            DataFrame with labels and scores
        label_col : str, optional
            Column name for true labels, by default "label"
        mqual_col : str, optional
            Column name for model quality scores, by default "mqual"
        prob_orig_col : str, optional
            Column name for original probabilities, by default "prob_orig"
        """
        # Extract scores (MQUAL or logit) for true positives and false positives
        if self.config["transform_mode"] == "mqual":
            self.scores_t = pd_df[pd_df[label_col]][mqual_col].to_numpy()
            self.scores_f = pd_df[~pd_df[label_col]][mqual_col].to_numpy()
        elif self.config["transform_mode"] == "logit":
            self.scores_t = prob_to_logit(pd_df.loc[pd_df[label_col], prob_orig_col].to_numpy(), phred=True)
            self.scores_f = prob_to_logit(pd_df.loc[~pd_df[label_col], prob_orig_col].to_numpy(), phred=True)
        else:
            raise ValueError(f"Invalid transform_mode: {self.config['transform_mode']}")

        if len(self.scores_t) == 0 or len(self.scores_f) == 0:
            raise ValueError("Insufficient data: empty true or false positive sets")

        logger.debug("Data: %d true positives, %d false positives", len(self.scores_t), len(self.scores_f))

    def create_grid(self) -> None:
        """
        Step 2: Set up computational grid.

        Requires that extract_scores() has been called first.
        """
        if self.scores_t is None or self.scores_f is None:
            raise ValueError("Must call extract_scores() before create_grid()")

        all_scores = np.concatenate([self.scores_t, self.scores_f])
        (self.grid, self.dx, self.to_grid, self.from_grid, self.grid_metadata) = make_grid_and_transform(
            all_scores, grid_size=self.config["grid_size"], transform_mode=self.config["transform_mode"]
        )

        logger.debug(
            "Grid: %d points, dx=%.6f, range=[%.1f, %.1f]",
            len(self.grid),
            self.dx,
            self.from_grid(self.grid[0]),
            self.from_grid(self.grid[-1]),
        )

    def bin_data_to_grid(self) -> None:
        """
        Step 3: Bin data to grid.

        Requires that extract_scores() and create_grid() have been called first.
        """
        if self.scores_t is None or self.scores_f is None:
            raise ValueError("Must call extract_scores() before bin_data_to_grid()")
        if self.grid is None or self.to_grid is None:
            raise ValueError("Must call create_grid() before bin_data_to_grid()")

        self.counts_true, self.bin_metadata_true = bin_data_to_grid(
            self.scores_t, to_grid_space=self.to_grid, grid_size=self.config["grid_size"]
        )
        self.counts_false, self.bin_metadata_false = bin_data_to_grid(
            self.scores_f, to_grid_space=self.to_grid, grid_size=self.config["grid_size"]
        )

        logger.debug(
            "Binning weight conservation error: true=%.0f, false=%.0f",
            self.bin_metadata_true.get("weight_conservation_error", 0),
            self.bin_metadata_false.get("weight_conservation_error", 0),
        )

    def calculate_smoothed_precision(self) -> None:
        """
        Step 4: Calculate smoothed precision using adaptive KDE.

        Requires that all previous steps have been completed.
        """
        if self.uncertainty_fn is None:
            raise ValueError("Must call create_uncertainty_function() before calculate_smoothed_precision()")
        if self.scores_t is None or self.scores_f is None:
            raise ValueError("Must call extract_scores() before calculate_smoothed_precision()")
        if self.grid is None or self.from_grid is None:
            raise ValueError("Must call create_grid() before calculate_smoothed_precision()")
        if self.counts_true is None or self.counts_false is None:
            raise ValueError("Must call bin_data_to_grid() before calculate_smoothed_precision()")

        (self.tpr, self.fpr, self.density_true, self.density_false, self.precision_grid, self.kde_metadata) = (
            calculate_smoothed_precision_kde(
                self.counts_true,
                self.counts_false,
                self.grid,
                self.dx,
                self.uncertainty_fn,
                self.from_grid,
                len(self.scores_t),
                len(self.scores_f),
                num_bandwidth_levels=self.config["num_bandwidth_levels"],
                enforce_monotonic=self.config["enforce_monotonic"],
                truncation_mode=self.config["truncation_mode"],
            )
        )

        logger.debug("Adaptive KDE completed successfully!")
        logger.debug("Conservation error: %.2e", self.kde_metadata.get("mixing", {}).get("conservation_error_total", 0))
        logger.debug("Precision range: [%.6f, %.6f]", np.nanmin(self.precision_grid), np.nanmax(self.precision_grid))

    def fit(
        self,
        pd_df: pd.DataFrame,
        num_cv_folds: int,
        label_col: str = LABEL_COL,
        fold_col: str = FOLD_COL,
        mqual_col: str = MQUAL,
        prob_orig_col: str = PROB_ORIG,
    ) -> None:
        """
        Fit the adaptive KDE precision estimator by orchestrating all steps.

        Parameters
        ----------
        pd_df : pd.DataFrame
            DataFrame with validation data including fold assignments and per-fold predictions
        label_col : str, optional
            Column name for true labels, by default "label"
        fold_col : str, optional
            Column name for fold assignments (NaN indicates validation set), by default "fold_id"
        num_cv_folds : int, optional
            Number of cross-validation folds, by default 5
        mqual_col : str, optional
            Column name for model quality scores, by default "mqual"
        prob_orig_col : str, optional
            Column name for original probabilities, by default "prob_orig"
        """
        logger.debug(
            "Fitting adaptive KDE precision estimator with grid_size=%d, bandwidth_levels=%d",
            self.config["grid_size"],
            self.config["num_bandwidth_levels"],
        )

        self.n_true = pd_df[label_col].sum()
        self.n_false = pd_df.shape[0] - self.n_true
        self.base_rate = self.n_true / (self.n_true + self.n_false) if (self.n_true + self.n_false) > 0 else 0.0

        # Execute all steps in sequence
        self.create_uncertainty_function(pd_df, num_cv_folds, fold_col=fold_col, label_col=label_col)
        self.extract_scores(pd_df, label_col=label_col, mqual_col=mqual_col, prob_orig_col=prob_orig_col)
        self.create_grid()
        self.bin_data_to_grid()
        self.calculate_smoothed_precision()

        # Define convenience methods for TPR, FPR, and precision lookup
        grid_coords = self.from_grid(self.grid)
        self.get_tp_density = partial(np.interp, xp=grid_coords, fp=self.density_true)
        self.get_fp_density = partial(np.interp, xp=grid_coords, fp=self.density_false)
        self.get_fpr = partial(np.interp, xp=grid_coords, fp=self.fpr)
        self.get_tpr = partial(np.interp, xp=grid_coords, fp=self.tpr)
        self.get_precision = partial(np.interp, xp=grid_coords, fp=self.precision_grid)

    def sample_from_density_interp(self, n_bootstrap: int) -> dict[str, np.ndarray]:
        """
        Sample N points from density_fn using precomputed CDF + interpolation.

        Parameters
        ----------
        n_bootstrap : int
            Number of samples to draw.
        x_min, x_max : float
            Domain bounds.
        n_discrete : int, optional
            Number of discretization points for precomputing the CDF (default: 10,000).

        Returns
        -------
        samples : np.ndarray
            Array of N samples distributed according to density_fn.
        """
        # Inverse probability transform
        self.n_bootstrap = n_bootstrap
        grid_coords = self.from_grid(self.grid)
        self.samples = {}
        for density, n, label in zip(
            [self.density_true, self.density_false], [self.n_true, self.n_false], ["tp", "fp"], strict=True
        ):
            if density is None:
                raise ValueError("Must run fit() before sampling")
            cdf = cumulative_trapezoid(density, grid_coords, initial=0)
            cdf /= cdf[-1]  # normalize

            u = self.rng.random(n * n_bootstrap)

            # Invert CDF using interpolation
            sample = np.interp(u, cdf, grid_coords)
            sample = sample.reshape(n_bootstrap, n)
            self.samples[label] = sample

    def get_bootstrap_kde_densities(self):
        """
        Get bootstrap densities for true positives and false positives.
        Has to be run after self.sample_from_density_interp().

        Returns
        -------
        dict
            Dictionary with keys 'tp' and 'fp' containing bootstrap density arrays.
        """
        if not hasattr(self, "samples"):
            raise ValueError("Must run sample_from_density_interp() before getting bootstrap densities")

        # counts_bootstrap = {}
        self.bootstrap_stats = {
            "fpr": [],
            "tpr": [],
            "precision": [],
            "density_true": [],
            "density_false": [],
            "precision_grid": [],
        }
        for i in range(self.samples["tp"].shape[0]):
            counts = {}
            for label in ["tp", "fp"]:
                counts[label], _ = bin_data_to_grid(
                    self.samples[label][i], to_grid_space=self.to_grid, grid_size=self.config["grid_size"]
                )

            (tpr, fpr, density_true, density_false, precision_grid, _) = calculate_smoothed_precision_kde(
                counts["tp"],
                counts["fp"],
                self.grid,
                self.dx,
                self.uncertainty_fn,
                self.from_grid,
                len(self.scores_t),
                len(self.scores_f),
                num_bandwidth_levels=self.config["num_bandwidth_levels"],
                enforce_monotonic=self.config["enforce_monotonic"],
                truncation_mode=self.config["truncation_mode"],
            )
            self.bootstrap_stats["tpr"].append(tpr)
            self.bootstrap_stats["fpr"].append(fpr)
            self.bootstrap_stats["density_true"].append(density_true)
            self.bootstrap_stats["density_false"].append(density_false)
            self.bootstrap_stats["precision_grid"].append(precision_grid)
        for key in self.bootstrap_stats:
            self.bootstrap_stats[key] = np.array(self.bootstrap_stats[key])
