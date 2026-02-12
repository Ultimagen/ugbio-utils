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
#    Unit tests for KDE smoothing consistency with counting methods
# CHANGELOG in reverse chronological order

"""
Tests for smoothing_utils.py KDE methods to ensure consistency with counting methods.

The main goal is to verify that:
1. KDE-based precision estimation produces reasonable smoothed histograms
2. KDE results are consistent with counting methods (within expected smoothing differences)
3. The smoothing reduces noise, especially in tail regions, without introducing bias
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ugbio_core.logger import logger
from ugbio_srsnv.smoothing_utils import (
    AdaptiveKDEPrecisionEstimator,
    bin_data_to_grid,
    make_grid_and_transform,
)


@pytest.fixture
def resources_dir():
    """Return path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def synthetic_cv_data():
    """
    Create synthetic cross-validation data with known properties.

    Generates:
    - 10,000 true positives with high MQUAL scores
    - 10,000 false positives with lower MQUAL scores
    - 3 folds for cross-validation
    - 5,000 point validation set with NaN fold_id
    - Heteroskedastic noise added to MQUAL values (higher at lower MQUAL)
    """
    rng = np.random.default_rng(42)
    n_true = 10000
    n_false = 10000
    n_folds = 3
    n_validation = 5000

    # True positives: higher MQUAL (mean≈25, std≈8)
    mqual_true = rng.gamma(shape=10, scale=2.5, size=n_true)
    # mqual_true = np.clip(mqual_true, 0.5, 60)

    # False positives: lower MQUAL (mean≈12, std≈6)
    mqual_false = rng.gamma(shape=4, scale=3, size=n_false)
    # mqual_false = np.clip(mqual_false, 0.5, 40)

    # Combine data
    labels = np.concatenate([np.ones(n_true, dtype=bool), np.zeros(n_false, dtype=bool)])
    mqual_all = np.concatenate([mqual_true, mqual_false])

    # Create fold assignments
    n_total = n_true + n_false
    fold_id = rng.integers(0, n_folds, size=n_total).astype(float)
    val_idxs = rng.choice(n_total, size=n_validation, replace=False)
    fold_id[val_idxs] = np.nan  # Validation set has no fold

    # Create per-fold predictions with heteroskedastic noise on MQUAL
    data = {
        "label": labels,
        "fold_id": fold_id,
        "MQUAL": np.zeros(n_total),
        "prob_orig": np.full(n_total, np.nan),
    }

    # Heteroskedastic noise: higher at lower MQUAL values
    # sigma(x) is piecewise: 2 for MQUAL < 15, 1 for MQUAL >= 15
    # For very low MQUAL < 8, proportional to MQUAL
    sigma = np.where(mqual_all < 15, 2.0, 1.0)
    sigma[mqual_all < 8] = 0.25 * mqual_all[mqual_all < 8]

    # Add per-fold MQUAL predictions with noise
    for k in range(n_folds):
        noise = rng.normal(0, sigma, size=n_total)
        mqual_fold = mqual_all + noise
        data[f"MQUAL_fold_{k}"] = mqual_fold
        data[f"prob_fold_{k}"] = 1 - 10 ** (-mqual_fold / 10)  # phred_to_prob
        data["MQUAL"][fold_id == k] = mqual_fold[fold_id == k]
        data["MQUAL"][np.isnan(fold_id)] += mqual_fold[np.isnan(fold_id)] / n_folds

    data["prob_orig"] = 1 - 10 ** (-data["MQUAL"] / 10)  # phred_to_prob

    data_df = pd.DataFrame(data)
    return data_df


@pytest.fixture
def real_test_data(resources_dir):
    """Load the real test parquet data."""
    parquet_path = resources_dir / "416119_L7402.test.featuremap_df.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Test data not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def calculate_counting_histogram(mqual_true, mqual_false, n_bins=50):
    """
    Calculate precision using simple counting (binning) method.

    This is equivalent to the _create_quality_lookup_table_count method
    but simplified for testing.

    Parameters
    ----------
    mqual_true : np.ndarray
        MQUAL scores for true positives
    mqual_false : np.ndarray
        MQUAL scores for false positives
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    tuple
        (bin_centers, tpr, fpr, precision)
    """
    # Determine range
    mqual_min = min(mqual_true.min(), mqual_false.min())
    mqual_max = max(mqual_true.max(), mqual_false.max())

    # Create bins
    bins = np.linspace(mqual_min, mqual_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate TPR and FPR at each bin center
    tpr = np.array([(mqual_true >= bc).mean() for bc in bin_centers])
    fpr = np.array([(mqual_false >= bc).mean() for bc in bin_centers])

    # Calculate precision
    eps = 1e-12
    ratio_false_to_true = len(mqual_false) / max(len(mqual_true), 1)
    precision = tpr / (tpr + ratio_false_to_true * fpr + eps)

    return bin_centers, tpr, fpr, precision


def calculate_kde_histogram(pd_df, num_cv_folds, grid_size=2048, num_bandwidth_levels=1, transform_mode="mqual"):
    """
    Calculate precision using KDE smoothing method.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame with cross-validation data
    num_cv_folds : int
        Number of CV folds
    grid_size : int
        Grid size for KDE
    num_bandwidth_levels : int
        Number of bandwidth levels
    transform_mode : str
        Transform mode ("mqual" or "logit")

    Returns
    -------
    tuple
        (mqual_values, tpr, fpr, precision, metadata)
    """
    # Use AdaptiveKDEPrecisionEstimator
    estimator = AdaptiveKDEPrecisionEstimator(
        grid_size=grid_size,
        num_bandwidth_levels=num_bandwidth_levels,
        lowess_frac=0.3,
        enforce_monotonic=False,
        truncation_mode="auto_detect",
        transform_mode=transform_mode,
    )

    # Fit estimator (passes min_val_size check now)
    estimator.fit(pd_df, num_cv_folds=num_cv_folds)

    # Convert grid to MQUAL values
    mqual_values = estimator.from_grid(estimator.grid)

    return mqual_values, estimator.tpr, estimator.fpr, estimator.precision_grid, estimator.kde_metadata


# ─────────────────────── Test Functions ───────────────────────


def test_kde_produces_smoother_results_than_counting(synthetic_cv_data):
    """
    Test that KDE produces smoother (less noisy) precision curves than counting.

    The smoothness is quantified by the variance of the first derivative,
    which should be lower for the smoothed KDE curve.
    """
    data_df = synthetic_cv_data
    num_cv_folds = 3

    # Get true and false MQUAL scores
    mqual_true = data_df[data_df["label"]]["MQUAL"].to_numpy()
    mqual_false = data_df[~data_df["label"]]["MQUAL"].to_numpy()

    # Calculate counting-based precision
    bin_centers, tpr_count, fpr_count, prec_count = calculate_counting_histogram(mqual_true, mqual_false, n_bins=50)

    # Calculate KDE-based precision
    mqual_kde, tpr_kde, fpr_kde, prec_kde, metadata = calculate_kde_histogram(
        data_df, num_cv_folds, grid_size=2048, num_bandwidth_levels=3
    )

    # Interpolate both to the same grid for comparison
    common_grid = np.linspace(max(bin_centers.min(), mqual_kde.min()), min(bin_centers.max(), mqual_kde.max()), 100)

    prec_count_interp = np.interp(common_grid, bin_centers, prec_count)
    prec_kde_interp = np.interp(common_grid, mqual_kde, prec_kde)

    # Calculate smoothness metric: variance of first derivative
    def calc_smoothness(y):
        """Lower values = smoother curve."""
        dy = np.diff(y)
        return np.nanvar(dy)

    smoothness_count = calc_smoothness(prec_count_interp)
    smoothness_kde = calc_smoothness(prec_kde_interp)

    # KDE should be smoother (lower variance in derivative)
    assert smoothness_kde < smoothness_count, (
        f"KDE should be smoother than counting: "
        f"KDE smoothness={smoothness_kde:.6f}, Count smoothness={smoothness_count:.6f}"
    )


def test_kde_maintains_overall_precision_level(synthetic_cv_data):
    """
    Test that KDE maintains similar overall precision level to counting method.

    While KDE smooths the curve, it should not systematically bias the precision
    estimates. We check that the mean precision is similar between methods.
    """
    data_df = synthetic_cv_data
    num_cv_folds = 3

    # Get true and false MQUAL scores
    mqual_true = data_df[data_df["label"]]["MQUAL"].to_numpy()
    mqual_false = data_df[~data_df["label"]]["MQUAL"].to_numpy()

    # Calculate counting-based precision
    bin_centers, tpr_count, fpr_count, prec_count = calculate_counting_histogram(mqual_true, mqual_false, n_bins=50)

    # Calculate KDE-based precision
    mqual_kde, tpr_kde, fpr_kde, prec_kde, metadata = calculate_kde_histogram(
        data_df, num_cv_folds, grid_size=2048, num_bandwidth_levels=3
    )

    # Compare mean precision in overlapping range
    mqual_min = max(bin_centers.min(), mqual_kde.min())
    mqual_max = min(bin_centers.max(), mqual_kde.max())

    # Filter to overlapping range
    mask_count = (bin_centers >= mqual_min) & (bin_centers <= mqual_max)
    mask_kde = (mqual_kde >= mqual_min) & (mqual_kde <= mqual_max)

    mean_prec_count = np.nanmean(prec_count[mask_count])
    mean_prec_kde = np.nanmean(prec_kde[mask_kde])

    # Mean precision should be similar (within 10%)
    relative_diff = abs(mean_prec_count - mean_prec_kde) / mean_prec_count
    assert relative_diff < 0.10, (
        f"Mean precision differs too much between methods: "
        f"Count={mean_prec_count:.4f}, KDE={mean_prec_kde:.4f}, "
        f"relative_diff={relative_diff:.4f}"
    )


# NOTE: I am not sure that the reduction of noise in the tail should actually be enforced, so removing the test for now.
# def test_kde_reduces_noise_in_tail_regions(synthetic_cv_data):
#     """
#     Test that KDE reduces noise in tail regions (high MQUAL).

#     In tail regions with sparse data, counting methods show high variance.
#     KDE should provide more stable estimates.
#     """
#     data_df = synthetic_cv_data
#     num_cv_folds = 3

#     # Get true and false MQUAL scores
#     mqual_true = data_df[data_df["label"]]["MQUAL"].to_numpy()
#     mqual_false = data_df[~data_df["label"]]["MQUAL"].to_numpy()

#     # Calculate both methods
#     bin_centers, tpr_count, fpr_count, prec_count = calculate_counting_histogram(mqual_true, mqual_false, n_bins=50)
#     mqual_kde, tpr_kde, fpr_kde, prec_kde, metadata = calculate_kde_histogram(
#         data_df, num_cv_folds, grid_size=2048, num_bandwidth_levels=3
#     )

#     # Focus on tail region (top 25% of MQUAL)
#     tail_threshold = np.percentile(np.concatenate([mqual_true, mqual_false]), 75)

#     # Get precision in tail for both methods
#     tail_mask_count = bin_centers >= tail_threshold
#     tail_mask_kde = mqual_kde >= tail_threshold

#     if tail_mask_count.sum() < 5 or tail_mask_kde.sum() < 5:
#         pytest.skip("Insufficient data in tail region")

#     prec_tail_count = prec_count[tail_mask_count]
#     prec_tail_kde = prec_kde[tail_mask_kde]

#     # Calculate coefficient of variation (std/mean) as noise metric
#     cv_count = np.nanstd(prec_tail_count) / (np.nanmean(prec_tail_count) + 1e-6)
#     cv_kde = np.nanstd(prec_tail_kde) / (np.nanmean(prec_tail_kde) + 1e-6)

#     # KDE should have similar or lower coefficient of variation (not significantly worse)
#     # With heteroskedastic noise, KDE might be slightly higher but should be within 20% of counting
#     assert cv_kde < cv_count * 1.2, (
#         f"KDE should not be significantly noisier in tail region: "
#         f"KDE CV={cv_kde:.4f}, Count CV={cv_count:.4f}, ratio={cv_kde/cv_count:.2f}"
#     )


def test_kde_and_counting_correlation(synthetic_cv_data):
    """
    Test that KDE and counting methods are highly correlated.

    While KDE smooths the curve, the overall shape should be similar.
    """
    data_df = synthetic_cv_data
    num_cv_folds = 3

    # Get true and false MQUAL scores
    mqual_true = data_df[data_df["label"]]["MQUAL"].to_numpy()
    mqual_false = data_df[~data_df["label"]]["MQUAL"].to_numpy()

    # Calculate both methods
    bin_centers, tpr_count, fpr_count, prec_count = calculate_counting_histogram(mqual_true, mqual_false, n_bins=50)
    mqual_kde, tpr_kde, fpr_kde, prec_kde, metadata = calculate_kde_histogram(
        data_df, num_cv_folds, grid_size=2048, num_bandwidth_levels=3
    )

    # Interpolate both to the same grid
    common_grid = np.linspace(max(bin_centers.min(), mqual_kde.min()), min(bin_centers.max(), mqual_kde.max()), 100)

    prec_count_interp = np.interp(common_grid, bin_centers, prec_count)
    prec_kde_interp = np.interp(common_grid, mqual_kde, prec_kde)

    # Calculate Pearson correlation (handle NaN values)
    valid_mask = ~np.isnan(prec_count_interp) & ~np.isnan(prec_kde_interp)
    correlation = np.corrcoef(prec_count_interp[valid_mask], prec_kde_interp[valid_mask])[0, 1]

    # Correlation should be high (> 0.85)
    assert correlation > 0.85, f"KDE and counting methods should be highly correlated: r={correlation:.4f}"


def test_kde_handles_small_dataset(real_test_data):
    """
    Test that KDE can handle the small real test dataset without failing.

    This is a smoke test to ensure the pipeline works on real data.
    Since the real test data has insufficient validation points, we artificially
    create a larger validation set by duplicating the data and adding noise.
    """
    data_df = real_test_data.copy()
    num_cv_folds = 2

    # Check if there's enough validation data
    n_validation = data_df["fold_id"].isna().sum()

    if n_validation < 5000:
        # Artificially create validation set by duplicating data
        # This is acceptable for a smoke test that verifies pipeline completion
        n_repeats = int(np.ceil(5000 / len(data_df))) + 1

        # Create duplicated datasets with added noise
        validation_dfs = []
        rng = np.random.default_rng(42)

        for i in range(n_repeats):
            df_dup = data_df.copy()

            # Set fold_id to NaN for validation set
            df_dup["fold_id"] = np.nan

            # Add noise to probability columns (variance ~1 on logit scale)
            # This translates to varying noise levels in probability space
            for col in ["prob_fold_0", "prob_fold_1", "prob_orig"]:
                if col in df_dup.columns:
                    # Convert to logit, add noise, convert back
                    from ugbio_srsnv.srsnv_utils import logit_to_prob, prob_to_logit

                    probs = df_dup[col].to_numpy()
                    # Clip to avoid extreme values
                    probs = np.clip(probs, 1e-6, 1 - 1e-6)

                    # Convert to logit (phred=True for consistent scaling)
                    logits = prob_to_logit(probs, phred=True)

                    # Add Gaussian noise with std=1 on logit scale
                    logits_noisy = logits + rng.normal(0, 1.0, size=len(logits))

                    # Convert back to probabilities
                    probs_noisy = logit_to_prob(logits_noisy, phred=True)
                    probs_noisy = np.clip(probs_noisy, 1e-6, 1 - 1e-6)

                    df_dup[col] = probs_noisy

            validation_dfs.append(df_dup)

        # Combine original data (with folds) and artificial validation data
        data_df = pd.concat([data_df] + validation_dfs, ignore_index=True)

        n_validation_new = data_df["fold_id"].isna().sum()
        logger.info(
            f"Artificially expanded validation set from {n_validation} to {n_validation_new} points " f"for smoke test"
        )

    # Use small grid size for efficiency
    estimator = AdaptiveKDEPrecisionEstimator(
        grid_size=512,
        num_bandwidth_levels=1,
        lowess_frac=0.4,
        enforce_monotonic=False,
        truncation_mode="none",
        transform_mode="mqual",
    )

    try:
        # Fit estimator
        estimator.fit(data_df, num_cv_folds=num_cv_folds)

        # Basic sanity checks
        assert estimator.tpr is not None, "TPR should be computed"
        assert estimator.fpr is not None, "FPR should be computed"
        assert estimator.precision_grid is not None, "Precision should be computed"

        # Check that results are in valid ranges
        assert np.all(np.isfinite(estimator.tpr)), "TPR should be finite"
        assert np.all(np.isfinite(estimator.fpr)), "FPR should be finite"
        assert np.all(np.isfinite(estimator.precision_grid)), "Precision should be finite"

        assert np.all(
            (estimator.precision_grid >= 0) & (estimator.precision_grid <= 1)
        ), "Precision should be in [0, 1]"

    except Exception as e:
        pytest.fail(f"KDE failed on real test data: {e}")


def test_grid_operations_conserve_mass():
    """
    Test that grid binning operations conserve total mass (counts).

    This verifies the bin_data_to_grid function works correctly.
    """
    rng = np.random.default_rng(123)

    # Generate test data
    data = rng.gamma(shape=5, scale=2, size=10000)

    # Create grid
    grid, dx, to_grid, from_grid, metadata = make_grid_and_transform(
        data, grid_size=1024, transform_mode="mqual", padding_factor=0.1
    )

    # Bin data to grid
    counts, bin_metadata = bin_data_to_grid(data, weights=None, grid_size=1024, to_grid_space=to_grid)

    # Total counts should equal number of data points (within numerical tolerance)
    total_counts = counts.sum()
    expected_counts = len(data)

    relative_error = abs(total_counts - expected_counts) / expected_counts
    assert relative_error < 1e-6, (
        f"Grid binning should conserve mass: "
        f"expected={expected_counts}, got={total_counts:.2f}, "
        f"relative_error={relative_error:.2e}"
    )


# NOTE: monotonicity enforcement is not currently supported. Might be in the future.
# def test_kde_monotonicity_enforcement():
#     """
#     Test that monotonicity enforcement works correctly.

#     When enforce_monotonic=True, precision should be non-increasing with score.
#     """
#     rng = np.random.default_rng(456)

#     # Create data with known properties
#     n_true = 10000
#     n_false = 10000
#     n_validation = 5000
#     n_folds = 3

#     mqual_true = rng.gamma(shape=8, scale=2, size=n_true)
#     mqual_false = rng.gamma(shape=3, scale=2, size=n_false)

#     labels = np.concatenate([np.ones(n_true, dtype=bool), np.zeros(n_false, dtype=bool)])
#     mqual_all = np.concatenate([mqual_true, mqual_false])

#     # Create fold assignments
#     n_total = n_true + n_false
#     fold_id = rng.integers(0, n_folds, size=n_total).astype(float)
#     val_idxs = rng.choice(n_total, size=n_validation, replace=False)
#     fold_id[val_idxs] = np.nan

#     data = {
#         "label": labels,
#         "fold_id": fold_id,
#         "MQUAL": np.zeros(n_total),
#         "prob_orig": np.full(n_total, np.nan),
#     }

#     # Add per-fold predictions with noise on MQUAL
#     sigma = np.ones(n_total) * 1.0  # Constant noise
#     for k in range(n_folds):
#         noise = rng.normal(0, sigma, size=n_total)
#         mqual_fold = mqual_all + noise
#         data[f"MQUAL_fold_{k}"] = mqual_fold
#         data[f"prob_fold_{k}"] = 1 - 10 ** (-mqual_fold / 10)
#         data["MQUAL"][fold_id == k] = mqual_fold[fold_id == k]
#         data["MQUAL"][np.isnan(fold_id)] += mqual_fold[np.isnan(fold_id)] / n_folds

#     data["prob_orig"] = 1 - 10 ** (-data["MQUAL"] / 10)

#     data_df = pd.DataFrame(data)

#     # Test with monotonicity enforcement
#     estimator_mono = AdaptiveKDEPrecisionEstimator(
#         grid_size=1024,
#         num_bandwidth_levels=1,
#         enforce_monotonic=True,
#         truncation_mode="none",
#         transform_mode="mqual",
#     )

#     estimator_mono.fit(data_df, num_cv_folds=n_folds)

#     # Verify that monotonic enforcement was enabled
#     assert estimator_mono.config["enforce_monotonic"] is True, "Monotonic enforcement should be enabled"

#     # Verify precision is in valid range
#     assert np.all((estimator_mono.precision_grid >= 0) & (estimator_mono.precision_grid <= 1)), (
#         "Precision should be in [0, 1]"
#     )

#     # Verify that precision generally decreases with score (check overall trend)
#     # Split grid into quartiles and check that later quartiles have higher mean precision
#     n_points = len(estimator_mono.precision_grid)
#     q1_precision = estimator_mono.precision_grid[:n_points//4].mean()
#     q4_precision = estimator_mono.precision_grid[3*n_points//4:].mean()

#     assert q4_precision >= q1_precision * 0.9, (
#         f"Precision should generally increase with score: Q1={q1_precision:.4f}, Q4={q4_precision:.4f}"
#     )


def test_kde_transforms_consistency():
    """
    Test that MQUAL and logit transform modes produce consistent results.

    While the internal processing differs, the final precision curves
    (when compared in the same coordinate system) should be similar.
    """
    rng = np.random.default_rng(789)

    # Create test data
    n_true = 10000
    n_false = 10000
    n_validation = 5000
    n_folds = 3

    mqual_true = rng.gamma(shape=7, scale=2.5, size=n_true)
    mqual_false = rng.gamma(shape=3, scale=2.5, size=n_false)

    labels = np.concatenate([np.ones(n_true, dtype=bool), np.zeros(n_false, dtype=bool)])
    mqual_all = np.concatenate([mqual_true, mqual_false])

    # Create fold assignments
    n_total = n_true + n_false
    fold_id = rng.integers(0, n_folds, size=n_total).astype(float)
    val_idxs = rng.choice(n_total, size=n_validation, replace=False)
    fold_id[val_idxs] = np.nan

    data = {
        "label": labels,
        "fold_id": fold_id,
        "MQUAL": np.zeros(n_total),
        "prob_orig": np.full(n_total, np.nan),
    }

    # Add per-fold predictions with noise on MQUAL
    sigma = np.ones(n_total) * 1.5
    for k in range(n_folds):
        noise = rng.normal(0, sigma, size=n_total)
        mqual_fold = mqual_all + noise
        data[f"MQUAL_fold_{k}"] = mqual_fold
        data[f"prob_fold_{k}"] = 1 - 10 ** (-mqual_fold / 10)
        data["MQUAL"][fold_id == k] = mqual_fold[fold_id == k]
        data["MQUAL"][np.isnan(fold_id)] += mqual_fold[np.isnan(fold_id)] / n_folds

    data["prob_orig"] = 1 - 10 ** (-data["MQUAL"] / 10)

    data_df = pd.DataFrame(data)

    # Test both transform modes
    estimator_mqual = AdaptiveKDEPrecisionEstimator(
        grid_size=1024, num_bandwidth_levels=1, enforce_monotonic=False, transform_mode="mqual"
    )

    estimator_logit = AdaptiveKDEPrecisionEstimator(
        grid_size=1024, num_bandwidth_levels=1, enforce_monotonic=False, transform_mode="logit"
    )

    estimator_mqual.fit(data_df, num_cv_folds=n_folds)
    estimator_logit.fit(data_df, num_cv_folds=n_folds)

    # Convert both to MQUAL coordinates
    mqual_values_mqual = estimator_mqual.from_grid(estimator_mqual.grid)
    mqual_values_logit = estimator_logit.from_grid(estimator_logit.grid)

    # Interpolate to common grid
    common_grid = np.linspace(
        max(mqual_values_mqual.min(), mqual_values_logit.min()),
        min(mqual_values_mqual.max(), mqual_values_logit.max()),
        100,
    )

    prec_mqual_interp = np.interp(common_grid, mqual_values_mqual, estimator_mqual.precision_grid)
    prec_logit_interp = np.interp(common_grid, mqual_values_logit, estimator_logit.precision_grid)

    # Calculate correlation (handle NaN values)
    valid_mask = ~np.isnan(prec_mqual_interp) & ~np.isnan(prec_logit_interp)
    correlation = np.corrcoef(prec_mqual_interp[valid_mask], prec_logit_interp[valid_mask])[0, 1]

    # Both transform modes should produce similar results
    assert correlation > 0.80, (
        f"MQUAL and logit transforms should produce consistent results: " f"correlation={correlation:.4f}"
    )


# def test_kde_with_extreme_class_imbalance():
#     """
#     Test KDE behavior with extreme class imbalance.

#     This tests robustness when one class is much more common than the other.
#     """
#     rng = np.random.default_rng(999)

#     # Create highly imbalanced data (10:1 ratio)
#     n_true = 2000
#     n_false = 20000
#     n_validation = 5000
#     n_folds = 3

#     mqual_true = rng.gamma(shape=6, scale=3, size=n_true)
#     mqual_false = rng.gamma(shape=2, scale=3, size=n_false)

#     labels = np.concatenate([np.ones(n_true, dtype=bool), np.zeros(n_false, dtype=bool)])
#     mqual_all = np.concatenate([mqual_true, mqual_false])

#     # Create fold assignments
#     n_total = n_true + n_false
#     fold_id = rng.integers(0, n_folds, size=n_total).astype(float)
#     val_idxs = rng.choice(n_total, size=n_validation, replace=False)
#     fold_id[val_idxs] = np.nan

#     data = {
#         "label": labels,
#         "fold_id": fold_id,
#         "MQUAL": np.zeros(n_total),
#         "prob_orig": np.full(n_total, np.nan),
#     }

#     # Add per-fold predictions with noise on MQUAL
#     sigma = np.ones(n_total) * 2.0
#     for k in range(n_folds):
#         noise = rng.normal(0, sigma, size=n_total)
#         mqual_fold = mqual_all + noise
#         data[f"MQUAL_fold_{k}"] = mqual_fold
#         data[f"prob_fold_{k}"] = 1 - 10 ** (-mqual_fold / 10)
#         data["MQUAL"][fold_id == k] = mqual_fold[fold_id == k]
#         data["MQUAL"][np.isnan(fold_id)] += mqual_fold[np.isnan(fold_id)] / n_folds

#     data["prob_orig"] = 1 - 10 ** (-data["MQUAL"] / 10)

#     data_df = pd.DataFrame(data)

#     # KDE should handle imbalanced data without errors
#     estimator = AdaptiveKDEPrecisionEstimator(
#         grid_size=1024, num_bandwidth_levels=1, enforce_monotonic=False, transform_mode="mqual"
#     )

#     try:
#         estimator.fit(data_df, num_cv_folds=n_folds)

#         # Basic sanity checks
#         assert np.all(np.isfinite(estimator.precision_grid)), "Precision should be finite"
# assert np.all(
#     (estimator.precision_grid >= 0) & (estimator.precision_grid <= 1)
# ), "Precision should be in [0, 1]"

#         # Verify that FPR and TPR are both computed
#         assert np.all(np.isfinite(estimator.fpr)), "FPR should be finite"
#         assert np.all(np.isfinite(estimator.tpr)), "TPR should be finite"

#         # With 10x imbalance, the ratio should be reflected in the data counts
#         ratio_in_data = len(data_df[~data_df['label']]) / len(data_df[data_df['label']])
#         assert 9.0 < ratio_in_data < 11.0, f"Data should reflect ~10:1 imbalance: {ratio_in_data:.1f}"

#     except Exception as e:
#         pytest.fail(f"KDE failed with imbalanced data: {e}")
