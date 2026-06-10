"""
MRD statistical detection framework.

Implements Phase 1 of the MRD algorithmic improvement plan:
- Empirical p-value from synthetic control distribution
- Detection call (MRD Detected / Not Detected / Indeterminate)
- Personal LOD estimation via Poisson simulation

The test statistic is the count of supporting reads passing quality
filters (SNVQ >= 60, MAPQ >= 60, filt > 0) — the same metric already
computed by the existing pipeline. The null distribution is derived
empirically from S synthetic control signatures processed through
the same cfDNA sample.

Reference: bfx-read-the-docs/docs/tumor-informed-mrd/mrd_dev_plan_analysis.md
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ugbio_core.logger import logger


@dataclass
class DetectionResult:
    """Result of MRD statistical detection analysis."""

    # Detection call
    detected: bool | None  # True/False/None (indeterminate)
    call: str  # "MRD Detected" / "MRD Not Detected" / "Indeterminate"

    # Empirical p-value
    p_value: float

    # Observed signal
    matched_supporting_reads: int
    matched_ctdna_vaf: float

    # Null distribution summary
    null_median_reads: float
    null_max_reads: int
    n_synthetic_controls: int

    # Personal LOD (95% power)
    personal_lod: float | None  # TF at which detection power >= 95%

    # Null distribution (raw supporting read counts for each synthetic control)
    null_reads: np.ndarray  # shape (n_synthetic_controls,), dtype int
    fitted_p_value: float | None  # Poisson-fitted p-value (MLE lambda)

    # Assay metrics
    signature_size: int  # number of loci in filtered signature
    mean_coverage: float  # mean coverage at signature loci
    corrected_coverage: float  # total corrected coverage


def compute_empirical_pvalue(
    observed_reads: int,
    null_reads: np.ndarray,
) -> float:
    """
    Compute one-sided empirical p-value.

    p-value = (# synthetics with reads >= observed + 1) / (S + 1)

    The +1 in numerator and denominator is the conservative correction
    (Phipson & Smyth, 2010) that accounts for the observed sample itself
    being a draw from the null when H0 is true. This avoids p-values of
    exactly 0 and ensures valid type-I error control.

    Parameters
    ----------
    observed_reads : int
        Number of supporting reads for the matched signature.
    null_reads : np.ndarray
        Array of supporting read counts from synthetic control signatures.

    Returns
    -------
    float
        Empirical p-value in range [1/(S+1), 1].
    """
    s = len(null_reads)
    if s == 0:
        logger.warning("No synthetic controls available for p-value computation")
        return 1.0
    n_ge = np.sum(null_reads >= observed_reads)
    # Conservative correction (Phipson & Smyth, 2010)
    p_value = (n_ge + 1) / (s + 1)
    return float(p_value)


def compute_personal_lod(
    signature_size: int,
    mean_coverage: float,
    denom_ratio: float,
    detection_threshold: int,
    n_simulations: int = 10000,
    target_power: float = 0.95,
    random_seed: int = 42,
) -> float | None:
    """
    Estimate personal LOD via Poisson simulation.

    For a grid of tumor fractions, simulate the expected number of
    supporting reads given the patient's signature size and coverage,
    then find the smallest TF at which detection power >= target.

    The model assumes reads arrive as Poisson(TF * corrected_coverage)
    across the full signature. This is conservative because:
    - It ignores SNVQ weighting (all reads counted equally)
    - It uses total coverage, not per-locus (averaging over loci)
    - No parametric fit to the null — uses the empirical threshold

    Parameters
    ----------
    signature_size : int
        Number of loci in the filtered signature.
    mean_coverage : float
        Mean corrected coverage per locus.
    denom_ratio : float
        Denominator correction ratio.
    detection_threshold : int
        Minimum supporting reads to call detected (from null).
    n_simulations : int
        Number of Monte Carlo simulations per TF level.
    target_power : float
        Required detection probability (default 0.95).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    float or None
        Personal LOD (tumor fraction) or None if not computable.
    """
    if signature_size <= 0 or mean_coverage <= 0:
        logger.warning(f"Cannot compute personal LOD: signature_size={signature_size}, mean_coverage={mean_coverage}")
        return None

    rng = np.random.default_rng(random_seed)
    total_corrected_coverage = signature_size * mean_coverage * denom_ratio

    # Log-spaced TF grid from 1e-7 to 1e-3 (50 points)
    tf_grid = np.logspace(-7, -3, 50)

    for tf in tf_grid:
        # Expected number of supporting reads at this TF
        expected_reads = tf * total_corrected_coverage
        # Simulate Poisson draws
        simulated_reads = rng.poisson(expected_reads, size=n_simulations)
        # Power = fraction of simulations exceeding threshold
        power = np.mean(simulated_reads >= detection_threshold)
        if power >= target_power:
            return float(tf)

    # Could not achieve target power even at TF=1e-3
    logger.warning(
        "Personal LOD could not be determined — "
        "detection power < 95% even at TF=1e-3. "
        f"signature_size={signature_size}, "
        f"mean_coverage={mean_coverage}, "
        f"threshold={detection_threshold}"
    )
    return None


def run_detection_analysis(  # noqa: PLR0912
    df_tf: pd.DataFrame,
    df_signatures_filt: pd.DataFrame,
    denom_ratio: float,
    alpha: float = 0.01,
) -> DetectionResult:
    """
    Run the full MRD detection analysis.

    Extracts matched and synthetic control supporting read counts from
    the existing df_tf dataframe, computes the empirical p-value,
    makes a detection call, and estimates personal LOD.

    Parameters
    ----------
    df_tf : pd.DataFrame
        Tumor fraction dataframe as produced by get_tf_from_filtered_data.
        Index: (signature_type, signature).
        Columns: supporting_reads, coverage, corrected_coverage, ctdna_vaf.
    df_signatures_filt : pd.DataFrame
        Filtered signature dataframe with coverage per locus.
    denom_ratio : float
        Denominator correction ratio from SRSNV training data.
    alpha : float
        Significance threshold for detection call (default 0.01).

    Returns
    -------
    DetectionResult
        Complete detection analysis results.
    """
    # Extract matched signature data
    try:
        matched_data = df_tf.loc["matched"]
    except KeyError:
        logger.error("No matched signature found in df_tf")
        return DetectionResult(
            detected=None,
            call="Indeterminate",
            p_value=1.0,
            matched_supporting_reads=0,
            matched_ctdna_vaf=0.0,
            null_median_reads=0.0,
            null_max_reads=0,
            n_synthetic_controls=0,
            null_reads=np.array([], dtype=int),
            fitted_p_value=None,
            personal_lod=None,
            signature_size=0,
            mean_coverage=0.0,
            corrected_coverage=0.0,
        )

    # Handle single or multiple matched signatures (take first)
    if isinstance(matched_data, pd.DataFrame):
        matched_row = matched_data.iloc[0]
    else:
        matched_row = matched_data

    matched_reads = int(matched_row["supporting_reads"])
    matched_vaf = float(matched_row["ctdna_vaf"])
    corrected_coverage = float(matched_row["corrected_coverage"])

    # Extract synthetic control (db_control) supporting reads
    try:
        db_control_data = df_tf.loc["db_control"]
        if isinstance(db_control_data, pd.Series):
            null_reads = np.array([int(db_control_data["supporting_reads"])])
        else:
            null_reads = db_control_data["supporting_reads"].to_numpy().astype(int)
    except KeyError:
        logger.warning("No db_control (synthetic) signatures found in df_tf. Cannot compute empirical p-value.")
        null_reads = np.array([])

    # Compute empirical p-value
    p_value = compute_empirical_pvalue(matched_reads, null_reads)

    # Compute Poisson-fitted p-value (MLE: lambda = mean of null reads)
    if len(null_reads) > 0:
        from scipy.stats import poisson as _poisson

        lambda_hat = float(np.mean(null_reads))
        if lambda_hat > 0:
            fitted_p_value: float | None = float(_poisson.sf(matched_reads - 1, lambda_hat))
        else:
            fitted_p_value = float(_poisson.sf(matched_reads - 1, 1e-9))  # near-zero lambda
    else:
        fitted_p_value = None

    # Detection call
    if len(null_reads) == 0:
        detected = None
        call = "Indeterminate"
    elif p_value <= alpha:
        detected = True
        call = "MRD Detected"
    else:
        detected = False
        call = "MRD Not Detected"

    # Assay metrics from filtered matched signature
    matched_sig_mask = df_signatures_filt["signature_type"] == "matched"
    if "signature_type" not in df_signatures_filt.columns:
        # signature_type may be in index or as column depending on context
        matched_sig_loci = df_signatures_filt
    else:
        matched_sig_loci = df_signatures_filt[matched_sig_mask]
    signature_size = len(matched_sig_loci)
    mean_coverage = (
        float(matched_sig_loci["coverage"].mean())
        if "coverage" in matched_sig_loci.columns and len(matched_sig_loci) > 0
        else 0.0
    )

    # Detection threshold for LOD: use (max of null + 1) as conservative
    # This is the minimum reads needed to achieve p < 1/(S+1)
    if len(null_reads) > 0:
        detection_threshold = int(np.max(null_reads)) + 1
    else:
        detection_threshold = 1

    # Personal LOD
    personal_lod = compute_personal_lod(
        signature_size=signature_size,
        mean_coverage=mean_coverage,
        denom_ratio=denom_ratio,
        detection_threshold=detection_threshold,
    )

    return DetectionResult(
        detected=detected,
        call=call,
        p_value=p_value,
        matched_supporting_reads=matched_reads,
        matched_ctdna_vaf=matched_vaf,
        null_median_reads=float(np.median(null_reads)) if len(null_reads) > 0 else 0.0,
        null_max_reads=int(np.max(null_reads)) if len(null_reads) > 0 else 0,
        n_synthetic_controls=len(null_reads),
        null_reads=null_reads,
        fitted_p_value=fitted_p_value,
        personal_lod=personal_lod,
        signature_size=signature_size,
        mean_coverage=mean_coverage,
        corrected_coverage=corrected_coverage,
    )


def plot_null_distribution(
    detection: "DetectionResult",
    df_tf: pd.DataFrame,
    ax=None,
):
    """
    Plot the null distribution of supporting reads with patient and control values.

    Shows a histogram of synthetic control (db_control) supporting read counts,
    a vertical line for the matched patient signal, and scatter points for any
    individual (non-db) controls. Directly visualises what the p-value measures.

    Parameters
    ----------
    detection : DetectionResult
        Result from run_detection_analysis().
    df_tf : pd.DataFrame
        Tumor fraction dataframe (index: signature_type, signature).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates a new figure if None.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import poisson as _poisson

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    null = detection.null_reads
    obs = detection.matched_supporting_reads

    # --- Null histogram ---
    if len(null) > 0:
        max_val = max(int(null.max()), obs, 1)
        bins = np.arange(-0.5, max_val + 2.5, 1)
        ax.hist(
            null,
            bins=bins,
            color="#3a9ad9",
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            label=f"Synthetic controls (n={len(null)})",
            zorder=2,
        )

        # --- Poisson fit ---
        lambda_hat = float(np.mean(null))
        x_fit = np.arange(0, max_val + 3)
        y_fit = _poisson.pmf(x_fit, max(lambda_hat, 1e-9)) * len(null)
        ax.plot(
            x_fit, y_fit,
            color="#1a3a5c", linewidth=1, linestyle="-",
            label=f"Synthetic control fit — Poisson(λ={lambda_hat:.2f})",
            zorder=3,
        )

    # --- Individual cohort controls (non-db, non-matched) ---
    try:
        ctrl_data = df_tf.loc["control"]["supporting_reads"]
        if isinstance(ctrl_data, (int, float, np.integer)):
            ctrl_data = pd.Series([ctrl_data])
        for v in ctrl_data.values:
            ax.axvline(v, color="#e67e22", linewidth=1.5, linestyle="--", alpha=0.8, zorder=4)
        ax.axvline(np.nan, color="#e67e22", linewidth=1.5, linestyle="--", label="Cohort control")
    except KeyError:
        pass

    # --- Matched patient signal ---
    ax.axvline(
        obs, color="#c0392b", linewidth=2.5, linestyle="-",
        label=f"Patient signal ({obs} reads)", zorder=5,
    )

    # --- Detection threshold ---
    if len(null) > 0:
        threshold = int(null.max()) + 1
        ax.axvline(
            threshold - 0.5, color="#7f8c8d", linewidth=1.2,
            linestyle=":", alpha=0.7, label=f"Detection threshold ({threshold})",
            zorder=4,
        )

    # --- Title: call status + both p-values ---
    emp_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"
    if detection.fitted_p_value is not None:
        fit_str = f"{detection.fitted_p_value:.3f}" if detection.fitted_p_value >= 0.001 else f"{detection.fitted_p_value:.2e}"
        title = f"{detection.call}  (p_empirical={emp_str},  p_Poisson={fit_str})"
    else:
        title = f"{detection.call}  (p={emp_str})"

    ax.set_xlabel("Supporting reads (test statistic)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(left=-0.5)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.xaxis.get_major_locator().set_params(integer=True)


def format_scientific(value: float, precision: int = 1) -> str:
    """Format a float in scientific notation for display."""
    if value == 0:
        return "0"
    if value is None:
        return "N/A"
    mantissa_near_one_tol = 0.05
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exp
    if abs(mantissa - 1.0) < mantissa_near_one_tol:
        return f"10\u207b{abs(exp)}"
    return f"{mantissa:.{precision}f} \u00d7 10\u207b{abs(exp)}"
