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
    fitted_p_value: float | None  # distribution-fitted p-value
    fitted_distribution: str  # "Poisson" or "NegativeBinomial"
    null_fit_params: dict  # distribution parameters used for the fit

    # Assay metrics
    signature_size: int  # number of loci in filtered signature
    mean_coverage: float  # mean coverage at signature loci
    corrected_coverage: float  # total corrected coverage


def _fit_null_distribution(null_reads: np.ndarray) -> tuple[str, dict]:
    """
    Fit the best count distribution to null_reads.

    Uses Poisson when the data is not significantly overdispersed
    (variance/mean <= 1.5), and Negative Binomial by method-of-moments
    otherwise.  The NB has a heavier tail and yields a more conservative
    (larger) p-value when the noise is overdispersed.

    Parameters
    ----------
    null_reads : np.ndarray
        Array of supporting read counts from synthetic controls.

    Returns
    -------
    dist_name : str
        "Poisson" or "NegativeBinomial"
    params : dict
        For Poisson: {"lambda": float}
        For NB: {"r": float, "p": float, "mu": float}
    """
    if len(null_reads) == 0:
        return "Poisson", {"lambda": 0.0}

    n = len(null_reads)
    # Jeffreys-prior Bayes estimate for Poisson rate: (sum + 0.5) / (n + 0.5).
    # When all controls have 0 reads, this gives 0.5/n rather than 0, which is
    # the principled non-informative-prior estimate. It avoids p_value=0 without
    # inflating the rate, and is strictly more conservative than adding 1 count.
    mu = float((np.sum(null_reads) + 0.5) / (n + 0.5))
    if n < 3:
        return "Poisson", {"lambda": mu}

    var = float(np.var(null_reads, ddof=1))
    disp_index = var / mu  # = 1 for Poisson; > 1 means overdispersed

    if disp_index <= 1.5:
        return "Poisson", {"lambda": mu}

    # Overdispersed: fit Negative Binomial by method of moments
    # var = mu + mu**2/r  =>  r = mu**2 / (var - mu)
    r_hat = max(mu**2 / (var - mu), 0.1)  # numerical safety floor
    p_hat = float(np.clip(r_hat / (r_hat + mu), 1e-9, 1 - 1e-9))
    logger.debug(
        "Overdispersed null (var/mean=%.2f): fitting NegativeBinomial(r=%.2f, p=%.4f)",
        disp_index,
        r_hat,
        p_hat,
    )
    return "NegativeBinomial", {"r": r_hat, "p": p_hat, "mu": mu}


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

    # Log-spaced TF grid from 1e-7 to 0.1 (80 points)
    # Upper bound 0.1 handles cases where threshold is very high (e.g. unfiltered)
    tf_grid = np.logspace(-7, -1, 80)

    for tf in tf_grid:
        # Expected number of supporting reads at this TF
        expected_reads = tf * total_corrected_coverage
        # Simulate Poisson draws
        simulated_reads = rng.poisson(expected_reads, size=n_simulations)
        # Power = fraction of simulations exceeding threshold
        power = np.mean(simulated_reads >= detection_threshold)
        if power >= target_power:
            return float(tf)

    # Could not achieve target power even at TF=0.1 — expected for unfiltered
    logger.debug(
        "Personal LOD could not be determined "
        "(detection power < 95%% even at TF=0.1; "
        "signature_size=%d, mean_coverage=%.1f, threshold=%d)",
        signature_size,
        mean_coverage,
        detection_threshold,
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
            fitted_distribution="Poisson",
            null_fit_params={},
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

    # Fit null distribution (Poisson or Negative Binomial if overdispersed)
    if len(null_reads) > 0:
        from scipy.stats import nbinom as _nbinom
        from scipy.stats import poisson as _poisson

        fitted_distribution, null_fit_params = _fit_null_distribution(null_reads)
        if fitted_distribution == "NegativeBinomial":
            r_fit, p_fit = null_fit_params["r"], null_fit_params["p"]
            fitted_p_value: float | None = float(_nbinom.sf(matched_reads - 1, r_fit, p_fit))
        else:
            lam = null_fit_params["lambda"]  # Jeffreys-prior estimate, always > 0
            fitted_p_value = float(_poisson.sf(matched_reads - 1, lam))
    else:
        fitted_p_value = None
        fitted_distribution = "Poisson"
        null_fit_params: dict = {}

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
        fitted_distribution=fitted_distribution,
        null_fit_params=null_fit_params,
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
    Vertical strip/violin plot: signal vs. noise for MRD detection.

    Y-axis (left):  cfDNA reads supporting signature (log scale).
    Y-axis (right): cfDNA fraction = reads / corrected_coverage.
    X-positions:    0=empirical synthetics, 0.55=fitted-null samples,
                    1.4=cohort controls, 2.2=patient signal.

    For synthetic controls two side-by-side violins are drawn:
    - Empirical (blue):  the actual supporting read counts.
    - Fitted null (purple): samples from the fitted Poisson/NB distribution,
      showing how well the parametric model matches the empirical data.

    Detection threshold: set at max(synthetic controls) + 1. This is the
    minimum number of reads needed so that the empirical p-value falls
    below 1/(N+1), i.e. the patient signal exceeds all N synthetic controls.

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
    import matplotlib.ticker as mticker

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    null = detection.null_reads
    obs = detection.matched_supporting_reads
    corr_cov = detection.corrected_coverage

    # Floor: map 0 reads to 1e-7 fraction on the log scale.
    # Computed from coverage so the fraction axis sits at 1e-7 for zero-read points.
    _fraction_floor = 1e-7
    _floor = max(_fraction_floor * corr_cov, 0.01) if corr_cov > 0 else 0.01

    def _safe(v):
        """Map 0 → _floor so log scale is well-defined."""
        return max(float(v), _floor)

    # --- Empirical synthetic controls: violin + jittered scatter ---
    x_emp = 0.0
    x_fit = 0.55
    if len(null) > 0:
        null_plot = np.array([_safe(v) for v in null])
        if len(null) >= 5:
            parts = ax.violinplot(null_plot, positions=[x_emp], widths=0.45,
                                  showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("#3a9ad9")
                pc.set_alpha(0.35)
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.14, 0.14, size=len(null))
        ax.scatter(x_emp + jitter, null_plot, color="#3a9ad9", s=30,
                   alpha=0.85, zorder=4, label=f"Synthetic controls – empirical (n={len(null)})")

        # --- Fitted-null violin: sample from parametric fit ---
        dist_name = getattr(detection, "fitted_distribution", "Poisson")
        fit_params = getattr(detection, "null_fit_params", {})
        n_samples = max(len(null) * 20, 500)
        rng2 = np.random.default_rng(7)
        if dist_name == "NegativeBinomial" and fit_params:
            from scipy.stats import nbinom as _nbinom_s
            r_f, p_f = fit_params["r"], fit_params["p"]
            mu_f = fit_params["mu"]
            fit_samples = _nbinom_s.rvs(r_f, p_f, size=n_samples, random_state=rng2)
            fit_label = f"Fitted null – NB (μ={mu_f:.2f})"
        else:
            lam = fit_params.get("lambda", float(np.mean(null)))
            from scipy.stats import poisson as _poisson_s
            fit_samples = _poisson_s.rvs(lam, size=n_samples, random_state=rng2)
            # Note whether Jeffreys prior was applied (all-zero controls give λ=0.5/N)
            all_zero = bool(np.all(null == 0))
            jeffreys_note = " †" if all_zero else ""
            fit_label = f"Fitted null – Poisson (λ={lam:.2f}{jeffreys_note})"
        fit_plot = np.array([_safe(v) for v in fit_samples])
        if n_samples >= 5:
            parts2 = ax.violinplot(fit_plot, positions=[x_fit], widths=0.45,
                                   showmedians=False, showextrema=False)
            for pc in parts2["bodies"]:
                pc.set_facecolor("#7b2d8b")
                pc.set_alpha(0.30)
        # small jittered scatter for a few samples to show the distribution
        idx = rng2.choice(len(fit_samples), size=min(len(null), 60), replace=False)
        jitter2 = rng2.uniform(-0.14, 0.14, size=len(idx))
        ax.scatter(x_fit + jitter2, fit_plot[idx], color="#7b2d8b", s=20,
                   alpha=0.55, zorder=3, label=fit_label)

    # --- Cohort controls ---
    x_cohort = 1.4
    try:
        ctrl_data = df_tf.loc["control"]["supporting_reads"]
        if isinstance(ctrl_data, (int, float, np.integer)):
            ctrl_data = pd.Series([ctrl_data])
        for i, v in enumerate(ctrl_data.values):
            ax.scatter([x_cohort], [_safe(v)], color="#e67e22", s=80,
                       marker="D", zorder=5, alpha=0.9,
                       label="Cohort control" if i == 0 else "_nolegend_")
    except KeyError:
        pass

    # --- Patient signal ---
    x_patient = 2.2
    ax.scatter([x_patient], [_safe(obs)], color="#c0392b", s=160,
               marker="*", zorder=6, label=f"Patient signal ({obs} reads)")

    # --- Detection threshold ---
    threshold = None
    if len(null) > 0:
        threshold = int(null.max()) + 1
        ax.axhline(_safe(threshold), color="#7f8c8d", linewidth=1.5,
                   linestyle=":", alpha=0.8, zorder=4,
                   label=f"Detection threshold ({threshold} reads)\n"
                         f"= max(synthetic) + 1 → p < 1/(N+1)")

    # --- Log scale + y limits ---
    ax.set_yscale("log")
    y_top = max(_safe(obs),
                float(null.max()) if len(null) > 0 else 1,
                threshold or 1) * 6
    ax.set_ylim(_floor * 0.6, y_top)

    # --- Grid behind all data ---
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="both", linestyle=":", linewidth=0.6, color="#dde1e7", alpha=0.9)
    ax.set_facecolor("#f4f6f8")

    # --- Primary Y-axis label ---
    ax.set_ylabel("cfDNA reads supporting signature", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(round(y))}" if y >= 0.9 else "0"
    ))

    # --- Secondary Y-axis: cfDNA fraction ---
    if corr_cov > 0:
        ax2 = ax.twinx()
        ax2.set_yscale("log")
        ax2.set_ylim(_floor * 0.6 / corr_cov, y_top / corr_cov)
        ax2.set_ylabel("cfDNA fraction", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, _: format_scientific(y) if y > 0 else "0"
        ))

    # --- X-axis labels ---
    ax.set_xlim(-0.4, 2.8)
    ax.set_xticks([0.275, 1.4, 2.2])
    ax.set_xticklabels(["Synthetic\ncontrols", "Cohort\ncontrols", "Patient\nsignal"])

    # --- Title ---
    emp_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"
    if detection.fitted_p_value is not None:
        fit_str = (f"{detection.fitted_p_value:.3f}"
                   if detection.fitted_p_value >= 0.001 else f"{detection.fitted_p_value:.2e}")
        dist_short = "NB" if detection.fitted_distribution == "NegativeBinomial" else "Poisson"
        title = f"{detection.call}  (p_empirical={emp_str},  p_{dist_short}={fit_str})"
    else:
        title = f"{detection.call}  (p={emp_str})"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.85, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.spines["top"].set_visible(False)


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
