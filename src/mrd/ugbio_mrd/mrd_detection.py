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
    detection_threshold: int  # minimum reads for p < alpha from fitted null

    # Binomial model fields
    noise_rate: float             # background error rate from db_control (p_err)
    n_effective: int              # N = sig_size × mean_cov × denom_ratio for Binomial


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
    p_err: float,
    target_power: float = 0.95,
    fpr: float = 0.05,
) -> float | None:
    """
    Estimate personal LOD via analytical Binomial model.

    Mirrors the notebook's ``find_lod_at_tpr`` approach:

    1. Compute total corrected coverage N = signature_size * mean_coverage * denom_ratio.
    2. Derive the detection threshold ``n_th`` as the smallest k such that
       Binomial.sf(k-1, N, p_err) < fpr  (analytic FPR control on the null).
    3. Find the smallest TF where detection power >= target_power, i.e.
       Binomial.sf(n_th-1, N, p_err + TF) >= target_power,
       solved exactly with scipy.optimize.fsolve.

    Unlike the original Poisson simulation the noise floor (p_err) is
    included in the power calculation, so the returned LOD is the TF that
    must be *added on top of the background error rate* to reach the target
    sensitivity.

    Parameters
    ----------
    signature_size : int
        Number of loci in the filtered signature.
    mean_coverage : float
        Mean corrected coverage per locus.
    denom_ratio : float
        Denominator correction ratio.
    p_err : float
        Background error rate estimated from db_control synthetic controls
        (total supporting reads / total corrected coverage).
    target_power : float
        Required detection probability (default 0.95).
    fpr : float
        False-positive rate used to set the detection threshold (default 0.05).

    Returns
    -------
    float or None
        Personal LOD (tumor fraction above background) or None if not computable.
    """
    from scipy.optimize import fsolve
    from scipy.stats import binom as _binom

    if signature_size <= 0 or mean_coverage <= 0:
        logger.warning(f"Cannot compute personal LOD: signature_size={signature_size}, mean_coverage={mean_coverage}")
        return None

    n = int(signature_size * mean_coverage * denom_ratio)
    if n <= 0:
        return None

    # Step 1: analytic detection threshold at the given FPR under the null
    # n_th = smallest k s.t. binom.sf(k-1, n, p_err) < fpr
    k_range = np.arange(0, min(n + 1, 10000))
    sf_values = _binom.sf(k_range - 1, n, p_err)
    hits = np.where(sf_values < fpr)[0]
    if len(hits) == 0:
        logger.debug(
            "Personal LOD: no threshold satisfies FPR<%.3f "
            "(N=%d, p_err=%.2e) — LOD indeterminate",
            fpr,
            n,
            p_err,
        )
        return None
    n_th = int(hits[0])

    # Step 2: exact solve for the smallest TF where power >= target_power
    # power(tf) = binom.sf(n_th - 1, n, p_err + tf) - target_power = 0
    def _power_residual(tf):
        return np.abs(_binom.sf(n_th - 1, n, p_err + tf[0]) - target_power)

    try:
        result = fsolve(_power_residual, x0=[1e-6], full_output=True)
        lod_tf = float(result[0][0])
        if lod_tf < 0 or lod_tf > 1:
            logger.debug(
                "Personal LOD fsolve returned out-of-range value %.2e; "
                "n=%d, p_err=%.2e, n_th=%d",
                lod_tf,
                n,
                p_err,
                n_th,
            )
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Personal LOD fsolve failed: %s", exc)
        return None

    return lod_tf


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
            detection_threshold=0,
            noise_rate=0.0,
            n_effective=0,
        )

    # Handle single or multiple matched signatures (take first)
    if isinstance(matched_data, pd.DataFrame):
        matched_row = matched_data.iloc[0]
    else:
        matched_row = matched_data

    matched_reads = int(matched_row["supporting_reads"])
    matched_vaf = float(matched_row["ctdna_vaf"])
    corrected_coverage = float(matched_row["corrected_coverage"])

    # Extract synthetic control (db_control) supporting reads + background error rate
    try:
        db_control_data = df_tf.loc["db_control"]
        if isinstance(db_control_data, pd.Series):
            null_reads = np.array([int(db_control_data["supporting_reads"])])
            db_total_reads = float(db_control_data["supporting_reads"])
            db_total_cov = float(db_control_data["corrected_coverage"])
        else:
            null_reads = db_control_data["supporting_reads"].to_numpy().astype(int)
            db_total_reads = float(db_control_data["supporting_reads"].sum())
            db_total_cov = float(db_control_data["corrected_coverage"].sum())
        # Jeffreys prior: (k + 0.5) / (N + 1) — avoids p_err=0 when no reads observed
        p_err = (db_total_reads + 0.5) / (db_total_cov + 1) if db_total_cov > 0 else 0.0
    except KeyError:
        logger.warning("No db_control (synthetic) signatures found in df_tf. Cannot compute p-value.")
        null_reads = np.array([])
        p_err = 0.0

    # Fit null distribution (Poisson or Negative Binomial — kept for scatter plot)
    if len(null_reads) > 0:
        from scipy.stats import nbinom as _nbinom
        from scipy.stats import poisson as _poisson

        fitted_distribution, null_fit_params = _fit_null_distribution(null_reads)
        if fitted_distribution == "NegativeBinomial":
            r_fit, p_fit = null_fit_params["r"], null_fit_params["p"]
            fitted_p_value: float | None = float(_nbinom.sf(matched_reads - 1, r_fit, p_fit))
        else:
            lam = null_fit_params["lambda"]
            fitted_p_value = float(_poisson.sf(matched_reads - 1, lam))
    else:
        fitted_p_value = None
        fitted_distribution = "Poisson"
        null_fit_params: dict = {}

    # Assay metrics from filtered matched signature
    matched_sig_mask = df_signatures_filt["signature_type"] == "matched"
    if "signature_type" not in df_signatures_filt.columns:
        matched_sig_loci = df_signatures_filt
    else:
        matched_sig_loci = df_signatures_filt[matched_sig_mask]
    signature_size = len(matched_sig_loci)
    mean_coverage = (
        float(matched_sig_loci["coverage"].mean())
        if "coverage" in matched_sig_loci.columns and len(matched_sig_loci) > 0
        else 0.0
    )

    # Binomial p-value: P(X >= observed | N, p_err) under null Binom(n_effective, p_err)
    n_effective = int(signature_size * mean_coverage * denom_ratio)
    if len(null_reads) == 0 or n_effective == 0:
        p_value = 1.0
    else:
        from scipy.stats import binom as _binom

        p_value = float(_binom.sf(matched_reads - 1, n_effective, p_err))

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

    # Detection threshold from fitted null: reads needed for p < alpha (kept for scatter plot)
    if len(null_reads) > 0:
        from scipy.stats import nbinom as _nbinom_t
        from scipy.stats import poisson as _poisson_t

        if fitted_distribution == "NegativeBinomial":
            detection_threshold = int(_nbinom_t.ppf(1 - alpha, null_fit_params["r"], null_fit_params["p"])) + 1
        else:
            lam_t = null_fit_params["lambda"]
            detection_threshold = int(_poisson_t.ppf(1 - alpha, lam_t)) + 1
    else:
        detection_threshold = 1

    # Personal LOD (Binomial model with noise floor, analytic threshold at FPR=5%)
    personal_lod = compute_personal_lod(
        signature_size=signature_size,
        mean_coverage=mean_coverage,
        denom_ratio=denom_ratio,
        p_err=p_err,
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
        detection_threshold=detection_threshold,
        noise_rate=p_err,
        n_effective=n_effective,
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
        _, ax = plt.subplots(figsize=(8, 6))

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

    # --- Empirical synthetic controls: jittered scatter only ---
    x_emp = 0.0
    x_fit = 0.6
    if len(null) > 0:
        null_plot = np.array([_safe(v) for v in null])
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.14, 0.14, size=len(null))
        ax.scatter(x_emp + jitter, null_plot, color="#3a9ad9", s=30,
                   alpha=0.85, zorder=4, label=f"Synthetic controls (n={len(null)})")

        # --- Binomial null distribution: boxplot ---
        n_eff = getattr(detection, "n_effective", 0)
        p_err_val = getattr(detection, "noise_rate", 0.0)
        n_samples = max(len(null) * 20, 500)
        rng2 = np.random.default_rng(7)
        if n_eff > 0:
            from scipy.stats import binom as _binom_s
            fit_samples = _binom_s.rvs(n_eff, p_err_val, size=n_samples, random_state=rng2)
            p_err_str = format_scientific(p_err_val) if p_err_val > 0 else "0"
            fit_label = f"Binomial null (N={n_eff:,}, p_err={p_err_str})"
        else:
            # Fallback: Poisson from empirical null mean
            lam = float(np.mean(null)) if len(null) > 0 else 0.01
            from scipy.stats import poisson as _poisson_s
            fit_samples = _poisson_s.rvs(max(lam, 1e-9), size=n_samples, random_state=rng2)
            fit_label = f"Poisson fallback (λ={lam:.2f})"
        fit_plot = np.array([_safe(v) for v in fit_samples])
        bp = ax.boxplot(fit_plot, positions=[x_fit], widths=0.35,
                        patch_artist=True, manage_ticks=False,
                        medianprops={"color": "#4a0e5c", "linewidth": 1.5},
                        flierprops={"marker": ""},
                        whiskerprops={"color": "#7b2d8b"},
                        capprops={"color": "#7b2d8b"})
        for patch in bp["boxes"]:
            patch.set_facecolor("#7b2d8b")
            patch.set_alpha(0.25)
        # Invisible scatter for legend entry
        ax.scatter([], [], color="#7b2d8b", s=30, alpha=0.6, marker="s", label=fit_label)

    # --- Cohort controls ---
    x_cohort = 1.5
    try:
        ctrl_data = df_tf.loc["control"]["supporting_reads"]
        if isinstance(ctrl_data, (int, float, np.integer)):
            ctrl_data = pd.Series([ctrl_data])
        rng3 = np.random.default_rng(99)
        jitter_c = rng3.uniform(-0.14, 0.14, size=len(ctrl_data))
        for i, v in enumerate(ctrl_data.values):
            ax.scatter([x_cohort + jitter_c[i]], [_safe(v)], color="#e67e22", s=30,
                       marker="D", zorder=5, alpha=0.9,
                       label="Cohort control" if i == 0 else "_nolegend_")
    except KeyError:
        pass

    # --- Patient signal ---
    x_patient = 2.3
    ax.scatter([x_patient], [_safe(obs)], color="#c0392b", s=160,
               marker="*", zorder=6, label=f"Patient signal ({obs} reads)")

    # --- LOD line: convert TF → expected reads = LOD_TF × n_effective ---
    lod_reads = None
    if detection.personal_lod is not None and detection.n_effective > 0:
        lod_reads = detection.personal_lod * detection.n_effective
        lod_str = format_scientific(detection.personal_lod)
        ax.axhline(_safe(lod_reads), color="#e67e22", linewidth=1.8,
                   linestyle="--", alpha=0.9, zorder=4,
                   label=f"LOD 95% (TF={lod_str}, {lod_reads:.1f} reads)")

    # --- Log scale + y limits ---
    ax.set_yscale("log")
    y_top = max(_safe(obs),
                float(null.max()) if len(null) > 0 else 1,
                lod_reads if lod_reads else 1) * 6
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

    # --- Secondary Y-axis: ctDNA VAF ---
    if corr_cov > 0:
        ax2 = ax.twinx()
        ax2.set_yscale("log")
        ax2.set_ylim(_floor * 0.6 / corr_cov, y_top / corr_cov)
        ax2.set_ylabel("ctDNA VAF", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, _: format_scientific(y) if y > 0 else "0"
        ))
        # Move secondary y-axis label inward so legend has space
        ax2.yaxis.set_label_coords(1.08, 0.5)

    # --- X-axis labels ---
    ax.set_xlim(-0.5, 3.0)
    ax.set_xticks([0.3, 1.5, 2.3])
    ax.set_xticklabels(["Synthetic\ncontrols", "Cohort\ncontrols", "Patient\nsignal"])

    # --- Title ---
    binom_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"
    title = f"{detection.call}  (p_binomial={binom_str})"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.85, loc="upper left",
              bbox_to_anchor=(1.18, 1), borderaxespad=0)
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
