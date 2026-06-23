"""
MRD statistical detection framework.

Implements the MRD detection procedure:
- Binomial p-value under a background noise model derived from db_control signatures
- Detection call (MRD Detected / Not Detected / Indeterminate)
- Personal LOD estimation via analytical Binomial model

The test statistic is the count of supporting reads passing quality
filters (SNVQ >= 60, MAPQ >= 60, filt > 0) — the same metric already
computed by the existing pipeline. The noise rate (p_err) is estimated
from the db_control synthetic signatures via Jeffreys-prior Bayes estimate.

Reference: bfx-read-the-docs/docs/tumor-informed-mrd/mrd_dev_plan_analysis.md
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ugbio_core.logger import logger

# Uniform false-positive rate / significance threshold used throughout all
# detection, LOD, and threshold calculations in this module.
DEFAULT_FPR: float = 0.05

# QC thresholds — results are called Indeterminate when any of these fail
MIN_SIGNATURE_SIZE: int = 500  # minimum filtered signature loci
MIN_MEAN_COVERAGE: float = 15.0  # minimum mean coverage at signature loci
MIN_SYNTHETIC_CONTROLS: int = 2  # minimum db_control replicates for reliable null


@dataclass
class DetectionResult:
    """Result of MRD statistical detection analysis."""

    # Detection call
    detected: bool | None  # True/False/None (indeterminate)
    call: str  # "MRD Detected" / "MRD Not Detected" / "Indeterminate"

    # Binomial p-value: P(X >= matched_reads | n_effective, noise_rate)
    p_value: float

    # Observed signal
    matched_supporting_reads: int
    matched_ctdna_vaf: float

    # Null distribution summary
    null_median_reads: float
    null_max_reads: int
    n_synthetic_controls: int

    # Personal LOD (95% recall)
    personal_lod: float | None  # TF at which recall >= 95%

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
    noise_rate: float  # background error rate from db_control (p_err)
    n_effective: int  # N = sig_size × mean_cov × denom_ratio for Binomial
    jeffreys_prior_applied: bool  # True when no db_control reads observed (p_err floor via prior)

    # QC flags
    qc_flags: list = field(default_factory=list)  # non-empty ⇒ Indeterminate call


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
    if n < 3:  # noqa: PLR2004
        return "Poisson", {"lambda": mu}

    var = float(np.var(null_reads, ddof=1))
    disp_index = var / mu  # = 1 for Poisson; > 1 means overdispersed

    if disp_index <= 1.5:  # noqa: PLR2004
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


def compute_personal_lod(
    signature_size: int,
    mean_coverage: float,
    denom_ratio: float,
    p_err: float,
    target_recall: float = 0.95,
    fpr: float = DEFAULT_FPR,
) -> float | None:
    """
    Estimate personal LOD via analytical Binomial model.

    Mirrors the notebook's ``find_lod_at_tpr`` approach:

    1. Compute total corrected coverage N = signature_size * mean_coverage * denom_ratio.
    2. Derive the detection threshold ``n_th`` as the smallest k such that
       Binomial.sf(k-1, N, p_err) < fpr  (analytic FPR control on the null).
    3. Find the smallest TF where recall >= target_recall, i.e.
       Binomial.sf(n_th-1, N, p_err + TF) >= target_recall,
       solved exactly with scipy.optimize.fsolve.

    Unlike the original Poisson simulation the noise floor (p_err) is
    included in the recall calculation, so the returned LOD is the TF that
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
    target_recall : float
        Required detection probability (default 0.95).
    fpr : float
        False-positive rate used to set the detection threshold (default 0.05).

    Returns
    -------
    float or None
        Personal LOD (tumor fraction above background) or None if not computable.
    """
    from scipy.optimize import fsolve  # noqa: PLC0415
    from scipy.stats import binom as _binom  # noqa: PLC0415

    if signature_size <= 0 or mean_coverage <= 0:
        logger.warning(f"Cannot compute personal LOD: signature_size={signature_size}, mean_coverage={mean_coverage}")
        return None

    n = int(signature_size * mean_coverage * denom_ratio)
    if n <= 0:
        return None

    # Step 1: analytic detection threshold at the given FPR under the null
    # n_th = smallest k s.t. binom.sf(k-1, n, p_err) < fpr
    # Use ppf to set the search upper bound dynamically (avoids missing n_th
    # when n*p_err is large and the 95th percentile exceeds a fixed cap).
    k_max = int(_binom.ppf(0.9999, n, max(p_err, 1e-12))) + 10
    k_range = np.arange(0, k_max + 1)
    sf_values = _binom.sf(k_range - 1, n, p_err)
    hits = np.where(sf_values < fpr)[0]
    if len(hits) == 0:
        logger.debug(
            "Personal LOD: no threshold satisfies FPR<%.3f (N=%d, p_err=%.2e) — LOD indeterminate",
            fpr,
            n,
            p_err,
        )
        return None
    n_th = int(hits[0])

    # Step 2: exact solve for the smallest TF where recall >= target_recall
    # recall(tf) = binom.sf(n_th - 1, n, p_err + tf) - target_recall = 0
    def _recall_residual(tf):
        return np.abs(_binom.sf(n_th - 1, n, p_err + tf[0]) - target_recall)

    try:
        result = fsolve(_recall_residual, x0=[1e-6], full_output=True)
        lod_tf = float(result[0][0])
        if lod_tf < 0 or lod_tf > 1:
            logger.debug(
                "Personal LOD fsolve returned out-of-range value %.2e; n=%d, p_err=%.2e, n_th=%d",
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


def run_detection_analysis(  # noqa: PLR0912, PLR0915, C901
    df_tf: pd.DataFrame,
    df_signatures_filt: pd.DataFrame,
    denom_ratio: float,
    alpha: float = 0.01,
) -> DetectionResult:
    """
    Run the full MRD detection analysis.

    Extracts matched and synthetic control supporting read counts from
    the existing df_tf dataframe, computes a Binomial p-value against
    a noise model derived from db_control signatures, makes a detection
    call, and estimates personal LOD.

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
            jeffreys_prior_applied=False,
            qc_flags=["No matched signature found in df_tf"],
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
        raw_reads_zero = db_total_reads == 0
        p_err = (db_total_reads + 0.5) / (db_total_cov + 1) if db_total_cov > 0 else 0.0
    except KeyError:
        logger.warning("No db_control (synthetic) signatures found in df_tf. Cannot compute p-value.")
        null_reads = np.array([])
        p_err = 0.0
        raw_reads_zero = False

    # Fit null distribution (Poisson or Negative Binomial — kept for scatter plot)
    if len(null_reads) > 0:
        from scipy.stats import nbinom as _nbinom  # noqa: PLC0415
        from scipy.stats import poisson as _poisson  # noqa: PLC0415

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
        from scipy.stats import binom as _binom  # noqa: PLC0415

        p_value = float(_binom.sf(matched_reads - 1, n_effective, p_err))

    # QC checks — any failure forces Indeterminate
    qc_flags = []
    if signature_size < MIN_SIGNATURE_SIZE:  # noqa: PLR2004
        qc_flags.append(
            f"Signature too small: {signature_size} loci (minimum {MIN_SIGNATURE_SIZE})"
        )
    if mean_coverage < MIN_MEAN_COVERAGE:  # noqa: PLR2004
        qc_flags.append(
            f"Low mean coverage: {mean_coverage:.1f}x (minimum {MIN_MEAN_COVERAGE:.0f}x)"
        )
    if len(null_reads) < MIN_SYNTHETIC_CONTROLS:  # noqa: PLR2004
        qc_flags.append(
            f"Insufficient synthetic controls: {len(null_reads)} (minimum {MIN_SYNTHETIC_CONTROLS})"
        )

    # Detection call
    if qc_flags:
        detected = None
        call = "Indeterminate"
    elif len(null_reads) == 0:
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
        from scipy.stats import nbinom as _nbinom_t  # noqa: PLC0415
        from scipy.stats import poisson as _poisson_t  # noqa: PLC0415

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
        jeffreys_prior_applied=raw_reads_zero,
        qc_flags=qc_flags,
    )


def plot_null_distribution(  # noqa: PLR0915, C901
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

    Detection threshold: minimum reads needed for Binomial p-value < alpha
    under the fitted Poisson/NB null (used only for scatter-plot annotation).

    Parameters
    ----------
    detection : DetectionResult
        Result from run_detection_analysis().
    df_tf : pd.DataFrame
        Tumor fraction dataframe (index: signature_type, signature).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates a new figure if None.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.ticker as mticker  # noqa: PLC0415

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
        ax.scatter(
            x_emp + jitter,
            null_plot,
            color="#3a9ad9",
            s=30,
            alpha=0.85,
            zorder=4,
            label=f"Synthetic controls (n={len(null)})",
        )

        # --- Binomial null distribution: boxplot ---
        n_eff = getattr(detection, "n_effective", 0)
        p_err_val = getattr(detection, "noise_rate", 0.0)
        n_samples = max(len(null) * 20, 500)
        rng2 = np.random.default_rng(7)
        if n_eff > 0:
            from scipy.stats import binom as _binom_s  # noqa: PLC0415

            fit_samples = _binom_s.rvs(n_eff, p_err_val, size=n_samples, random_state=rng2)
            p_err_str = format_scientific(p_err_val) if p_err_val > 0 else "0"
            fit_label = f"Binomial null (N={n_eff:,}, p_err={p_err_str})"
        else:
            # Fallback: Poisson from empirical null mean
            lam = float(np.mean(null)) if len(null) > 0 else 0.01
            from scipy.stats import poisson as _poisson_s  # noqa: PLC0415

            fit_samples = _poisson_s.rvs(max(lam, 1e-9), size=n_samples, random_state=rng2)
            fit_label = f"Poisson fallback (λ={lam:.2f})"
        fit_plot = np.array([_safe(v) for v in fit_samples])
        bp = ax.boxplot(
            fit_plot,
            positions=[x_fit],
            widths=0.35,
            patch_artist=True,
            manage_ticks=False,
            medianprops={"color": "#4a0e5c", "linewidth": 1.5},
            flierprops={"marker": ""},
            whiskerprops={"color": "#7b2d8b"},
            capprops={"color": "#7b2d8b"},
        )
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
            ax.scatter(
                [x_cohort + jitter_c[i]],
                [_safe(v)],
                color="#e67e22",
                s=30,
                marker="D",
                zorder=5,
                alpha=0.9,
                label="Cohort control" if i == 0 else "_nolegend_",
            )
    except KeyError:
        pass

    # --- Patient signal ---
    x_patient = 2.3
    ax.scatter(
        [x_patient], [_safe(obs)], color="#c0392b", s=160, marker="*", zorder=6, label=f"Patient signal ({obs} reads)"
    )

    # --- Detection threshold / LOD line ---
    # n_th: smallest k s.t. P(X >= k | n_eff, p_err) < 5% (95th percentile of null).
    # k_max is set dynamically via ppf to avoid missing the percentile when
    # n_eff*p_err is large (fixed caps like 10000 fail for unfiltered reads panels).
    n_th_plot = None
    lod_tf_plot = None
    n_eff_plot = getattr(detection, "n_effective", 0)
    p_err_plot = getattr(detection, "noise_rate", 0.0)
    if n_eff_plot > 0:
        from scipy.stats import binom as _binom_lod  # noqa: PLC0415

        k_max = int(_binom_lod.ppf(0.9999, n_eff_plot, max(p_err_plot, 1e-12))) + 10
        k_range = np.arange(0, k_max + 1)
        sf_vals = _binom_lod.sf(k_range - 1, n_eff_plot, p_err_plot)
        hits = np.where(sf_vals < DEFAULT_FPR)[0]
        if len(hits) > 0:
            n_th_plot = int(hits[0])
            lod_tf_plot = compute_personal_lod(
                signature_size=1,
                mean_coverage=float(n_eff_plot),
                denom_ratio=1.0,
                p_err=p_err_plot,
            )
            # Detection threshold line: minimum reads to call a positive (DEFAULT_FPR)
            n_th_vaf = n_th_plot / n_eff_plot if n_eff_plot > 0 else 0.0
            ax.axhline(
                _safe(n_th_plot),
                color="#e67e22",
                linewidth=1.8,
                linestyle="--",
                alpha=0.9,
                zorder=4,
                label=f"Detection threshold ({format_scientific(n_th_vaf)}) | {DEFAULT_FPR * 100:.0f}% FPR",
            )
            if lod_tf_plot is not None:
                lod_str = format_scientific(lod_tf_plot)
                # LOD line: expected reads at the LOD TF (n_eff × (p_err + LOD_TF))
                n_lod = float(n_eff_plot) * (p_err_plot + lod_tf_plot)
                ax.axhline(
                    _safe(n_lod),
                    color="#8e44ad",
                    linewidth=1.8,
                    linestyle="-.",
                    alpha=0.9,
                    zorder=4,
                    label=f"LOD signal ({n_lod:.1f} reads) = {lod_str} | 95% recall",
                )

    # --- Log scale + y limits ---
    ax.set_yscale("log")
    n_lod_top = float(n_eff_plot) * (p_err_plot + lod_tf_plot) if (lod_tf_plot and n_eff_plot) else 1
    y_top = max(_safe(obs), float(null.max()) if len(null) > 0 else 1, n_th_plot if n_th_plot else 1, n_lod_top) * 6
    ax.set_ylim(_floor * 0.6, y_top)

    # --- Grid behind all data ---
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="both", linestyle=":", linewidth=0.6, color="#dde1e7", alpha=0.9)  # noqa: FBT003
    ax.set_facecolor("#f4f6f8")

    # --- Primary Y-axis label ---
    ax.set_ylabel("cfDNA reads supporting signature", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(round(y))}" if y >= 0.9 else "0"))  # noqa: PLR2004

    # --- Secondary Y-axis: ctDNA VAF ---
    if corr_cov > 0:
        ax2 = ax.twinx()
        ax2.set_yscale("log")
        ax2.set_ylim(_floor * 0.6 / corr_cov, y_top / corr_cov)
        ax2.set_ylabel("ctDNA VAF", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: format_scientific(y) if y > 0 else "0"))
        # Move secondary y-axis label further out to avoid overlap with tick labels
        ax2.yaxis.set_label_coords(1.18, 0.5)

    # --- X-axis labels ---
    ax.set_xlim(-0.5, 3.0)
    ax.set_xticks([0.3, 1.5, 2.3])
    ax.set_xticklabels(["Synthetic\ncontrols", "Cohort\ncontrols", "Patient\nsignal"])

    # --- Title ---
    binom_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"  # noqa: PLR2004
    title = f"Patient signal vs. controls  (p={binom_str})"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.85, loc="upper left", bbox_to_anchor=(1.18, 1), borderaxespad=0)
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
