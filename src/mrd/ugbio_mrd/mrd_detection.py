"""
MRD statistical detection framework.

Implements the MRD detection procedure:
- Binomial p-value under a background noise model derived from db_control signatures
- Detection call (MRD Detected / Not Detected / Indeterminate)
- Personal LOD estimation via analytical Binomial model

The test statistic is the count of supporting reads passing quality
filters (SNVQ >= 60, MAPQ >= 60, filt > 0) — the same metric already
computed by the existing pipeline. The noise rate (p_err) is estimated
from the db_control synthetic signatures via MLE; a Jeffreys-prior floor is applied
only when zero background reads are observed to avoid a degenerate null distribution.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom, poisson
from ugbio_core.logger import logger

# Significance threshold (alpha) for the MRD detection call.
# The call is made by comparing the Binomial p-value to this value.
DEFAULT_ALPHA: float = 0.01

# FPR used *only* for personal LOD estimation (kept at 5% to match the
# recall-threshold approach: LOD is the TF that achieves 95% recall at
# the 5th-percentile detection boundary, independent of the call alpha).
DEFAULT_LOD_FPR: float = 0.05

# QC thresholds — displayed as checkboxes in the report
MIN_SIGNATURE_SIZE: int = 500  # minimum filtered signature loci
MIN_MEAN_COVERAGE: float = 15.0  # minimum mean coverage at signature loci
MIN_SYNTHETIC_CONTROLS: int = 20  # minimum db_control replicates for reliable null
# For multi-read QC: Bonferroni family-wise error rate for per-locus outlier detection.
# The check fails when any locus has Poisson p-value < threshold / signature_size,
# i.e., its read count is a significant outlier (e.g. a germline variant).
MULTI_READ_ENRICHMENT_PVALUE_THRESHOLD: float = 0.01


@dataclass
class QcCheck:
    """Single QC check result shown as a pass/fail checkbox in the report."""

    label: str
    value_str: str  # formatted observed value
    threshold_str: str  # formatted threshold for display
    passed: bool


def _binom_detection_threshold(n: int, p_err: float, alpha: float) -> int | None:
    """Return the smallest read count k at which the Binomial right-tail probability falls below *alpha*.

    Formally, returns ``min{k : P(X ≥ k | Binom(n, p_err)) < alpha}``, i.e., the minimum number of
    supporting reads required to reject the noise-only null hypothesis at significance level *alpha*.

    The search range upper bound is set dynamically via the Binomial PPF at the 99.99th percentile
    plus a small safety buffer of 10, so the function remains correct even when ``n * p_err`` is large
    (a fixed cap would miss the true threshold in high-coverage or high-noise-rate regimes).

    Parameters
    ----------
    n : int
        Effective number of trials (signature_size × mean_coverage × denom_ratio).
    p_err : float
        Per-locus background noise rate (MLE from db_control signatures, or Jeffreys-prior floor
        when zero background reads were observed).
    alpha : float
        Significance level; the caller supplies either the detection alpha or the LOD FPR.

    Returns
    -------
    int or None
        Detection threshold k, or ``None`` when no k satisfies the criterion (meaning the noise
        distribution is so diffuse that significance cannot be achieved at the given alpha).
    """
    k_max = int(binom.ppf(0.9999, n, max(p_err, 1e-12))) + 10
    sf_vals = binom.sf(np.arange(k_max + 1) - 1, n, p_err)
    hits = np.where(sf_vals < alpha)[0]
    return int(hits[0]) if len(hits) > 0 else None


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

    # Assay metrics
    signature_size: int  # number of loci in filtered signature
    mean_coverage: float  # mean coverage at signature loci
    corrected_coverage: float  # total corrected coverage
    detection_threshold: int  # minimum reads for p < alpha from fitted null

    # Binomial model fields
    noise_rate: float  # background error rate from db_control (p_err)
    n_effective: int  # N = sig_size × mean_cov × denom_ratio for Binomial
    jeffreys_prior_applied: bool  # True when no db_control reads observed (p_err floor via prior)

    # QC checks (shown as pass/fail checkboxes in the report)
    qc_checks: list = field(default_factory=list)  # list[QcCheck]

    # Significance threshold used for this detection call (stored for plot labels)
    alpha: float = DEFAULT_ALPHA


def compute_personal_lod(  # noqa: PLR0911
    n: int,
    p_err: float,
    target_recall: float = 0.95,
    fpr: float = DEFAULT_LOD_FPR,
) -> float | None:
    """
    Estimate personal LOD via analytical Binomial model.

    Mirrors the notebook's ``find_lod_at_tpr`` approach:

    1. Use the pre-computed effective trial count N (= corrected_coverage for the matched signature).
    2. Derive the detection threshold ``n_th`` as the smallest k such that
       Binomial.sf(k-1, N, p_err) < fpr  (analytic FPR control on the null).
    3. Find the smallest TF where recall >= target_recall, i.e.
       Binomial.sf(n_th-1, N, p_err + TF) = target_recall.
       Since recall is monotone increasing in TF, the root is bracketed on
       [0, 1 - p_err] and solved with ``scipy.optimize.brentq``.

    Unlike the original Poisson simulation the noise floor (p_err) is
    included in the recall calculation, so the returned LOD is the TF that
    must be *added on top of the background error rate* to reach the target
    sensitivity.

    Parameters
    ----------
    n : int
        Effective number of Binomial trials.  Callers should pass the
        ``corrected_coverage`` value already computed by
        ``get_tf_from_filtered_data`` (= ceil(sum(coverage) * denom_ratio))
        so that the LOD and the reported ctDNA VAF share exactly the same
        denominator.
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
    if n <= 0:
        logger.warning("Cannot compute personal LOD: n=%d", n)
        return None

    # Step 1: analytic detection threshold at the given FPR under the null.
    # n_th = smallest k s.t. P(X >= k | Binom(n, p_err)) < fpr
    n_th = _binom_detection_threshold(n, p_err, fpr)
    if n_th is None:
        logger.debug(
            "Personal LOD: no threshold satisfies FPR<%.3f (N=%d, p_err=%.2e) — LOD indeterminate",
            fpr,
            n,
            p_err,
        )
        return None

    # Step 2: find the smallest TF where recall >= target_recall.
    # recall(tf) = binom.sf(n_th - 1, n, p_err + tf) is monotone increasing in tf.
    # Use brentq on the signed residual over the bracket [0, 1 - p_err]:
    #   at tf=0  recall = binom.sf(n_th-1, n, p_err) < fpr <= target_recall → residual < 0
    #   at tf=1-p_err  p=1  recall=1 >= target_recall                        → residual > 0
    def _recall_residual(tf):
        return binom.sf(n_th - 1, n, p_err + tf) - target_recall

    tf_lo, tf_hi = 0.0, 1.0 - p_err
    try:
        if _recall_residual(tf_lo) >= 0:
            # recall is already >= target_recall with no tumor fraction
            return 0.0
        if _recall_residual(tf_hi) < 0:
            logger.debug(
                "Personal LOD: cannot reach target recall even at tf=1; n=%d, p_err=%.2e, n_th=%d",
                n,
                p_err,
                n_th,
            )
            return None
        lod_tf = brentq(_recall_residual, tf_lo, tf_hi, xtol=1e-10, rtol=1e-8)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Personal LOD brentq failed: %s", exc)
        return None

    return float(lod_tf)


def run_detection_analysis(  # noqa: PLR0912, PLR0915, C901
    df_tf: pd.DataFrame,
    df_signatures_filt: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    df_supporting_reads_per_locus: pd.DataFrame | None = None,
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
        The ``corrected_coverage`` column is the single authoritative Binomial
        trial count (= ceil(sum(coverage) * denom_ratio)) and is used directly
        for the p-value, detection threshold, and LOD calculations so that all
        three share exactly the same denominator as the reported ctDNA VAF.
    df_signatures_filt : pd.DataFrame
        Filtered signature dataframe with per-locus coverage (used for QC metrics).
    alpha : float
        Significance threshold for detection call (default ``DEFAULT_ALPHA`` = 0.01).
        Personal LOD always uses ``DEFAULT_LOD_FPR`` (5%) independently of this.

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
            personal_lod=None,
            signature_size=0,
            mean_coverage=0.0,
            corrected_coverage=0.0,
            detection_threshold=0,
            noise_rate=0.0,
            n_effective=0,
            jeffreys_prior_applied=False,
            qc_checks=[],
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
        # Noise rate estimation:
        # - When background reads are observed, use MLE: p_err = k / N.
        # - When zero reads are observed, apply Jeffreys prior: p_err = 0.5 / (N + 1)
        #   to avoid a hard zero that would make the null Binomial degenerate.
        # If total corrected coverage is zero the null model has no depth; treat as
        # missing controls (p_err=0.0 + null_reads kept empty) so the call is Indeterminate.
        raw_reads_zero = db_total_reads == 0
        if db_total_cov > 0:
            if raw_reads_zero:
                p_err = 0.5 / (db_total_cov + 1)  # Jeffreys prior floor
            else:
                p_err = db_total_reads / db_total_cov  # MLE
        else:
            logger.warning(
                "db_control corrected_coverage is zero despite %d synthetic control(s) present — "
                "null model depth invalid; setting call to Indeterminate.",
                len(null_reads),
            )
            null_reads = np.array([])  # force Indeterminate path
            p_err = 0.0
    except KeyError:
        logger.warning("No db_control (synthetic) signatures found in df_tf. Cannot compute p-value.")
        null_reads = np.array([])
        p_err = 0.0
        raw_reads_zero = False

    # Assay metrics from filtered matched signature
    if "signature_type" not in df_signatures_filt.columns:
        matched_sig_loci = df_signatures_filt
    else:
        matched_sig_loci = df_signatures_filt[df_signatures_filt["signature_type"] == "matched"]
    signature_size = len(matched_sig_loci)
    mean_coverage = (
        float(matched_sig_loci["coverage"].mean())
        if "coverage" in matched_sig_loci.columns and len(matched_sig_loci) > 0
        else 0.0
    )

    # Binomial p-value: P(X >= observed | N, p_err) under null Binom(n_effective, p_err).
    # n_effective comes directly from df_tf corrected_coverage, which is the same denominator
    # used to compute ctdna_vaf — ensuring p-value, LOD, and VAF all share a single N.
    n_effective = int(corrected_coverage)
    if len(null_reads) == 0 or n_effective == 0:
        p_value = 1.0
    else:
        p_value = float(binom.sf(matched_reads - 1, n_effective, p_err))

    # QC checks — displayed as pass/fail checkboxes; do NOT force Indeterminate
    qc_checks: list[QcCheck] = [
        QcCheck(
            label="Signature size",
            value_str=f"{signature_size:,} loci",
            threshold_str=f"≥ {MIN_SIGNATURE_SIZE:,} loci",
            passed=signature_size >= MIN_SIGNATURE_SIZE,  # noqa: PLR2004
        ),
        QcCheck(
            label="Mean coverage",
            value_str=f"{mean_coverage:.1f}×",
            threshold_str=f"≥ {MIN_MEAN_COVERAGE:.0f}×",
            passed=mean_coverage >= MIN_MEAN_COVERAGE,  # noqa: PLR2004
        ),
        QcCheck(
            label="Synthetic controls",
            value_str=str(len(null_reads)),
            threshold_str=f"≥ {MIN_SYNTHETIC_CONTROLS}",
            passed=len(null_reads) >= MIN_SYNTHETIC_CONTROLS,  # noqa: PLR2004
        ),
    ]
    if df_supporting_reads_per_locus is not None and signature_size > 0:
        # --- Check: matched signature outlier detection ---
        try:
            matched_per_locus = df_supporting_reads_per_locus.query("signature_type == 'matched'")
            lam = mean_coverage * matched_vaf  # expected reads per locus under MRD model
            tf_str = f"{matched_vaf:.2%}" if matched_vaf >= 1e-4 else f"{matched_vaf:.2e}"  # noqa: PLR2004
            reads_arr = matched_per_locus["supporting_reads"].to_numpy()
            if lam > 0 and len(reads_arr) > 0:
                # Per-locus outlier detection: Bonferroni-corrected Poisson right-tail test.
                # Flags loci with significantly more reads than expected at the measured TF.
                per_locus_pvals = poisson.sf(reads_arr - 1, lam)
                n_outliers = int((per_locus_pvals * signature_size < MULTI_READ_ENRICHMENT_PVALUE_THRESHOLD).sum())
                max_reads = int(reads_arr.max())
                min_pval_corrected = float(per_locus_pvals.min()) * signature_size
            else:
                n_outliers = 0
                max_reads = int(reads_arr.max()) if len(reads_arr) > 0 else 0
                min_pval_corrected = 1.0
            qc_checks.append(
                QcCheck(
                    label="Expected multi-read support distribution (matched)",
                    value_str=(
                        f"{n_outliers} outlier {'locus' if n_outliers == 1 else 'loci'}"
                        f" (max {max_reads} reads/locus, Bonferroni p={min_pval_corrected:.3f})"
                    ),
                    threshold_str=(
                        f"0 outlier loci"
                        f" (Bonferroni-corrected p \u2265 {MULTI_READ_ENRICHMENT_PVALUE_THRESHOLD:.0%},"
                        f" expected \u03bb={lam:.3f} reads/locus at TF={tf_str})"
                    ),
                    passed=n_outliers == 0,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not compute matched multi-read support QC check: %s", exc)

        # --- Checks: control outlier detection (synthetic and cohort, when present) ---
        lam_ctrl = mean_coverage * p_err  # expected reads per locus at background noise rate
        for ctrl_type, ctrl_label in [("db_control", "synthetic controls"), ("control", "cohort controls")]:
            try:
                ctrl_per_locus = df_supporting_reads_per_locus.query(f"signature_type == '{ctrl_type}'")
                if len(ctrl_per_locus) == 0 or lam_ctrl <= 0:
                    continue
                # Take max reads per unique locus across all control signatures of this type
                ctrl_max_per_locus = (
                    ctrl_per_locus.reset_index().groupby(["chrom", "pos"])["supporting_reads"].max().to_numpy()
                )
                n_ctrl_loci = len(ctrl_max_per_locus)
                ctrl_pvals = poisson.sf(ctrl_max_per_locus - 1, lam_ctrl)
                n_ctrl_outliers = int((ctrl_pvals * n_ctrl_loci < MULTI_READ_ENRICHMENT_PVALUE_THRESHOLD).sum())
                max_ctrl_reads = int(ctrl_max_per_locus.max())
                min_ctrl_pval = float(ctrl_pvals.min()) * n_ctrl_loci
                qc_checks.append(
                    QcCheck(
                        label=f"Expected multi-read support distribution ({ctrl_label})",
                        value_str=(
                            f"{n_ctrl_outliers} outlier {'locus' if n_ctrl_outliers == 1 else 'loci'}"
                            f" (max {max_ctrl_reads} reads/locus, Bonferroni p={min_ctrl_pval:.3f})"
                        ),
                        threshold_str=(
                            f"0 outlier loci"
                            f" (Bonferroni-corrected p \u2265 {MULTI_READ_ENRICHMENT_PVALUE_THRESHOLD:.0%},"
                            f" expected \u03bb={lam_ctrl:.3f} reads/locus at noise rate)"
                        ),
                        passed=n_ctrl_outliers == 0,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not compute %s multi-read support QC check: %s", ctrl_type, exc)

    # Detection call
    if len(null_reads) == 0 or n_effective == 0:
        detected = None
        call = "Indeterminate"
    elif p_value <= alpha:
        detected = True
        call = "MRD Detected"
    else:
        detected = False
        call = "MRD Not Detected"

    # Detection threshold from Binomial model: smallest k s.t. Binom.sf(k-1, n_effective, p_err) < alpha
    if n_effective > 0 and p_err > 0:
        detection_threshold = _binom_detection_threshold(n_effective, p_err, alpha) or 1
    else:
        detection_threshold = 1

    # Personal LOD — always at DEFAULT_LOD_FPR (5%), independent of detection alpha.
    # Uses the same n_effective as the p-value and detection threshold.
    personal_lod = compute_personal_lod(
        n=n_effective,
        p_err=p_err,
        fpr=DEFAULT_LOD_FPR,
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
        personal_lod=personal_lod,
        signature_size=signature_size,
        mean_coverage=mean_coverage,
        corrected_coverage=corrected_coverage,
        detection_threshold=detection_threshold,
        noise_rate=p_err,
        n_effective=n_effective,
        jeffreys_prior_applied=raw_reads_zero,
        qc_checks=qc_checks,
        alpha=alpha,
    )


def plot_null_distribution(  # noqa: PLR0915, PLR0912, C901
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
    (read from detection.alpha) drawn as a dashed line on the scatter plot.
    LOD line uses DEFAULT_LOD_FPR (5%) regardless of detection alpha.

    Parameters
    ----------
    detection : DetectionResult
        Result from run_detection_analysis().
    df_tf : pd.DataFrame
        Tumor fraction dataframe (index: signature_type, signature).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates a new figure if None.
    """
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
            alpha=0.8,
            zorder=4,
            label=f"Synthetic controls (n={len(null)})",
        )

        # --- Binomial null distribution: boxplot ---
        n_eff = getattr(detection, "n_effective", 0)
        p_err_val = getattr(detection, "noise_rate", 0.0)
        n_samples = max(len(null) * 20, 500)
        rng2 = np.random.default_rng(7)
        if n_eff > 0:
            fit_samples = binom.rvs(n_eff, p_err_val, size=n_samples, random_state=rng2)
            p_err_str = format_scientific(p_err_val) if p_err_val > 0 else "0"
            fit_label = f"Binomial null (N={n_eff:,}, p_err={p_err_str})"
        else:
            # Fallback: Poisson from empirical null mean
            lam = float(np.mean(null)) if len(null) > 0 else 0.01
            fit_samples = poisson.rvs(max(lam, 1e-9), size=n_samples, random_state=rng2)
            fit_label = f"Poisson fallback (λ={lam:.2f})"
        fit_plot = np.array([_safe(v) for v in fit_samples])
        vp = ax.violinplot(
            fit_plot,
            positions=[x_fit],
            widths=0.35,
            showmedians=True,
            showextrema=True,
        )
        for body in vp["bodies"]:
            body.set_facecolor("#3a9ad9")
            body.set_edgecolor("#3a9ad9")
            body.set_alpha(0.4)
        for part in ("cbars", "cmins", "cmaxes"):
            vp[part].set_color("#3a9ad9")
        vp["cmedians"].set_color("#1e6e9e")
        # Invisible scatter for legend entry
        ax.scatter([], [], color="#3a9ad9", s=30, alpha=0.4, marker="s", label=fit_label)

    # --- Cohort controls ---
    x_cohort = 1.5
    try:
        ctrl_data = df_tf.loc["control"]["supporting_reads"]
        if isinstance(ctrl_data, int | float | np.integer):
            ctrl_data = pd.Series([ctrl_data])
        rng3 = np.random.default_rng(99)
        jitter_c = rng3.uniform(-0.14, 0.14, size=len(ctrl_data))
        for i, v in enumerate(ctrl_data.values):
            ax.scatter(
                [x_cohort + jitter_c[i]],
                [_safe(v)],
                color="#9b59b6",
                s=30,
                marker="D",
                zorder=5,
                alpha=0.8,
                label="Cohort control" if i == 0 else "_nolegend_",
            )
    except KeyError:
        logger.debug("Cohort control data not found in df_tf; plotting without cohort controls.")

    # --- Patient signal ---
    x_patient = 2.3
    ax.scatter([x_patient], [_safe(obs)], color="#c0392b", s=160, marker="*", zorder=6, label=f"Patient ({obs} reads)")

    # --- Detection threshold / LOD line ---
    # n_th: smallest k s.t. P(X >= k | n_eff, p_err) < detection.alpha (via _binom_detection_threshold).
    # LOD uses DEFAULT_LOD_FPR (5%) independently.
    n_th_plot = None
    lod_tf_plot = None
    n_eff_plot = getattr(detection, "n_effective", 0)
    p_err_plot = getattr(detection, "noise_rate", 0.0)
    if n_eff_plot > 0:
        _alpha_plot = getattr(detection, "alpha", DEFAULT_ALPHA)
        n_th_plot = _binom_detection_threshold(n_eff_plot, p_err_plot, _alpha_plot)
        if n_th_plot is not None:
            lod_tf_plot = compute_personal_lod(
                n=int(n_eff_plot),
                p_err=p_err_plot,
                fpr=DEFAULT_LOD_FPR,
            )
            # Detection threshold line: minimum reads to call a positive (alpha)
            n_th_vaf = n_th_plot / n_eff_plot
            ax.axhline(
                _safe(n_th_plot),
                color="#e67e22",
                linewidth=1.8,
                linestyle="--",
                alpha=0.9,
                zorder=4,
                label=f"Detection threshold ({format_scientific(n_th_vaf)}) | α={_alpha_plot * 100:.0f}%",
            )
            if lod_tf_plot is not None:
                lod_str = format_scientific(lod_tf_plot)
                # LOD line: expected reads at the LOD TF (n_eff × (p_err + LOD_TF))
                n_lod = float(n_eff_plot) * (p_err_plot + lod_tf_plot)
                ax.axhline(
                    _safe(n_lod),
                    color="#27ae60",
                    linewidth=1.8,
                    linestyle="-.",
                    alpha=0.9,
                    zorder=4,
                    label=f"LOD ({n_lod:.1f} reads) = {lod_str} | 95% recall",
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
    ax.set_xticklabels(["Synthetic\ncontrols", "Cohort\ncontrols", "Patient"])

    # --- Title ---
    binom_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"  # noqa: PLR2004
    title = f"Patient vs. controls  (p={binom_str})"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.85, loc="upper left", bbox_to_anchor=(1.18, 1), borderaxespad=0)
    ax.spines["top"].set_visible(False)


_SUPERSCRIPT_MINUS = "\u207b"
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def format_scientific(value: float, precision: int = 1) -> str:
    """Format a float in scientific notation with correct sign on the exponent."""
    if value is None:
        return "N/A"
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exp
    exp_str = str(abs(exp)).translate(_SUPERSCRIPT_DIGITS)
    sign_str = _SUPERSCRIPT_MINUS if exp < 0 else "+"
    if abs(mantissa - 1.0) < 0.05:  # noqa: PLR2004
        return f"10{sign_str}{exp_str}"
    return f"{mantissa:.{precision}f} \u00d7 10{sign_str}{exp_str}"
