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
MIN_SYNTHETIC_CONTROLS: int = 30  # minimum db_control replicates for reliable null
# For multi-read QC: Bonferroni family-wise error rate for per-locus outlier detection.
# The check fails when any locus has Poisson p-value < threshold / signature_size,
# i.e., its read count is a significant outlier (e.g. a germline variant).
MULTI_READ_ENRICHMENT_PVALUE_THRESHOLD: float = 0.01

# Minimum supporting-read count to flag a locus as a QC outlier.
# Mirrors _MIN_READS_TO_FILTER in mrd_utils.apply_multi_read_locus_filter: loci with
# only one supporting read are indistinguishable from background noise and are never
# removed by the filter, so the QC check must not count them as outliers either.
# At very low estimated TF (λ → 0) the Poisson p-value for a single read would
# otherwise fall below the Bonferroni threshold even though the filter never acts on it.
_MIN_READS_TO_QC: int = 2


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

    # Personal LOD (recall and FPR controlled by lod_recall / lod_fpr)
    personal_lod: float | None  # TF at which recall >= lod_recall (incremental, above p_err)

    # Null distribution (raw supporting read counts for each synthetic control)
    null_reads: np.ndarray  # shape (n_synthetic_controls,), dtype int

    # Assay metrics
    signature_size: int  # number of loci in filtered signature
    mean_coverage: float  # mean coverage at signature loci
    corrected_coverage: float  # total corrected coverage
    detection_threshold: int | None  # minimum reads for p < alpha from fitted null; None when no threshold exists

    # Binomial model fields
    noise_rate: float  # background error rate from db_control (p_err)
    n_effective: int  # N = sig_size × mean_cov × denom_ratio for Binomial
    jeffreys_prior_applied: bool  # True when no db_control reads observed (p_err floor via prior)

    # QC checks (shown as pass/fail checkboxes in the report)
    qc_checks: list = field(default_factory=list)  # list[QcCheck]

    # Significance threshold used for this detection call (stored for plot labels)
    alpha: float = DEFAULT_ALPHA
    lod_fpr: float = DEFAULT_LOD_FPR
    lod_recall: float = 0.95


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
    lod_fpr: float = DEFAULT_LOD_FPR,
    lod_recall: float = 0.95,
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
    lod_fpr : float
        FPR used for the personal LOD calculation (default ``DEFAULT_LOD_FPR`` = 0.05).
    lod_recall : float
        Target recall used for the personal LOD calculation (default 0.95).

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
            detection_threshold=None,
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

    # Assay metrics from filtered matched signature.
    # Use n_loci from df_tf when available (accounts for loci removed by noise/multi-read
    # filters); fall back to df_signatures_filt count for backwards compatibility.
    if "n_loci" in df_tf.columns:
        try:
            signature_size = int(df_tf.loc["matched", "n_loci"].iloc[0])
        except (KeyError, IndexError):
            signature_size = 0
        try:
            raw_coverage = float(df_tf.loc["matched", "coverage"].iloc[0])
        except (KeyError, IndexError):
            raw_coverage = 0.0
        mean_coverage = raw_coverage / signature_size if signature_size > 0 else 0.0
    else:
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
                # Guard: require >= _MIN_READS_TO_QC reads so that single-read loci are never
                # counted as outliers — consistent with apply_multi_read_locus_filter which
                # also never removes loci with fewer than _MIN_READS_TO_FILTER supporting reads.
                per_locus_pvals = poisson.sf(reads_arr - 1, lam)
                n_outliers = int(
                    (
                        (per_locus_pvals * signature_size < MULTI_READ_ENRICHMENT_PVALUE_THRESHOLD)
                        & (reads_arr >= _MIN_READS_TO_QC)
                    ).sum()
                )
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
        ctrl_sig_sizes = (
            df_signatures_filt.groupby(["signature_type", "signature"]).size()
            if "signature_type" in df_signatures_filt.columns
            else df_signatures_filt.groupby(level="signature").size()
        )
        for ctrl_type, ctrl_label in [("db_control", "synthetic controls"), ("control", "cohort controls")]:
            try:
                ctrl_per_locus = df_supporting_reads_per_locus.query(f"signature_type == '{ctrl_type}'")
                if len(ctrl_per_locus) == 0 or lam_ctrl <= 0:
                    continue
                # Per-signature Bonferroni: each signature is corrected by its own locus count.
                # A locus is flagged as an outlier if any single signature's test fires.
                all_reads = ctrl_per_locus.reset_index()[["chrom", "pos", "signature", "supporting_reads"]]
                outlier_locus_set: set = set()
                min_corrected_pval = 1.0
                max_ctrl_reads = 0
                for sig_name, sig_df in all_reads.groupby("signature"):
                    try:
                        n_sig = int(ctrl_sig_sizes.loc[ctrl_type, sig_name])
                    except (KeyError, TypeError):
                        n_sig = len(sig_df)
                    reads_arr = sig_df["supporting_reads"].to_numpy()
                    raw_pvals = poisson.sf(reads_arr - 1, lam_ctrl)
                    bonf_pvals = raw_pvals * n_sig
                    outlier_mask = (bonf_pvals < MULTI_READ_ENRICHMENT_PVALUE_THRESHOLD) & (
                        reads_arr >= _MIN_READS_TO_QC
                    )
                    outlier_locus_set.update(
                        zip(sig_df.loc[outlier_mask, "chrom"], sig_df.loc[outlier_mask, "pos"], strict=False)
                    )
                    min_corrected_pval = min(min_corrected_pval, float(bonf_pvals.min()))
                    max_ctrl_reads = max(max_ctrl_reads, int(reads_arr.max()))
                n_ctrl_outliers = len(outlier_locus_set)
                qc_checks.append(
                    QcCheck(
                        label=f"Expected multi-read support distribution ({ctrl_label})",
                        value_str=(
                            f"{n_ctrl_outliers} outlier {'locus' if n_ctrl_outliers == 1 else 'loci'}"
                            f" (max {max_ctrl_reads} reads/locus, Bonferroni p={min_corrected_pval:.3f})"
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

    # Detection threshold from Binomial model: smallest k s.t. Binom.sf(k-1, n_effective, p_err) < alpha.
    # Returns None when no integer threshold satisfies the criterion (e.g. noise too diffuse for alpha).
    if n_effective > 0 and p_err > 0:
        detection_threshold = _binom_detection_threshold(n_effective, p_err, alpha)
    else:
        detection_threshold = None

    # Personal LOD: total VAF (p_err + incremental TF) at which recall >= lod_recall.
    # Stored as total VAF so it sits on the same scale as matched_ctdna_vaf and the
    # LOD line shown in the patient vs controls plot.
    _lod_incremental = compute_personal_lod(
        n=n_effective,
        p_err=p_err,
        target_recall=lod_recall,
        fpr=lod_fpr,
    )
    personal_lod = (p_err + _lod_incremental) if _lod_incremental is not None else None

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
        lod_fpr=lod_fpr,
        lod_recall=lod_recall,
    )


def plot_patient_vs_control_vaf(  # noqa: PLR0915, PLR0912, C901
    detection: "DetectionResult",
    df_tf: pd.DataFrame,  # kept for API compatibility; cohort controls shown in separate scatter  # noqa: ARG001
    ax=None,
):
    """
    Vertical strip/violin plot: patient vs. synthetic controls.

    Left Y-axis (log): ctDNA VAF.
    Right Y-axis (log): Signature supporting reads (= VAF × corrected_coverage).

    * Synthetic controls: null_reads / corrected_coverage (VAF scale).
    * Patient: matched_ctdna_vaf; legend includes read count.
    * Cohort controls are shown in a separate scatter plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    null = detection.null_reads
    obs = detection.matched_supporting_reads
    corr_cov = detection.corrected_coverage
    _vaf_floor = 1e-7

    def _safe_vaf(v):
        return max(float(v), _vaf_floor)

    # ── Synthetic controls ────────────────────────────────────────────────────
    x_emp, x_fit = 0.0, 0.6
    if len(null) > 0 and corr_cov > 0:
        null_vafs = np.array([_safe_vaf(v / corr_cov) for v in null])
        rng = np.random.default_rng(42)
        ax.scatter(
            x_emp + rng.uniform(-0.14, 0.14, size=len(null)),
            null_vafs,
            color="#3a9ad9",
            s=30,
            alpha=0.8,
            zorder=4,
            label=f"Synthetic controls (n={len(null)})",
        )

        n_eff = getattr(detection, "n_effective", 0)
        p_err_val = getattr(detection, "noise_rate", 0.0)
        rng2 = np.random.default_rng(7)
        if n_eff > 0:
            fit_reads = binom.rvs(n_eff, p_err_val, size=max(len(null) * 20, 500), random_state=rng2)
            p_err_str = format_scientific(p_err_val) if p_err_val > 0 else "0"
            fit_label = f"Binomial null (p_err={p_err_str})"
            fit_vafs = np.array([_safe_vaf(r / n_eff) for r in fit_reads])
        else:
            lam = float(np.mean(null)) if len(null) > 0 else 0.01
            fit_reads = poisson.rvs(max(lam, 1e-9), size=max(len(null) * 20, 500), random_state=rng2)
            fit_label = f"Poisson fallback (λ={lam:.2f})"
            fit_vafs = np.array([_safe_vaf(r / max(corr_cov, 1)) for r in fit_reads])
        vp = ax.violinplot(fit_vafs, positions=[x_fit], widths=0.35, showmedians=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_facecolor("#3a9ad9")
            body.set_edgecolor("#3a9ad9")
            body.set_alpha(0.4)
        for part in ("cbars", "cmins", "cmaxes"):
            vp[part].set_color("#3a9ad9")
        vp["cmedians"].set_color("#1e6e9e")
        ax.scatter([], [], color="#3a9ad9", s=30, alpha=0.4, marker="s", label=fit_label)

    # ── Compute threshold in VAF before plotting ──────────────────────────────
    det_vaf = None
    n_eff_plot = getattr(detection, "n_effective", 0)
    p_err_plot = getattr(detection, "noise_rate", 0.0)
    _alpha_plot = getattr(detection, "alpha", DEFAULT_ALPHA)
    if n_eff_plot > 0 and corr_cov > 0:
        n_th = _binom_detection_threshold(n_eff_plot, p_err_plot, _alpha_plot)
        if n_th is not None:
            det_vaf = n_th / corr_cov

    # ── Patient signal ────────────────────────────────────────────────────────
    x_pat = 1.5
    pat_vaf = _safe_vaf(detection.matched_ctdna_vaf)
    pat_vaf_str = format_scientific(detection.matched_ctdna_vaf) if detection.matched_ctdna_vaf > 0 else "0"
    ax.scatter(
        [x_pat], [pat_vaf], color="#c0392b", s=160, marker="*", zorder=6, label=f"Patient ({obs} reads, {pat_vaf_str})"
    )

    # ── Threshold / LOD lines in VAF ─────────────────────────────────────────
    if det_vaf is not None and n_eff_plot > 0:
        n_th_reads = int(round(det_vaf * corr_cov))
        _det_label = (
            f"Detection threshold ({format_scientific(det_vaf)}, {n_th_reads} reads)" f" | α={_alpha_plot * 100:.0f}%"
        )
        ax.axhline(
            _safe_vaf(det_vaf), color="#e67e22", linewidth=1.8, linestyle="--", alpha=0.9, zorder=4, label=_det_label
        )
    if detection.personal_lod is not None and n_eff_plot > 0:
        # personal_lod is already total VAF (p_err + incremental TF).
        lod_total_vaf = detection.personal_lod
        n_lod_reads = int(round(n_eff_plot * lod_total_vaf))
        _lod_recall_plot = getattr(detection, "lod_recall", 0.95)
        _lod_label = (
            f"LOD = {format_scientific(lod_total_vaf)} ({n_lod_reads} reads)" f" | {_lod_recall_plot * 100:.0f}% recall"
        )
        ax.axhline(
            _safe_vaf(lod_total_vaf),
            color="#27ae60",
            linewidth=1.8,
            linestyle="-.",
            alpha=0.9,
            zorder=4,
            label=_lod_label,
        )

    # ── Scale / labels ────────────────────────────────────────────────────────
    ax.set_yscale("log")
    y_vals = [pat_vaf] + (list(null / corr_cov) if len(null) > 0 and corr_cov > 0 else [])
    if detection.personal_lod:
        y_vals.append(detection.personal_lod)
    ax.set_ylim(_vaf_floor * 0.5, max(y_vals) * 8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="both", linestyle=":", linewidth=0.6, color="#dde1e7", alpha=0.9)  # noqa: FBT003
    ax.set_facecolor("#f4f6f8")
    ax.set_ylabel("ctDNA VAF", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: format_scientific(y) if y > 0 else "0"))
    ax.set_xlim(-0.5, 2.3)
    ax.set_xticks([0.3, x_pat])
    ax.set_xticklabels(["Synthetic\ncontrols", "Patient"])
    binom_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"  # noqa: PLR2004
    ax.set_title(f"Patient vs. controls  (p={binom_str})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.85, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    # ── Right axis: supporting reads aligned to left VAF axis ─────────────────
    if corr_cov > 0:
        ax2 = ax.twinx()
        y_min, y_max = ax.get_ylim()
        ax2.set_yscale("log")
        ax2.set_ylim(y_min * corr_cov, y_max * corr_cov)
        ax2.set_ylabel("Signature supporting reads", fontsize=10)
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{int(round(y))}" if y >= 0.5 else "")  # noqa: PLR2004
        )
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)


def plot_cohort_scatter(
    detection: "DetectionResult",
    df_tf: pd.DataFrame,
    ctrl_n_loci: "pd.Series",
    ax=None,
):
    """
    Scatter plot: cohort control signature size (x) vs ctDNA VAF (y).

    Each cohort control is plotted as a purple diamond at its own (signature_size, ctdna_vaf).
    The patient is shown as a red star at (patient_signature_size, patient_ctdna_vaf).

    Parameters
    ----------
    detection : DetectionResult
        Used for patient VAF, read count and signature size.
    df_tf : pd.DataFrame
        Tumor fraction table indexed by (signature_type, signature); must contain
        a "control" level with columns ``ctdna_vaf`` and ``supporting_reads``.
    ctrl_n_loci : pd.Series
        Number of loci per cohort control signature (index = signature name).
    ax : matplotlib Axes, optional
        If None a new figure is created.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    _vaf_floor = 1e-7

    try:
        ctrl_mask = df_tf.index.get_level_values(0) == "control"
        if not ctrl_mask.any():
            logger.debug("plot_cohort_scatter: no cohort controls in df_tf")
            return
        ctrl_tf = df_tf[ctrl_mask].droplevel(0)  # signature level only; always a DataFrame
    except (KeyError, IndexError):
        logger.debug("plot_cohort_scatter: no cohort controls in df_tf")
        return

    xs = np.array([int(ctrl_n_loci.get(name, 0)) for name in ctrl_tf.index])
    ys_raw = ctrl_tf["ctdna_vaf"].to_numpy(dtype=float)
    ys = np.array([max(float(v), _vaf_floor) for v in ys_raw])

    ax.scatter(
        xs,
        ys,
        color="#9b59b6",
        s=60,
        marker="D",
        alpha=0.85,
        zorder=5,
        label=f"Cohort controls (n={len(xs)})",
    )

    # Patient star
    pat_vaf = max(float(detection.matched_ctdna_vaf), _vaf_floor)
    pat_size = detection.signature_size
    pat_vaf_str = format_scientific(detection.matched_ctdna_vaf) if detection.matched_ctdna_vaf > 0 else "0"
    ax.scatter(
        [pat_size],
        [pat_vaf],
        color="#c0392b",
        s=180,
        marker="*",
        zorder=6,
        label=f"Patient ({detection.matched_supporting_reads} reads, {pat_vaf_str})",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Signature size (loci)", fontsize=10)
    ax.set_ylabel("ctDNA VAF", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: format_scientific(y) if y > 0 else "0"))
    ax.set_title("Cohort controls: signature size vs. ctDNA VAF", fontsize=11, fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="both", linestyle=":", linewidth=0.6, color="#dde1e7", alpha=0.9)  # noqa: FBT003
    ax.set_facecolor("#f4f6f8")
    ax.legend(fontsize=8, framealpha=0.85, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


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
