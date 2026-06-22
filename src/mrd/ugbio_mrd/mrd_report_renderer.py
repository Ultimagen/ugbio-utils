"""
MRD report renderer — generates self-contained HTML from analysis results.

Replaces the nbconvert-based approach with a clean Jinja2 HTML template,
giving full control over styling and layout.
"""

import base64
import datetime
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from ugbio_core.logger import logger

from ugbio_mrd.mrd_detection import DetectionResult, format_scientific, plot_null_distribution

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _fig_to_base64(fig, dpi=120) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def render_binomial_distribution(detection: DetectionResult, alpha: float = 0.01) -> str:
    """Render Binomial null distribution PMF with observed reads and detection threshold marked."""
    from scipy.stats import binom as _binom

    n = detection.n_effective
    p = detection.noise_rate
    obs = detection.matched_supporting_reads

    if n <= 0:
        return ""

    # Detection threshold: smallest k s.t. binom.sf(k-1, n, p) < alpha
    k_range = np.arange(0, min(n + 1, 10000))
    sf_vals = _binom.sf(k_range - 1, n, p)
    hits = np.where(sf_vals < alpha)[0]
    n_th = int(hits[0]) if len(hits) > 0 else 0

    # Plot range: 0 to max(n_th+3, obs+2)
    x_max = max(n_th + 3, obs + 2, 8)
    x = np.arange(0, x_max + 1)
    pmf = _binom.pmf(x, n, p)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f4f6f8")
    ax.set_facecolor("#f4f6f8")

    below = x < n_th
    above = x >= n_th
    if any(below):
        ax.bar(x[below], pmf[below], color="#3a9ad9", alpha=0.75, width=0.7, label="Null (not significant)")
    if any(above):
        ax.bar(
            x[above], pmf[above], color="#e74c3c", alpha=0.75, width=0.7,
            label=f"Detected region (\u2265{n_th} reads, p < {alpha})",
        )

    # Observed reads marker
    if 0 <= obs <= x_max:
        p_str = f"{detection.p_value:.2e}"
        ax.axvline(
            obs, color="#2c3e50", linewidth=2.5, linestyle="--",
            label=f"Observed ({obs} reads, p={p_str})",
        )
    elif obs > x_max:
        # Arrow annotation for obs outside range
        ax.annotate(
            f"Observed = {obs} reads\u2192",
            xy=(x_max, pmf[x_max] if x_max < len(pmf) else 0),
            xytext=(x_max - 2, max(pmf) * 0.6),
            arrowprops={"arrowstyle": "->", "color": "#2c3e50"},
            color="#2c3e50", fontsize=9,
        )

    noise_label = format_scientific(p) if p > 0 else "0"
    ax.set_xlabel("Supporting reads", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title(
        f"Binomial null  (N={n:,}, noise rate={noise_label})",
        fontsize=11, fontweight="bold",
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_signal_vs_noise(detection: DetectionResult, df_tf: pd.DataFrame) -> str:
    """Render signal vs. noise plot to base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f4f6f8")
    plot_null_distribution(detection, df_tf, ax=ax)
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_sbs_profile(df_signatures: pd.DataFrame, signature_filter_query: str, plot_sbs_fn) -> str:
    """Render SBS mutational profile to base64 PNG."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#f4f6f8")
    plot_sbs_fn(df_signatures, title="Signature Mutational Profile", ax=ax, query=signature_filter_query)
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_vaf_plot(df_signatures: pd.DataFrame, signature_filter_query: str, plot_af_fn) -> str:
    """Render signature VAF plot to base64 PNG."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#f4f6f8")
    plot_af_fn(df_signatures, signature_filter_query, panel="filtered", ax=ax)
    n_filt = df_signatures.query(signature_filter_query).shape[0]
    ax.set_title(f"Signature VAF (filtered, n={n_filt:,})", fontsize=10)
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_sbs_vaf_combined(df_signatures: pd.DataFrame, signature_filter_query: str, plot_sbs_fn, plot_af_fn) -> str:
    """Render SBS profile and VAF distribution side by side."""
    fig, (ax_sbs, ax_vaf) = plt.subplots(1, 2, figsize=(12, 3))
    fig.patch.set_facecolor("#f4f6f8")
    plot_sbs_fn(df_signatures, title="Mutation Profile", ax=ax_sbs, query=signature_filter_query)
    plot_af_fn(df_signatures, signature_filter_query, panel="filtered", ax=ax_vaf)
    n_filt = df_signatures.query(signature_filter_query).shape[0]
    ax_vaf.set_title(f"VAF Distribution (n={n_filt:,})", fontsize=10)
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_intersection_af(
    df_supporting_reads_per_locus: pd.DataFrame,
    df_signatures: pd.DataFrame,
) -> list[dict]:
    """Render individual cfDNA intersection AF histograms, return list of {label, description, img_b64}."""
    queries = [
        ("All variants", "All signature loci with ≥1 cfDNA read", None),
        ("Matched", "Loci from the matched (tumor) signature — signal", "signature_type == 'matched'"),
        ("Control", "Loci from control signature(s) — noise baseline", "signature_type != 'matched'"),
    ]
    colors = {"All variants": "#3498db", "Matched": "#c0392b", "Control": "#27ae60"}
    results = []

    for label, desc, query in queries:
        if query:
            idx = df_supporting_reads_per_locus.query(query).index
        else:
            idx = df_supporting_reads_per_locus.index
        af_data = df_signatures.loc[idx]["af"].dropna()
        if len(af_data) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 2))
        fig.patch.set_facecolor("#f4f6f8")
        ax.hist(af_data, bins=50, range=(0, 1), color=colors[label], alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{label} (n={len(af_data):,})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Allele Fraction (AF)")
        ax.set_ylabel("Count")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")
        ax.set_facecolor("#f4f6f8")
        plt.tight_layout()
        results.append({"label": label, "description": desc, "img_b64": _fig_to_base64(fig)})

    return results


def render_intersection_snvq_combined(df_features_filt: pd.DataFrame) -> str:
    """Render SNVQ distribution histogram: matched (red) vs control (blue) with legend."""
    if "snvq" not in df_features_filt.columns:
        return ""

    matched = df_features_filt.query("signature_type == 'matched'")["snvq"].dropna()
    control = df_features_filt.query("signature_type != 'matched'")["snvq"].dropna()

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#f4f6f8")

    bins = np.arange(0, 101, 2)
    if len(control) > 0:
        ax.hist(control, bins=bins, color="#3498db", alpha=0.6,
                edgecolor="white", linewidth=0.5, label=f"Other signatures (n={len(control):,})",
                density=True)
    if len(matched) > 0:
        ax.hist(matched, bins=bins, color="#c0392b", alpha=0.7,
                edgecolor="white", linewidth=0.5, label=f"Matched signature (n={len(matched):,})",
                density=True)

    ax.set_xlabel("SNVQ", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("cfDNA Intersection SNVQ Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(0, 100)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")
    ax.set_facecolor("#f4f6f8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_intersection_af_combined(
    df_supporting_reads_per_locus: pd.DataFrame,
    df_signatures: pd.DataFrame,
) -> str:
    """Render a single combined AF histogram: matched (red) vs control (blue) with legend."""
    matched_idx = df_supporting_reads_per_locus.query("signature_type == 'matched'").index
    control_idx = df_supporting_reads_per_locus.query("signature_type != 'matched'").index

    matched_af = df_signatures.loc[matched_idx.intersection(df_signatures.index)]["af"].dropna()
    control_af = df_signatures.loc[control_idx.intersection(df_signatures.index)]["af"].dropna()

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#f4f6f8")

    if len(control_af) > 0:
        ax.hist(control_af, bins=50, range=(0, 1), color="#3498db", alpha=0.6,
                edgecolor="white", linewidth=0.5, label=f"Other signatures (n={len(control_af):,})")
    if len(matched_af) > 0:
        ax.hist(matched_af, bins=50, range=(0, 1), color="#c0392b", alpha=0.7,
                edgecolor="white", linewidth=0.5, label=f"Matched signature (n={len(matched_af):,})")

    ax.set_xlabel("Allele Fraction (AF)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("cfDNA Intersection Allele Fraction", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")
    ax.set_facecolor("#f4f6f8")
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_analysis_report(
    detection: DetectionResult,
    df_tf: pd.DataFrame,
    df_signatures: pd.DataFrame,
    df_signatures_filt: pd.DataFrame,
    df_supporting_reads_per_locus: pd.DataFrame,
    basename: str,
    signature_filter_query: str,
    read_filter_query: str,
    denom_ratio: float,
    filt_ratio: float,
    plot_sbs_fn,
    plot_af_fn,
    applied_filters: dict | None = None,
    df_features_filt: pd.DataFrame | None = None,
) -> str:
    """
    Render the MRD analysis report as a self-contained HTML string.

    Parameters
    ----------
    detection : DetectionResult
        Output from run_detection_analysis().
    df_tf : pd.DataFrame
        Tumor fraction table (index: signature_type, signature).
    df_signatures : pd.DataFrame
        Full (unfiltered) signature dataframe.
    df_signatures_filt : pd.DataFrame
        Filtered signature dataframe.
    df_supporting_reads_per_locus : pd.DataFrame
        Per-locus supporting reads dataframe.
    basename : str
        Sample basename for display.
    signature_filter_query : str
        The signature filter query applied.
    read_filter_query : str
        The read filter query applied.
    denom_ratio : float
        Denominator correction ratio.
    filt_ratio : float
        Filtering ratio.
    plot_sbs_fn : callable
        Function to plot SBS profile (from notebook cell).
    plot_af_fn : callable
        Function to plot allele fractions.
    applied_filters : dict, optional
        Dict of {filter_name: description} for display.

    Returns
    -------
    str
        Complete HTML document.
    """
    logger.info("Rendering MRD analysis report with Jinja2 template")

    # Generate plots
    signal_noise_img = render_signal_vs_noise(detection, df_tf)

    # SBS + VAF side-by-side for matched signatures
    matched_sigs = df_signatures.query("signature_type == 'matched'")["signature"].unique()
    sbs_vaf_plots = []
    for sig in matched_sigs:
        sig_df = df_signatures.query(f"signature == '{sig}'")
        sbs_vaf_plots.append(render_sbs_vaf_combined(sig_df, signature_filter_query, plot_sbs_fn, plot_af_fn))

    # Intersection AF — single combined plot
    intersection_af_img = render_intersection_af_combined(df_supporting_reads_per_locus, df_signatures_filt)

    # SNVQ distribution
    intersection_snvq_img = render_intersection_snvq_combined(df_features_filt) if df_features_filt is not None else ""

    # Format values for template
    binom_p_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"
    noise_rate_str = format_scientific(detection.noise_rate) if detection.noise_rate > 0 else "0"

    context = {
        "report_title": "MRD Analysis Report",
        "report_date": datetime.date.today().isoformat(),
        "basename": basename or "N/A",
        "detection": detection,
        "binom_p_str": binom_p_str,
        "noise_rate_str": noise_rate_str,
        "vaf_str": format_scientific(detection.matched_ctdna_vaf) if detection.matched_ctdna_vaf > 0 else "0",
        "lod_str": format_scientific(detection.personal_lod) if detection.personal_lod else "N/A",
        "signal_noise_img": signal_noise_img,
        "sbs_vaf_plots": sbs_vaf_plots,
        "intersection_af_img": intersection_af_img,
        "intersection_snvq_img": intersection_snvq_img,
        "signature_filter_query": signature_filter_query,
        "read_filter_query": read_filter_query,
        "denom_ratio": denom_ratio,
        "filt_ratio": filt_ratio,
        "applied_filters": applied_filters or {},
    }

    template_path = TEMPLATE_DIR / "mrd_analysis_report.html"
    template = Template(template_path.read_text())
    return template.render(**context)


def render_qc_report(
    detection: DetectionResult,
    detection_unfilt: DetectionResult,
    detection_unfilt2: DetectionResult,
    df_tf_filt: pd.DataFrame,
    df_tf_unfilt: pd.DataFrame,
    df_tf_unfilt2: pd.DataFrame,
    df_signatures: pd.DataFrame,
    df_signatures_filt: pd.DataFrame,
    df_features: pd.DataFrame,
    df_features_filt: pd.DataFrame,
    df_supporting_reads_per_locus_unfilt: pd.DataFrame,
    df_supporting_reads_per_locus_unfilt2: pd.DataFrame,
    basename: str,
    signature_filter_query: str,
    read_filter_query: str,
    denom_ratio: float,
    filt_ratio: float,
    plot_sbs_fn,
    plot_af_fn,
) -> str:
    """
    Render the MRD QC report as a self-contained HTML string.

    Parameters
    ----------
    detection : DetectionResult
        Primary detection (filtered reads + filtered signatures).
    detection_unfilt : DetectionResult
        Detection with filtered reads + unfiltered signatures.
    detection_unfilt2 : DetectionResult
        Detection with unfiltered reads + filtered signatures.
    df_tf_filt, df_tf_unfilt, df_tf_unfilt2 : pd.DataFrame
        Tumor fraction tables for each analysis variant.
    df_signatures, df_signatures_filt : pd.DataFrame
        Full and filtered signature dataframes.
    df_features, df_features_filt : pd.DataFrame
        Full and filtered features dataframes.
    df_supporting_reads_per_locus_unfilt, df_supporting_reads_per_locus_unfilt2 : pd.DataFrame
        Per-locus supporting reads for the two unfiltered analyses.
    basename : str
        Sample basename.
    signature_filter_query, read_filter_query : str
        Filter queries.
    denom_ratio, filt_ratio : float
        Ratio values.
    plot_sbs_fn : callable
        SBS profile plotting function.
    plot_af_fn : callable
        Allele fraction plotting function.

    Returns
    -------
    str
        Complete HTML document.
    """
    logger.info("Rendering MRD QC report with Jinja2 template")

    # ── Control signature profiles: SBS unfiltered + filtered side-by-side ──
    control_profiles = []
    cohort_sigs = sorted(df_signatures.query("signature_type == 'control'")["signature"].unique())
    syn_sigs = sorted(df_signatures.query("signature_type == 'db_control'")["signature"].unique())[:1]
    ctrl_sigs = cohort_sigs + syn_sigs
    for sig in ctrl_sigs:
        sig_df = df_signatures.query(f"signature == '{sig}'")
        sig_type = sig_df["signature_type"].iloc[0]
        label = f"{sig}  ({sig_type})"
        fig, axs = plt.subplots(1, 2, figsize=(10, 2.4))
        fig.patch.set_facecolor("#f4f6f8")
        plot_sbs_fn(sig_df, title="Unfiltered", ax=axs[0])
        plot_sbs_fn(sig_df, title="After QC filters", ax=axs[1], query=signature_filter_query)
        plt.tight_layout()
        control_profiles.append({"label": label, "img_b64": _fig_to_base64(fig)})

    # ── Fragment length distributions (2x2 grid) ──
    fragment_length_img = None
    if "rl" in df_features.columns:
        fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
        fig.patch.set_facecolor("#f4f6f8")
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        panels = [
            ("Matched reads\nunfiltered", df_features.query("signature_type=='matched'")["rl"]),
            ("Matched reads\nfiltered", df_features.query(f"signature_type=='matched' and {read_filter_query}")["rl"]),
            ("Unmatched reads\nunfiltered", df_features.query("signature_type!='matched'")["rl"]),
            ("Unmatched reads\nfiltered", df_features.query(f"signature_type!='matched' and {read_filter_query}")["rl"]),
        ]
        max_val = max(s.max() for _, s in panels if len(s) > 0) if any(len(s) > 0 for _, s in panels) else 250
        for ax, (title, data) in zip(axs.flatten(), panels):
            if len(data) > 0:
                ax.hist(data, bins=range(int(max_val) + 1), color="#3a9ad9", alpha=0.7, edgecolor="white", linewidth=0.3)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_facecolor("#f4f6f8")
        for ax in axs[-1, :]:
            ax.set_xlabel("Read length", fontsize=9)
        plt.tight_layout()
        fragment_length_img = _fig_to_base64(fig)

    # ── QC Analysis — Unfiltered Signature ──
    unfilt_sig_signal_noise_img = _render_signal_noise_internal(detection_unfilt, df_tf_unfilt)

    matched_sigs = df_signatures.query("signature_type == 'matched'")["signature"].unique()
    # SBS + VAF side-by-side for matched (using unfiltered signatures)
    unfilt_sig_sbs_vaf_parts = []
    for sig in matched_sigs:
        sig_df = df_signatures.query(f"signature == '{sig}'")
        unfilt_sig_sbs_vaf_parts.append(
            render_sbs_vaf_combined(sig_df, signature_filter_query, plot_sbs_fn, plot_af_fn)
        )
    unfilt_sig_sbs_vaf_img = unfilt_sig_sbs_vaf_parts[0] if unfilt_sig_sbs_vaf_parts else None

    unfilt_sig_intersection_img = render_intersection_af_combined(
        df_supporting_reads_per_locus_unfilt, df_signatures
    )

    # ── QC Analysis — Unfiltered Reads ──
    unfilt_reads_signal_noise_img = _render_signal_noise_internal(detection_unfilt2, df_tf_unfilt2)

    unfilt_reads_sbs_vaf_parts = []
    for sig in matched_sigs:
        sig_df = df_signatures_filt.query(f"signature == '{sig}'")
        if len(sig_df) > 0:
            unfilt_reads_sbs_vaf_parts.append(
                render_sbs_vaf_combined(sig_df, signature_filter_query, plot_sbs_fn, plot_af_fn)
            )
    unfilt_reads_sbs_vaf_img = unfilt_reads_sbs_vaf_parts[0] if unfilt_reads_sbs_vaf_parts else None

    unfilt_reads_intersection_img = render_intersection_af_combined(
        df_supporting_reads_per_locus_unfilt2, df_signatures_filt
    )

    # ── Format values ──
    binom_p_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"
    noise_rate_str = format_scientific(detection.noise_rate) if detection.noise_rate > 0 else "0"

    context = {
        "report_title": "MRD QC Report",
        "report_date": datetime.date.today().isoformat(),
        "basename": basename or "N/A",
        "detection": detection,
        "binom_p_str": binom_p_str,
        "noise_rate_str": noise_rate_str,
        "vaf_str": format_scientific(detection.matched_ctdna_vaf) if detection.matched_ctdna_vaf > 0 else "0",
        "lod_str": format_scientific(detection.personal_lod) if detection.personal_lod else "N/A",
        "filt_ratio": filt_ratio,
        "control_profiles": control_profiles,
        "fragment_length_img": fragment_length_img,
        "unfilt_sig_signal_noise_img": unfilt_sig_signal_noise_img,
        "unfilt_sig_sbs_vaf_img": unfilt_sig_sbs_vaf_img,
        "unfilt_sig_intersection_img": unfilt_sig_intersection_img,
        "unfilt_reads_signal_noise_img": unfilt_reads_signal_noise_img,
        "unfilt_reads_sbs_vaf_img": unfilt_reads_sbs_vaf_img,
        "unfilt_reads_intersection_img": unfilt_reads_intersection_img,
    }

    template_path = TEMPLATE_DIR / "mrd_qc_report.html"
    template = Template(template_path.read_text())
    return template.render(**context)


def _render_signal_noise_internal(detection: DetectionResult, df_tf: pd.DataFrame) -> str:
    """Render signal vs noise plot (internal helper, same as render_signal_vs_noise)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f4f6f8")
    plot_null_distribution(detection, df_tf, ax=ax)
    plt.tight_layout()
    return _fig_to_base64(fig)
