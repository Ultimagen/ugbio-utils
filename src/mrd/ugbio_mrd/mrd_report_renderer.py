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
from jinja2 import Environment, FileSystemLoader
from ugbio_core.logger import logger

from ugbio_mrd.mrd_detection import DetectionResult, format_scientific, plot_null_distribution

TEMPLATE_DIR = Path(__file__).parent / "templates"
_JINJA_ENV = Environment(  # noqa: S701  (autoescape disabled: values are pre-sanitised base64/numbers/safe HTML fragments)
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=False,  # noqa: S701
)


def _fig_to_base64(fig, dpi=120) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── COSMIC SBS96 constants ──────────────────────────────────────────────────
_COSMIC_MUT_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
_COSMIC_COLORS_SBS = {
    "C>A": "#1EBFF0",
    "C>G": "#050708",
    "C>T": "#E62725",
    "T>A": "#CBCACB",
    "T>C": "#A1C935",
    "T>G": "#ECC6C5",
}
_FLANKS = ["A", "C", "G", "T"]
_SBS96_CHANNELS = [f"{f5}[{mt}]{f3}" for mt in _COSMIC_MUT_TYPES for f5 in _FLANKS for f3 in _FLANKS]
_RC = str.maketrans("ACGT", "TGCA")


def _count_sbs96(df: pd.DataFrame) -> pd.Series | None:
    """Count SBS96 channels from df.

    Accepts two column conventions:
    - ``x_prev1`` / ``x_next1``  — single-base flanks (featuremap parquets)
    - ``left_motif`` / ``right_motif`` — multi-base motifs from signature VCFs;
      the *last* character of left_motif is the 5' flank and the *first* character
      of right_motif is the 3' flank.
    """
    has_xprev = "x_prev1" in df.columns and "x_next1" in df.columns
    has_motif = "left_motif" in df.columns and "right_motif" in df.columns
    if not ("ref" in df.columns and "alt" in df.columns and (has_xprev or has_motif)):
        return None

    v = df[["ref", "alt"]].copy()
    if has_xprev:
        v["x_prev1"] = df["x_prev1"].astype(str).str.upper()
        v["x_next1"] = df["x_next1"].astype(str).str.upper()
    else:
        # Extract single flanking bases from the motif strings
        v["x_prev1"] = df["left_motif"].astype(str).str[-1].str.upper()
        v["x_next1"] = df["right_motif"].astype(str).str[0].str.upper()
    for col in ("ref", "alt"):
        v[col] = v[col].astype(str).str.upper()
    valid = set("ACGT")
    mask = (
        v["ref"].isin(valid)
        & v["alt"].isin(valid)
        & v["x_prev1"].isin(valid)
        & v["x_next1"].isin(valid)
        & (v["ref"] != v["alt"])
    )
    v = v[mask]
    if v.empty:
        return None
    # Normalise to pyrimidine strand: flip A/G refs to their complement
    needs_flip = v["ref"].isin({"A", "G"})
    ref = v["ref"].copy()
    alt = v["alt"].copy()
    f5 = v["x_prev1"].copy()
    f3 = v["x_next1"].copy()

    def _comp(s: pd.Series) -> pd.Series:
        return s.str.translate(_RC)

    ref[needs_flip] = _comp(v["ref"][needs_flip])
    alt[needs_flip] = _comp(v["alt"][needs_flip])
    f5[needs_flip] = _comp(v["x_next1"][needs_flip])  # 5' ← former 3' after RC
    f3[needs_flip] = _comp(v["x_prev1"][needs_flip])  # 3' ← former 5' after RC

    labels = f5 + "[" + ref + ">" + alt + "]" + f3
    counts = labels.value_counts()
    return counts.reindex(_SBS96_CHANNELS, fill_value=0)


def render_sbs96_profile(df_features_filt: pd.DataFrame) -> str:
    """Render a COSMIC-style SBS96 mutational profile from matched cfDNA reads.

    Uses x_prev1 and x_next1 as flanking bases (from the featuremap intersection
    parquet), with pyrimidine-strand normalisation identical to COSMIC convention.
    """
    if df_features_filt is None or df_features_filt.empty:
        return ""
    # Use only matched reads (patient signal)
    if "signature_type" in df_features_filt.columns:
        df_m = df_features_filt.query("signature_type == 'matched'")
    else:
        df_m = df_features_filt

    counts = _count_sbs96(df_m)
    if counts is None or counts.sum() == 0:
        return ""

    total = int(counts.sum())
    fracs = counts / total

    fig, ax = plt.subplots(figsize=(14, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(96)
    bar_colors = [_COSMIC_COLORS_SBS[ch[ch.index("[") + 1 : ch.index("]")]] for ch in _SBS96_CHANNELS]
    ax.bar(x, fracs.to_numpy(), color=bar_colors, width=0.9, linewidth=0, zorder=2)

    # Vertical separators between mutation-type groups
    for i in range(1, 6):
        ax.axvline(i * 16 - 0.5, color="#cccccc", linewidth=1.0, zorder=3)

    # Mutation-type header labels above each group
    y_max = fracs.to_numpy().max() if fracs.to_numpy().max() > 0 else 0.01
    ax.set_ylim(0, y_max * 1.25)
    for i, mt in enumerate(_COSMIC_MUT_TYPES):
        cx = i * 16 + 7.5
        col = _COSMIC_COLORS_SBS[mt]
        ax.text(
            cx,
            y_max * 1.18,
            mt,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=col if mt != "C>G" else "#444444",
        )

    # X-axis: show 5'-base labels (16 per group × 6 groups)
    tick_labels = [f"{ch[0]}{ch[ch.index('[') + 1]}{ch[-1]}" for ch in _SBS96_CHANNELS]  # trinucleotide context
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=5.5, rotation=90)
    ax.set_xlim(-0.5, 95.5)
    ax.set_ylabel("Fraction", fontsize=9)
    ax.set_title(f"Mutational Profile — matched cfDNA reads (n={total:,})", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)  # noqa: FBT003
    ax.set_axisbelow(True)
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_binomial_distribution(detection: DetectionResult, alpha: float = 0.01) -> str:
    """Render Binomial null distribution PMF with observed reads and detection threshold marked."""
    from scipy.stats import binom as _binom  # noqa: PLC0415

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
            x[above],
            pmf[above],
            color="#e74c3c",
            alpha=0.75,
            width=0.7,
            label=f"Detected region (\u2265{n_th} reads, p < {alpha})",
        )

    # Observed reads marker
    if 0 <= obs <= x_max:
        p_str = f"{detection.p_value:.2e}"
        ax.axvline(
            obs,
            color="#2c3e50",
            linewidth=2.5,
            linestyle="--",
            label=f"Observed ({obs} reads, p={p_str})",
        )
    elif obs > x_max:
        # Arrow annotation for obs outside range
        ax.annotate(
            f"Observed = {obs} reads\u2192",
            xy=(x_max, pmf[x_max] if x_max < len(pmf) else 0),
            xytext=(x_max - 2, max(pmf) * 0.6),
            arrowprops={"arrowstyle": "->", "color": "#2c3e50"},
            color="#2c3e50",
            fontsize=9,
        )

    noise_label = format_scientific(p) if p > 0 else "0"
    ax.set_xlabel("Supporting reads", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title(
        f"Binomial null  (N={n:,}, noise rate={noise_label})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")  # noqa: FBT003
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
    ax.set_title(f"Signature VAF (filtered, n={n_filt:,})", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_sbs_vaf_combined(
    df_signatures: pd.DataFrame,
    signature_filter_query: str,
    plot_sbs_fn,
    plot_af_fn,
    df_features: pd.DataFrame | None = None,
    df_features_filt: pd.DataFrame | None = None,
) -> tuple[str, str]:
    """Return (sbs96_img, sbs6_vaf_img) as two separate base64 PNG strings.

    SBS96 is built from df_signatures directly (all filtered signature variants).
    df_signatures carries left_motif / right_motif from the signature VCF INFO fields,
    so trinucleotide context is available for every locus — not just those that happened
    to have a cfDNA read in the intersection.
    """
    # ── SBS96 figure ──────────────────────────────────────────────────────────
    df_filt = df_signatures.query(signature_filter_query) if signature_filter_query else df_signatures
    counts = _count_sbs96(df_filt)
    df_for_sbs6 = df_filt
    sbs6_query = signature_filter_query

    fig96, ax96 = plt.subplots(figsize=(14, 3.5))
    fig96.patch.set_facecolor("white")
    ax96.set_facecolor("white")
    if counts is not None and counts.sum() > 0:
        total = int(counts.sum())
        fracs = counts / total
        x = np.arange(96)
        bar_colors = [_COSMIC_COLORS_SBS[ch[ch.index("[") + 1 : ch.index("]")]] for ch in _SBS96_CHANNELS]
        ax96.bar(x, fracs.to_numpy(), color=bar_colors, width=0.9, linewidth=0, zorder=2)
        for i in range(1, 6):
            ax96.axvline(i * 16 - 0.5, color="#cccccc", linewidth=1.0, zorder=3)
        y_max = fracs.to_numpy().max() if fracs.to_numpy().max() > 0 else 0.01
        ax96.set_ylim(0, y_max * 1.30)
        for i, mt in enumerate(_COSMIC_MUT_TYPES):
            ax96.text(
                i * 16 + 7.5,
                y_max * 1.22,
                mt,
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color=_COSMIC_COLORS_SBS[mt] if mt != "C>G" else "#444444",
            )
        tick_labels = [f"{ch[0]}{ch[ch.index('[') + 1]}{ch[-1]}" for ch in _SBS96_CHANNELS]
        ax96.set_xticks(x)
        ax96.set_xticklabels(tick_labels, fontsize=5.5, rotation=90)
        ax96.set_xlim(-0.5, 95.5)
        ax96.set_ylabel("Fraction", fontsize=9)
        ax96.set_title(f"Mutational Profile (SBS96) — {total:,} signature variants", fontsize=10, fontweight="bold")
        ax96.spines[["top", "right"]].set_visible(False)
        ax96.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)  # noqa: FBT003
        ax96.set_axisbelow(True)
    else:
        ax96.text(
            0.5,
            0.5,
            "Insufficient trinucleotide context data for SBS96",
            ha="center",
            va="center",
            transform=ax96.transAxes,
            color="#999",
        )
        ax96.set_axis_off()
    plt.tight_layout()
    sbs96_img = _fig_to_base64(fig96)

    # ── 6-bar SBS + VAF figure ────────────────────────────────────────────────
    fig2, (ax_sbs6, ax_vaf) = plt.subplots(1, 2, figsize=(12, 3))
    fig2.patch.set_facecolor("#f4f6f8")
    plot_sbs_fn(df_for_sbs6, title="Substitution Types", ax=ax_sbs6, query=sbs6_query)
    plot_af_fn(df_signatures, signature_filter_query, panel="filtered", ax=ax_vaf)
    ax_vaf.set_title(f"VAF Distribution (n={df_filt.shape[0]:,})", fontsize=10, fontweight="bold")
    plt.tight_layout()
    sbs6_vaf_img = _fig_to_base64(fig2)

    return sbs96_img, sbs6_vaf_img


def render_intersection_af(
    df_supporting_reads_per_locus: pd.DataFrame,
    df_signatures: pd.DataFrame,
) -> list[dict]:
    """Render individual cfDNA intersection AF histograms, return list of {label, description, img_b64}."""
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    queries = [
        ("Patient", "Loci from the patient (tumor) signature — signal", "signature_type == 'matched'"),
        ("Control", "Loci from control signature(s) — noise baseline", "signature_type != 'matched'"),
    ]
    bar_colors = {"Matched": "#c0392b", "Control": "#3498db"}
    kde_colors = {"Matched": "#7b241c", "Control": "#1a5276"}
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
        counts, bin_edges, _ = ax.hist(
            af_data,
            bins=50,
            range=(0, 1),
            color=bar_colors[label],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.5,
        )
        # Smoothed KDE line scaled to histogram counts
        if len(af_data) >= 5:  # noqa: PLR2004
            try:
                from matplotlib import patheffects  # noqa: PLC0415

                kde = gaussian_kde(af_data, bw_method=0.3)
                x_kde = np.linspace(0, 1, 500)
                bin_width = bin_edges[1] - bin_edges[0]
                ax.plot(
                    x_kde,
                    kde(x_kde) * len(af_data) * bin_width,
                    color=kde_colors[label],
                    linewidth=1.2,
                    zorder=4,
                    label="KDE",
                    path_effects=[
                        patheffects.withStroke(linewidth=2.5, foreground="white"),
                        patheffects.Normal(),
                    ],
                )
                ax.legend(fontsize=8, framealpha=0.85)
            except Exception as e:  # noqa: BLE001
                logger.debug("KDE line skipped: %s", e)
        ax.set_title(f"{label} (n={len(af_data):,})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Allele Fraction (AF)")
        ax.set_ylabel("Count")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")  # noqa: FBT003
        ax.set_facecolor("#f4f6f8")
        plt.tight_layout()
        results.append({"label": label, "description": desc, "img_b64": _fig_to_base64(fig)})

    return results


def render_intersection_snvq_combined(df_features_filt: pd.DataFrame) -> str:
    """Render SNVQ distribution histogram: matched (red) vs control (blue) with legend."""
    if "snvq" not in df_features_filt.columns:
        return ""

    from matplotlib import patheffects  # noqa: PLC0415
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    matched = df_features_filt.query("signature_type == 'matched'")["snvq"].dropna()
    control = df_features_filt.query("signature_type != 'matched'")["snvq"].dropna()

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#f4f6f8")

    bins = np.arange(0, 101, 2)
    if len(control) > 0:
        ax.hist(
            control,
            bins=bins,
            color="#3498db",
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
            label=f"Other signatures (n={len(control):,})",
            density=True,
        )
        if len(control) >= 5:  # noqa: PLR2004
            try:
                kde = gaussian_kde(control, bw_method=0.3)
                x_kde = np.linspace(0, 100, 1000)
                ax.plot(
                    x_kde,
                    kde(x_kde),
                    color="#1a5276",
                    linewidth=1.2,
                    zorder=4,
                    label="KDE (other)",
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white"), patheffects.Normal()],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("KDE line skipped (control snvq): %s", e)
    if len(matched) > 0:
        ax.hist(
            matched,
            bins=bins,
            color="#c0392b",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            label=f"Patient signature (n={len(matched):,})",
            density=True,
        )
        if len(matched) >= 5:  # noqa: PLR2004
            try:
                kde = gaussian_kde(matched, bw_method=0.3)
                x_kde = np.linspace(0, 100, 1000)
                ax.plot(
                    x_kde,
                    kde(x_kde),
                    color="#7b241c",
                    linewidth=1.2,
                    zorder=4,
                    label="KDE (matched)",
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white"), patheffects.Normal()],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("KDE line skipped (matched snvq): %s", e)

    ax.set_xlabel("SNVQ", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("cfDNA Intersection SNVQ Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(0, 100)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")  # noqa: FBT003
    ax.set_facecolor("#f4f6f8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_read_length_histogram(df_features_filt: pd.DataFrame) -> str:
    """Render read length histogram: patient signature (red) vs controls (blue) with KDE lines."""
    if "X_LENGTH" not in df_features_filt.columns:
        return ""

    from matplotlib import patheffects  # noqa: PLC0415
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    matched = df_features_filt.query("signature_type == 'matched'")["X_LENGTH"].dropna()
    control = df_features_filt.query("signature_type != 'matched'")["X_LENGTH"].dropna()

    if len(matched) == 0 and len(control) == 0:
        return ""

    all_lengths = pd.concat([matched, control])
    x_min = max(0, int(all_lengths.min()) - 5)
    x_max = min(600, int(all_lengths.max()) + 5)  # cap at 600 bp
    bins = np.arange(x_min, x_max + 2, 2)

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#f4f6f8")
    ax.set_facecolor("#f4f6f8")

    if len(control) > 0:
        ax.hist(
            control.clip(upper=x_max),
            bins=bins,
            color="#3498db",
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
            label=f"Other signatures (n={len(control):,})",
            density=True,
        )
        if len(control) >= 5:  # noqa: PLR2004
            try:
                kde = gaussian_kde(control.clip(upper=x_max), bw_method=0.15)
                x_kde = np.linspace(x_min, x_max, 1000)
                ax.plot(
                    x_kde,
                    kde(x_kde),
                    color="#1a5276",
                    linewidth=1.2,
                    zorder=4,
                    label="KDE (other)",
                    path_effects=[
                        patheffects.withStroke(linewidth=2.5, foreground="white"),
                        patheffects.Normal(),
                    ],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("KDE line skipped (control read length): %s", e)
    if len(matched) > 0:
        ax.hist(
            matched.clip(upper=x_max),
            bins=bins,
            color="#c0392b",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            label=f"Patient signature (n={len(matched):,})",
            density=True,
        )
        if len(matched) >= 5:  # noqa: PLR2004
            try:
                kde = gaussian_kde(matched.clip(upper=x_max), bw_method=0.15)
                x_kde = np.linspace(x_min, x_max, 1000)
                ax.plot(
                    x_kde,
                    kde(x_kde),
                    color="#7b241c",
                    linewidth=1.2,
                    zorder=4,
                    label="KDE (matched)",
                    path_effects=[
                        patheffects.withStroke(linewidth=2.5, foreground="white"),
                        patheffects.Normal(),
                    ],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("KDE line skipped (matched read length): %s", e)

    ax.set_xlabel("Read length (bp)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Read Length Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(x_min, x_max)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")  # noqa: FBT003
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    img = _fig_to_base64(fig)
    plt.close(fig)
    return img


def render_intersection_af_combined(
    df_supporting_reads_per_locus: pd.DataFrame,
    df_signatures: pd.DataFrame,
) -> str:
    """Render a single combined AF histogram: matched (blue) vs control (red) with KDE lines and legend."""
    from matplotlib import patheffects  # noqa: PLC0415
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    matched_idx = df_supporting_reads_per_locus.query("signature_type == 'matched'").index
    control_idx = df_supporting_reads_per_locus.query("signature_type != 'matched'").index

    matched_af = df_signatures.loc[matched_idx.intersection(df_signatures.index)]["af"].dropna()
    control_af = df_signatures.loc[control_idx.intersection(df_signatures.index)]["af"].dropna()

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#f4f6f8")

    bin_edges = np.linspace(0, 1, 51)
    bin_width = bin_edges[1] - bin_edges[0]

    if len(control_af) > 0:
        ax.hist(
            control_af,
            bins=bin_edges,
            color="#3498db",
            alpha=0.55,
            edgecolor="white",
            linewidth=0.5,
            label=f"Other signatures (n={len(control_af):,})",
        )
        if len(control_af) >= 5:  # noqa: PLR2004
            try:
                kde = gaussian_kde(control_af, bw_method=0.3)
                x_kde = np.linspace(0, 1, 500)
                ax.plot(
                    x_kde,
                    kde(x_kde) * len(control_af) * bin_width,
                    color="#1a5276",
                    linewidth=1.2,
                    zorder=4,
                    label="KDE (other)",
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white"), patheffects.Normal()],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("KDE line skipped (control): %s", e)

    if len(matched_af) > 0:
        ax.hist(
            matched_af,
            bins=bin_edges,
            color="#c0392b",
            alpha=0.65,
            edgecolor="white",
            linewidth=0.5,
            label=f"Patient signature (n={len(matched_af):,})",
        )
        if len(matched_af) >= 5:  # noqa: PLR2004
            try:
                kde = gaussian_kde(matched_af, bw_method=0.3)
                x_kde = np.linspace(0, 1, 500)
                ax.plot(
                    x_kde,
                    kde(x_kde) * len(matched_af) * bin_width,
                    color="#7b241c",
                    linewidth=1.2,
                    zorder=4,
                    label="KDE (matched)",
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white"), patheffects.Normal()],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("KDE line skipped (matched): %s", e)

    ax.set_xlabel("Allele Fraction (AF)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("cfDNA Intersection Allele Fraction", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")  # noqa: FBT003
    ax.set_facecolor("#f4f6f8")
    plt.tight_layout()
    return _fig_to_base64(fig)


def render_supporting_reads_histogram(
    df_supporting_reads_per_locus: pd.DataFrame,
    signature_size: int,
) -> str:
    """
    Histogram of alt-supporting read counts per variant locus.

    Shows how many signature loci have 1, 2, 3, ... alt-supporting reads
    in the cfDNA intersection, separately for matched (signal) and control
    (noise) loci.  Loci with zero supporting reads are annotated but not
    plotted (they dominate and would compress the axis).
    """
    matched = df_supporting_reads_per_locus.query("signature_type == 'matched'")["supporting_reads"]
    control = df_supporting_reads_per_locus.query("signature_type != 'matched'")["supporting_reads"]

    if len(matched) == 0 and len(control) == 0:
        return ""

    max_reads = max(
        matched.max() if len(matched) > 0 else 1,
        control.max() if len(control) > 0 else 1,
    )
    x_cap = min(int(max_reads) + 1, 20)  # cap display at 20 reads
    bins = list(range(1, x_cap + 2))  # edges: 1, 2, ..., x_cap+1

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#f4f6f8")
    ax.set_facecolor("#f4f6f8")

    n_ctrl_with_reads = len(control)
    n_matched_with_reads = len(matched)
    n_matched_zero = max(0, signature_size - n_matched_with_reads)

    if n_ctrl_with_reads > 0:
        ax.hist(
            control.clip(upper=x_cap),
            bins=bins,
            color="#3498db",
            alpha=0.65,
            label=f"Control (n={n_ctrl_with_reads:,} loci with reads)",
            align="left",
        )
    if n_matched_with_reads > 0:
        ax.hist(
            matched.clip(upper=x_cap),
            bins=bins,
            color="#c0392b",
            alpha=0.7,
            label=f"Patient signature (n={n_matched_with_reads:,}/{signature_size:,} loci with reads)",
            align="left",
        )

    if n_matched_zero > 0:
        ax.text(
            0.97,
            0.96,
            f"{n_matched_zero:,} matched loci have 0 reads (not shown)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#7f8c8d",
            style="italic",
        )

    ax.set_xlabel("Alt-supporting reads per variant locus", fontsize=10)
    ax.set_ylabel("Number of loci", fontsize=10)
    ax.set_title("Alt-Supporting Reads per Variant Locus", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xticks(bins[:-1])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#dde1e7")  # noqa: FBT003
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    img = _fig_to_base64(fig)
    plt.close(fig)
    return img


def render_analysis_report(  # noqa: PLR0913
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
    df_features: pd.DataFrame | None = None,
    df_features_filt: pd.DataFrame | None = None,
    inputs_info: dict | None = None,
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

    # SBS96 + (6-bar SBS + VAF) for matched signatures — now two separate images per sig
    matched_sigs = df_signatures.query("signature_type == 'matched'")["signature"].unique()
    sbs96_plots = []
    sbs6_vaf_plots = []
    for sig in matched_sigs:
        sig_df = df_signatures.query(f"signature == '{sig}'")
        sbs96_img, sbs6_vaf_img = render_sbs_vaf_combined(
            sig_df,
            signature_filter_query,
            plot_sbs_fn,
            plot_af_fn,
            df_features=df_features,
            df_features_filt=df_features_filt,
        )
        sbs96_plots.append(sbs96_img)
        sbs6_vaf_plots.append(sbs6_vaf_img)

    # Intersection AF — single combined plot
    intersection_af_img = render_intersection_af_combined(df_supporting_reads_per_locus, df_signatures_filt)

    # Supporting reads per locus histogram
    supporting_reads_hist_img = render_supporting_reads_histogram(
        df_supporting_reads_per_locus, detection.signature_size
    )

    # Read length histogram
    read_length_img = render_read_length_histogram(df_features_filt) if df_features_filt is not None else ""

    # SNVQ distribution
    intersection_snvq_img = render_intersection_snvq_combined(df_features_filt) if df_features_filt is not None else ""

    # Format values for template
    binom_p_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"  # noqa: PLR2004
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
        "sbs96_plots": sbs96_plots,
        "sbs6_vaf_plots": sbs6_vaf_plots,
        "intersection_af_img": intersection_af_img,
        "supporting_reads_hist_img": supporting_reads_hist_img,
        "read_length_img": read_length_img,
        "intersection_snvq_img": intersection_snvq_img,
        "signature_filter_query": signature_filter_query,
        "read_filter_query": read_filter_query,
        "denom_ratio": denom_ratio,
        "filt_ratio": filt_ratio,
        "applied_filters": applied_filters or {},
        "inputs_info": inputs_info or {},
    }

    template = _JINJA_ENV.get_template("mrd_analysis_report.html")
    return template.render(**context)


def render_qc_report(  # noqa: PLR0913
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
            (
                "Unmatched reads\nfiltered",
                df_features.query(f"signature_type!='matched' and {read_filter_query}")["rl"],
            ),
        ]
        max_val = max(s.max() for _, s in panels if len(s) > 0) if any(len(s) > 0 for _, s in panels) else 250
        for ax, (title, data) in zip(axs.flatten(), panels, strict=False):
            if len(data) > 0:
                ax.hist(
                    data, bins=range(int(max_val) + 1), color="#3a9ad9", alpha=0.7, edgecolor="white", linewidth=0.3
                )
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
        _sbs96, sbs6_vaf = render_sbs_vaf_combined(sig_df, signature_filter_query, plot_sbs_fn, plot_af_fn)
        unfilt_sig_sbs_vaf_parts.append(sbs6_vaf)
    unfilt_sig_sbs_vaf_img = unfilt_sig_sbs_vaf_parts[0] if unfilt_sig_sbs_vaf_parts else None

    unfilt_sig_intersection_img = render_intersection_af_combined(df_supporting_reads_per_locus_unfilt, df_signatures)

    # ── QC Analysis — Unfiltered Reads ──
    unfilt_reads_signal_noise_img = _render_signal_noise_internal(detection_unfilt2, df_tf_unfilt2)

    unfilt_reads_sbs_vaf_parts = []
    for sig in matched_sigs:
        sig_df = df_signatures_filt.query(f"signature == '{sig}'")
        if len(sig_df) > 0:
            _sbs96, sbs6_vaf = render_sbs_vaf_combined(sig_df, signature_filter_query, plot_sbs_fn, plot_af_fn)
            unfilt_reads_sbs_vaf_parts.append(sbs6_vaf)
    unfilt_reads_sbs_vaf_img = unfilt_reads_sbs_vaf_parts[0] if unfilt_reads_sbs_vaf_parts else None

    unfilt_reads_intersection_img = render_intersection_af_combined(
        df_supporting_reads_per_locus_unfilt2, df_signatures_filt
    )

    # ── Format values ──
    binom_p_str = f"{detection.p_value:.3f}" if detection.p_value >= 0.001 else f"{detection.p_value:.2e}"  # noqa: PLR2004
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

    template = _JINJA_ENV.get_template("mrd_qc_report.html")
    return template.render(**context)


def _render_signal_noise_internal(detection: DetectionResult, df_tf: pd.DataFrame) -> str:
    """Render signal vs noise plot (internal helper, same as render_signal_vs_noise)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f4f6f8")
    plot_null_distribution(detection, df_tf, ax=ax)
    plt.tight_layout()
    return _fig_to_base64(fig)
