"""Split featuremap parquet by VAF bins and generate trinucleotide substitution profiles."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from ugbio_core.logger import logger

VAF_BINS = [
    (0.0, 0.005, "0-0.5%"),
    (0.005, 0.05, "0.5-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.30, "10-30%"),
    (0.30, 0.50, "30-50%"),
    (0.50, 1.0000001, "50-100%"),
]

SUBSTITUTION_ORDER = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
TRINUC_ORDER = [f"{left}[{sub}]{right}" for sub in SUBSTITUTION_ORDER for left in "ACGT" for right in "ACGT"]


def _revcomp(seq: str) -> str:
    table = str.maketrans("ACGT", "TGCA")
    return seq.translate(table)[::-1]


def _canonical_trinuc_change(left: str, ref: str, right: str, alt: str) -> str | None:
    """Canonicalize a trinucleotide substitution to pyrimidine context."""
    bases = {"A", "C", "G", "T"}
    left, ref, right, alt = left.upper(), ref.upper(), right.upper(), alt.upper()
    if {left, ref, right, alt} - bases:
        return None
    if ref == alt:
        return None
    if ref in {"A", "G"}:
        context = _revcomp(left + ref + right)
        alt = _revcomp(alt)
    else:
        context = left + ref + right
    return f"{context[0]}[{context[1]}>{alt}]{context[2]}"


def _assign_vaf_bin(vaf: float) -> str | None:
    for lower, upper, label in VAF_BINS:
        if lower <= vaf < upper:
            return label
    return None


def split_by_vaf(
    input_parquet: str,
    output_dir: str,
    basename: str,
    snvq_threshold: float = 60,
    mapq_threshold: float = 60,
) -> tuple[str, str]:
    """Read featuremap parquet, filter, split by VAF, produce trinuc counts CSV and histogram PNG.

    Parameters
    ----------
    input_parquet : str
        Path to the featuremap parquet file.
    output_dir : str
        Directory for output files.
    basename : str
        Base name for output files.
    snvq_threshold : float
        Minimum SNVQ quality score.
    mapq_threshold : float
        Minimum MAPQ mapping quality.

    Returns
    -------
    tuple[str, str]
        Paths to (trinuc_counts_csv, trinuc_histogram_png).
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reading parquet: {input_parquet}")
    df_region = pl.read_parquet(input_parquet)
    logger.info(f"Input shape: {df_region.shape}")

    required_columns = ["CHROM", "POS", "REF", "ALT", "X_PREV1", "X_NEXT1", "VAF", "SNVQ", "FILT", "MAPQ"]
    missing = [c for c in required_columns if c not in df_region.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_plot = (
        df_region.select(required_columns)
        .with_columns(
            [
                pl.col("REF").cast(pl.Utf8).str.to_uppercase(),
                pl.col("ALT").cast(pl.Utf8).str.to_uppercase(),
                pl.col("X_PREV1").cast(pl.Utf8).str.to_uppercase(),
                pl.col("X_NEXT1").cast(pl.Utf8).str.to_uppercase(),
                pl.col("VAF").cast(pl.Float64).alias("vaf"),
                pl.col("SNVQ").cast(pl.Float64),
                pl.col("FILT").cast(pl.Float64),
                pl.col("MAPQ").cast(pl.Float64),
            ]
        )
        .filter(
            pl.col("REF").str.len_chars() == 1,
            pl.col("ALT").str.len_chars() == 1,
            pl.col("X_PREV1").str.len_chars() == 1,
            pl.col("X_NEXT1").str.len_chars() == 1,
            pl.col("SNVQ") > snvq_threshold,
            pl.col("FILT") > 0,
            pl.col("MAPQ") >= mapq_threshold,
        )
        .filter((pl.col("vaf") >= 0) & (pl.col("vaf") <= 1))
        .to_pandas()
    )

    logger.info(f"After quality filtering: {len(df_plot):,} rows")

    # Compute canonical trinucleotide substitution
    df_plot["trinuc_substitution"] = [
        _canonical_trinuc_change(left, ref, right, alt)
        for left, ref, right, alt in zip(
            df_plot["X_PREV1"], df_plot["REF"], df_plot["X_NEXT1"], df_plot["ALT"], strict=False
        )
    ]
    df_plot = df_plot.dropna(subset=["trinuc_substitution"]).copy()

    # Assign VAF bin
    df_plot["vaf_bin"] = df_plot["vaf"].map(_assign_vaf_bin)
    df_plot = df_plot.dropna(subset=["vaf_bin"]).copy()
    if len(df_plot) > 0:
        df_plot["substitution"] = df_plot["trinuc_substitution"].str.extract(r"\[([CT]>[ACGT])\]", expand=False)
    else:
        df_plot["substitution"] = pd.Series(dtype=str)

    logger.info(f"Rows with usable context: {len(df_plot):,}")

    # Build profile table
    bin_labels = [label for _, _, label in VAF_BINS]
    profile_table = (
        df_plot.groupby(["vaf_bin", "trinuc_substitution"])
        .size()
        .rename("count")
        .reset_index()
        .pivot_table(index="trinuc_substitution", columns="vaf_bin", values="count")
        .reindex(TRINUC_ORDER)
        .fillna(0)
        .astype(int)
        .reindex(columns=bin_labels, fill_value=0)
        .reset_index()
    )

    counts_csv = os.path.join(output_dir, f"{basename}.trinuc_counts.csv")
    profile_table.to_csv(counts_csv, index=False)
    logger.info(f"Wrote trinuc counts: {counts_csv}")

    # Plot trinucleotide histogram
    palette = dict(zip(SUBSTITUTION_ORDER, sns.color_palette("Set2", n_colors=len(SUBSTITUTION_ORDER)), strict=False))
    fig, axes = plt.subplots(len(bin_labels), 1, figsize=(24, 2.8 * len(bin_labels)), sharex=True)
    if len(bin_labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, bin_labels, strict=False):
        subset = df_plot.loc[df_plot["vaf_bin"] == label]
        counts = subset["trinuc_substitution"].value_counts().reindex(TRINUC_ORDER, fill_value=0)
        colors = [palette[item[2:5]] for item in counts.index]
        ax.bar(range(len(counts)), counts.values, color=colors, width=0.9)
        ax.set_ylabel(label)
        ax.set_title(f"{label} VAF (n={len(subset):,} reads)", loc="left")
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xticks(range(len(TRINUC_ORDER)))
    axes[-1].set_xticklabels(TRINUC_ORDER, rotation=90, fontsize=8)
    fig.suptitle("Tumor-agnostic MRD trinucleotide substitution profile by VAF", y=1.02)
    fig.tight_layout()

    histogram_png = os.path.join(output_dir, f"{basename}.trinuc_histogram.png")
    fig.savefig(histogram_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote histogram: {histogram_png}")

    return counts_csv, histogram_png


def main():
    parser = argparse.ArgumentParser(description="Split featuremap by VAF bins and generate trinuc profiles.")
    parser.add_argument("--input", required=True, help="Input featuremap parquet file.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--basename", required=True, help="Base name for output files.")
    parser.add_argument("--snvq-threshold", type=float, default=60, help="Minimum SNVQ threshold (default: 60).")
    parser.add_argument("--mapq-threshold", type=float, default=60, help="Minimum MAPQ threshold (default: 60).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    split_by_vaf(
        input_parquet=args.input,
        output_dir=args.output_dir,
        basename=args.basename,
        snvq_threshold=args.snvq_threshold,
        mapq_threshold=args.mapq_threshold,
    )


if __name__ == "__main__":
    main()
