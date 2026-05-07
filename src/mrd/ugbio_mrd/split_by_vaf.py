"""Split featuremap parquet by VAF bins and generate trinucleotide substitution profiles."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
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

FIRST_BIN_SINGLE_READ_LABEL = "0-0.5% (1 read)"
FIRST_BIN_MULTI_READ_LABEL = "0-0.5% (>1 reads)"
VARIANT_KEY_COLUMNS = ["CHROM", "POS", "REF", "ALT"]

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


def get_vaf_bin_labels() -> list[str]:
    """Return output bin labels, with the first VAF bin split by read support."""
    return [
        FIRST_BIN_SINGLE_READ_LABEL,
        FIRST_BIN_MULTI_READ_LABEL,
        *[label for _, _, label in VAF_BINS[1:]],
    ]


def _assign_vaf_bin(vaf: float, supporting_read_count: int | None = None) -> str | None:
    """Assign a VAF bin, splitting the first bin by supporting read count."""
    for index, (lower, upper, label) in enumerate(VAF_BINS):
        if lower <= vaf < upper:
            if index == 0 and supporting_read_count is not None:
                if supporting_read_count <= 1:
                    return FIRST_BIN_SINGLE_READ_LABEL
                return FIRST_BIN_MULTI_READ_LABEL
            return label
    return None


def split_by_vaf(  # noqa: PLR0915
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

    required_columns = [
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "X_PREV1",
        "X_NEXT1",
        "VAF",
        "SNVQ",
        "FILT",
        "MAPQ",
    ]

    # Check for missing columns first by reading schema
    logger.info(f"Reading parquet: {input_parquet}")
    schema = pl.read_parquet_schema(input_parquet)
    missing = [c for c in required_columns if c not in schema]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Read only required columns and convert to lazy for the computation pipeline
    df_region = pl.read_parquet(input_parquet, columns=required_columns)
    logger.info(f"Input shape: {df_region.shape}")
    lf = df_region.lazy()

    # Quality filtering with predicate pushdown
    lf = (
        lf.with_columns(
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
    )

    # Compute supporting read count per variant in Polars
    lf = lf.with_columns(pl.col("POS").count().over(["CHROM", "POS", "REF", "ALT"]).alias("supporting_read_count"))

    # Compute canonical trinucleotide substitution entirely in Polars
    # Logic: if ref is purine (A/G), reverse-complement context and alt
    is_purine = pl.col("REF").is_in(["A", "G"])

    def _complement_expr(col_name):
        """Build a Polars expression that complements a single-base column."""
        expr = pl.col(col_name)
        # Use temporary lowercase to avoid double-substitution
        for orig, comp in [("A", "t"), ("T", "a"), ("C", "g"), ("G", "c")]:
            expr = expr.str.replace(orig, comp, literal=True)
        return expr.str.to_uppercase()

    lf = lf.with_columns(
        [
            pl.when(is_purine).then(_complement_expr("X_NEXT1")).otherwise(pl.col("X_PREV1")).alias("ctx_left"),
            pl.when(is_purine).then(_complement_expr("REF")).otherwise(pl.col("REF")).alias("ctx_ref"),
            pl.when(is_purine).then(_complement_expr("X_PREV1")).otherwise(pl.col("X_NEXT1")).alias("ctx_right"),
            pl.when(is_purine).then(_complement_expr("ALT")).otherwise(pl.col("ALT")).alias("canon_alt"),
        ]
    )

    # Build the canonical trinuc string: "X[Y>Z]W"
    lf = lf.with_columns(
        (
            pl.col("ctx_left")
            + pl.lit("[")
            + pl.col("ctx_ref")
            + pl.lit(">")
            + pl.col("canon_alt")
            + pl.lit("]")
            + pl.col("ctx_right")
        ).alias("trinuc_substitution")
    )

    # Filter to valid trinuc substitutions and exclude ref==alt (already canonical)
    valid_trinucs = set(TRINUC_ORDER)
    lf = lf.filter(pl.col("trinuc_substitution").is_in(valid_trinucs))

    # Assign VAF bins in Polars
    vaf_col = pl.col("vaf")
    src_col = pl.col("supporting_read_count")

    # Build a chained when/then for VAF bin assignment
    # First bin is special: split by supporting_read_count
    first_lower, first_upper, _ = VAF_BINS[0]
    vaf_bin_expr = (
        pl.when((vaf_col >= first_lower) & (vaf_col < first_upper) & (src_col <= 1))
        .then(pl.lit(FIRST_BIN_SINGLE_READ_LABEL))
        .when((vaf_col >= first_lower) & (vaf_col < first_upper) & (src_col > 1))
        .then(pl.lit(FIRST_BIN_MULTI_READ_LABEL))
    )
    for lower, upper, label in VAF_BINS[1:]:
        vaf_bin_expr = vaf_bin_expr.when((vaf_col >= lower) & (vaf_col < upper)).then(pl.lit(label))
    vaf_bin_expr = vaf_bin_expr.otherwise(pl.lit(None)).alias("vaf_bin")

    lf = lf.with_columns(vaf_bin_expr).filter(pl.col("vaf_bin").is_not_null())

    # Collect the final result
    logger.info("Collecting filtered and annotated data...")
    df_with_counts = lf.collect()
    logger.info(f"After quality filtering and trinuc computation: {df_with_counts.height:,} rows")

    # Build profile table directly in Polars
    bin_labels = get_vaf_bin_labels()

    if df_with_counts.height > 0:
        grouped = df_with_counts.group_by(["vaf_bin", "trinuc_substitution"]).len()
        profile_polars = grouped.pivot(  # noqa: PD010
            on="vaf_bin", index="trinuc_substitution", values="len"
        ).sort("trinuc_substitution")
        profile_table = profile_polars.to_pandas().set_index("trinuc_substitution")
        profile_table = (
            profile_table.reindex(TRINUC_ORDER)
            .fillna(0)
            .astype(int)
            .reindex(columns=bin_labels, fill_value=0)
            .reset_index()
        )
    else:
        profile_table = pd.DataFrame({"trinuc_substitution": TRINUC_ORDER})
        for label in bin_labels:
            profile_table[label] = 0

    counts_csv = os.path.join(output_dir, f"{basename}.trinuc_counts.csv")
    profile_table.to_csv(counts_csv, index=False)
    logger.info(f"Wrote trinuc counts: {counts_csv}")

    # Build histogram from profile_table to reduce memory footprint
    palette = dict(
        zip(
            SUBSTITUTION_ORDER,
            sns.color_palette("Set2", n_colors=len(SUBSTITUTION_ORDER)),
            strict=False,
        )
    )
    fig, axes = plt.subplots(len(bin_labels), 1, figsize=(24, 2.8 * len(bin_labels)), sharex=True)
    if len(bin_labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, bin_labels, strict=False):
        # Count variants per trinuc from profile_table instead of subsetting large dataframe
        counts = profile_table.set_index("trinuc_substitution")[label].reindex(TRINUC_ORDER, fill_value=0)
        colors = [palette[item[2:5]] for item in counts.index]
        total = counts.sum()
        ax.bar(range(len(counts)), counts.values, color=colors, width=0.9)
        ax.set_ylabel(label)
        ax.set_title(f"{label} VAF (n={int(total):,} reads)", loc="left")
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
