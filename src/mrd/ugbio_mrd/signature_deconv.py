"""Signature deconvolution using MuSiCal on trinucleotide substitution counts."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import musical
import pandas as pd
import seaborn as sns
from ugbio_core.logger import logger

from ugbio_mrd.split_by_vaf import get_vaf_bin_labels


def signature_deconv(  # noqa: PLR0915
    trinuc_counts_csv: str,
    cosmic_signatures_file: str,
    signatures_to_include: list[str],
    output_dir: str,
    basename: str,
    method: str = "likelihood_bidirectional",
    threshold: float = 0.0,
) -> tuple[str, str]:
    """Run MuSiCal signature deconvolution on trinuc counts.

    Parameters
    ----------
    trinuc_counts_csv : str
        CSV file with trinucleotide substitution counts per VAF bin (from split_by_vaf).
    cosmic_signatures_file : str
        Tab-separated COSMIC mutational signatures catalog file.
    signatures_to_include : list[str]
        List of signature names to include in the deconvolution.
    output_dir : str
        Output directory.
    basename : str
        Base name for output files.
    method : str
        MuSiCal refit method.
    threshold : float
        MuSiCal refit threshold.

    Returns
    -------
    tuple[str, str]
        Paths to (signature_weights_csv, signature_plot_png).
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reading trinuc counts: {trinuc_counts_csv}")
    profile_table = pd.read_csv(trinuc_counts_csv)

    logger.info(f"Reading COSMIC signatures: {cosmic_signatures_file}")
    cosmic_catalog = pd.read_csv(cosmic_signatures_file, sep="\t").set_index("Type")

    # Prepare counts matrix: set index to trinuc type, align to cosmic catalog order
    df_counts = profile_table.copy()
    df_counts = df_counts.rename(columns={"trinuc_substitution": "Type"})
    df_counts = df_counts.set_index("Type").reindex(cosmic_catalog.index)

    # Drop non-VAF-bin columns if present (e.g. 'substitution')
    bin_labels = get_vaf_bin_labels()
    cols_to_keep = [c for c in df_counts.columns if c in bin_labels]
    df_counts = df_counts[cols_to_keep].fillna(0).astype(int)

    # Filter cosmic catalog to requested signatures
    missing_sigs = [s for s in signatures_to_include if s not in cosmic_catalog.columns]
    if missing_sigs:
        logger.warning(f"Signatures not found in catalog (skipping): {missing_sigs}")
    available_sigs = [s for s in signatures_to_include if s in cosmic_catalog.columns]
    if not available_sigs:
        raise ValueError(f"None of the requested signatures found in catalog: {signatures_to_include}")

    # Identify empty bins (all zeros) — skip them from refit to avoid crashes
    nonempty_cols = [c for c in cols_to_keep if df_counts[c].sum() > 0]
    empty_cols = [c for c in cols_to_keep if c not in nonempty_cols]
    if empty_cols:
        logger.warning(f"Skipping empty (all-zero) VAF bins: {empty_cols}")

    logger.info(f"Running MuSiCal refit with method={method}, threshold={threshold}, signatures={available_sigs}")
    sig_matrix = cosmic_catalog[available_sigs]

    if nonempty_cols:
        exposures, _model = musical.refit.refit(df_counts[nonempty_cols], W=sig_matrix, method=method, thresh=threshold)
    else:
        exposures = pd.DataFrame(index=available_sigs, columns=cols_to_keep, dtype=float)

    # Fill NaN for empty bins
    for c in empty_cols:
        exposures[c] = float("nan")

    # Reorder columns to match original order
    exposures = exposures[[c for c in cols_to_keep if c in exposures.columns]]

    # Save signature weights
    weights_csv = os.path.join(output_dir, f"{basename}.signature_weights.csv")
    exposures.to_csv(weights_csv)
    logger.info(f"Wrote signature weights: {weights_csv}")

    # Plot signature contributions per VAF bin (only non-empty bins)
    if nonempty_cols:
        fig, axes = plt.subplots(len(nonempty_cols), 1, figsize=(6, 1.5 * len(nonempty_cols)), sharex=True)
        if len(nonempty_cols) == 1:
            axes = [axes]

        for i, bin_label in enumerate(nonempty_cols):
            sns.barplot(data=exposures, x=exposures.index, y=exposures[bin_label], ax=axes[i])
            axes[i].set_title(f"VAF bin: {bin_label}", fontsize=14)
            axes[i].set_ylabel("")
            if i == len(nonempty_cols) - 1:
                axes[i].set_xlabel("Signature")
            else:
                axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=45)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, "All VAF bins are empty", ha="center", va="center", fontsize=14)
        ax.set_axis_off()

    plot_png = os.path.join(output_dir, f"{basename}.signature_deconv.png")
    fig.savefig(plot_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote signature plot: {plot_png}")

    return weights_csv, plot_png


def main():
    parser = argparse.ArgumentParser(description="Signature deconvolution with MuSiCal on trinuc counts.")
    parser.add_argument("--trinuc-counts", required=True, help="Trinucleotide counts CSV file.")
    parser.add_argument("--cosmic-signatures", required=True, help="COSMIC signatures TSV file.")
    parser.add_argument(
        "--signatures-to-include",
        required=True,
        help="Comma-separated list of signature names to include.",
    )
    parser.add_argument("--method", default="likelihood_bidirectional", help="MuSiCal refit method.")
    parser.add_argument("--threshold", type=float, default=0.0, help="MuSiCal refit threshold.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--basename", required=True, help="Base name for output files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    signatures_to_include = [s.strip() for s in args.signatures_to_include.split(",")]

    signature_deconv(
        trinuc_counts_csv=args.trinuc_counts,
        cosmic_signatures_file=args.cosmic_signatures,
        signatures_to_include=signatures_to_include,
        output_dir=args.output_dir,
        basename=args.basename,
        method=args.method,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
