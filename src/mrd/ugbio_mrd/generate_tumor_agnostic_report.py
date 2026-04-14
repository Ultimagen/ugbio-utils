"""Generate tumor-agnostic MRD HTML report from trinuc histogram and signature deconvolution results."""

import argparse
import os
from pathlib import Path

from ugbio_core.logger import logger
from ugbio_core.reports.report_utils import generate_report

HTML_REPORT_SUFFIX = ".tumor_agnostic_mrd.report.html"
BASE_PATH = Path(__file__).parent
TEMPLATE_NOTEBOOK = BASE_PATH / "reports" / "tumor_agnostic_mrd_report.ipynb"


def generate_tumor_agnostic_report(
    trinuc_histogram_png: str,
    trinuc_counts_csv: str,
    signature_plot_png: str,
    signature_weights_csv: str,
    output_dir: str,
    basename: str,
) -> Path:
    """Generate HTML report from tumor-agnostic MRD analysis results.

    Parameters
    ----------
    trinuc_histogram_png : str
        Path to trinucleotide histogram PNG.
    trinuc_counts_csv : str
        Path to trinucleotide counts CSV.
    signature_plot_png : str
        Path to signature deconvolution plot PNG.
    signature_weights_csv : str
        Path to signature weights CSV.
    output_dir : str
        Output directory.
    basename : str
        Base name for the report.

    Returns
    -------
    Path
        Path to the generated HTML report.
    """
    os.makedirs(output_dir, exist_ok=True)

    parameters = {
        "trinuc_histogram_png": trinuc_histogram_png,
        "trinuc_counts_csv": trinuc_counts_csv,
        "signature_plot_png": signature_plot_png,
        "signature_weights_csv": signature_weights_csv,
        "basename": basename,
    }
    output_report_html_path = Path(output_dir) / (basename + HTML_REPORT_SUFFIX)

    logger.info(f"Generating tumor-agnostic MRD report: {output_report_html_path}")
    generate_report(
        template_notebook_path=TEMPLATE_NOTEBOOK,
        parameters=parameters,
        output_report_html_path=output_report_html_path,
    )
    return output_report_html_path


def main():
    parser = argparse.ArgumentParser(description="Generate tumor-agnostic MRD HTML report.")
    parser.add_argument("--trinuc-histogram", required=True, help="Trinucleotide histogram PNG file.")
    parser.add_argument("--trinuc-counts", required=True, help="Trinucleotide counts CSV file.")
    parser.add_argument("--signature-plot", required=True, help="Signature deconvolution plot PNG file.")
    parser.add_argument("--signature-weights", required=True, help="Signature weights CSV file.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--basename", required=True, help="Base name for the report.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    generate_tumor_agnostic_report(
        trinuc_histogram_png=args.trinuc_histogram,
        trinuc_counts_csv=args.trinuc_counts,
        signature_plot_png=args.signature_plot,
        signature_weights_csv=args.signature_weights,
        output_dir=args.output_dir,
        basename=args.basename,
    )


if __name__ == "__main__":
    main()
