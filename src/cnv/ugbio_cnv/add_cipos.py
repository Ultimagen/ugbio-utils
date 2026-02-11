"""Add CIPOS (confidence interval around position) to CNV VCF files."""

import argparse
import sys

import pysam
from ugbio_cnv.cnv_vcf_consts import INFO_TAG_REGISTRY


def add_cipos_to_vcf(input_vcf: str, output_vcf: str, window_size: int) -> None:
    """Add CIPOS field to all records in a VCF file based on window size.

    Adds CIPOS INFO field to the header if not already present, then adds CIPOS
    values to all records.

    Parameters
    ----------
    input_vcf : str
        Input VCF file path.
    output_vcf : str
        Output VCF file path.
    window_size : int
        Window size (bin size) used for CNV calling. Used to calculate CIPOS as
        (-window_size/2, window_size/2+1).
    """
    with pysam.VariantFile(input_vcf) as vcf_in:
        header = vcf_in.header.copy()

        # Add CIPOS to header if not already present
        if "CIPOS" not in header.info:
            cipos_info = INFO_TAG_REGISTRY["CIPOS"]
            header.info.add(cipos_info[0], cipos_info[1], cipos_info[2], cipos_info[3])

        with pysam.VariantFile(output_vcf, "w", header=header) as vcf_out:
            for record in vcf_in:
                record.info["CIPOS"] = (round(-window_size / 2), round(window_size / 2 + 1))
                vcf_out.write(record)


def run(argv: list[str]) -> None:
    """Entry point for add_cipos_to_vcf command.

    Parameters
    ----------
    argv : list[str]
        Command line arguments.
    """
    parser = argparse.ArgumentParser(description="Add CIPOS field to VCF records based on window size")
    parser.add_argument("--input_vcf", required=True, help="Input VCF file")
    parser.add_argument("--output_vcf", required=True, help="Output VCF file")
    parser.add_argument("--window_size", required=True, type=int, help="Window size for CIPOS calculation")
    args = parser.parse_args(argv[1:])
    add_cipos_to_vcf(args.input_vcf, args.output_vcf, args.window_size)


def main() -> None:
    """Main entry point for command line."""
    run(sys.argv)


if __name__ == "__main__":
    main()
