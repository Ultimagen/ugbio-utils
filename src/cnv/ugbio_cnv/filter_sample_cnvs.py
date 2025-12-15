# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    CLI interface to filter CNV calls from BED file using LCR regions and length filters.
# CHANGELOG in reverse chronological order

import argparse
import logging
import sys

from ugbio_cnv.cnmops_utils import annotate_bed
from ugbio_core.logger import logger


def get_parser():
    """Create and return argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="filter_sample_cnvs",
        description="Filter CNV calls from BED file using LCR regions and/or length filters",
    )
    parser.add_argument(
        "--input_bed_file",
        help="Input BED file with CNV calls (4 columns: chr, start, end, copy-number)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--intersection_cutoff",
        help="Intersection cutoff for bedtools subtract function (default: 0.5)",
        required=False,
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--cnv_lcr_file",
        help="UG-CNV-LCR BED file for filtering low-complexity regions",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--min_cnv_length",
        help="Minimum CNV length filter (default: 10000)",
        required=False,
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--out_directory",
        help="Output directory where filtered files will be saved. "
        "If not supplied, files will be written to current directory",
        required=False,
        type=str,
        default="",
    )
    parser.add_argument(
        "--verbosity",
        help="Verbosity level: ERROR, WARNING, INFO, DEBUG (default: INFO)",
        required=False,
        default="INFO",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
    )
    return parser


def main():
    """Main entry point for the CLI."""
    parser = get_parser()
    args = parser.parse_args()

    # Set logging level
    logger.setLevel(getattr(logging, args.verbosity))

    # Prepare prefix for output files
    prefix = ""
    if args.out_directory:
        prefix = args.out_directory.rstrip("/") + "/"

    logger.info(f"Filtering CNV calls from: {args.input_bed_file}")
    if args.cnv_lcr_file:
        logger.info(f"Using LCR file: {args.cnv_lcr_file} (cutoff: {args.intersection_cutoff})")
    if args.min_cnv_length:
        logger.info(f"Minimum CNV length filter: {args.min_cnv_length}")

    # Call the annotate_bed function
    output_file = annotate_bed(
        bed_file=args.input_bed_file,
        lcr_cutoff=args.intersection_cutoff,
        lcr_file=args.cnv_lcr_file,
        prefix=prefix,
        length_cutoff=args.min_cnv_length,
    )

    logger.info(f"Filtered CNV file created: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
