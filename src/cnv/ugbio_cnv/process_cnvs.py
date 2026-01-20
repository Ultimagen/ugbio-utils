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
#    Process CNV calls in BED format like from cn.mops and ControlFREEC:
#    filter by length and UG-CNV-LCR, annotate, and convert to VCF.
# CHANGELOG in reverse chronological order

import argparse
import logging
import sys
import warnings

import ugbio_core.misc_utils as mu
from ugbio_cnv import cnv_bed_format_utils
from ugbio_core import vcf_utils
from ugbio_core.logger import logger

warnings.filterwarnings("ignore")

bedtools = "bedtools"
bedmap = "bedmap"


def get_parser():
    parser = argparse.ArgumentParser(
        prog="process_cnvs.py",
        description="Process CNV calls from cn.mops and ControlFREEC: filter, annotate, and convert to VCF",
    )
    parser.add_argument("--input_bed_file", help="input bed file with .bed suffix", required=True, type=str)
    parser.add_argument(
        "--intersection_cutoff",
        help="intersection cutoff for bedtools subtract function",
        required=False,
        type=float,
        default=0.5,
    )
    parser.add_argument("--cnv_lcr_file", help="UG-CNV-LCR bed file", required=False, type=str)
    parser.add_argument("--min_cnv_length", required=False, type=int, default=10000)
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--sample_norm_coverage_file", help="sample normalized coverage file (BED)", required=False, type=str
    )
    parser.add_argument("--cohort_avg_coverage_file", help="Cohort average coverage (BED)", required=False, type=str)

    parser.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )
    parser.add_argument(
        "--fasta_index_file",
        help="tab delimited file holding reference genome chr ids with their lengths. (.fai file)",
        required=True,
        type=str,
    )
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)

    return parser


def run(argv):
    """
    Given a CNV BED file from cn.mops or ControlFREEC, this script will filter it by:
    1. lcr bed (ug_cnv_lcr) file
    2. length
    Output is a VCF file with filtering tags.
    """

    parser = get_parser()
    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    prefix = ""
    if args.out_directory:
        prefix = args.out_directory
        prefix = prefix.rstrip("/") + "/"

    # Annotate with lcr_file or min_cnv_length are provided, sort otherwise
    out_annotate_bed_file = cnv_bed_format_utils.annotate_bed(
        args.input_bed_file, args.intersection_cutoff, args.cnv_lcr_file, prefix, args.min_cnv_length
    )

    coverage_annotations = []
    if args.sample_norm_coverage_file and args.cohort_avg_coverage_file:
        coverage_annotations = cnv_bed_format_utils.aggregate_coverages(
            out_annotate_bed_file, args.sample_norm_coverage_file, args.cohort_avg_coverage_file, args.out_directory
        )

    cnv_df = cnv_bed_format_utils.aggregate_annotations_in_df(out_annotate_bed_file, coverage_annotations)
    cnv_df = cnv_bed_format_utils.add_ids(cnv_df)
    cnv_df["SVLEN"] = cnv_df["end"] - cnv_df["start"]
    out_vcf_file = out_annotate_bed_file.replace(".bed", ".vcf.gz")
    cnv_bed_format_utils.write_cnv_vcf(out_vcf_file, cnv_df, args.sample_name, args.fasta_index_file)
    vu = vcf_utils.VcfUtils()
    vu.index_vcf(out_vcf_file)
    logger.info("Cleaning temporary files...")
    mu.cleanup_temp_files([out_annotate_bed_file] + [cov[2] for cov in coverage_annotations])
    logger.info(f"output files: {out_vcf_file}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
