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
#    Process CNV calls from cn.mops: filter by length and UG-CNV-LCR, annotate, and convert to VCF.
# CHANGELOG in reverse chronological order

import argparse
import logging
import os
import sys
import warnings

import ugbio_core.misc_utils as mu
from ugbio_cnv import cnmops_utils
from ugbio_core import bed_utils, vcf_utils
from ugbio_core.logger import logger

warnings.filterwarnings("ignore")

bedtools = "bedtools"
bedmap = "bedmap"


def get_parser():
    parser = argparse.ArgumentParser(
        prog="process_cnmops_cnvs.py",
        description="Process CNV calls from cn.mops: filter, annotate, and convert to VCF",
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


def annotate_bed(bed_file, lcr_cutoff, lcr_file, prefix, length_cutoff=10000):
    # get filters regions
    filter_files = []
    bu = bed_utils.BedUtils()

    if lcr_file is not None:
        lcr_bed_file = bu.filter_by_bed_file(bed_file, lcr_cutoff, lcr_file, prefix, "UG-CNV-LCR")
        filter_files.append(lcr_bed_file)

    if length_cutoff is not None and length_cutoff > 0:
        length_bed_file = bu.filter_by_length(bed_file, length_cutoff, prefix)
        filter_files.append(length_bed_file)

    if not filter_files:
        # No filters to apply, just return sorted bed file
        out_bed_file_sorted = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".annotate.bed"
        bu.bedtools_sort(bed_file, out_bed_file_sorted)
        return out_bed_file_sorted

    out_combined_info = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".unsorted.annotate.combined.bed"
    cnmops_utils.merge_filter_files(bed_file, filter_files, out_combined_info)
    # merge all filters and sort

    out_annotate = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".annotate.bed"
    bu.bedtools_sort(out_combined_info, out_annotate)
    os.unlink(out_combined_info)
    for f in filter_files:
        os.unlink(f)

    return out_annotate


def _aggregate_coverages(
    annotated_bed_file: str, sample_norm_coverage_file: str, cohort_avg_coverage_file: str, tempdir: str
) -> list[tuple[str, str, str]]:
    """
    Prepare coverage annotations for aggregation.
    Parameters
    ----------
    annotated_bed_file : str
        Path to the annotated bed file.
    sample_norm_coverage_file : str
        Path to the sample normalized coverage bed file.
    cohort_avg_coverage_file : str
        Path to the cohort average coverage bed file.
    tempdir : str
        Directory to store intermediate files.
    Returns
    -------
    list of tuple
        List of tuples containing (sample/cohort cvg type, operation, bed_file_path) for coverage annotations.
    """
    coverage_annotations = []
    # annotate with coverage info
    input_sample = ["sample", "cohort"]
    output_param = ["mean", "stdev"]

    for isamp in input_sample:
        for oparam in output_param:
            out_annotate_bed_file_cov = annotated_bed_file.replace(".annotate.bed", f".annotate.{isamp}.{oparam}.bed")
            input_cov_file = sample_norm_coverage_file if isamp == "sample" else cohort_avg_coverage_file
            bed_utils.BedUtils().bedtools_map(
                a_bed=annotated_bed_file,
                b_bed=input_cov_file,
                output_bed=out_annotate_bed_file_cov,
                operation=oparam,
                presort=True,
                tempdir_prefix=tempdir,
                column=5,
            )
            coverage_annotations.append((isamp, oparam, out_annotate_bed_file_cov))
    return coverage_annotations


def run(argv):
    """
    Given a cn.mops bed file, this script will filter it by :
    1. lcr bed (ug_cnv_lcr) file
    3. length
    output is a VCF file with filtering tags
    """

    parser = get_parser()
    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    prefix = ""
    if args.out_directory:
        prefix = args.out_directory
        prefix = prefix.rstrip("/") + "/"

    # Annotate with lcr_file or min_cnv_length are provided, sort otherwise
    out_annotate_bed_file = annotate_bed(
        args.input_bed_file, args.intersection_cutoff, args.cnv_lcr_file, prefix, args.min_cnv_length
    )

    coverage_annotations = []
    if args.sample_norm_coverage_file and args.cohort_avg_coverage_file:
        coverage_annotations = _aggregate_coverages(
            out_annotate_bed_file, args.sample_norm_coverage_file, args.cohort_avg_coverage_file, args.out_directory
        )

    cnmops_cnv_df = cnmops_utils.aggregate_annotations_in_df(out_annotate_bed_file, coverage_annotations)
    cnmops_cnv_df = cnmops_utils.add_ids(cnmops_cnv_df)
    cnmops_cnv_df["SVLEN"] = cnmops_cnv_df["end"] - cnmops_cnv_df["start"]
    out_vcf_file = out_annotate_bed_file.replace(".bed", ".vcf.gz")
    cnmops_utils.write_cnv_vcf(out_vcf_file, cnmops_cnv_df, args.sample_name, args.fasta_index_file)
    vu = vcf_utils.VcfUtils()
    vu.index_vcf(out_vcf_file)
    logger.info("Cleaning temporary files...")
    mu.cleanup_temp_files([out_annotate_bed_file] + [cov[2] for cov in coverage_annotations])
    logger.info(f"output files: {out_vcf_file}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
