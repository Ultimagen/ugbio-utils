#!/env/python
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
#    MRD data preparation pipeline
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse
import os
import sys

from ugbio_mrd.mrd_utils import prepare_data_from_mrd_pipeline


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="prepare_data", description=run.__doc__)
    parser.add_argument(
        "-f",
        "--intersected-featuremaps",
        nargs="+",
        type=str,
        required=True,
        help="Input signature and featuemaps vcf files",
    )
    parser.add_argument(
        "--matched-signatures-vcf",
        nargs="+",
        type=str,
        default=None,
        help="Input signature vcf file/s (matched)",
    )
    parser.add_argument(
        "--control-signatures-vcf",
        nargs="+",
        type=str,
        default=None,
        help="Input signature vcf file/s (control)",
    )
    parser.add_argument(
        "--db-control-signatures-vcf",
        nargs="+",
        type=str,
        default=None,
        help="Input signature vcf file/s (db control)",
    )
    parser.add_argument(
        "--coverage-csv",
        type=str,
        default=None,
        required=False,
        help="Coverage csv file generated with gatk ExtractCoverageOverVcfFiles",
    )
    parser.add_argument(
        "--tumor-sample",
        type=str,
        required=False,
        default=None,
        help=""" sample name in the vcf to take allele fraction (AF) from. Checked with "a in b" so it doesn't have to
    be the full sample name, but does have to return a unique result. Default: None (auto-discovered) """,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="""Path to which output files will be written.""",
    )
    parser.add_argument(
        "-b",
        "--output-basename",
        type=str,
        default=None,
        help="""Basename of output files that will be created.""",
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """Aggregate the outputs from the MRDFeatureMap pipeline and prepare dataframes for analysis"""
    args_in = parse_args(argv)
    prepare_data_from_mrd_pipeline(
        intersected_featuremaps_parquet=args_in.intersected_featuremaps,
        matched_signatures_vcf_files=args_in.matched_signatures_vcf,
        control_signatures_vcf_files=args_in.control_signatures_vcf,
        db_control_signatures_vcf_files=args_in.db_control_signatures_vcf,
        coverage_csv=args_in.coverage_csv,
        tumor_sample=args_in.tumor_sample,
        output_dir=args_in.output_dir,
        output_basename=args_in.output_basename,
    )
    sys.stdout.write("DONE" + os.linesep)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
