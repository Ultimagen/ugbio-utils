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
#    Run CNVpytor on a single sample.
# CHANGELOG in reverse chronological order

import argparse
import logging
import os
import sys
from os.path import join as pjoin

import cnvpytor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_cnvpytor.py",
        description=run.__doc__,
    )
    parser.add_argument("--input_bam_cram_file", help="input cram/bam file", required=True, type=str)
    parser.add_argument("--sample_name", help="sample_name", required=True, type=str)
    parser.add_argument("--ref_fasta", help="reference fasta file", required=True, type=str)
    parser.add_argument(
        "--bin_size", help="window size for coverage calculation", required=False, type=int, default=500
    )
    parser.add_argument(
        "--chr_list",
        help="chromosomes list sepereated by comma",
        required=False,
        type=str,
        default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY",
    )
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """
    Given a bam/cram file, this script will run cnvpytor on the sample:
    output consists of 2 files:
    - cnv calls in tsv format
    """
    args = __parse_args(argv)
    logger = logging.getLogger("cnvpytor")

    app = cnvpytor.Root(pjoin(args.out_directory, f"{args.sample_name}.pytor"), create=True, max_cores=os.cpu_count())

    chroms = args.chr_list.split(",")
    bin_size = int(args.bin_size)
    logger.info("chromosomoes list:")
    logger.info(chroms)

    # import RD signal from bam file:
    app.rd([args.input_bam_cram_file], chroms=chroms, reference_filename=args.ref_fasta)

    # Calculate histograms with bin sizes=bin_size:
    app.calculate_histograms([bin_size])

    # Calculate partition for the same bin sizes:
    app.partition([bin_size])

    # Calculate and print CNV calls:
    calls = app.call([bin_size])
    for bin_size in calls:
        out_file = pjoin(args.out_directory, f"{args.sample_name}.pytor.bin{bin_size}.CNVs.tsv")
        with open(out_file, "w") as file:
            for call in calls[bin_size]:
                # Use file.write instead of print
                file.write(("{:13}{:>5}{:10}{:10}{:10}{:15.2f}{:15.2e}{:15.2e}\n").format(*tuple(call)))

    logger.info("output files:")
    logger.info(out_file)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
