#!/usr/bin/env python
# Copyright 2024 Ultima Genomics Inc.
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
#    Prepare data for training CNV filtering model

from __future__ import annotations

import argparse
import logging
import sys

from ugbio_core.logger import logger

from ugbio_filtering import training_prep


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(
        prog="training_prep_cnv_pipeline.py", description="Prepare training data for CNV filtering model"
    )
    ap_var.add_argument("--call_vcf", help="Call VCF file", type=str, required=True)
    ap_var.add_argument("--base_vcf", help="Truth VCF file", type=str, required=True)
    ap_var.add_argument("--hcr", help="High confidence regions BED file", type=str, required=False)
    ap_var.add_argument(
        "--custom_annotations",
        help="Custom INFO annotations to read from the VCF (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
    )

    ap_var.add_argument(
        "--train_fraction",
        help="Fraction of CNVs to use for training (rest will be used for testing)",
        type=float,
        default=0.25,
    )

    ap_var.add_argument("--output_prefix", help="Output HDF5 files prefix", type=str, required=True)

    ap_var.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )

    ap_var.add_argument(
        "--ignore_cnv_type",
        help="Ignore CNV type when matching to truth",
        required=False,
        default=False,
        action="store_true",
    )

    ap_var.add_argument(
        "--skip_collapse",
        help="Skip collapsing variants before comparison",
        required=False,
        default=False,
        action="store_true",
    )

    args = ap_var.parse_args(argv)
    if args.custom_annotations is None:
        args.custom_annotations = [
            "REGION_ANNOTATIONS",
            "CNV_SOURCE",
            "RoundedCopyNumber",
            "CopyNumber",
            "CNMOPS_SAMPLE_STDEV",
            "CNMOPS_SAMPLE_MEAN",
            "CNMOPS_COHORT_STDEV",
            "CNMOPS_COHORT_MEAN",
            "pytorQ0",
            "pytorP2",
            "pytorRD",
            "pytorP1",
            "pytorP3",
            "CN",
            "GAP_PERCENTAGE",
            "CNV_DUP_READS",
            "CNV_DEL_READS",
            "CNV_DUP_FRAC",
            "CNV_DEL_FRAC",
            "JALIGN_DUP_SUPPORT",
            "JALIGN_DEL_SUPPORT",
            "JALIGN_DUP_SUPPORT_STRONG",
            "JALIGN_DEL_SUPPORT_STRONG",
            "SVTYPE",
            "SVLEN",
            "DISCORDANT_OVERLAP",
            "CONCORDANT_OVERLAP",
        ]

    return args


def run(argv: list[str]):
    """Run function"""
    args = parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))
    logger.debug(args)

    logger.info("Prepare training data started")
    training_prep.training_prep_cnv(
        call_vcf=args.call_vcf,
        base_vcf=args.base_vcf,
        hcr=args.hcr,
        custom_annotations=args.custom_annotations,
        train_fraction=args.train_fraction,
        output_prefix=args.output_prefix,
        ignore_cnv_type=args.ignore_cnv_type,
        skip_collapse=args.skip_collapse,
    )
    logger.info("Prepare training data finished")
    return 0


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
