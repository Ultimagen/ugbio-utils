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
#    Intersects featuremap vcf-like with pre-defined signature VCF-like
# CHANGELOG in reverse chronological order
import argparse
import sys

from ugbio_mrd.mrd_utils import intersect_featuremap_with_signature


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="intersect_with_signature", description=run.__doc__)
    parser.add_argument(
        "-f",
        "--featuremap",
        type=str,
        required=True,
        help="""Featuremap vcf file""",
    )
    parser.add_argument(
        "-s",
        "--signature",
        type=str,
        required=True,
        help="""Signature vcf file""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=False,
        help="""Output intersection vcf file (lines from featuremap propagated)""",
    )
    parser.add_argument(
        "-stype",
        "--signature_type",
        type=str,
        default=None,
        required=False,
        help="""matched, control or db_control""",
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """Intersect featuremap and signature vcf files on position and matching ref and alts"""
    args_in = __parse_args(argv)
    intersect_featuremap_with_signature(
        featuremap_file=args_in.featuremap,
        signature_file=args_in.signature,
        output_intersection_file=args_in.output,
        signature_type=args_in.signature_type,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
