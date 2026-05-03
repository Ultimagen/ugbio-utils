#!/env/python  # noqa: N999
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
#    Run the ppmSeq QC analysis pipeline on the subsampled SAM produced by sorter.
import argparse
import sys

from ugbio_ppmseq.ppmSeq_utils import (
    STRAND_RATIO_LOWER_THRESH,
    STRAND_RATIO_UPPER_THRESH,
    ppmseq_qc_analysis,
    supported_adapter_versions,
)


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ppmSeq_qc_analysis", description=run.__doc__)
    parser.add_argument(
        "--adapter-version",
        choices=supported_adapter_versions,
        help="Library adapter version",
    )
    parser.add_argument(
        "--subsampled-sam",
        type=str,
        required=True,
        help="path to the subsampled sam.gz emitted by sorter (requires demux --sample-nr-reads=N)",
    )
    parser.add_argument(
        "--trimmer-failure-codes-csv",
        type=str,
        required=False,
        help="Trimmer failure codes csv file",
    )
    parser.add_argument(
        "--sorter-stats-csv",
        type=str,
        required=False,
        help="path to a Sorter stats csv file",
    )
    parser.add_argument(
        "--sorter-stats-json",
        type=str,
        required=False,
        help="path to a Sorter stats json file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="path (folder) to which data and report will be written to",
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        default=None,
        help="basename for output files",
    )
    parser.add_argument(
        "--input-material-ng",
        type=float,
        required=False,
        default=None,
        help="Optional - input material in ng, will be included in statistics and report",
    )
    parser.add_argument(
        "--sr-lower",
        type=float,
        default=STRAND_RATIO_LOWER_THRESH,
        help="lower strand ratio threshold for determining strand ratio category",
    )
    parser.add_argument(
        "--sr-upper",
        type=float,
        default=STRAND_RATIO_UPPER_THRESH,
        help="upper strand ratio threshold for determining strand ratio category",
    )
    parser.add_argument(
        "--generate-report",
        type=bool,
        required=False,
        default=True,
        help="""generate an html + jupyter report""",
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """Create ppmSeq QC report"""
    args_in = __parse_args(argv)

    ppmseq_qc_analysis(
        adapter_version=args_in.adapter_version,
        subsampled_sam=args_in.subsampled_sam,
        sorter_stats_csv=args_in.sorter_stats_csv,
        sorter_stats_json=args_in.sorter_stats_json,
        trimmer_failure_codes_csv=args_in.trimmer_failure_codes_csv,
        output_path=args_in.output_path,
        output_basename=args_in.output_basename,
        generate_report=args_in.generate_report,
        sr_lower=args_in.sr_lower,
        sr_upper=args_in.sr_upper,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
