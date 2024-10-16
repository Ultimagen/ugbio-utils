#!/env/python
# Copyright 2023 Ultima Genomics Inc.
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
#    Add ml qual to SNVs according to features in featuremap
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse
import sys

from ugbio_srsnv.srsnv_inference_utils import single_read_snv_inference


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="srsnv_inference.py", description=run.__doc__)
    parser.add_argument(
        "-f",
        "--featuremap_path",
        type=str,
        required=True,
        help="""input featuremap file""",
    )
    parser.add_argument(
        "--model_joblib_path",
        type=str,
        required=False,
        default=None,
        help="""Path to joblib file containing model information, i.e., trained models,
        training parameters, and quality interpolation function. If --model_jl_path is provided
        and also any of --model_path, --params_path, --test_set_mrd_simulation_dataframe_file,
        then the latter values will override the values sepcified in the model joblib file.
        """,
    )
    parser.add_argument(
        "-p",
        "--params_path",
        type=str,
        required=False,
        default=None,
        help="""params file path. If not provided, must provide --model_joblib_path.""",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=False,
        help="""model file path. If not provided, must provide --model_joblib_path""",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="""Path to which output files will be written to""",
    )
    parser.add_argument(
        "-d",
        "--test_set_mrd_simulation_dataframe_file",
        type=str,
        required=False,
        help="""Path to MRD simulation dataframe, required to assign qualities and filters""",
    )
    parser.add_argument(
        "-@",
        "--process_number",
        type=int,
        default=0,
        help="""Number of processes to use for parallelization.
             If N < 1, use all-available - abs(N) cores. Default 0""",
    )

    return parser.parse_args(argv[1:])


def run(argv):
    """Add ml qual to SNVs according to features in featuremap"""
    args = parse_args(argv)

    single_read_snv_inference(
        featuremap_path=args.featuremap_path,
        model_joblib_path=args.model_joblib_path,
        params_path=args.params_path,
        model_path=args.model_path,
        out_path=args.output_path,
        test_set_mrd_simulation_dataframe_file=args.test_set_mrd_simulation_dataframe_file,
        process_number=args.process_number,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
