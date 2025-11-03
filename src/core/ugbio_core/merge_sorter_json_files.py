from __future__ import annotations

import argparse
import sys

from ugbio_core.sorter_utils import merge_sorter_json_files


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="merge-sorter-json",
        description=(
            "Merge two or more sorter statistics JSON files by summing integer counts, "
            "zero-padding and summing lists (including nested), and merging nested dicts. "
            "The 'extra_information' field is treated as strict metadata and must match across inputs."
        ),
    )
    parser.add_argument(
        "-i",
        "--inputs",
        dest="inputs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input sorter JSON files (two or more).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        help="Path to the output merged JSON file.",
    )
    parser.add_argument(
        "--stringent",
        dest="stringent",
        action="store_true",
        help=(
            "Enable stringent mode: raise on missing keys or metadata mismatches. "
            "Default is non-stringent (log warnings)."
        ),
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]) -> None:
    args_in = __parse_args(argv)
    if len(args_in.inputs) < 2:  # noqa: PLR2004
        raise ValueError("At least two input JSON files are required")
    merge_sorter_json_files(
        sorter_json_stats_files=args_in.inputs,
        output_json_file=args_in.output,
        stringent_mode=args_in.stringent,
    )


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
