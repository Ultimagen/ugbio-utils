import argparse
import logging
import multiprocessing as mp
import os
import sys

import pandas as pd
from ugbio_core.logger import logger


def parse_bases(bases):
    ref_count = 0
    nonref_count = 0
    i = 0
    while i < len(bases):
        c = bases[i]
        if c in ".,":
            ref_count += 1
        elif c in "ACGTNacgtn*":
            nonref_count += 1
        elif c == "^":
            i += 1
        elif c == "$":
            pass
        elif c in "+-":
            i += 1
            length = ""
            nonref_count += 1  # count the indel itself
            while i < len(bases) and bases[i].isdigit():
                length += bases[i]
                i += 1
            i += int(length) - 1
        i += 1
    return ref_count, nonref_count


def process_mpileup_file(mpiluep_filepath):
    """Process mpileup file and add ref/nonref counts using parallel processing."""
    df_mpileup = pd.read_csv(
        mpiluep_filepath, sep="\t", names=["chrom", "pos", "ref", "depth", "bases"], usecols=[0, 1, 2, 3, 4]
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(parse_bases, df_mpileup["bases"])

    df_mpileup[["ref_count", "nonref_count"]] = pd.DataFrame(results, columns=["ref_count", "nonref_count"])
    df_mpileup.drop(columns=["bases", "depth"])  # Drop unnecessary columns
    return df_mpileup


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="convert_mpileup_to_ref_nonref_counts.py",
        description="given mpileup, count ref/non-ref #bases per location",
    )
    parser.add_argument(
        "--input_mpileup", required=True, help="input mpileup file. tsv file format: chrom, pos, ref, depth, bases"
    )
    parser.add_argument("--out_directory", required=True, help="Output directory")
    return parser.parse_args(argv[1:])


def run(argv: list[str]) -> str:
    """ """
    args = __parse_args(argv)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(logging.INFO)

    logger.info(f"Output directory: {args.out_directory}")
    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)
        logger.info(f"Created output directory: {args.out_directory}")

    # Process input tumor/normal mpileup to ref/non-ref counts in tsv format
    # Process tumor and normal mpileup files
    df_mpileup = process_mpileup_file(args.input_mpileup)

    # Write processed tumor mpileup to TSV file
    mpileup_output_path = os.path.join(
        args.out_directory, {args.input_mpileup}.replace(".pileup", ".ref_nonref_counts.tsv")
    )
    df_mpileup[["chrom", "pos", "ref", "ref_count", "nonref_count"]].to_csv(
        mpileup_output_path, sep="\t", header=False, index=False
    )
    logger.info(f"Wrote mpileup ref,nonref counts to: {mpileup_output_path}")

    return mpileup_output_path


def main() -> None:
    """
    Entry point for command line execution.
    """
    run(sys.argv)


if __name__ == "__main__":  # pragma: no cover
    main()
