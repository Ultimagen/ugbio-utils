import argparse
import logging
import multiprocessing as mp
import os
import sys

import pandas as pd
from ugbio_core.logger import logger


def parse_bases(bases: str) -> tuple[int, int]:
    """
    Parse base calls from mpileup format and count reference and non-reference bases.

    Parameters
    ----------
    bases : str
        String containing base calls from mpileup format. Can include:
        - '.' or ',' for reference bases
        - 'ACGTNacgtn*' for non-reference bases
        - '^' followed by mapping quality for read start
        - '$' for read end
        - '+' or '-' followed by length digits and inserted/deleted sequence

    Returns
    -------
    ref_count : int
        Number of reference base calls (. or ,)
    nonref_count : int
        Number of non-reference base calls (substitutions and indels)

    Notes
    -----
    The function handles mpileup format special characters:
    - Skips mapping quality after '^'
    - Ignores '$' markers
    - Properly parses indel notation (+/-) with length specification
    """
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


def process_mpileup_file(mpileup_filepath: str) -> pd.DataFrame:
    """
    Process mpileup file and add ref/nonref counts using parallel processing.

    Parameters
    ----------
    mpileup_filepath : str
        Path to the mpileup file to be processed.

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns: chrom, pos, ref, ref_count, nonref_count.
        The 'bases' and 'depth' columns are dropped from the original data.

    Notes
    -----
    The function reads a mpileup file with tab-separated values containing
    chromosome, position, reference base, depth, and bases information.
    It uses multiprocessing to parse the bases column in parallel and
    calculate reference and non-reference counts for each position.
    """
    df_mpileup = pd.read_csv(
        mpileup_filepath, sep="\t", names=["chrom", "pos", "ref", "depth", "bases"], usecols=[0, 1, 2, 3, 4]
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(parse_bases, df_mpileup["bases"])

    df_mpileup[["ref_count", "nonref_count"]] = pd.DataFrame(results, columns=["ref_count", "nonref_count"])
    df_mpileup.drop(columns=["bases", "depth"])  # Drop unnecessary columns
    return df_mpileup


def __parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments for mpileup to ref/non-ref counts conversion.

    Parameters
    ----------
    argv : list[str]
        Command line arguments passed to the script, typically sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing:
        - input_mpileup: Path to input mpileup file (tsv format with columns: chrom, pos, ref, depth, bases)
        - out_directory: Path to output directory where results will be saved

    Notes
    -----
    The input mpileup file should be in TSV format with the following columns:
    - chrom: Chromosome identifier
    - pos: Genomic position
    - ref: Reference base
    - depth: Read depth at position
    - bases: Observed bases at position
    """
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
    """
    Process mpileup file and convert to reference/non-reference allele counts.

    Parameters
    ----------
    argv : list[str]
        Command line arguments containing:
        - input_mpileup: Path to input mpileup file
        - out_directory: Path to output directory for results

    Returns
    -------
    str
        Path to the output TSV file containing reference and non-reference counts

    Notes
    -----
    The function performs the following operations:
    1. Creates output directory if it doesn't exist
    2. Processes the input mpileup file to extract reference and non-reference counts
    3. Writes results to a TSV file with columns: chrom, pos, ref, ref_count, nonref_count
    4. Output filename is derived from input by replacing '.pileup' with '.ref_nonref_counts.tsv'
    """
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
