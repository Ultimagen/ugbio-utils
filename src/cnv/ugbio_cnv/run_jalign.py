#!/usr/bin/env python3
"""CLI for batch processing CNV regions with jump alignment.

This script processes multiple CNV regions from a BED file, running jump
alignment on each region and writing results to output files.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import pyfaidx
import pysam
from ugbio_cnv.jalign import JAlignConfig, process_cnv
from ugbio_core.logger import logger


def get_parser() -> argparse.ArgumentParser:
    """Create and return argument parser for the CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="run_jalign",
        description=(
            "Process CNV regions with jump alignment to identify supporting evidence for duplications and deletions"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "input_cram",
        type=str,
        help="Input CRAM or BAM file with aligned reads",
    )
    parser.add_argument(
        "cnv_bed",
        type=str,
        help="BED file with CNV candidates (chr, start, end)",
    )
    parser.add_argument(
        "ref_fasta",
        type=str,
        help="Reference genome FASTA file",
    )
    parser.add_argument(
        "output_prefix",
        type=str,
        help="Output prefix for .bed and .log files",
    )

    # Configuration parameters
    config_group = parser.add_argument_group("alignment configuration")
    config_group.add_argument(
        "--match-score",
        type=int,
        default=2,
        help="Score for matching bases",
    )
    config_group.add_argument(
        "--mismatch-score",
        type=int,
        default=-8,
        help="Penalty for mismatched bases",
    )
    config_group.add_argument(
        "--gap-open-score",
        type=int,
        default=-18,
        help="Penalty for opening a gap",
    )
    config_group.add_argument(
        "--gap-extend-score",
        type=int,
        default=-1,
        help="Penalty for extending a gap",
    )
    config_group.add_argument(
        "--jump-score",
        type=int,
        default=0,
        help="Score for jump operation",
    )
    config_group.add_argument(
        "--min-mismatches",
        type=int,
        default=1,
        help="Minimum mismatches required to accept a read",
    )
    config_group.add_argument(
        "--softclip-threshold",
        type=int,
        default=10,
        help="Minimum soft-clip length to consider substantial",
    )
    config_group.add_argument(
        "--fetch-read-padding",
        type=int,
        default=500,
        help="Padding around breakpoint when fetching reads (bp)",
    )
    config_group.add_argument(
        "--fetch-ref-padding",
        type=int,
        default=0,
        help="Padding when fetching reference sequences (bp)",
    )
    config_group.add_argument(
        "--min-seq-len-jump-align-component",
        type=int,
        default=30,
        help="Minimum length for each jump alignment component",
    )
    config_group.add_argument(
        "--max-reads-per-cnv",
        type=int,
        default=4000,
        help="Maximum reads to process per CNV (triggers subsampling)",
    )
    config_group.add_argument(
        "--max-score-fraction",
        type=float,
        default=0.9,
        help="Fraction of theoretical maximal score (all matches) that the jump alignment needs "
        "to achieve to be considered support",
    )
    config_group.add_argument(
        "--stringent-max-score-fraction",
        type=float,
        default=0.95,
        help="Fraction of theoretical maximal score (all matches) that the jump alignment needs "
        "to achieve to be considered strong support",
    )
    config_group.add_argument(
        "--tool-path",
        type=str,
        default=os.environ.get("TOOL", "jump_align"),
        help="Path or name of jump alignment tool executable",
    )
    config_group.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for reproducible subsampling",
    )

    # Runtime options
    runtime_group = parser.add_argument_group("runtime options")
    runtime_group.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory for temporary files (default: system temp)",
    )
    runtime_group.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level",
    )

    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input files and parameters.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Raises
    ------
    FileNotFoundError
        If required input files do not exist
    ValueError
        If parameters are invalid
    """
    # Check input files exist
    for filepath, name in [
        (args.input_cram, "Input CRAM/BAM"),
        (args.cnv_bed, "CNV BED"),
        (args.ref_fasta, "Reference FASTA"),
    ]:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"{name} file not found: {filepath}")

    # Validate output directory exists
    output_dir = Path(args.output_prefix).parent
    if output_dir != Path("") and not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    """Main entry point for the CLI.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments (default: sys.argv)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors)
    """
    parser = get_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logger.setLevel(getattr(logging, args.verbosity))

    try:
        # Validate inputs
        validate_inputs(args)

        # Create configuration
        config = JAlignConfig(
            match_score=args.match_score,
            mismatch_score=args.mismatch_score,
            gap_open_score=args.gap_open_score,
            gap_extend_score=args.gap_extend_score,
            jump_score=args.jump_score,
            min_mismatches=args.min_mismatches,
            softclip_threshold=args.softclip_threshold,
            fetch_read_padding=args.fetch_read_padding,
            fetch_ref_padding=args.fetch_ref_padding,
            min_seq_len_jump_align_component=(args.min_seq_len_jump_align_component),
            max_reads_per_cnv=args.max_reads_per_cnv,
            max_score_fraction=args.max_score_fraction,
            stringent_max_score_fraction=args.stringent_max_score_fraction,
            tool_path=args.tool_path,
            random_seed=args.random_seed,
        )

        # Check for local jump_align tool (for backward compatibility)
        local_tool_path = Path("jump_align") / config.tool_path
        if local_tool_path.exists():
            logger.info(f"Using local jump alignment tool: {local_tool_path}")
            config.tool_path = str(local_tool_path)
            # Rebuild command template with updated path
            config.__post_init__()

        # Set up temporary directory
        if args.temp_dir:
            temp_dir = Path(args.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.gettempdir())

        logger.info(f"Using temporary directory: {temp_dir}")

        # Open input files
        logger.info("Opening input files...")
        reads_file = pysam.AlignmentFile(args.input_cram, "rb", reference_filename=args.ref_fasta)
        reference = pyfaidx.Fasta(args.ref_fasta)

        # Set up output files
        output_bed = args.output_prefix + ".bed"
        output_log = args.output_prefix + ".log"

        logger.info(f"Processing CNV regions from {args.cnv_bed}")
        logger.info(f"Writing results to {output_bed}")

        # Process each CNV region
        cnv_count = 0
        with open(output_bed, "w") as out_bed, open(output_log, "w") as flog:
            with open(args.cnv_bed) as f:
                for line in f:
                    # Skip comments
                    if line.startswith("#"):
                        out_bed.write(line)
                        continue

                    # Parse BED line
                    bed_line = line.strip().split()
                    bed_chrom, bed_start, bed_end = bed_line[:3]
                    bed_start = int(bed_start)
                    bed_end = int(bed_end)

                    # Process this CNV region
                    try:
                        (
                            (
                                jump_better,
                                djump_better,
                                jump_much_better,
                                djump_much_better,
                            ),
                            _,
                        ) = process_cnv(
                            bed_chrom,
                            bed_start,
                            bed_end,
                            reads_file,
                            reference,
                            config,
                            temp_dir,
                            flog,
                        )

                        # Write output line
                        outline = (
                            f"{line.rstrip()}\t{jump_better}\t{djump_better}\t{jump_much_better}\t{djump_much_better}\n"
                        )
                        out_bed.write(outline)
                        logger.info(outline.rstrip())

                        cnv_count += 1

                    except Exception as e:
                        logger.error(
                            f"Error processing {bed_chrom}:{bed_start}-{bed_end}: {e}",
                            exc_info=True,
                        )
                        # Write zeros for failed regions
                        outline = f"{line.rstrip()}\t0\t0\t0\t0\n"
                        out_bed.write(outline)

        # Close files
        reads_file.close()
        reference.close()

        logger.info(f"Successfully processed {cnv_count} CNV regions")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
