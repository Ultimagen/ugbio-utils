#!/usr/bin/env python3
"""CLI for batch processing CNV regions with jump alignment.

This script processes multiple CNV regions from a BED file, running jump
alignment on each region and writing results to output files.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyfaidx
import pysam
from ugbio_cnv.jalign import JAlignConfig, process_cnv
from ugbio_core.logger import logger


def create_bam_header(
    reads_file: pysam.AlignmentFile,
) -> pysam.AlignmentHeader:
    """Create BAM header with required read groups for jump alignment.

    Parameters
    ----------
    reads_file : pysam.AlignmentFile
        Input alignment file to copy header from

    Returns
    -------
    pysam.AlignmentHeader
        BAM header with REF1, REF2, DUP, and DEL read groups
    """
    header_dict = reads_file.header.to_dict()

    # Ensure required read groups exist
    if "RG" not in header_dict:
        header_dict["RG"] = []

    existing_rg_ids = {rg["ID"] for rg in header_dict.get("RG", [])}
    for rgid in ["REF1", "REF2", "DUP", "DEL"]:
        if rgid not in existing_rg_ids:
            header_dict["RG"].append(
                {
                    "ID": rgid,
                    "SM": "SAMPLE",
                    "PL": "ULTIMA",
                }
            )

    return pysam.AlignmentHeader.from_dict(header_dict)


def process_single_cnv(
    rec_data: tuple,
    input_cram: str,
    ref_fasta: str,
    config: JAlignConfig,
    temp_dir: Path,
) -> tuple:
    """Process a single CNV region.

    Parameters
    ----------
    rec_data : tuple
        Tuple of (index, chrom, start, end) for the CNV
    input_cram : str
        Path to input CRAM/BAM file
    ref_fasta : str
        Path to reference FASTA file
    config : JAlignConfig
        Configuration for jump alignment
    temp_dir : Path
        Directory for temporary files

    Returns
    -------
    tuple
        (index, chrom, start, end, fwd_better, rev_better, fwd_strong_better,
         rev_strong_better, alignment_results, realigned_reads, cycle_time, success, error_msg)
    """
    idx, chrom, start, end = rec_data

    try:
        # Open files in worker (each process needs its own file handles)
        reads_file = pysam.AlignmentFile(input_cram, "rb", reference_filename=ref_fasta)
        reference = pyfaidx.Fasta(ref_fasta)

        # Create BAM header in worker to avoid pickling issues
        bam_header = create_bam_header(reads_file)

        cycle_start_time = time.time()

        (
            (
                fwd_better,
                rev_better,
                fwd_strong_better,
                rev_strong_better,
            ),
            alignment_results,
            realigned_reads,
            _,
        ) = process_cnv(
            chrom,
            start,
            end,
            reads_file,
            reference,
            config,
            temp_dir,
            None,  # No log file in parallel mode
            bam_header,
        )

        cycle_time = time.time() - cycle_start_time

        # Close files
        reads_file.close()
        reference.close()

        return (
            idx,
            chrom,
            start,
            end,
            fwd_better,
            rev_better,
            fwd_strong_better,
            rev_strong_better,
            alignment_results,
            realigned_reads,
            cycle_time,
            True,
            None,
        )

    except Exception as e:
        return (idx, chrom, start, end, 0, 0, 0, 0, None, [], 0.0, False, str(e))


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
        "cnv_vcf",
        type=str,
        help="VCF file with CNV candidates (chr, start, end)",
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
        default=-12,
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


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915, C901
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
            temp_dir = Path(args.output_prefix).resolve()

        logger.info(f"Using temporary directory: {temp_dir}")

        # Open input files
        logger.info("Opening input files...")
        reads_file = pysam.AlignmentFile(args.input_cram, "rb", reference_filename=args.ref_fasta)
        reference = pyfaidx.Fasta(args.ref_fasta)
        bam_header = create_bam_header(reads_file)
        # Set up output files
        output_vcf = args.output_prefix + ".vcf.gz"
        realigned_bam = args.output_prefix + ".realigned.bam"
        logger.info(f"Processing CNV regions from {args.cnv_vcf}")
        logger.info(f"Writing results to {output_vcf}")

        # Read all VCF records first
        logger.info("Reading CNV records...")
        cnv_records = []
        with pysam.VariantFile(args.cnv_vcf) as f:
            vcf_header = f.header
            vcf_header.info.add(
                "JALIGN_DUP_SUPPORT", 1, "Integer", "Number of reads supporting the duplication via jump alignment"
            )
            vcf_header.info.add(
                "JALIGN_DEL_SUPPORT", 1, "Integer", "Number of reads supporting the deletion via jump alignment"
            )
            vcf_header.info.add(
                "JALIGN_DUP_SUPPORT_STRONG",
                1,
                "Integer",
                "Number of reads strongly supporting the duplication via jump alignment",
            )
            vcf_header.info.add(
                "JALIGN_DEL_SUPPORT_STRONG",
                1,
                "Integer",
                "Number of reads strongly supporting the deletion via jump alignment",
            )

            for idx, rec in enumerate(f):
                # Store record data as tuple for parallel processing
                rec_data = (idx, rec.chrom, rec.start, rec.stop)
                cnv_records.append((rec, rec_data))

        logger.info(f"Processing {len(cnv_records)} CNV regions...")

        # Process CNVs sequentially
        processing_results = []
        for _, rec_data in cnv_records:
            result = process_single_cnv(
                rec_data,
                args.input_cram,
                args.ref_fasta,
                config,
                temp_dir,
            )
            processing_results.append(result)

        # Write results
        logger.info("Writing results...")
        cnv_count = 0
        failed_count = 0
        alignment_results_list = []

        with pysam.VariantFile(output_vcf, "w", header=vcf_header) as out_vcf:
            with pysam.AlignmentFile(realigned_bam, "w", header=bam_header) as realigned_bam_file:
                for (rec, _), result in zip(cnv_records, processing_results, strict=False):
                    (
                        idx,
                        chrom,
                        start,
                        end,
                        fwd_better,
                        rev_better,
                        fwd_strong_better,
                        rev_strong_better,
                        alignment_results,
                        realigned_reads,
                        cycle_time,
                        success,
                        error_msg,
                    ) = result

                    if success:
                        # Update VCF record
                        rec.info["JALIGN_DUP_SUPPORT"] = rev_better
                        rec.info["JALIGN_DEL_SUPPORT"] = fwd_better
                        rec.info["JALIGN_DUP_SUPPORT_STRONG"] = rev_strong_better
                        rec.info["JALIGN_DEL_SUPPORT_STRONG"] = fwd_strong_better

                        # Write realigned reads
                        for rc in realigned_reads:
                            realigned_bam_file.write(rc)

                        out_vcf.write(rec)
                        cnv_count += 1
                        alignment_results_list.append(alignment_results)

                        logger.info(
                            f"{chrom}:{start}-{end} - DUP:{rev_better}/{rev_strong_better} \
                            DEL:{fwd_better}/{fwd_strong_better} - "
                            f"Realigned reads: {len(realigned_reads)} - Time: {cycle_time:.2f}s"
                        )
                    else:
                        failed_count += 1
                        logger.error(f"Error processing {chrom}:{start}-{end}: {error_msg}")

        # Close files
        reads_file.close()
        reference.close()

        # Save alignment results
        if alignment_results_list:
            alignment_results = pd.concat(alignment_results_list)
            alignment_results.to_csv(args.output_prefix + ".csv", index=False)

        logger.info(f"Successfully processed {cnv_count} CNV regions")
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} CNV regions")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
