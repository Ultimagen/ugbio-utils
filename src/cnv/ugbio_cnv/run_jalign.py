#!/usr/bin/env python3
"""CLI for batch processing CNV regions with jump alignment.

This script annotates CNV VCF file with the counts of jump alignments around the breakpoints.
"""

import argparse
import logging
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
import pyfaidx
import pysam
from ugbio_cnv import jalign as jalign_module
from ugbio_cnv.jalign import JAlignConfig, create_bam_header, process_cnv
from ugbio_core.logger import logger


# Worker-process-local file handles (opened once per worker via Pool initializer)
_worker_reads_file: pysam.AlignmentFile | None = None
_worker_reference: pyfaidx.Fasta | None = None
_worker_bam_header: pysam.AlignmentHeader | None = None
_worker_open_counts: dict[str, int] = {}


def _worker_init(
    input_cram: str,
    ref_fasta: str,
    verbosity: str,
) -> None:
    """Pool initializer: open CRAM and FASTA once per worker process."""
    global _worker_reads_file, _worker_reference, _worker_bam_header, _worker_open_counts  # noqa: PLW0603
    logger.setLevel(getattr(logging, verbosity, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    _worker_reads_file = pysam.AlignmentFile(input_cram, "rb", reference_filename=ref_fasta)
    _worker_reference = pyfaidx.Fasta(ref_fasta, rebuild=False)
    _worker_bam_header = create_bam_header(_worker_reads_file.header)
    _worker_open_counts = {
        "cram_opens": 1,
        "fasta_opens": 1,
        "bam_write_opens": 0,
        "bam_read_opens": 0,
    }


def _chunk_records(records: list[tuple], chunk_size: int) -> list[list[tuple]]:
    """Split records into fixed-size groups."""
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def _create_bam_records_from_df(
    alignment_df: pd.DataFrame,
    reads_in_order: list[pysam.AlignedSegment],
    chrom: str,
    ref1_start: int,
    ref2_start: int,
    header: pysam.AlignmentHeader,
    config: JAlignConfig,
) -> list[pysam.AlignedSegment]:
    """Create BAM records directly from an alignment DataFrame for a single CNV."""
    best_alignments = jalign_module.determine_best_alignments(alignment_df, config)
    read_map = {read.query_name: read for read in reads_in_order}
    alignment_types = ["align1", "align2", "jump_forward", "jump_backward"]
    bam_records = []

    for _, row in alignment_df.iterrows():
        qname = row.get("qname")
        if not qname or qname not in read_map:
            continue

        read = read_map[qname]
        seq = read.query_sequence

        if config.output_all_alignments:
            for idx, alignment_type in enumerate(alignment_types, start=1):
                score_field = f"{alignment_type}.score"
                if pd.notna(row.get(score_field)) and int(row.get(score_field, 0)) > 0:
                    records = jalign_module._create_bam_records_for_alignment_type(
                        alignment_type=alignment_type,
                        row=row,
                        qname=f"{qname}/{idx}",
                        seq=seq,
                        chrom=chrom,
                        ref1_start=ref1_start,
                        ref2_start=ref2_start,
                        header=header,
                    )
                    bam_records.extend(records)
        else:
            best_alignment_type = best_alignments.get(qname)
            if not best_alignment_type:
                continue

            records = jalign_module._create_bam_records_for_alignment_type(
                alignment_type=best_alignment_type,
                row=row,
                qname=qname,
                seq=seq,
                chrom=chrom,
                ref1_start=ref1_start,
                ref2_start=ref2_start,
                header=header,
            )
            bam_records.extend(records)

    return bam_records


def process_cnv_group(
    rec_group_data: list[tuple[int, str, int, int]],
    input_cram: str,
    ref_fasta: str,
    config: JAlignConfig,
    temp_dir: Path,
    verbosity: str = "INFO",
    *,
    use_worker_handles: bool = False,
) -> tuple[list[tuple], float, float, float, dict[str, int]]:
    """Process a group of CNVs with one C++ tool invocation.

    Returns
    -------
    tuple
        - Per-CNV results list with one entry per input record
        - Group total time
        - Group C++ tool time
        - Group Python time
        - Open counters for this group
    """
    default_result = lambda rec, err=None: (
        rec[0],
        rec[1],
        rec[2],
        rec[3],
        0,
        0,
        0,
        0,
        pd.DataFrame(),
        None,
        0.0,
        0.0,
        0.0,
        err is None,
        err,
    )

    try:
        logger.setLevel(getattr(logging, verbosity, logging.INFO))
        if not logger.handlers:
            handler = logging.StreamHandler(stream=sys.stderr)
            formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        open_counts: dict[str, int]
        if use_worker_handles:
            reads_file = _worker_reads_file
            reference = _worker_reference
            bam_header = _worker_bam_header
            open_counts = {"cram_opens": 0, "fasta_opens": 0, "bam_write_opens": 0, "bam_read_opens": 0}
        else:
            reads_file = pysam.AlignmentFile(input_cram, "rb", reference_filename=ref_fasta)
            reference = pyfaidx.Fasta(ref_fasta, rebuild=False)
            bam_header = create_bam_header(reads_file.header)
            open_counts = {"cram_opens": 1, "fasta_opens": 1, "bam_write_opens": 0, "bam_read_opens": 0}

        cycle_start_time = time.time()

        cnv_meta = []
        input_file = temp_dir / (
            f"jalign_group_{os.getpid()}_{rec_group_data[0][0]}_{rec_group_data[-1][0]}.txt"
        )
        output_file = temp_dir / (
            f"jalign_group_{os.getpid()}_{rec_group_data[0][0]}_{rec_group_data[-1][0]}_output.json"
        )

        with open(input_file, "w") as f:
            for rec in rec_group_data:
                idx, chrom, start, end = rec
                reads, refs_extents = jalign_module._fetch_reads_at_breakpoints(chrom, start, end, reads_file, config)
                if not reads:
                    cnv_meta.append(
                        {
                            "rec": rec,
                            "reads_in_order": [],
                            "ref1_start": 0,
                            "ref2_start": 0,
                            "row_count": 0,
                        }
                    )
                    continue

                refs = jalign_module._extract_references(chrom, refs_extents, reference)

                subsample_ratio = 1.0
                if len(reads) > config.max_reads_per_cnv:
                    subsample_ratio = config.max_reads_per_cnv / len(reads)

                reads_in_order = []
                ref_emitted = False
                for read in reads.values():
                    if random.random() > subsample_ratio:  # noqa: S311
                        continue

                    reads_in_order.append(read)
                    if not ref_emitted:
                        line = f"{read.query_name}\t{read.query_sequence}\t{refs[0][1]}\t{refs[1][1]}\n"
                        ref_emitted = True
                    else:
                        line = f"{read.query_name}\t{read.query_sequence}\t=\n"
                    f.write(line)

                cnv_meta.append(
                    {
                        "rec": rec,
                        "reads_in_order": reads_in_order,
                        "ref1_start": refs[0][0],
                        "ref2_start": refs[1][0],
                        "row_count": len(reads_in_order),
                    }
                )

        cpp_time = 0.0
        grouped_df = pd.DataFrame()
        if any(meta["row_count"] > 0 for meta in cnv_meta):
            cpp_start_time = time.time()
            alignment_cmd = config.build_alignment_command(input_file, output_file)
            prev_no_cigars = os.environ.get("NO_CIGARS")
            os.environ["NO_CIGARS"] = "1"
            try:
                jalign_module.run_alignment_tool(alignment_cmd, None)
            finally:
                if prev_no_cigars is None:
                    os.environ.pop("NO_CIGARS", None)
                else:
                    os.environ["NO_CIGARS"] = prev_no_cigars
            cpp_time = time.time() - cpp_start_time
            grouped_df = jalign_module._parse_alignment_results(output_file)

        result_rows = []
        cursor = 0

        for meta in cnv_meta:
            rec = meta["rec"]
            idx, chrom, start, end = rec
            row_count = meta["row_count"]

            if row_count == 0:
                result_rows.append(default_result(rec))
                continue

            cnv_df = grouped_df.iloc[cursor : cursor + row_count].copy()
            cursor += row_count

            if "qname" not in cnv_df.columns:
                result_rows.append(default_result(rec, "Missing qname column in grouped alignment output"))
                continue

            counts = jalign_module._count_supporting_alignments(cnv_df, config)
            bam_records = _create_bam_records_from_df(
                alignment_df=cnv_df,
                reads_in_order=meta["reads_in_order"],
                chrom=chrom,
                ref1_start=meta["ref1_start"],
                ref2_start=meta["ref2_start"],
                header=bam_header,
                config=config,
            )

            temp_bam_file = None
            if bam_records:
                temp_bam_file = temp_dir / f"jalign_realigned_{chrom}_{start}_{end}_{os.getpid()}_{idx}.bam"
                with pysam.AlignmentFile(str(temp_bam_file), "wb", header=bam_header) as temp_bam:
                    open_counts["bam_write_opens"] += 1
                    for read in bam_records:
                        temp_bam.write(read)

            cnv_df["chrom"] = chrom
            cnv_df["start"] = start
            cnv_df["end"] = end

            result_rows.append(
                (
                    idx,
                    chrom,
                    start,
                    end,
                    counts[0],
                    counts[1],
                    counts[2],
                    counts[3],
                    cnv_df,
                    temp_bam_file,
                    0.0,
                    0.0,
                    0.0,
                    True,
                    None,
                )
            )

        cycle_time = time.time() - cycle_start_time
        python_time = cycle_time - cpp_time

        # Distribute group timings uniformly across successful CNVs for per-CNV logs.
        successful_results = [r for r in result_rows if r[13]]
        if successful_results:
            per_cnv_cycle = cycle_time / len(successful_results)
            per_cnv_cpp = cpp_time / len(successful_results)
            per_cnv_python = python_time / len(successful_results)
            updated_results = []
            for r in result_rows:
                if r[13]:
                    updated_results.append((*r[:10], per_cnv_cycle, per_cnv_cpp, per_cnv_python, r[13], r[14]))
                else:
                    updated_results.append(r)
            result_rows = updated_results

        if not use_worker_handles:
            reads_file.close()
            reference.close()

        if input_file.exists():
            input_file.unlink()
        if output_file.exists():
            output_file.unlink()

        return result_rows, cycle_time, cpp_time, python_time, open_counts
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed group: {e}")
        return [default_result(rec, str(e)) for rec in rec_group_data], 0.0, 0.0, 0.0, {
            "cram_opens": 0,
            "fasta_opens": 0,
            "bam_write_opens": 0,
            "bam_read_opens": 0,
        }


def process_single_cnv(
    rec_data: tuple,
    input_cram: str,
    ref_fasta: str,
    config: JAlignConfig,
    temp_dir: Path,
) -> tuple:
    """Process a single CNV region with jump alignment.

    This function is designed to be called in parallel by multiprocessing workers.
    Each worker opens its own file handles and processes one CNV region independently.


    Parameters
    ----------
    rec_data : tuple of (int, str, int, int)
        CNV record data containing (index, chromosome, start, end)
    input_cram : str
        Path to input CRAM or BAM file with aligned reads
    ref_fasta : str
        Path to reference genome FASTA file (must have .fai index)
    config : JAlignConfig
        Configuration object containing alignment parameters and thresholds
    temp_dir : Path
        Directory path for writing temporary BAM files

    Returns
    -------
    tuple of (int, str, int, int, int, int, int, int, pd.DataFrame or None, Path or None, float, float, float, bool, str or None)
        Results tuple containing:
        - idx : int - CNV record index
        - chrom : str - Chromosome name
        - start : int - CNV start position
        - end : int - CNV end position
        - fwd_better : int - Number of reads supporting deletion
        - rev_better : int - Number of reads supporting duplication
        - fwd_strong_better : int - Number of reads strongly supporting deletion
        - rev_strong_better : int - Number of reads strongly supporting duplication
        - alignment_results : pd.DataFrame or None - Detailed alignment statistics
        - temp_bam_file : Path or None - Path to temporary BAM file with realigned reads
        - cycle_time : float - Total processing time in seconds
        - cpp_time : float - Time spent in C++ jalign tool
        - python_time : float - Time spent in Python code around C++ call
        - success : bool - Whether processing succeeded
        - error_msg : str or None - Error message if processing failed, None otherwise

    Notes
    -----
    - Each parallel worker creates its own pysam.AlignmentFile and pyfaidx.Fasta handles
    - Temporary BAM files are named with PID and index to avoid collisions. This is preferred
      instead of pysam.AlignedSegment due to pickling issues with multiprocessing.
    - File handles are explicitly closed after processing to prevent resource leaks
    """
    idx, chrom, start, end = rec_data

    try:
        # Open files in worker (each process needs its own file handles)
        reads_file = pysam.AlignmentFile(input_cram, "rb", reference_filename=ref_fasta)
        reference = pyfaidx.Fasta(ref_fasta, rebuild=False)

        # Create BAM header in worker to avoid pickling issues
        bam_header = create_bam_header(reads_file.header)

        cycle_start_time = time.time()

        # Measure time spent in C++ jalign tool
        cpp_start_time = time.time()
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
        cpp_time = time.time() - cpp_start_time

        # Write realigned reads to temporary BAM file
        temp_bam_file = None
        if realigned_reads:
            temp_bam_file = temp_dir / f"jalign_realigned_{chrom}_{start}_{end}_{os.getpid()}_{idx}.bam"
            with pysam.AlignmentFile(str(temp_bam_file), "wb", header=bam_header) as temp_bam:
                for read in realigned_reads:
                    temp_bam.write(read)

        cycle_time = time.time() - cycle_start_time
        python_time = cycle_time - cpp_time

        logger.info(
            f"{chrom}:{start}-{end} timing - C++ tool: {cpp_time:.3f}s, Python overhead: {python_time:.3f}s, Total: {cycle_time:.3f}s"
        )

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
            temp_bam_file,
            cycle_time,
            cpp_time,
            python_time,
            True,
            None,
        )

    except Exception as e:
        return (idx, chrom, start, end, 0, 0, 0, 0, None, None, 0.0, 0.0, 0.0, False, str(e))


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
        help="Reference genome FASTA file (need to have FAI index)",
    )
    parser.add_argument(
        "output_prefix",
        type=str,
        help="Output prefix for the output files",
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
        default=os.environ.get("TOOL", "para_jalign"),
        help="Path or name of para_jalign tool executable",
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
        "--threads",
        type=int,
        default=1,
        help="Number of parallel threads for processing CNVs (default: 1 for sequential processing)",
    )
    runtime_group.add_argument(
        "--cnvs-per-invocation",
        type=int,
        default=8,
        help="Number of CNVs to group into a single C++ tool invocation",
    )
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


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915, C901, PLR0912
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
    main_start_time = time.perf_counter()
    parser = get_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logger.setLevel(getattr(logging, args.verbosity))
    # Ensure handler exists for multiprocessing compatibility
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

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

        # Resolve and log the actual path of the alignment tool
        tool_real_path = shutil.which(config.tool_path)
        if tool_real_path:
            logger.info(f"Using alignment tool: {tool_real_path}")
        else:
            logger.error(f"Alignment tool '{config.tool_path}' not found, please modify PATH or provide full path")
            raise RuntimeError(f"Alignment tool '{config.tool_path}' not found")

        # Set up temporary directory
        if args.temp_dir:
            temp_dir = Path(args.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(args.output_prefix).parent

        logger.info(f"Using temporary directory: {temp_dir}")

        # Open input files
        logger.info("Opening input files...")
        reads_file = pysam.AlignmentFile(args.input_cram, "rb", reference_filename=args.ref_fasta)
        reference = pyfaidx.Fasta(args.ref_fasta, rebuild=False)
        bam_header = create_bam_header(reads_file.header)
        # Set up output files
        output_vcf = args.output_prefix + ".jalign.vcf.gz"
        realigned_bam = args.output_prefix + ".jalign.bam"
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
        logger.info(f"Grouping {args.cnvs_per_invocation} CNVs per C++ invocation")

        if args.cnvs_per_invocation < 1:
            raise ValueError("--cnvs-per-invocation must be >= 1")

        rec_groups = _chunk_records([rec_data for _, rec_data in cnv_records], args.cnvs_per_invocation)
        logger.info(f"Created {len(rec_groups)} CNV groups for processing")

        # Process CNVs in parallel or sequentially
        processing_start_time = time.perf_counter()
        if args.threads > 1:
            logger.info(f"Using {args.threads} parallel threads")
            # Create partial function with fixed arguments
            worker_func = partial(
                process_cnv_group,
                input_cram=args.input_cram,
                ref_fasta=args.ref_fasta,
                config=config,
                temp_dir=temp_dir,
                verbosity=args.verbosity,
                use_worker_handles=True,
            )

            # Process in parallel using multiprocessing
            # _worker_init opens CRAM/FASTA once per worker process
            # chunksize=1 ensures dynamic dispatch: workers pull the next group only
            # when they are idle, preventing fast workers from sitting unused while
            # slow workers work through a pre-assigned batch.
            with mp.Pool(
                processes=args.threads,
                initializer=_worker_init,
                initargs=(args.input_cram, args.ref_fasta, args.verbosity),
            ) as pool:
                group_processing_results = list(pool.imap_unordered(worker_func, rec_groups, chunksize=1))
        else:
            logger.info("Processing sequentially (single thread)")
            # Process CNVs sequentially
            group_processing_results = []
            for rec_group in rec_groups:
                result = process_cnv_group(
                    rec_group,
                    args.input_cram,
                    args.ref_fasta,
                    config,
                    temp_dir,
                    args.verbosity,
                    use_worker_handles=False,
                )
                group_processing_results.append(result)
        processing_wall_time = time.perf_counter() - processing_start_time

        # Write results
        logger.info("Writing results...")
        cnv_count = 0
        failed_count = 0
        alignment_results_list = []
        temp_bam_files = []
        total_cpp_time = 0.0
        total_python_time = 0.0
        total_cycle_time = 0.0
        # Main process CRAM/FASTA opens: 1 for header read; workers account for theirs separately
        # In threaded mode workers open via initializer (1 per worker), not reported here per-group
        total_cram_opens = 1 + (args.threads if args.threads > 1 else 0)
        total_fasta_opens = 1 + (args.threads if args.threads > 1 else 0)
        total_bam_write_opens = 0
        total_bam_read_opens = 0

        processing_results_by_idx = {}
        for group_results, group_cycle_time, group_cpp_time, group_python_time, group_open_counts in group_processing_results:
            total_cycle_time += group_cycle_time
            total_cpp_time += group_cpp_time
            total_python_time += group_python_time
            total_cram_opens += group_open_counts.get("cram_opens", 0)
            total_fasta_opens += group_open_counts.get("fasta_opens", 0)
            total_bam_write_opens += group_open_counts.get("bam_write_opens", 0)
            total_bam_read_opens += group_open_counts.get("bam_read_opens", 0)
            for result in group_results:
                processing_results_by_idx[result[0]] = result

        with pysam.VariantFile(output_vcf, "w", header=vcf_header) as out_vcf:
            with pysam.AlignmentFile(realigned_bam, "wb", header=bam_header) as realigned_bam_file:
                total_bam_write_opens += 1
                for rec, rec_data in cnv_records:
                    result = processing_results_by_idx.get(rec_data[0])
                    if result is None:
                        failed_count += 1
                        logger.error(
                            f"Error processing {rec_data[1]}:{rec_data[2]}-{rec_data[3]}: missing processing result"
                        )
                        continue

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
                        temp_bam_file,
                        cycle_time,
                        cpp_time,
                        python_time,
                        success,
                        error_msg,
                    ) = result

                    if success:
                        # Update VCF record
                        rec.info["JALIGN_DUP_SUPPORT"] = rev_better
                        rec.info["JALIGN_DEL_SUPPORT"] = fwd_better
                        rec.info["JALIGN_DUP_SUPPORT_STRONG"] = rev_strong_better
                        rec.info["JALIGN_DEL_SUPPORT_STRONG"] = fwd_strong_better

                        # Read and write realigned reads from temporary BAM file
                        read_count = 0
                        if temp_bam_file and temp_bam_file.exists():
                            with pysam.AlignmentFile(temp_bam_file, "rb") as temp_bam:
                                total_bam_read_opens += 1
                                for read in temp_bam:
                                    realigned_bam_file.write(read)
                                    read_count += 1
                            temp_bam_files.append(temp_bam_file)

                        out_vcf.write(rec)
                        cnv_count += 1
                        alignment_results_list.append(alignment_results)

                        logger.info(
                            f"{chrom}:{start}-{end} - DUP:{rev_better}/{rev_strong_better} "
                            f"DEL:{fwd_better}/{fwd_strong_better} - "
                            f"Realigned reads: {read_count} - Time: {cycle_time:.2f}s"
                        )
                    else:
                        failed_count += 1
                        logger.error(f"Error processing {chrom}:{start}-{end}: {error_msg}")

        # Close files
        reads_file.close()
        reference.close()

        # Clean up temporary BAM files
        for temp_bam_file in temp_bam_files:
            if temp_bam_file and temp_bam_file.exists():
                temp_bam_file.unlink()

        logger.info(f"Cleaned up {len(temp_bam_files)} temporary BAM files")

        # Save alignment results
        if alignment_results_list:
            alignment_results = pd.concat(alignment_results_list)
            alignment_results.to_csv(args.output_prefix + ".jalign.csv", index=False)

        logger.info(f"Successfully processed {cnv_count} CNV regions")
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} CNV regions")

        main_wall_time = time.perf_counter() - main_start_time
        non_cnv_overhead = main_wall_time - total_cycle_time
        cnv_vs_processing_delta = total_cycle_time - processing_wall_time

        logger.info(
            "JALIGN_TIMING_SUMMARY | Total timing across run - "
            f"C++ tool: {total_cpp_time:.3f}s, "
            f"Python overhead: {total_python_time:.3f}s, "
            f"Total: {total_cycle_time:.3f}s"
        )
        logger.info(
            "JALIGN_TIMING_SUMMARY | Wall-clock reconciliation - "
            f"Main wall time: {main_wall_time:.3f}s, "
            f"CNV processing wall time: {processing_wall_time:.3f}s, "
            f"Summed per-CNV time: {total_cycle_time:.3f}s"
        )
        if args.threads > 1:
            logger.info(
                "Multithread timing note - "
                f"Summed per-CNV minus CNV wall time: {cnv_vs_processing_delta:.3f}s "
                "(positive is expected with parallel workers)"
            )
        logger.info(
            "JALIGN_TIMING_SUMMARY | Non-CNV overhead estimate - "
            f"Main wall time minus summed per-CNV time: {non_cnv_overhead:.3f}s"
        )
        logger.info(
            "JALIGN_OPEN_SUMMARY | "
            f"CRAM opens: {total_cram_opens}, "
            f"FASTA opens: {total_fasta_opens}, "
            f"BAM write opens: {total_bam_write_opens}, "
            f"BAM read opens: {total_bam_read_opens}, "
            f"BAM total opens: {total_bam_write_opens + total_bam_read_opens}"
        )
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
