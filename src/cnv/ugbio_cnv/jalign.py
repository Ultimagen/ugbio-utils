#!/usr/bin/env python3
"""Jump alignment for CNV breakpoint analysis.

This module performs jump alignment on reads at CNV breakpoints to identify
supporting evidence for duplications using an external C alignment tool.
The core functionality operates on individual CNV regions and can be used
programmatically or through the run_jalign.py CLI script.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import pysam
from ugbio_core.logger import logger

# CIGAR operation codes
CIGAR_SOFT_CLIP = 4

# Alignment parsing constants
MIN_ALIGNMENT_FIELDS = 9  # Minimum fields expected in alignment output


@dataclass
class JAlignConfig:
    """Configuration for jump alignment algorithm.

    Attributes
    ----------
    match_score : int
        Score for matching bases in alignment
    mismatch_score : int
        Penalty for mismatched bases (negative value)
    gap_open_score : int
        Penalty for opening a gap (negative value)
    gap_extend_score : int
        Penalty for extending a gap (negative value)
    jump_score : int
        Score for jump operation in alignment
    min_mismatches : int
        Minimum number of mismatches required to accept a read
    softclip_threshold : int
        Minimum soft-clip length to consider substantial
    fetch_read_padding : int
        Padding around breakpoint when fetching reads (bp)
    fetch_ref_padding : int
        Padding when fetching reference sequences (bp)
    min_seq_len_jump_align_component : int
        Minimum length for each component of jump alignment
    min_gap_len : int
        Minimum gap length to consider
    max_reads_per_cnv : int
        Maximum reads to process per CNV (subsampling threshold)
    max_score_fraction : float
        Fraction of max possible score for minimal threshold
    stringent_max_score_fraction : float
        Fraction of max possible score for stringent threshold
    tool_path : str
        Path or name of external alignment tool executable
    random_seed : int
        Random seed for reproducible subsampling
    """

    match_score: int = 2
    mismatch_score: int = -8
    gap_open_score: int = -18
    gap_extend_score: int = -1
    jump_score: int = 0
    min_mismatches: int = 5
    softclip_threshold: int = 30
    fetch_read_padding: int = 500
    fetch_ref_padding: int = 0
    min_seq_len_jump_align_component: int = 30
    min_gap_len: int = 30
    max_reads_per_cnv: int = 4000
    max_score_fraction: float = 0.9
    stringent_max_score_fraction: float = 0.95
    tool_path: str = "jump_align"
    random_seed: int = 0
    _alignment_cmd_template: list[str] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize random seed and build alignment command template."""
        random.seed(self.random_seed)
        self._alignment_cmd_template = [
            self.tool_path,
            str(self.match_score),
            str(self.mismatch_score),
            str(self.gap_open_score),
            str(self.gap_extend_score),
            "1",
            str(self.jump_score),
            "",  # Placeholder for input file path
        ]

    def build_alignment_command(self, input_file: Path) -> list[str]:
        """Build alignment command with specific input file.

        Parameters
        ----------
        input_file : Path
            Path to input file for alignment tool

        Returns
        -------
        list[str]
            Command with arguments ready for execution
        """
        cmd = self._alignment_cmd_template.copy()
        cmd[-1] = str(input_file)
        return cmd


def count_md_mismatches(read: pysam.AlignedSegment) -> int | None:
    """Count the number of mismatches using the MD tag.

    Indels are not counted, only substitutions.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Aligned read from BAM/CRAM file

    Returns
    -------
    int or None
        Number of mismatches found, or None if MD tag is missing
    """
    try:
        md_tag = read.get_tag("MD")
    except KeyError:
        return None

    # Find all letters in the MD string, which represent mismatches
    mismatches = re.findall(r"[A-Z]", str(md_tag))
    return len(mismatches)


def count_nm_mismatches(read: pysam.AlignedSegment) -> int | None:
    """Count the number of mismatches using the NM tag.

    Adjusts for indels to count only substitutions.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Aligned read from BAM/CRAM file

    Returns
    -------
    int or None
        Number of mismatches found, or None if NM tag is missing
    """
    try:
        nm_tag = int(read.get_tag("NM"))
        # Subtract indel lengths (CIGAR operations 1-3, excluding 4 which is soft clip)
        if read.cigartuples:
            non_mismatches = [x for x in read.cigartuples if 0 < x[0] < CIGAR_SOFT_CLIP]
            nm_tag -= sum(x[1] for x in non_mismatches if x[0] != CIGAR_SOFT_CLIP)
    except KeyError:
        return None

    return nm_tag


def count_softclip_mismatches(read: pysam.AlignedSegment, reference: pysam.FastaFile) -> int:
    """Count mismatches in soft-clipped regions of a read.

    Examines both left and right soft-clipped portions.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Aligned read from BAM/CRAM file
    reference : pysam.FastaFile
        Reference genome FASTA file

    Returns
    -------
    int
        Number of mismatches in soft-clipped regions
    """
    if read.is_unmapped:
        return 0

    seq = read.query_sequence
    if not seq:
        return 0

    mismatches = 0
    ref_name = read.reference_name
    start = read.reference_start
    end = read.reference_end

    cigartuples = read.cigartuples
    if not cigartuples or not start or not end:
        return 0

    soft_clip = CIGAR_SOFT_CLIP

    # Left soft clip
    if cigartuples[0][0] == soft_clip:
        clip_len = cigartuples[0][1]
        clipped_bases = seq[:clip_len]
        ref_start = max(0, start - clip_len)
        ref_bases = reference.fetch(ref_name, ref_start, start)
        for rb, qb in zip(ref_bases, clipped_bases, strict=False):
            if rb.upper() != qb.upper():
                mismatches += 1

    # Right soft clip
    if cigartuples[-1][0] == soft_clip:
        clip_len = cigartuples[-1][1]
        clipped_bases = seq[-clip_len:]
        ref_bases = reference.fetch(ref_name, end, end + clip_len)
        for rb, qb in zip(ref_bases, clipped_bases, strict=False):
            if rb.upper() != qb.upper():
                mismatches += 1

    return mismatches


def is_softclipped(read: pysam.AlignedSegment) -> bool:
    """Check if read has any soft clipping.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Aligned read from BAM/CRAM file

    Returns
    -------
    bool
        True if read has soft clipping on either end
    """
    if not read.cigartuples:
        return False
    return read.cigartuples[0][0] == pysam.CSOFT_CLIP or read.cigartuples[-1][0] == pysam.CSOFT_CLIP


def is_substantial_softclipped(read: pysam.AlignedSegment, threshold: int) -> bool:
    """Check if read has substantial soft clipping.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Aligned read from BAM/CRAM file
    threshold : int
        Minimum soft-clip length to consider substantial

    Returns
    -------
    bool
        True if soft clip length meets or exceeds threshold
    """
    if not read.cigartuples:
        return False
    return (read.cigartuples[0][0] == pysam.CSOFT_CLIP and read.cigartuples[0][1] >= threshold) or (
        read.cigartuples[-1][0] == pysam.CSOFT_CLIP and read.cigartuples[-1][1] >= threshold
    )


def accept_read(read: pysam.AlignedSegment, min_mismatches: int) -> bool:
    """Determine if read should be accepted for jump alignment.

    Filters out duplicates and reads with insufficient mismatches.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Aligned read from BAM/CRAM file
    min_mismatches : int
        Minimum number of mismatches required

    Returns
    -------
    bool
        True if read should be accepted for analysis
    """
    if read.is_duplicate:
        return False
    if min_mismatches <= 0:
        return True
    nm = count_nm_mismatches(read)
    if nm is None:
        nm = 0
    return nm >= min_mismatches


def run_alignment_tool(command: list[str], log_file: TextIO | None = None) -> tuple[str, int]:
    """Execute external alignment tool and capture output.

    Parameters
    ----------
    command : list[str]
        Command and arguments to execute
    log_file : file-like object, optional
        Log file for recording command execution

    Returns
    -------
    tuple[str, int]
        Tuple of (stdout, return_code)

    Raises
    ------
    RuntimeError
        If command execution fails
    """
    logger.info(f"Executing: {' '.join(command)}")
    if log_file:
        log_file.write(f"<<< {command}\n")

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        return process.stdout, process.returncode
    except subprocess.CalledProcessError as e:
        error_msg = f"Alignment tool failed with return code {e.returncode}"
        logger.error(f"{error_msg}: {e.stderr}")
        raise RuntimeError(error_msg) from e


def _fetch_reads_at_breakpoints(
    chrom: str,
    start: int,
    end: int,
    reads_file: pysam.AlignmentFile,
    config: JAlignConfig,
) -> tuple[dict[str, pysam.AlignedSegment], list[set[str]], list[list[int]]]:
    """Fetch reads crossing CNV breakpoints.

    Parameters
    ----------
    chrom : str
        Chromosome name
    start : int
        CNV start position
    end : int
        CNV end position
    reads_file : pysam.AlignmentFile
        Opened alignment file (BAM/CRAM)
    config : JAlignConfig
        Configuration parameters

    Returns
    -------
    tuple
        - dict mapping read names to AlignedSegment objects
        - list of two sets containing read names at each breakpoint
        - list of two lists with [min_pos, max_pos] for each breakpoint
    """
    reads = {}
    reads_in_ref = [set(), set()]
    refs_extents = []

    for ref_id, loc in enumerate([start, end]):
        rmin = max(0, loc - config.fetch_read_padding)
        rmax = loc + config.fetch_read_padding

        for read in reads_file.fetch(
            chrom,
            max(0, loc - config.fetch_read_padding),
            loc + config.fetch_read_padding,
        ):
            if read.is_duplicate:
                continue
            if not is_substantial_softclipped(read, config.softclip_threshold) and not accept_read(
                read, config.min_mismatches
            ):
                continue

            reads[read.query_name] = read
            if read.reference_start is not None:
                rmin = min(rmin, read.reference_start)
            if read.reference_end is not None:
                rmax = max(rmax, read.reference_end)
            reads_in_ref[ref_id].add(read.query_name)

        refs_extents.append([rmin, rmax])

    # Extend references with additional padding
    refs_extents[0][0] = max(0, refs_extents[0][0] - config.fetch_ref_padding)
    refs_extents[1][1] = refs_extents[1][1] + config.fetch_ref_padding

    return reads, reads_in_ref, refs_extents


def _extract_references(
    chrom: str,
    refs_extents: list[list[int]],
    fasta_file: pysam.FastaFile,
) -> list[tuple[int, str]]:
    """Extract reference sequences for breakpoint regions.

    Parameters
    ----------
    chrom : str
        Chromosome name
    refs_extents : list of lists
        Extent coordinates [[start1, end1], [start2, end2]]
    fasta_file : pysam.FastaFile
        Opened reference FASTA file

    Returns
    -------
    list of tuples
        List of (start_position, sequence) tuples for each region
    """
    refs = []
    for rmin, rmax in refs_extents:
        ref_seq = fasta_file.fetch(chrom, rmin, rmax)
        refs.append((rmin, ref_seq))
    return refs


def _write_alignment_input(
    reads: dict[str, pysam.AlignedSegment],
    refs: list[tuple[int, str]],
    temp_dir: Path,
    chrom: str,
    start: int,
    end: int,
    config: JAlignConfig,
    log_file: TextIO | None = None,
) -> tuple[Path, list[pysam.AlignedSegment]]:
    """Write input file for jump alignment tool.

    Parameters
    ----------
    reads : dict
        Mapping of read names to AlignedSegment objects
    refs : list of tuples
        Reference sequences with (start_pos, sequence)
    temp_dir : Path
        Directory for temporary files
    chrom : str
        Chromosome name
    start : int
        CNV start position
    end : int
        CNV end position
    config : JAlignConfig
        Configuration parameters
    log_file : file-like object, optional
        Log file for recording input

    Returns
    -------
    tuple
        - Path to created input file
        - List of reads in order written
    """
    # Subsample if too many reads
    subsample_ratio = 1.0
    if len(reads) > config.max_reads_per_cnv:
        subsample_ratio = config.max_reads_per_cnv / len(reads)
        logger.info(f"Subsampling reads with ratio {subsample_ratio:.3f}")

    # Create input file
    input_file = temp_dir / f"jalign_{chrom}_{start}_{end}_{os.getpid()}.txt"
    reads_in_order = []
    ref_emitted = False

    with open(input_file, "w") as f:
        for read in reads.values():
            if subsample_ratio < 1.0 and random.random() > subsample_ratio:  # noqa: S311
                continue

            reads_in_order.append(read)

            if not ref_emitted:
                # First line includes both reference sequences
                line = f"{read.query_name}\t{read.query_sequence}\t{refs[1][1]}\t{refs[0][1]}\n"
                ref_emitted = True
            else:
                # Subsequent lines reference the same sequences
                line = f"{read.query_name}\t{read.query_sequence}\t=\n"

            f.write(line)
            if log_file:
                log_file.write(line)

    logger.debug(f"Wrote {len(reads_in_order)} reads to {input_file}")
    return input_file, reads_in_order


def _parse_alignment_results(
    alignment_output: str,
    reads_in_order: list[pysam.AlignedSegment],
    reads_in_ref: list[set[str]],
    refs: list[tuple[int, str]],
    log_file: TextIO | None = None,
) -> list[list]:
    """Parse alignment tool output into structured results.

    Parameters
    ----------
    alignment_output : str
        Raw output from alignment tool
    reads_in_order : list
        Reads in the order they were written to input
    reads_in_ref : list of sets
        Sets of read names present at each breakpoint
    refs : list of tuples
        Reference sequences with (start_pos, sequence)
    log_file : file-like object, optional
        Log file for recording parsed results

    Returns
    -------
    list
        List of realignment results, each containing:
        [read, ref1_start, ref2_start, alignment_info, in_ref1, in_ref2]
    """
    realignments = []
    lines = alignment_output.split("\n")

    # Skip header line
    for alignment, read in zip(lines[1:], reads_in_order, strict=False):
        if log_file:
            log_file.write(alignment + "\n")

        if not alignment.strip():
            continue

        alignment_fields = alignment.split("\t")
        in_ref1 = read.query_name in reads_in_ref[0]
        in_ref2 = read.query_name in reads_in_ref[1]

        realignments.append([read, refs[0][0], refs[1][0], alignment_fields, in_ref1, in_ref2])

    return realignments


def _count_supporting_alignments(  # noqa: C901, PLR0912
    realignments: list[list], config: JAlignConfig
) -> tuple[int, int, int, int]:
    """Count alignments supporting DUP and DEL hypotheses.

    Parameters
    ----------
    realignments : list
        Parsed alignment results from _parse_alignment_results
    config : JAlignConfig
        Configuration parameters

    Returns
    -------
    tuple[int, int, int, int]
        Counts of (jump_better, djump_better, jump_much_better,
        djump_much_better)
    """
    jump_better = 0
    djump_better = 0
    jump_much_better = 0
    djump_much_better = 0

    for realignment in realignments:
        read, ref1_start, ref2_start, ainfo, in_ref1, in_ref2 = realignment

        # Skip if insufficient alignment info
        if len(ainfo) < MIN_ALIGNMENT_FIELDS:
            continue

        # Parse alignment scores
        try:
            (
                score,
                jreadlen1,
                jreadlen2,
                dscore,
                djreadlen1,
                djreadlen2,
                score1,
                score2,
            ) = map(int, ainfo[1:9])
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse alignment info for {ainfo[0]}")
            continue

        # Evaluate DUP jump alignment
        minimal_score = config.max_score_fraction * (jreadlen1 + jreadlen2) * config.match_score
        stringent_score = config.stringent_max_score_fraction * (jreadlen1 + jreadlen2) * config.match_score

        if score > 0 and score > max(score1, score2) + config.min_seq_len_jump_align_component:
            if min(jreadlen1, jreadlen2) >= config.min_seq_len_jump_align_component:
                if score > minimal_score:
                    jump_better += 1
                if score > stringent_score:
                    jump_much_better += 1

        # Evaluate DEL jump alignment
        minimal_score = config.max_score_fraction * (djreadlen1 + djreadlen2) * config.match_score
        stringent_score = config.stringent_max_score_fraction * (djreadlen1 + djreadlen2) * config.match_score

        if dscore > 0 and dscore > max(score1, score2) + config.min_seq_len_jump_align_component:
            if min(djreadlen1, djreadlen2) >= config.min_seq_len_jump_align_component:
                if dscore > minimal_score:
                    djump_better += 1
                if dscore > stringent_score:
                    djump_much_better += 1

    return jump_better, djump_better, jump_much_better, djump_much_better


def process_cnv(
    chrom: str,
    start: int,
    end: int,
    reads_file: pysam.AlignmentFile,
    fasta_file: pysam.FastaFile,
    config: JAlignConfig,
    temp_dir: Path,
    log_file: TextIO | None = None,
) -> tuple[int, int, int, int]:
    """Process a single CNV region with jump alignment.

    Analyzes reads at CNV breakpoints to identify supporting evidence
    for duplications and deletions using jump alignment.

    Parameters
    ----------
    chrom : str
        Chromosome name
    start : int
        CNV start position (0-based)
    end : int
        CNV end position (0-based)
    reads_file : pysam.AlignmentFile
        Opened alignment file (BAM/CRAM)
    fasta_file : pysam.FastaFile
        Opened reference FASTA file
    config : JAlignConfig
        Configuration parameters for alignment
    temp_dir : Path
        Directory for temporary files
    log_file : file-like object, optional
        Log file for detailed output

    Returns
    -------
    tuple[int, int, int, int]
        Counts of supporting alignments:
        (jump_better, djump_better, jump_much_better, djump_much_better)
    """
    if log_file:
        log_file.write(f"\n>>> {chrom}:{start}-{end}\n")

    logger.debug(f"Processing CNV: {chrom}:{start}-{end}")

    # Fetch reads at breakpoints
    reads, reads_in_ref, refs_extents = _fetch_reads_at_breakpoints(chrom, start, end, reads_file, config)

    if not reads:
        logger.debug(f"No reads found for {chrom}:{start}-{end}")
        return 0, 0, 0, 0

    # Extract reference sequences
    refs = _extract_references(chrom, refs_extents, fasta_file)

    # Write alignment input file
    input_file, reads_in_order = _write_alignment_input(reads, refs, temp_dir, chrom, start, end, config, log_file)

    # Run jump alignment tool
    try:
        alignment_cmd = config.build_alignment_command(input_file)
        alignment_output, _ = run_alignment_tool(alignment_cmd, log_file)
    finally:
        # Clean up temporary file
        if input_file.exists():
            input_file.unlink()

    # Parse results
    if log_file:
        log_file.write(f">>> alignments: {chrom}:{start}-{end}\n")

    realignments = _parse_alignment_results(alignment_output, reads_in_order, reads_in_ref, refs, log_file)

    if log_file:
        log_file.write(f"<<< alignments: {chrom}:{start}-{end}\n")

    # Count supporting alignments
    return _count_supporting_alignments(realignments, config)
