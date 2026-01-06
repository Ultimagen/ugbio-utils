#!/usr/bin/env python3
"""Jump alignment for CNV breakpoint analysis.

This module performs jump alignment on reads at CNV breakpoints to identify
supporting evidence for duplications using an external C alignment tool.
The core functionality operates on individual CNV regions and can be used
programmatically or through the run_jalign.py CLI script.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import pandas as pd
import pyfaidx
import pysam
from ugbio_core.logger import logger

# CIGAR operation codes
CIGAR_SOFT_CLIP = 4


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
    output_all_alignments : bool
        If True, output all alignments; if False, output only the best alignment per read
    """

    match_score: int = 2
    mismatch_score: int = -8
    gap_open_score: int = -12
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
    output_all_alignments: bool = False
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
            "",  # Placeholder for output JSON file path
        ]

    def build_alignment_command(self, input_file: Path, output_file: Path) -> list[str]:
        """Build alignment command with specific input and output files.

        Parameters
        ----------
        input_file : Path
            Path to input file for alignment tool
        output_file : Path
            Path to output JSON file for alignment results

        Returns
        -------
        list[str]
            Command with arguments ready for execution
        """
        cmd = self._alignment_cmd_template.copy()
        input_file_idx = 7
        output_file_idx = 8
        cmd[input_file_idx] = str(input_file)
        cmd[output_file_idx] = str(output_file)
        return cmd


def create_bam_record_from_alignment(
    qname: str,
    seq: str,
    chrom: str,
    ref_start: int,
    score: int,
    begin: int,
    cigar: str,
    rgid: str,
    header: pysam.AlignmentHeader,
    *,
    is_supplementary: bool = False,
) -> pysam.AlignedSegment:
    """Create a BAM record from alignment data.

    Parameters
    ----------
    qname : str
        Query (read) name
    seq : str
        Read sequence
    chrom : str
        Chromosome/reference name
    ref_start : int
        Reference sequence start position (0-based genomic coordinate)
    score : int
        Alignment score
    begin : int
        Alignment begin position relative to reference start
    cigar : str
        CIGAR string
    rgid : str
        Read group ID (e.g., 'REF1', 'REF2', 'DEL', 'DUP')
    header : pysam.AlignmentHeader
        BAM header with reference sequence information
    is_supplementary : bool
        Whether this is a supplementary alignment

    Returns
    -------
    pysam.AlignedSegment
        BAM record with alignment information
    """
    record = pysam.AlignedSegment(header)
    record.query_name = qname
    record.query_sequence = seq
    record.reference_name = chrom
    record.reference_start = ref_start + begin
    record.cigarstring = cigar
    record.mapping_quality = 60

    # Set flags
    if is_supplementary:
        record.flag = pysam.FSUPPLEMENTARY

    # Set tags
    record.set_tag("AS", score, value_type="i")
    record.set_tag("RG", rgid, value_type="Z")

    return record


def create_bam_records_from_simple_alignment(
    qname: str,
    seq: str,
    chrom: str,
    ref_start: int,
    score: int,
    begin: int,
    cigar: str,
    rgid: str,
    header: pysam.AlignmentHeader,
) -> list[pysam.AlignedSegment]:
    """Create BAM record from simple alignment (align1/align2).

    Parameters
    ----------
    qname : str
        Query (read) name
    seq : str
        Read sequence
    chrom : str
        Chromosome/reference name
    ref_start : int
        Reference sequence start position (0-based genomic coordinate)
    score : int
        Alignment score
    begin : int
        Alignment begin position relative to reference start
    cigar : str
        CIGAR string
    rgid : str
        Read group ID ('REF1' or 'REF2')
    header : pysam.AlignmentHeader
        BAM header with reference sequence information

    Returns
    -------
    list[pysam.AlignedSegment]
        Single-element list containing the BAM record
    """
    record = create_bam_record_from_alignment(
        qname=qname,
        seq=seq,
        chrom=chrom,
        ref_start=ref_start,
        score=score,
        begin=begin,
        cigar=cigar,
        rgid=rgid,
        header=header,
        is_supplementary=False,
    )
    return [record]


def create_bam_records_from_jump_alignment(  # noqa: PLR0913
    qname: str,
    seq: str,
    chrom: str,
    ref1_start: int,
    ref2_start: int,
    score: int,
    begin1: int,
    cigar1: str,
    begin2: int,
    cigar2: str,
    rgid: str,
    header: pysam.AlignmentHeader,
) -> list[pysam.AlignedSegment]:
    """Create BAM records from jump alignment (jump_forward/jump_backward).

    Creates two records: primary alignment to first reference and supplementary
    alignment to second reference.

    Parameters
    ----------
    qname : str
        Query (read) name
    seq : str
        Read sequence
    chrom : str
        Chromosome/reference name
    ref1_start : int
        First reference sequence start position (0-based genomic coordinate)
    ref2_start : int
        Second reference sequence start position (0-based genomic coordinate)
    score : int
        Alignment score
    begin1 : int
        First alignment begin position relative to ref1_start
    cigar1 : str
        First alignment CIGAR string
    begin2 : int
        Second alignment begin position relative to ref2_start
    cigar2 : str
        Second alignment CIGAR string
    rgid : str
        Read group ID ('DEL' or 'DUP')
    header : pysam.AlignmentHeader
        BAM header with reference sequence information

    Returns
    -------
    list[pysam.AlignedSegment]
        Two-element list: [primary_record, supplementary_record]
    """
    # Primary alignment (first component)
    primary = create_bam_record_from_alignment(
        qname=qname,
        seq=seq,
        chrom=chrom,
        ref_start=ref1_start,
        score=score,
        begin=begin1,
        cigar=cigar1,
        rgid=rgid,
        header=header,
        is_supplementary=False,
    )

    # Supplementary alignment (second component)
    supplementary = create_bam_record_from_alignment(
        qname=qname,
        seq=seq,
        chrom=chrom,
        ref_start=ref2_start,
        score=score,
        begin=begin2,
        cigar=cigar2,
        rgid=rgid,
        header=header,
        is_supplementary=True,
    )

    # Set SA tag on primary to point to supplementary
    sa_tag = f"{chrom},{ref2_start + begin2 + 1},{'+' if not supplementary.is_reverse else '-'},{cigar2},{60},{0};"
    primary.set_tag("SA", sa_tag, value_type="Z")

    return [primary, supplementary]


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
    nm = count_nm_mismatches(read)
    if nm is None:
        nm = 0
    return nm >= min_mismatches


def run_alignment_tool(command: list[str], log_file: TextIO | None = None) -> int:
    """Execute external alignment tool and capture output.

    Parameters
    ----------
    command : list[str]
        Command and arguments to execute
    log_file : file-like object, optional
        Log file for recording command execution

    Returns
    -------
    int
        Return code from command execution

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
        return process.returncode
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
) -> tuple[dict[str, pysam.AlignedSegment], list[list[int]]]:
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
        - list of two lists with [min_pos, max_pos] for each breakpoint
    """
    reads = {}
    refs_extents = []

    for loc in [start, end]:
        rmin = max(0, loc - config.fetch_read_padding)
        rmax = loc + config.fetch_read_padding

        for read in reads_file.fetch(
            chrom,
            rmin,
            rmax,
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

        refs_extents.append([rmin, rmax])

    # Extend references with additional padding
    refs_extents[0][0] = max(0, refs_extents[0][0] - config.fetch_ref_padding)
    refs_extents[1][1] = refs_extents[1][1] + config.fetch_ref_padding

    return reads, refs_extents


def _extract_references(
    chrom: str,
    refs_extents: list[list[int]],
    fasta_file: pyfaidx.Fasta,
) -> list[tuple[int, str]]:
    """Extract reference sequences for breakpoint regions.

    Parameters
    ----------
    chrom : str
        Chromosome name
    refs_extents : list of lists
        Extent coordinates [[start1, end1], [start2, end2]]
    fasta_file : pyfaidx.Fasta
        Opened reference FASTA file

    Returns
    -------
    list of tuples
        List of (start_position, sequence) tuples for each region
    """
    refs = []
    for rmin, rmax in refs_extents:
        ref_seq = fasta_file[chrom][rmin:rmax].seq.upper().replace("N", "A")  # type: ignore
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
            if random.random() > subsample_ratio:  # noqa: S311
                continue

            reads_in_order.append(read)

            if not ref_emitted:
                # First line includes both reference sequences
                line = f"{read.query_name}\t{read.query_sequence}\t{refs[0][1]}\t{refs[1][1]}\n"
                ref_emitted = True
            else:
                # Subsequent lines reference the same sequences
                line = f"{read.query_name}\t{read.query_sequence}\t=\n"

            f.write(line)
            if log_file:
                log_file.write(line)

    logger.info(f"Wrote {len(reads_in_order)} reads to {input_file}")
    return input_file, reads_in_order


def _evaluate_alignment_scores(
    row: pd.Series,
    config: JAlignConfig,
) -> tuple[str, bool, bool, bool, bool]:
    """Evaluate alignment scores for a single read.

    Parameters
    ----------
    row : pd.Series
        DataFrame row containing alignment data
    config : JAlignConfig
        Configuration parameters

    Returns
    -------
    tuple[str, bool, bool, bool, bool]
        - qname: Query name
        - jump_forward_is_best: Whether jump_forward alignment meets criteria
        - jump_backward_is_best: Whether jump_backward alignment meets criteria
        - jump_forward_is_stringent: Whether jump_forward meets stringent criteria
        - jump_backward_is_stringent: Whether jump_backward meets stringent criteria
    """
    # Extract values from DataFrame row
    qname = row.get("qname", "")
    score = row.get("jump_forward.score", 0)
    jreadlen1 = row.get("jump_forward.readlen1", 0)
    jreadlen2 = row.get("jump_forward.readlen2", 0)
    dscore = row.get("jump_backward.score", 0)
    djreadlen1 = row.get("jump_backward.readlen1", 0)
    djreadlen2 = row.get("jump_backward.readlen2", 0)
    score1 = row.get("align1.score", 0)
    score2 = row.get("align2.score", 0)

    # Evaluate jump_forward (DUP) alignment
    jump_forward_is_best = False
    jump_forward_is_stringent = False
    if score > 0 and score > max(score1, score2) + config.min_seq_len_jump_align_component:
        if min(jreadlen1, jreadlen2) >= config.min_seq_len_jump_align_component:
            minimal_score = config.max_score_fraction * (jreadlen1 + jreadlen2) * config.match_score
            stringent_score = config.stringent_max_score_fraction * (jreadlen1 + jreadlen2) * config.match_score
            if score > minimal_score:
                jump_forward_is_best = True
            if score > stringent_score:
                jump_forward_is_stringent = True

    # Evaluate jump_backward (DEL) alignment
    jump_backward_is_best = False
    jump_backward_is_stringent = False
    if dscore > 0 and dscore > max(score1, score2) + config.min_seq_len_jump_align_component:
        if min(djreadlen1, djreadlen2) >= config.min_seq_len_jump_align_component:
            minimal_score = config.max_score_fraction * (djreadlen1 + djreadlen2) * config.match_score
            stringent_score = config.stringent_max_score_fraction * (djreadlen1 + djreadlen2) * config.match_score
            if dscore > minimal_score:
                jump_backward_is_best = True
            if dscore > stringent_score:
                jump_backward_is_stringent = True

    return qname, jump_forward_is_best, jump_backward_is_best, jump_forward_is_stringent, jump_backward_is_stringent


def determine_best_alignments(df: pd.DataFrame, config: JAlignConfig) -> dict[str, str]:
    """Determine the best alignment type for each read using scoring logic.

    Uses the same criteria as _count_supporting_alignments to determine
    which alignment type is best for each read.

    Parameters
    ----------
    df : pd.DataFrame
        Alignment results from _parse_alignment_results
    config : JAlignConfig
        Configuration parameters

    Returns
    -------
    dict[str, str]
        Dictionary mapping read names to best alignment type:
        'align1', 'align2', 'jump_forward', or 'jump_backward'
    """
    best_alignments = {}

    for _, row in df.iterrows():
        # Evaluate alignment scores using shared logic
        qname, jump_forward_is_best, jump_backward_is_best, _, _ = _evaluate_alignment_scores(row, config)

        if not qname:
            continue

        # Determine best alignment based on evaluation results
        if jump_forward_is_best and jump_backward_is_best:
            # Both jump alignments are good, choose the one with higher score
            score = row.get("jump_forward.score", 0)
            dscore = row.get("jump_backward.score", 0)
            best_alignments[qname] = "jump_forward" if score >= dscore else "jump_backward"
        elif jump_forward_is_best:
            best_alignments[qname] = "jump_forward"
        elif jump_backward_is_best:
            best_alignments[qname] = "jump_backward"
        else:
            # Neither jump alignment is good enough, choose better simple alignment
            score1 = row.get("align1.score", 0)
            score2 = row.get("align2.score", 0)
            best_alignments[qname] = "align1" if score1 >= score2 else "align2"

    return best_alignments


def _create_bam_records_for_alignment_type(
    alignment_type: str,
    row: pd.Series,
    qname: str,
    seq: str,
    chrom: str,
    ref1_start: int,
    ref2_start: int,
    header: pysam.AlignmentHeader,
) -> list[pysam.AlignedSegment]:
    """Create BAM records for a specific alignment type.

    Parameters
    ----------
    alignment_type : str
        Alignment type: 'align1', 'align2', 'jump_forward', or 'jump_backward'
    row : pd.Series
        DataFrame row containing alignment data
    qname : str
        Query (read) name
    seq : str
        Read sequence
    chrom : str
        Chromosome name
    ref1_start : int
        First reference start position
    ref2_start : int
        Second reference start position
    header : pysam.AlignmentHeader
        BAM header

    Returns
    -------
    list[pysam.AlignedSegment]
        List of BAM records for this alignment type
    """
    # Configuration for each alignment type
    alignment_configs = {
        "align1": {
            "is_simple": True,
            "ref_start": ref1_start,
            "score_field": "align1.score",
            "begin_field": "align1.begin",
            "cigar_field": "align1.cigar",
            "rgid": "REF1",
        },
        "align2": {
            "is_simple": True,
            "ref_start": ref2_start,
            "score_field": "align2.score",
            "begin_field": "align2.begin",
            "cigar_field": "align2.cigar",
            "rgid": "REF2",
        },
        "jump_forward": {
            "is_simple": False,
            "ref1_start": ref1_start,
            "ref2_start": ref2_start,
            "score_field": "jump_forward.score",
            "begin1_field": "jump_forward.begin1",
            "cigar1_field": "jump_forward.cigar1",
            "begin2_field": "jump_forward.begin2",
            "cigar2_field": "jump_forward.cigar2",
            "rgid": "DEL",
        },
        "jump_backward": {
            "is_simple": False,
            "ref1_start": ref2_start,
            "ref2_start": ref1_start,
            "score_field": "jump_backward.score",
            "begin1_field": "jump_backward.begin1",
            "cigar1_field": "jump_backward.cigar1",
            "begin2_field": "jump_backward.begin2",
            "cigar2_field": "jump_backward.cigar2",
            "rgid": "DUP",
        },
    }

    config = alignment_configs[alignment_type]

    if config["is_simple"]:
        return create_bam_records_from_simple_alignment(
            qname=qname,
            seq=seq,
            chrom=chrom,
            ref_start=config["ref_start"],
            score=int(row.get(config["score_field"], 0)),
            begin=int(row.get(config["begin_field"], 0)),
            cigar=str(row.get(config["cigar_field"], "*")),
            rgid=config["rgid"],
            header=header,
        )
    else:
        return create_bam_records_from_jump_alignment(
            qname=qname,
            seq=seq,
            chrom=chrom,
            ref1_start=config["ref1_start"],
            ref2_start=config["ref2_start"],
            score=int(row.get(config["score_field"], 0)),
            begin1=int(row.get(config["begin1_field"], 0)),
            cigar1=str(row.get(config["cigar1_field"], "*")),
            begin2=int(row.get(config["begin2_field"], 0)),
            cigar2=str(row.get(config["cigar2_field"], "*")),
            rgid=config["rgid"],
            header=header,
        )


def create_all_bam_records_from_json(
    json_file: Path,
    reads_in_order: list[pysam.AlignedSegment],
    chrom: str,
    ref1_start: int,
    ref2_start: int,
    header: pysam.AlignmentHeader,
    best_alignments: dict[str, str],
    *,
    output_all_alignments: bool = False,
) -> list[pysam.AlignedSegment]:
    """Create BAM records from JSON alignment output.

    Creates BAM records for either all alignments or only the best alignment
    (determined by scoring logic) for each read.

    Parameters
    ----------
    json_file : Path
        Path to JSON file containing alignment results
    reads_in_order : list
        Reads in the order they were written to input
    chrom : str
        Chromosome name
    ref1_start : int
        First reference sequence start position (0-based genomic coordinate)
    ref2_start : int
        Second reference sequence start position (0-based genomic coordinate)
    header : pysam.AlignmentHeader
        BAM header with reference sequence information
    best_alignments : dict[str, str]
        Dictionary mapping read names to best alignment type
    output_all_alignments : bool
        If True, output all alignments; if False, output only best alignment per read

    Returns
    -------
    list[pysam.AlignedSegment]
        List of BAM records
    """
    alignment_df = _parse_alignment_results(json_file)

    # Create a mapping from qname to read
    read_map = {read.query_name: read for read in reads_in_order}

    # Define alignment types to process
    alignment_types = ["align1", "align2", "jump_forward", "jump_backward"]

    bam_records = []

    for _, row in alignment_df.iterrows():
        qname = row.get("qname")
        if not qname or qname not in read_map:
            continue

        read = read_map[qname]
        seq = read.query_sequence

        if output_all_alignments:
            # Output all alignments for this read
            for idx, alignment_type in enumerate(alignment_types, start=1):
                score_field = f"{alignment_type}.score"
                if pd.notna(row.get(score_field)) and int(row.get(score_field, 0)) > 0:
                    records = _create_bam_records_for_alignment_type(
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
            # Output only the best alignment for this read
            best_alignment_type = best_alignments.get(qname)
            if not best_alignment_type:
                continue

            records = _create_bam_records_for_alignment_type(
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


def _parse_alignment_results(
    json_file: Path,
) -> pd.DataFrame:
    """Parse alignment tool JSON output into DataFrame.

    Parameters
    ----------
    json_file : Path
        Path to JSON file containing alignment results

    Returns
    -------
    pd.DataFrame
        Alignment results with flattened nested structure
    """
    # Load and normalize JSON data
    with open(json_file) as f:
        json_data = json.load(f)

    # Use pandas to flatten the nested JSON structure
    alignment_df = pd.json_normalize(json_data)

    return alignment_df


def _count_supporting_alignments(  # noqa: C901, PLR0912
    df: pd.DataFrame, config: JAlignConfig
) -> tuple[int, int, int, int]:
    """Count alignments supporting DUP and DEL hypotheses.

    Parameters
    ----------
    df : pd.DataFrame
        Alignment results from _parse_alignment_results
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

    for _, row in df.iterrows():
        # Evaluate alignment scores using shared logic
        _, jump_forward_is_best, jump_backward_is_best, jump_forward_is_stringent, jump_backward_is_stringent = (
            _evaluate_alignment_scores(row, config)
        )

        # Count DUP jump alignments
        if jump_forward_is_best:
            jump_better += 1
        if jump_forward_is_stringent:
            jump_much_better += 1

        # Count DEL jump alignments
        if jump_backward_is_best:
            djump_better += 1
        if jump_backward_is_stringent:
            djump_much_better += 1

    return jump_better, djump_better, jump_much_better, djump_much_better


def process_cnv(  # noqa: C901
    chrom: str,
    start: int,
    end: int,
    reads_file: pysam.AlignmentFile,
    fasta_file: pyfaidx.Fasta,
    config: JAlignConfig,
    temp_dir: Path,
    log_file: TextIO | None = None,
    header: pysam.AlignmentHeader | None = None,
) -> tuple[tuple[int, int, int, int], pd.DataFrame, list[pysam.AlignedSegment], pysam.AlignmentHeader | None]:
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
    fasta_file : pyfaidx.Fasta
        Opened reference FASTA file
    config : JAlignConfig
        Configuration parameters for alignment
    temp_dir : Path
        Directory for temporary files
    log_file : file-like object, optional
        Log file for detailed output

    Returns
    -------
    tuple[tuple[int, int, int, int], pd.DataFrame, list[pysam.AlignedSegment], pysam.AlignmentHeader]
        Tuple containing:
        - Counts of supporting alignments: (jump_better, djump_better, jump_much_better, djump_much_better)
        - Alignment info for each read (simplified list structure)
        - List of BAM records
        - BAM header
    """
    if log_file:
        log_file.write(f"\n>>> {chrom}:{start}-{end}\n")

    logger.debug(f"Processing CNV: {chrom}:{start}-{end}")

    # Fetch reads at breakpoints
    reads, refs_extents = _fetch_reads_at_breakpoints(chrom, start, end, reads_file, config)

    if not reads:
        logger.debug(f"No reads found for {chrom}:{start}-{end}")
        return (0, 0, 0, 0), pd.DataFrame(), [], header

    # Extract reference sequences
    refs = _extract_references(chrom, refs_extents, fasta_file)

    # Create BAM header for generating alignment records
    if header is None:
        header_dict = reads_file.header.to_dict()
        # Add read groups if not present
        if "RG" not in header_dict:
            header_dict["RG"] = []

        # Ensure all required read groups are present
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

        header = pysam.AlignmentHeader.from_dict(header_dict)

    # Write alignment input file
    input_file, reads_in_order = _write_alignment_input(reads, refs, temp_dir, chrom, start, end, config, log_file)

    # Create output JSON file path
    output_file = temp_dir / f"jalign_{chrom}_{start}_{end}_{os.getpid()}_output.json"

    # Run jump alignment tool
    try:
        alignment_cmd = config.build_alignment_command(input_file, output_file)
        run_alignment_tool(alignment_cmd, log_file)

        # Parse results from JSON file
        if log_file:
            log_file.write(f">>> alignments: {chrom}:{start}-{end}\n")

        # Parse alignment results into DataFrame
        result_df = _parse_alignment_results(output_file)

        # Determine best alignment for each read using scoring logic
        best_alignments = determine_best_alignments(result_df, config)

        # Create BAM records from alignments
        bam_records = create_all_bam_records_from_json(
            output_file,
            reads_in_order,
            chrom,
            refs[0][0],
            refs[1][0],
            header,
            best_alignments,
            output_all_alignments=config.output_all_alignments,
        )

        # Count supporting alignments
        counts = _count_supporting_alignments(result_df, config)
    finally:
        # Clean up temporary files
        if input_file.exists():
            input_file.unlink()
        if output_file.exists():
            output_file.unlink()

    if log_file:
        log_file.write(f"<<< alignments: {chrom}:{start}-{end}\n")
    result_df["chrom"] = chrom
    result_df["start"] = start
    result_df["end"] = end
    return counts, result_df, bam_records, header
