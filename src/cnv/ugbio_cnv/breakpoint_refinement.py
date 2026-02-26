"""
CNV breakpoint refinement using windowed split-read analysis.

BIOIN-2676: Refine CNV breakpoints by analyzing split reads (SA tags) from windowed
regions at CNV boundaries. Uses median position + max deviation for CIPOS estimation.
"""

from statistics import median

import pysam
from ugbio_cnv.analyze_cnv_breakpoint_reads import (
    _calculate_breakpoint_regions,
    _should_skip_read,
    get_supplementary_alignments,
)

# Constants for read filtering
MIN_READS_PER_BREAKPOINT = 3  # Minimum reads required for reliable breakpoint estimation


def _extract_sa_positions(reads: list[pysam.AlignedSegment]) -> list[int]:
    """
    Extract supplementary alignment positions from SA tags.

    Parameters
    ----------
    reads : list[pysam.AlignedSegment]
        Reads with SA tags

    Returns
    -------
    list[int]
        List of 1-based positions extracted from SA tags
    """
    positions = []
    for read in reads:
        # Use existing helper to parse SA tags robustly
        supp_alns = get_supplementary_alignments(read)
        for _chrom, pos, _end, _left_clip, _right_clip, _is_reverse in supp_alns:
            # get_supplementary_alignments returns 0-based positions, convert to 1-based
            positions.append(pos + 1)
    return positions


def _process_reads_from_window(
    bam: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    cnv_type: str,
    seen_pairs: set[tuple[str, str]],
) -> list[pysam.AlignedSegment]:
    """
    Process reads from a window, filtering by RG tag and deduplicating.

    Parameters
    ----------
    bam : pysam.AlignmentFile
        Open BAM file
    chrom : str
        Chromosome
    start : int
        Window start
    end : int
        Window end
    cnv_type : str
        CNV type for RG filtering ("DEL" or "DUP")
    seen_pairs : set[tuple[str, str]]
        Set of seen (read_name, RG) pairs for deduplication

    Returns
    -------
    list[pysam.AlignedSegment]
        Filtered reads from window
    """
    reads = []
    for read in bam.fetch(chrom, start, end):
        if _should_skip_read(read):
            continue

        # Skip reads without query name
        if not read.query_name:
            continue

        # Get RG tag and filter by CNV type
        try:
            rg_tag = read.get_tag("RG")
            rg = str(rg_tag) if rg_tag is not None else "UNKNOWN"
        except KeyError:
            rg = "UNKNOWN"

        if rg != cnv_type:
            continue

        # Deduplication: track (read_name, RG) pairs
        read_key = (read.query_name, rg)
        if read_key in seen_pairs:
            continue
        seen_pairs.add(read_key)

        reads.append(read)
    return reads


def extract_reads_windowed(
    bam_files: list[str],
    cnv_chrom: str,
    cnv_start: int,
    cnv_end: int,
    cnv_type: str,
    cushion: int = 2500,
) -> tuple[list[pysam.AlignedSegment], list[pysam.AlignedSegment], set[tuple[str, str]]]:
    """
    Extract reads from two 5kb windows at CNV boundaries with fixed deduplication.

    Parameters
    ----------
    bam_files : list[str]
        BAM file paths (supports multi-sample BAMs)
    cnv_chrom : str
        Chromosome
    cnv_start : int
        CNV start position
    cnv_end : int
        CNV end position
    cnv_type : str
        "DEL" or "DUP" for RG filtering
    cushion : int
        Half-window size (default: 2500 bp)

    Returns
    -------
    left_reads : list[pysam.AlignedSegment]
        Reads from left window [START-cushion, START+cushion]
    right_reads : list[pysam.AlignedSegment]
        Reads from right window [END-cushion, END+cushion]
    seen_pairs : set[tuple[str, str]]
        Deduplication tracker (read_name, RG)
    """
    left_reads = []
    right_reads = []
    seen_pairs = set()

    # Calculate breakpoint regions using existing helper
    start_region_start, start_region_end, end_region_start, end_region_end = _calculate_breakpoint_regions(
        cnv_start, cnv_end, cushion
    )

    for bam_path in bam_files:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            # Process left window
            left_reads.extend(
                _process_reads_from_window(bam, cnv_chrom, start_region_start, start_region_end, cnv_type, seen_pairs)
            )
            # Process right window
            right_reads.extend(
                _process_reads_from_window(bam, cnv_chrom, end_region_start, end_region_end, cnv_type, seen_pairs)
            )

    return left_reads, right_reads, seen_pairs


def estimate_refined_breakpoints(
    left_reads: list[pysam.AlignedSegment],
    right_reads: list[pysam.AlignedSegment],
    original_cipos: tuple[int, int],
) -> tuple[int, int, tuple[int, int]] | None:
    """
    Estimate refined breakpoints using median + max deviation from SA tags.

    Only returns refined values if CIPOS interval size < original interval size.

    Parameters
    ----------
    left_reads : list[pysam.AlignedSegment]
        Reads from left window
    right_reads : list[pysam.AlignedSegment]
        Reads from right window
    original_cipos : tuple[int, int]
        Original CIPOS bounds (left, right)

    Returns
    -------
    (refined_start, refined_end, refined_cipos) or None if not improved
    """
    # Extract SA tag positions using helper function
    left_positions = _extract_sa_positions(left_reads)
    right_positions = _extract_sa_positions(right_reads)

    # Need at least MIN_READS_PER_BREAKPOINT positions per breakpoint
    if len(left_positions) < MIN_READS_PER_BREAKPOINT or len(right_positions) < MIN_READS_PER_BREAKPOINT:
        return None

    # Calculate median and max deviation
    left_median = round(median(left_positions))
    right_median = round(median(right_positions))

    left_max_dev = max(abs(pos - left_median) for pos in left_positions)
    right_max_dev = max(abs(pos - right_median) for pos in right_positions)

    # Refined CIPOS: symmetric (-max_dev, +max_dev)
    refined_cipos = (-max(left_max_dev, right_max_dev), max(left_max_dev, right_max_dev))

    # Only update if refined interval is tighter than original
    original_interval_size = original_cipos[1] - original_cipos[0]
    refined_interval_size = refined_cipos[1] - refined_cipos[0]

    if refined_interval_size >= original_interval_size:
        return None

    return left_median, right_median, refined_cipos


def refine_cnv_breakpoints_from_vcf(
    input_vcf: str, bam_files: list[str], output_vcf: str, sv_vcf: str | None = None, cushion: int = 2500
) -> None:
    """
    Refine CNV breakpoints from VCF using windowed split-read analysis.

    Parameters
    ----------
    input_vcf : str
        Input VCF with CNVs (must have CIPOS field)
    bam_files : list[str]
        BAM file paths for evidence extraction
    output_vcf : str
        Output VCF with refined breakpoints
    sv_vcf : str, optional
        SV calls VCF (not implemented - raises NotImplementedError)
    cushion : int
        Half-window size (default: 2500 bp)

    Raises
    ------
    NotImplementedError
        If sv_vcf is provided
    ValueError
        If input VCF lacks CIPOS field
    """
    if sv_vcf is not None:
        raise NotImplementedError("SV VCF integration not yet implemented")

    vcf_in = pysam.VariantFile(input_vcf)
    vcf_out = pysam.VariantFile(output_vcf, "w", header=vcf_in.header)

    for record in vcf_in:
        # Extract CIPOS (required)
        if "CIPOS" not in record.info:
            raise ValueError(f"Record {record.chrom}:{record.pos} missing CIPOS field")

        cipos = record.info["CIPOS"]
        original_start = record.pos
        original_end = record.stop

        # Get CNV type from SVTYPE
        svtype = record.info.get("SVTYPE", "").upper()
        if svtype not in ("DEL", "DUP"):
            # Not a DEL/DUP, write unchanged
            vcf_out.write(record)
            continue

        # Extract reads from windowed regions
        left_reads, right_reads, _ = extract_reads_windowed(
            bam_files, record.chrom, original_start, original_end, svtype, cushion
        )

        # Attempt refinement
        result = estimate_refined_breakpoints(left_reads, right_reads, cipos)

        if result is not None:
            refined_start, refined_end, refined_cipos = result
            # Update record
            record.pos = refined_start
            record.stop = refined_end
            record.info["CIPOS"] = refined_cipos

        vcf_out.write(record)

    vcf_in.close()
    vcf_out.close()
