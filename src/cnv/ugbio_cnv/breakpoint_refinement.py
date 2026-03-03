"""
CNV breakpoint refinement using windowed split-read analysis.

BIOIN-2676: Refine CNV breakpoints by analyzing soft-clipped reads from windowed
regions at CNV boundaries. Uses median position + max deviation for CIPOS estimation.
"""

import argparse
import statistics
import sys
from dataclasses import dataclass

import pysam
from ugbio_cnv.analyze_cnv_breakpoint_reads import _calculate_breakpoint_regions
from ugbio_core.logger import logger

# Constants for read filtering
MIN_READS_PER_BREAKPOINT = 3  # Minimum reads required for reliable breakpoint estimation
BAM_CSOFT_CLIP = 4  # CIGAR operation code for soft clipping


@dataclass
class BamRefinementResult:
    """Per-BAM breakpoint refinement result."""

    bam_index: int
    bam_path: str
    read_count: int  # Number of matched read pairs
    refined_start: int
    refined_end: int
    refined_cipos: tuple[int, int]
    ci_size: int  # refined_cipos[1] - refined_cipos[0]


def _extract_softclip_positions(reads: list[pysam.AlignedSegment]) -> list[int]:
    """
    Extract breakpoint positions from soft-clipped regions in reads.

    For each read, identifies the reference position where soft clipping occurs.
    If a read has soft clips on both ends, uses the position corresponding to
    the longest soft clip.

    Parameters
    ----------
    reads : list[pysam.AlignedSegment]
        Reads with soft-clipped alignments

    Returns
    -------
    list[int]
        List of 1-based reference positions where soft clips occur
    """
    positions = []
    for read in reads:
        if read.cigartuples is None:
            positions.append(None)
            continue

        # Check for soft clips at start and end
        left_clip = 0
        right_clip = 0

        # First operation
        if read.cigartuples[0][0] == BAM_CSOFT_CLIP:
            left_clip = read.cigartuples[0][1]

        # Last operation
        if read.cigartuples[-1][0] == BAM_CSOFT_CLIP:
            right_clip = read.cigartuples[-1][1]

        # Skip reads with no soft clipping
        if left_clip == 0 and right_clip == 0:
            positions.append(None)
            continue

        # Use position of longest soft clip
        if left_clip >= right_clip:
            # Breakpoint is at the start of alignment (1-based)
            positions.append(read.reference_start + 1)
        else:
            # Breakpoint is at the end of alignment (1-based)
            # reference_end is already exclusive (0-based), so it's the correct 1-based position
            positions.append(read.reference_end)

    return positions


def _process_reads_from_window(
    bam: pysam.AlignmentFile, chrom: str, start: int, end: int, cnv_type: str
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
    seen_pairs : set[tuple[str, str, int]]
        Set of seen (read_name, RG, position) tuples for deduplication

    Returns
    -------
    list[pysam.AlignedSegment]
        Filtered reads from window
    """
    seen_pairs: set[tuple[str, str, int]] = set()
    reads = []
    for read in bam.fetch(chrom, start, end):
        # Skip unmapped, secondary, and duplicate reads (but NOT supplementary)
        if read.is_unmapped or read.is_secondary or read.is_duplicate:
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

        # Deduplication: track (read_name, RG, position) tuples to distinguish alignments
        # This allows both primary and supplementary alignments from the same read
        read_key = (read.query_name, rg, read.flag)
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
) -> dict[int, tuple[list[pysam.AlignedSegment], list[pysam.AlignedSegment]]]:
    """
    Extract reads from two 5kb windows at CNV boundaries with separate deduplication.

    For small CNVs where windows overlap, reads in the overlap can contribute
    to both left and right breakpoint evidence.

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
    dict[int, tuple[list[pysam.AlignedSegment], list[pysam.AlignedSegment]]]
        Dictionary mapping BAM index to (left_reads, right_reads) tuple.
        Each BAM's reads are kept separate for independent refinement.
    """
    per_bam_reads = {}

    # Calculate breakpoint regions using existing helper
    start_region_start, start_region_end, end_region_start, end_region_end = _calculate_breakpoint_regions(
        cnv_start, cnv_end, cushion
    )

    for i, bam_path in enumerate(bam_files):
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            # Process left window with its own deduplication set
            left_reads = _process_reads_from_window(bam, cnv_chrom, start_region_start, start_region_end, cnv_type)
            # Process right window with its own deduplication set
            right_reads = _process_reads_from_window(bam, cnv_chrom, end_region_start, end_region_end, cnv_type)
            # Store per-BAM reads
            per_bam_reads[i] = (left_reads, right_reads)

    return per_bam_reads


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
    # Extract breakpoint positions from soft-clipped reads
    left_positions = _extract_softclip_positions(left_reads)
    right_positions = _extract_softclip_positions(right_reads)

    matching_positions = [
        (left_positions[i], right_positions[i])
        for i in range(len(left_positions))
        if left_positions[i] is not None and right_positions[i] is not None and left_positions[i] < right_positions[i]
    ]
    left_positions = [x[0] for x in matching_positions]
    right_positions = [x[1] for x in matching_positions]

    # Need at least MIN_READS_PER_BREAKPOINT positions per breakpoint
    if len(left_positions) < MIN_READS_PER_BREAKPOINT or len(right_positions) < MIN_READS_PER_BREAKPOINT:
        return None

    # Calculate median and max deviation
    left_median = round(statistics.median(left_positions))
    right_median = round(statistics.median(right_positions))

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


def select_best_bam(
    per_bam_results: list[BamRefinementResult],
) -> BamRefinementResult | None:
    """
    Select the BAM with the smallest confidence interval.

    Parameters
    ----------
    per_bam_results : list[BamRefinementResult]
        List of refinement results per BAM

    Returns
    -------
    BamRefinementResult or None
        Best BAM's refinement result, or None if no BAM meets criteria

    Notes
    -----
    Selection criteria:
    1. Must have >= MIN_READS_PER_BREAKPOINT matched read pairs
    2. Among qualifying BAMs, select smallest CI size
    3. Ties broken by BAM index (first BAM wins)
    """
    if not per_bam_results:
        return None

    # Filter by read count threshold
    qualifying_bams = [result for result in per_bam_results if result.read_count >= MIN_READS_PER_BREAKPOINT]

    if not qualifying_bams:
        return None

    # Select BAM with minimum CI size (ties broken by first occurrence)
    best_bam = min(qualifying_bams, key=lambda r: (r.ci_size, r.bam_index))

    return best_bam


def _get_jalign_support_info(
    record: pysam.VariantRecord,
) -> tuple[int, int, bool]:
    """
    Extract JALIGN support information from VCF record.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to extract from

    Returns
    -------
    tuple[int, int, bool]
        (jalign_del_reads, jalign_dup_reads, has_strong_support)
    """
    try:
        jalign_del_reads = record.info.get("JALIGN_DEL_SUPPORT", 0)
        jalign_dup_reads = record.info.get("JALIGN_DUP_SUPPORT", 0)
        has_strong_jalign_support = (
            jalign_del_reads >= MIN_READS_PER_BREAKPOINT or jalign_dup_reads >= MIN_READS_PER_BREAKPOINT
        )
    except (ValueError, KeyError):
        jalign_del_reads = 0
        jalign_dup_reads = 0
        has_strong_jalign_support = False

    return jalign_del_reads, jalign_dup_reads, has_strong_jalign_support


def _process_bam_refinement(
    bam_idx: int,
    bam_path: str,
    left_reads: list[pysam.AlignedSegment],
    right_reads: list[pysam.AlignedSegment],
    cipos: tuple[int, int],
) -> BamRefinementResult | None:
    """
    Process single BAM for CNV breakpoint refinement.

    Parameters
    ----------
    bam_idx : int
        BAM file index
    bam_path : str
        Path to BAM file
    left_reads : list[pysam.AlignedSegment]
        Reads from left window
    right_reads : list[pysam.AlignedSegment]
        Reads from right window
    cipos : tuple[int, int]
        Original CIPOS bounds

    Returns
    -------
    BamRefinementResult or None
        Refinement result if successful, None otherwise
    """
    matched_left, matched_right = match_reads(left_reads, right_reads)
    refinement_result = estimate_refined_breakpoints(matched_left, matched_right, cipos)

    if refinement_result is None:
        return None

    refined_start, refined_end, refined_cipos = refinement_result
    ci_size = refined_cipos[1] - refined_cipos[0]

    return BamRefinementResult(
        bam_index=bam_idx,
        bam_path=bam_path,
        read_count=len(matched_left),
        refined_start=refined_start,
        refined_end=refined_end,
        refined_cipos=refined_cipos,
        ci_size=ci_size,
    )


def _log_refinement_failure(
    record: pysam.VariantRecord,
    original_start: int,
    original_end: int,
    svtype: str,
    cipos: tuple[int, int],
    jalign_del_reads: int,
    jalign_dup_reads: int,
):
    """
    Log and categorize CNV refinement failure.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record that failed refinement
    original_start : int
        CNV start position
    original_end : int
        CNV end position
    svtype : str
        CNV type ("DEL" or "DUP")
    cipos : tuple[int, int]
        Original CIPOS bounds
    jalign_del_reads : int
        JALIGN DEL support count
    jalign_dup_reads : int
        JALIGN DUP support count
    """
    base_msg = "CNV with strong JALIGN support not refined (no CIPOS improvement)"

    position_info = (
        f"{record.chrom}:{original_start}-{original_end} ({svtype}), "
        f"JALIGN_DEL={jalign_del_reads}, JALIGN_DUP={jalign_dup_reads}, "
    )

    position_info += f" original_CIPOS={cipos}"

    logger.debug(f"{base_msg}: {position_info}")


def _update_record_with_refinement(
    record: pysam.VariantRecord,
    best_bam: BamRefinementResult,
    original_interval_size: int,
) -> int:
    """
    Update VCF record with refined breakpoints from best BAM.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to update (mutated in-place)
    best_bam : BamRefinementResult
        Best BAM's refinement result
    original_interval_size : int
        Original CIPOS interval size

    Returns
    -------
    int
        CIPOS interval size reduction (bp)
    """
    record.pos = best_bam.refined_start
    record.stop = best_bam.refined_end
    record.info["CIPOS"] = best_bam.refined_cipos

    reduction = original_interval_size - best_bam.ci_size
    return reduction


def _log_refinement_summary(
    stats: dict[str, int],
    cipos_reductions: list[int],
    output_vcf: str,
) -> None:
    """
    Log summary statistics for CNV breakpoint refinement.

    Parameters
    ----------
    stats: dict[str, int]
        Refinement statistics
    cipos_reductions : list[int]
        List of CIPOS interval size reductions (in bp)
    output_vcf : str
        Path to output VCF file
    """
    logger.info("=" * 70)
    logger.info("CNV Breakpoint Refinement Summary")
    logger.info("=" * 70)
    logger.info(f"Total CNVs processed:           {stats['total_cnvs']}")
    logger.info(f"  Non-DEL/DUP (skipped):        {stats['non_del_dup']}")
    logger.info(f"  DEL/DUP analyzed:             {stats['total_cnvs'] - stats['non_del_dup']}")

    refinement_pct = (
        stats["refined_cnvs"] / (stats["total_cnvs"] - stats["non_del_dup"]) * 100
        if (stats["total_cnvs"] - stats["non_del_dup"]) > 0
        else 0
    )
    logger.info(f"  Successfully refined:         {stats['refined_cnvs']} ({refinement_pct:.1f}%)")
    logger.info(f"  Insufficient evidence (<{MIN_READS_PER_BREAKPOINT} reads): {stats['insufficient_evidence']}")
    logger.info(f"  No improvement (CIPOS):       {stats['no_improvement']}")

    if cipos_reductions:
        logger.info("")
        logger.info("CIPOS Interval Size Reductions:")
        logger.info(f"  Mean reduction:   {statistics.mean(cipos_reductions):.1f} bp")
        logger.info(f"  Median reduction: {statistics.median(cipos_reductions):.1f} bp")
        logger.info(f"  Min reduction:    {min(cipos_reductions)} bp")
        logger.info(f"  Max reduction:    {max(cipos_reductions)} bp")

    logger.info("=" * 70)
    logger.info(f"Output written to: {output_vcf}")


def match_reads(
    left_reads: list[pysam.AlignedSegment],
    right_reads: list[pysam.AlignedSegment],
) -> tuple[list[pysam.AlignedSegment], list[pysam.AlignedSegment]]:
    """
    Match reads from left and right windows based on query name and RG tag.

    This allows both primary and supplementary alignments from the same read to be included.

    Parameters
    ----------
    left_reads : list[pysam.AlignedSegment]
        Reads from left window
    right_reads : list[pysam.AlignedSegment]
        Reads from right window

    Returns
    -------
    matched_left : list[pysam.AlignedSegment]
        Matched reads from left window
    matched_right : list[pysam.AlignedSegment]
        Matched reads from right window
    """

    def key(read):
        return (read.query_name, read.get_tag("RG"))

    matches = []
    right_reads_lookup = {}
    for read in right_reads:
        right_reads_lookup[key(read)] = right_reads_lookup.get(key(read), []) + [read]
    for read in left_reads:
        for other_read in right_reads_lookup.get(key(read), []):
            if other_read.flag != read.flag:
                matches.append((read, other_read))
    return ([match[0] for match in matches], [match[1] for match in matches])


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

    logger.info("Starting CNV breakpoint refinement")
    logger.info(f"  Input VCF: {input_vcf}")
    logger.info(f"  BAM files: {', '.join(bam_files)}")
    logger.info(f"  Processing {len(bam_files)} BAM file(s)")
    logger.info(f"  Output VCF: {output_vcf}")

    vcf_in = pysam.VariantFile(input_vcf)
    vcf_out = pysam.VariantFile(output_vcf, "w", header=vcf_in.header)

    # Statistics tracking
    stats = {
        "total_cnvs": 0,
        "non_del_dup": 0,
        "refined_cnvs": 0,
        "insufficient_evidence": 0,
        "no_improvement": 0,
    }

    cipos_reductions = []  # Track interval size reductions
    progress_interval = 100  # Log progress every N records

    for record in vcf_in:
        stats["total_cnvs"] += 1

        # Progress reporting
        if stats["total_cnvs"] % progress_interval == 0:
            logger.info(f"Processed {stats['total_cnvs']} CNVs ({stats['refined_cnvs']} refined so far)...")

        # Extract CIPOS (required)
        if "CIPOS" not in record.info:
            raise ValueError(f"Record {record.chrom}:{record.pos} missing CIPOS field")

        cipos = record.info["CIPOS"]
        original_start = record.pos
        original_end = record.stop
        original_interval_size = cipos[1] - cipos[0]

        # Get CNV type from SVTYPE
        svtype = record.info.get("SVTYPE", "").upper()
        if svtype not in ("DEL", "DUP"):
            # Not a DEL/DUP, write unchanged
            stats["non_del_dup"] += 1
            vcf_out.write(record)
            continue

        # Extract reads from windowed regions (per-BAM)
        per_bam_reads = extract_reads_windowed(bam_files, record.chrom, original_start, original_end, svtype, cushion)

        # Process each BAM independently using helper
        per_bam_results = [
            result
            for result in [
                _process_bam_refinement(idx, bam_files[idx], left, right, cipos)
                for idx, (left, right) in per_bam_reads.items()
            ]
            if result is not None
        ]

        # Select best BAM
        best_bam = select_best_bam(per_bam_results)

        if best_bam is not None:
            # Update record with best BAM's refined breakpoints using helper
            reduction = _update_record_with_refinement(record, best_bam, original_interval_size)

            stats["refined_cnvs"] += 1
            cipos_reductions.append(reduction)

        else:
            # Extract JALIGN support info using helper
            jalign_del_reads, jalign_dup_reads, has_strong_jalign_support = _get_jalign_support_info(record)

            # Update counters based on category
            if has_strong_jalign_support:
                stats["no_improvement"] += 1
                _log_refinement_failure(
                    record,
                    original_start,
                    original_end,
                    svtype,
                    cipos,
                    jalign_del_reads,
                    jalign_dup_reads,
                )

            else:
                stats["insufficient_evidence"] += 1

        vcf_out.write(record)

    vcf_in.close()
    vcf_out.close()

    # Log summary statistics
    _log_refinement_summary(stats, cipos_reductions, output_vcf)


def get_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """
    Create or populate argument parser for CNV breakpoint refinement.

    Parameters
    ----------
    parser : argparse.ArgumentParser, optional
        Existing parser to add arguments to. If None, creates a new parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all arguments added.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    parser.add_argument(
        "--input-vcf",
        required=True,
        help="Input VCF file with CNV calls (must have CIPOS field)",
    )
    parser.add_argument(
        "--bam-files",
        required=True,
        nargs="+",
        help="One or more BAM file paths for evidence extraction",
    )
    parser.add_argument(
        "--output-vcf",
        required=True,
        help="Output VCF file with refined breakpoints",
    )
    parser.add_argument(
        "--cushion",
        type=int,
        default=2500,
        help="Half-window size for read extraction (default: 2500 bp)",
    )
    parser.add_argument(
        "--sv-vcf",
        default=None,
        help="SV calls VCF for integration (not yet implemented)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for CNV breakpoint refinement CLI.

    Parameters
    ----------
    argv : list[str], optional
        Command line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    parser = get_parser()
    args = parser.parse_args(argv)

    try:
        refine_cnv_breakpoints_from_vcf(
            input_vcf=args.input_vcf,
            bam_files=args.bam_files,
            output_vcf=args.output_vcf,
            sv_vcf=args.sv_vcf,
            cushion=args.cushion,
        )
        return 0
    except Exception as e:
        logger.error(f"Error refining CNV breakpoints: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
