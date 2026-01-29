"""
Analyze reads at CNV breakpoints for duplication and deletion evidence.

This script analyzes single-ended reads at CNV breakpoints to identify
supporting evidence for duplications and deletions based on read orientation
and position patterns. It takes a VCF file as input and outputs an annotated
VCF with breakpoint evidence in INFO fields.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pysam
from ugbio_core.dna_sequence_utils import CIGAR_OPS, parse_cigar_string
from ugbio_core.logger import logger

# CIGAR operation constants
CIGAR_SOFT_CLIP = CIGAR_OPS["S"]


@dataclass
class BreakpointEvidence:
    """Evidence for CNV at an interval breakpoint."""

    chrom: str
    start: int
    end: int
    duplication_reads: int
    deletion_reads: int
    total_reads: int


def has_right_soft_clip(cigar_tuples: list[tuple[int, int]] | None) -> bool:
    """
    Check if read has soft clipping on the right side.

    Parameters
    ----------
    cigar_tuples : list of tuples
        CIGAR tuples from pysam (operation, length)

    Returns
    -------
    bool
        True if last operation is soft clip
    """
    if not cigar_tuples:
        return False
    return cigar_tuples[-1][0] == CIGAR_SOFT_CLIP


def has_left_soft_clip(cigar_tuples: list[tuple[int, int]] | None) -> bool:
    """
    Check if read has soft clipping on the left side.

    Parameters
    ----------
    cigar_tuples : list of tuples
        CIGAR tuples from pysam (operation, length)

    Returns
    -------
    bool
        True if first operation is soft clip
    """
    if not cigar_tuples:
        return False
    return cigar_tuples[0][0] == CIGAR_SOFT_CLIP


def get_supplementary_alignments(
    alignment_file: pysam.AlignmentFile,
    read: pysam.AlignedSegment,
) -> list[tuple[str, int, bool, bool, bool]]:
    """
    Get supplementary alignments for a read from the SA tag.

    Parameters
    ----------
    alignment_file : pysam.AlignmentFile
        Open BAM/CRAM file
    read : pysam.AlignedSegment
        The read to analyze

    Returns
    -------
    list of tuples
        List of (chrom, pos, has_left_soft_clip, has_right_soft_clip, is_reverse) for supplementary alignments
    """
    if not read.has_tag("SA"):
        return []

    sa_tag = read.get_tag("SA")
    if not isinstance(sa_tag, str):
        return []

    supplementary_alns = []

    # SA tag format: (rname,pos,strand,CIGAR,mapQ,NM;)+
    for sa_entry in sa_tag.rstrip(";").split(";"):
        parts = sa_entry.split(",")

        chrom = parts[0]
        pos = int(parts[1]) - 1  # Convert to 0-based
        strand = parts[2]
        cigar_str = parts[3]

        is_reverse = strand == "-"

        # Parse CIGAR to get soft clip info
        cigar_tups = parse_cigar_string(cigar_str)
        left_soft_clip = has_left_soft_clip(cigar_tups)
        right_soft_clip = has_right_soft_clip(cigar_tups)

        supplementary_alns.append((chrom, pos, left_soft_clip, right_soft_clip, is_reverse))

    return supplementary_alns


def _prepare_read_for_cnv_check(
    read: pysam.AlignedSegment,
    interval_start: int,
    interval_end: int,
    cushion: int,
) -> tuple[str, int, bool, bool, bool, int, int, int, int] | None:
    """
    Prepare and validate read information for CNV consistency checking.

    Parameters
    ----------
    read : pysam.AlignedSegment
        The primary alignment of the read
    interval_start : int
        Start position of the interval (0-based)
    interval_end : int
        End position of the interval (0-based)
    cushion : int
        Number of bases to extend search around breakpoints

    Returns
    -------
    tuple or None
        (chrom, start, is_reverse, has_left_clip, has_right_clip,
         start_region_start, start_region_end, end_region_start, end_region_end)
        or None if read is not valid for checking
    """
    primary_chrom = read.reference_name
    primary_start = read.reference_start
    primary_is_reverse = read.is_reverse

    if primary_chrom is None or primary_start is None or read.cigartuples is None:
        raise RuntimeError(f"Corrupt read {read.query_name} with no CIGAR tuples found")

    # Determine if primary is first or second part based on soft clipping
    primary_has_left_clip = has_left_soft_clip(read.cigartuples)
    primary_has_right_clip = has_right_soft_clip(read.cigartuples)

    # Check if primary alignment start is near interval breakpoints
    start_region_start = interval_start - cushion
    start_region_end = interval_start + cushion
    end_region_start = interval_end - cushion
    end_region_end = interval_end + cushion

    primary_near_start = start_region_start <= primary_start <= start_region_end
    primary_near_end = end_region_start <= primary_start <= end_region_end

    if not (primary_near_start or primary_near_end):
        return None

    return (
        primary_chrom,
        primary_start,
        primary_is_reverse,
        primary_has_left_clip,
        primary_has_right_clip,
        start_region_start,
        start_region_end,
        end_region_start,
        end_region_end,
    )


def check_read_cnv_consistency(
    read: pysam.AlignedSegment,
    interval_start: int,
    interval_end: int,
    cushion: int,
    supplementary_alns: list[tuple[str, int, bool, bool, bool]],
) -> tuple[bool, bool]:
    """
    Check if a split read is consistent with duplication or deletion at this interval.

    For duplications: the first part FOLLOWS the second part on reference (positions reversed).
    For deletions: the first part PRECEDES the second part on reference (positions in order).
    Both parts must be in the same direction (both forward or both reverse).

    First part = alignment with right soft clip (end of read sequence)
    Second part = alignment with left soft clip (start of read sequence)

    Parameters
    ----------
    read : pysam.AlignedSegment
        The primary alignment of the read
    interval_start : int
        Start position of the interval (0-based)
    interval_end : int
        End position of the interval (0-based)
    cushion : int
        Number of bases to extend search around breakpoints
    supplementary_alns : list of tuples
        List of (chrom, pos, has_left_soft_clip, has_right_soft_clip, is_reverse) for supplementary alignments

    Returns
    -------
    tuple[bool, bool]
        (is_duplication, is_deletion)
    """
    if not supplementary_alns:
        return False, False

    # Prepare and validate read information
    prep_result = _prepare_read_for_cnv_check(read, interval_start, interval_end, cushion)
    if prep_result is None:
        return False, False

    (
        primary_chrom,
        primary_start,
        primary_is_reverse,
        primary_has_left_clip,
        primary_has_right_clip,
        start_region_start,
        start_region_end,
        end_region_start,
        end_region_end,
    ) = prep_result

    # Check supplementary alignments
    for supp_chrom, supp_start, supp_left_clip, supp_right_clip, supp_is_reverse in supplementary_alns:
        # Must be on same chromosome and same strand
        primary_near_start = start_region_start <= primary_start <= start_region_end
        primary_near_end = end_region_start <= primary_start <= end_region_end
        supp_near_start = start_region_start <= supp_start <= start_region_end
        supp_near_end = end_region_start <= supp_start <= end_region_end
        if supp_chrom != primary_chrom or supp_is_reverse != primary_is_reverse:
            continue

        if not (supp_near_start or supp_near_end):
            continue

        # The two alignments should be on opposite breakpoints
        if (primary_near_start and not supp_near_end) or (primary_near_end and not supp_near_start):
            continue

        # Determine which is first part (right clip) and which is second part (left clip)
        is_dup = _alignment_consistent_with_dup(
            primary_start,
            supp_start,
            primary_has_left_clip,
            primary_has_right_clip,
            supp_left_clip,
            supp_right_clip,
        )
        is_del = _alignment_consistent_with_del(
            primary_start,
            supp_start,
            primary_has_left_clip,
            primary_has_right_clip,
            supp_left_clip,
            supp_right_clip,
        )
        if is_dup or is_del:
            return is_dup, is_del
    return False, False


def _alignment_consistent_with_dup(
    primary_start, supp_start, primary_left_clip, primary_right_clip, supp_left_clip, supp_right_clip
) -> bool:
    # Case 1: Primary has right clip (first part), supplementary has left clip (second part)
    if primary_right_clip and supp_left_clip:
        if primary_start > supp_start:
            return True

    # Case 2: Primary has left clip (second part), supplementary has right clip (first part)
    if primary_left_clip and supp_right_clip:
        if supp_start > primary_start:
            return True  # Duplication: first part AFTER second part

    return False


def _alignment_consistent_with_del(
    primary_start, supp_start, primary_left_clip, primary_right_clip, supp_left_clip, supp_right_clip
) -> bool:
    # Case 1: Primary has right clip (first part), supplementary has left clip (second part)
    if primary_right_clip and supp_left_clip:
        if primary_start < supp_start:
            return True

    # Case 2: Primary has left clip (second part), supplementary has right clip (first part)
    if primary_left_clip and supp_right_clip:
        if supp_start < primary_start:
            return True  # Duplication: first part AFTER second part

    return False


def analyze_interval_breakpoints(
    alignment_file: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    cushion: int,
) -> BreakpointEvidence:
    """
    Analyze reads at interval breakpoints for CNV evidence.

    Parameters
    ----------
    alignment_file : pysam.AlignmentFile
        Open BAM/CRAM file
    chrom : str
        Chromosome name
    start : int
        Interval start position (0-based)
    end : int
        Interval end position (0-based)
    cushion : int
        Number of bases to extend search around breakpoints

    Returns
    -------
    BreakpointEvidence
        Counts of reads supporting duplication and deletion
    """
    duplication_reads = 0
    deletion_reads = 0

    # Track reads we've already processed to avoid double counting
    processed_reads = set()

    # Define the two breakpoint regions
    start_region_start = max(0, start - cushion)
    start_region_end = start + cushion
    end_region_start = end - cushion
    end_region_end = end + cushion

    # Fetch reads from the combined region covering both breakpoints
    fetch_start = start_region_start
    fetch_end = end_region_end

    try:
        for read in alignment_file.fetch(chrom, fetch_start, fetch_end):
            # Skip unmapped, secondary, supplementary, duplicate reads
            if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate:
                continue

            # Skip if we already processed this read
            if read.query_name in processed_reads:
                continue

            read_start = read.reference_start
            if read_start is None:
                continue

            # Check if read start is near either breakpoint
            near_start = start_region_start <= read_start <= start_region_end
            near_end = end_region_start <= read_start <= end_region_end

            if not (near_start or near_end):
                continue

            # Add to processed set
            processed_reads.add(read.query_name)

            # Get supplementary alignments for this read
            supplementary_alns = get_supplementary_alignments(alignment_file, read)

            # Check for duplication and deletion evidence
            is_dup, is_del = check_read_cnv_consistency(read, start, end, cushion, supplementary_alns)
            if is_dup:
                duplication_reads += 1
            elif is_del:
                deletion_reads += 1

    except Exception as e:
        logger.warning(f"Error fetching reads for {chrom}:{start}-{end}: {e}")

    total_reads = len(processed_reads)

    return BreakpointEvidence(
        chrom=chrom,
        start=start,
        end=end,
        duplication_reads=duplication_reads,
        deletion_reads=deletion_reads,
        total_reads=total_reads,
    )


def analyze_cnv_breakpoints(
    bam_file: str,
    vcf_file: str,
    cushion: int = 100,
    output_file: str | None = None,
    reference_fasta: str | None = None,
) -> None:
    """
    Analyze all CNV intervals in a VCF file for breakpoint evidence.

    Parameters
    ----------
    bam_file : str
        Path to BAM or CRAM file
    vcf_file : str
        Path to VCF file with CNV variants
    cushion : int, optional
        Number of bases to extend search around breakpoints (default: 100)
    output_file : str, optional
        Path to output VCF file (default: None, writes to stdout)
    reference_fasta : str, optional
        Path to reference FASTA file (required for CRAM files)
    """
    # Determine file mode based on extension
    file_ext = Path(bam_file).suffix.lower()
    if file_ext == ".cram":
        if reference_fasta is None:
            raise ValueError("Reference FASTA is required for CRAM files")
        mode = "rc"
        alignment_file = pysam.AlignmentFile(bam_file, mode, reference_filename=reference_fasta)
    elif file_ext == ".bam":
        mode = "rb"
        alignment_file = pysam.AlignmentFile(bam_file, mode)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only .bam and .cram are supported.")

    # Open input VCF and add new INFO fields to header
    with pysam.VariantFile(vcf_file) as vcf_in:
        hdr = vcf_in.header
        hdr.info.add("CNV_DUP_READS", "1", "Integer", "Number of reads supporting duplication at breakpoints")
        hdr.info.add("CNV_DEL_READS", "1", "Integer", "Number of reads supporting deletion at breakpoints")
        hdr.info.add("CNV_TOTAL_READS", "1", "Integer", "Total reads analyzed at breakpoints")
        hdr.info.add("CNV_DUP_FRAC", "1", "Float", "Fraction of reads supporting duplication")
        hdr.info.add("CNV_DEL_FRAC", "1", "Float", "Fraction of reads supporting deletion")

        # Open output VCF with modified header
        if output_file:
            vcf_out = pysam.VariantFile(output_file, "w", header=hdr)
        else:
            vcf_out = pysam.VariantFile("-", "w", header=hdr)

        # Process each variant
        variant_count = 0
        for record in vcf_in:
            variant_count += 1
            if variant_count % 100 == 0:
                logger.info(f"Processing variant {variant_count}: {record.chrom}:{record.start}-{record.stop}")

            # Analyze breakpoints for this variant
            evidence = analyze_interval_breakpoints(alignment_file, record.chrom, record.start, record.stop, cushion)

            # Add new INFO fields directly to the record
            record.info["CNV_DUP_READS"] = evidence.duplication_reads
            record.info["CNV_DEL_READS"] = evidence.deletion_reads
            record.info["CNV_TOTAL_READS"] = evidence.total_reads

            if evidence.total_reads > 0:
                record.info["CNV_DUP_FRAC"] = evidence.duplication_reads / evidence.total_reads
                record.info["CNV_DEL_FRAC"] = evidence.deletion_reads / evidence.total_reads
            else:
                record.info["CNV_DUP_FRAC"] = 0.0
                record.info["CNV_DEL_FRAC"] = 0.0

            # Write annotated record
            vcf_out.write(record)

        vcf_out.close()

    # Close alignment file
    alignment_file.close()

    logger.info(f"Processed {variant_count} variants")
    if output_file:
        logger.info(f"Annotated VCF written to {output_file}")


def get_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Create or populate argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser, optional
        Existing parser to add arguments to. If None, creates a new parser.
        This allows reusing the argument definitions in subparser contexts.

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
        "--bam-file",
        required=True,
        help="Path to BAM or CRAM file with single-ended reads",
    )
    parser.add_argument(
        "--vcf-file",
        required=True,
        help="Path to VCF file with CNV variants to analyze",
    )
    parser.add_argument(
        "--cushion",
        type=int,
        default=100,
        help="Number of bases to extend search around breakpoints (default: 100)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to output VCF file (default: stdout)",
    )
    parser.add_argument(
        "--reference-fasta",
        default=None,
        help="Path to reference FASTA file (required for CRAM files)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args(argv)

    try:
        analyze_cnv_breakpoints(
            bam_file=args.bam_file,
            vcf_file=args.vcf_file,
            cushion=args.cushion,
            output_file=args.output_file,
            reference_fasta=args.reference_fasta,
        )
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
