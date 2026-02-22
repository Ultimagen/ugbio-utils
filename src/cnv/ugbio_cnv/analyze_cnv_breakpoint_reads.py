"""
Analyze reads at CNV breakpoints for duplication and deletion evidence.

This script analyzes single-ended reads at CNV breakpoints to identify
supporting evidence for duplications and deletions based on read orientation
and position patterns. It takes a VCF file as input and outputs an annotated
VCF with breakpoint evidence in INFO fields.
"""

import argparse
import sys
from dataclasses import dataclass, field
from statistics import median

import pysam
from ugbio_cnv.jalign import create_bam_header
from ugbio_core.dna_sequence_utils import CIGAR_OPS, get_reference_alignment_end, parse_cigar_string
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
    dup_insert_sizes: list[int] = field(default_factory=list)
    del_insert_sizes: list[int] = field(default_factory=list)
    supporting_reads: list[tuple[pysam.AlignedSegment, str]] = field(default_factory=list)
    supplementary_reads: dict[str, list[pysam.AlignedSegment]] = field(default_factory=dict)

    @property
    def dup_median_insert_size(self) -> float | None:
        """Calculate median insert size for duplication-supporting reads."""
        if len(self.dup_insert_sizes) < 1:
            return None
        return median(self.dup_insert_sizes)

    @property
    def del_median_insert_size(self) -> float | None:
        """Calculate median insert size for deletion-supporting reads."""
        if len(self.del_insert_sizes) < 1:
            return None
        return median(self.del_insert_sizes)


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
) -> list[tuple[str, int, int, bool, bool, bool]]:
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
        List of (chrom, pos, end, has_left_soft_clip, has_right_soft_clip, is_reverse) for supplementary alignments
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
        end = get_reference_alignment_end(pos, cigar_str)
        supplementary_alns.append((chrom, pos, end, left_soft_clip, right_soft_clip, is_reverse))

    return supplementary_alns


def _prepare_read_for_cnv_check(
    read: pysam.AlignedSegment,
    interval_start: int,
    interval_end: int,
    cushion: int,
) -> tuple[str, int, int, bool, bool, bool, int, int, int, int] | None:
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
        (chrom, start, end, is_reverse, has_left_clip, has_right_clip,
         start_region_start, start_region_end, end_region_start, end_region_end)
        or None if read is not valid for checking
    """
    primary_chrom = read.reference_name
    primary_start = read.reference_start
    if read.reference_end is None:
        raise RuntimeError(f"Corrupt read {read.query_name} with no reference end found")
    primary_end = read.reference_end
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
        primary_end,
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
    supplementary_alns: list[tuple[str, int, int, bool, bool, bool]],
) -> tuple[bool, bool, int | None]:
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
    tuple[bool, bool, int | None]
        (is_duplication, is_deletion, insert_size)
    """
    if not supplementary_alns:
        return False, False, None
    # Prepare and validate read information
    prep_result = _prepare_read_for_cnv_check(read, interval_start, interval_end, cushion)
    if prep_result is None:
        return False, False, None

    (
        primary_chrom,
        primary_start,
        primary_end,
        primary_is_reverse,
        primary_has_left_clip,
        primary_has_right_clip,
        start_region_start,
        start_region_end,
        end_region_start,
        end_region_end,
    ) = prep_result

    # Check supplementary alignments
    for supp_chrom, supp_start, supp_end, supp_left_clip, supp_right_clip, supp_is_reverse in supplementary_alns:
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
        dup_insert_size = _alignment_consistent_with_dup(
            primary_start,
            primary_end,
            supp_start,
            supp_end,
            primary_has_left_clip,
            primary_has_right_clip,
            supp_left_clip,
            supp_right_clip,
        )
        del_insert_size = _alignment_consistent_with_del(
            primary_start,
            primary_end,
            supp_start,
            supp_end,
            primary_has_left_clip,
            primary_has_right_clip,
            supp_left_clip,
            supp_right_clip,
        )
        if dup_insert_size is not None:
            return True, False, dup_insert_size
        if del_insert_size is not None:
            return False, True, del_insert_size
    return False, False, None


def _alignment_consistent_with_dup(
    primary_start,
    primary_end,
    supp_start,
    supp_end,
    primary_left_clip,
    primary_right_clip,
    supp_left_clip,
    supp_right_clip,
) -> int | None:
    """Check if alignment is consistent with duplication and return insert size if so."""
    # Case 1: Primary has right clip (first part), supplementary has left clip (second part)
    if primary_right_clip and supp_left_clip:
        insert_size = primary_end - supp_start
        if primary_start > supp_end:
            return insert_size

    # Case 2: Primary has left clip (second part), supplementary has right clip (first part)
    if primary_left_clip and supp_right_clip:
        insert_size = supp_end - primary_start
        if supp_start > primary_end:
            return insert_size  # Duplication: first part AFTER second part

    return None


def _alignment_consistent_with_del(
    primary_start,
    primary_end,
    supp_start,
    supp_end,
    primary_left_clip,
    primary_right_clip,
    supp_left_clip,
    supp_right_clip,
) -> int | None:
    """Check if alignment is consistent with deletion and return insert size if so."""
    # Case 1: Primary has right clip (first part), supplementary has left clip (second part)
    if primary_right_clip and supp_left_clip:
        insert_size = supp_start - primary_end
        if primary_end < supp_start:
            return insert_size

    # Case 2: Primary has left clip (second part), supplementary has right clip (first part)
    if primary_left_clip and supp_right_clip:
        insert_size = primary_start - supp_end
        if supp_start < primary_start:
            return insert_size  # Deletion: first part BEFORE second part

    return None


def _should_skip_read(read: pysam.AlignedSegment) -> bool:
    """
    Check if a read should be skipped during breakpoint analysis.

    Skips unmapped, secondary, supplementary, and duplicate reads.

    Parameters
    ----------
    read : pysam.AlignedSegment
        The read to check

    Returns
    -------
    bool
        True if read should be skipped
    """
    return read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate


def _is_read_near_breakpoint(
    read_start: int,
    start_region_start: int,
    start_region_end: int,
    end_region_start: int,
    end_region_end: int,
) -> bool:
    """
    Check if read start position is near either breakpoint region.

    Parameters
    ----------
    read_start : int
        The reference start position of the read
    start_region_start : int
        Start of the interval start region
    start_region_end : int
        End of the interval start region
    end_region_start : int
        Start of the interval end region
    end_region_end : int
        End of the interval end region

    Returns
    -------
    bool
        True if read is near either breakpoint
    """
    near_start = start_region_start <= read_start <= start_region_end
    near_end = end_region_start <= read_start <= end_region_end
    return near_start or near_end


def _process_read_for_cnv_evidence(
    read: pysam.AlignedSegment,
    alignment_file: pysam.AlignmentFile,
    start: int,
    end: int,
    cushion: int,
) -> tuple[bool, bool, int | None]:
    """
    Process a single read to determine CNV evidence.

    Parameters
    ----------
    read : pysam.AlignedSegment
        The read to process
    alignment_file : pysam.AlignmentFile
        Open BAM/CRAM file for fetching supplementary alignments
    start : int
        Interval start position (0-based)
    end : int
        Interval end position (0-based)
    cushion : int
        Number of bases to extend search around breakpoints

    Returns
    -------
    tuple[bool, bool, int | None]
        (is_duplication, is_deletion, insert_size)
    """
    supplementary_alns = get_supplementary_alignments(alignment_file, read)
    return check_read_cnv_consistency(read, start, end, cushion, supplementary_alns)


def _calculate_breakpoint_regions(
    start: int,
    end: int,
    cushion: int,
) -> tuple[int, int, int, int]:
    """
    Calculate the breakpoint search regions.

    Parameters
    ----------
    start : int
        Interval start position (0-based)
    end : int
        Interval end position (0-based)
    cushion : int
        Number of bases to extend search around breakpoints

    Returns
    -------
    tuple[int, int, int, int]
        (start_region_start, start_region_end, end_region_start, end_region_end)
    """
    start_region_start = max(0, start - cushion)
    start_region_end = start + cushion
    end_region_start = max(0, end - cushion)
    end_region_end = end + cushion
    return start_region_start, start_region_end, end_region_start, end_region_end


def _annotate_vcf_record_with_evidence(record: pysam.VariantRecord, evidence: BreakpointEvidence) -> None:
    """
    Annotate a VCF record with CNV breakpoint evidence.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to annotate
    evidence : BreakpointEvidence
        Breakpoint evidence data to add to record
    """
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

    # Add insert size statistics (use 0.0 if None to ensure downstream processing)
    record.info["DUP_READS_MEDIAN_INSERT_SIZE"] = (
        evidence.dup_median_insert_size if evidence.dup_median_insert_size is not None else 0.0
    )
    record.info["DEL_READS_MEDIAN_INSERT_SIZE"] = (
        evidence.del_median_insert_size if evidence.del_median_insert_size is not None else 0.0
    )


def _process_primary_read_for_evidence(
    read: pysam.AlignedSegment,
    alignment_file: pysam.AlignmentFile,
    start: int,
    end: int,
    cushion: int,
    duplication_reads: int,
    deletion_reads: int,
    dup_insert_sizes: list[int],
    del_insert_sizes: list[int],
    supporting_reads: list[tuple[pysam.AlignedSegment, str]],
) -> tuple[int, int]:
    """
    Process a primary read for CNV evidence and update counters.

    Returns
    -------
    tuple[int, int]
        Updated (duplication_reads, deletion_reads) counts
    """
    is_dup, is_del, insert_size = _process_read_for_cnv_evidence(read, alignment_file, start, end, cushion)

    if is_dup:
        duplication_reads += 1
        if insert_size is not None and insert_size > 0:
            dup_insert_sizes.append(insert_size)
        supporting_reads.append((read, "DUP"))
    elif is_del:
        deletion_reads += 1
        if insert_size is not None and insert_size > 0:
            del_insert_sizes.append(insert_size)
        supporting_reads.append((read, "DEL"))

    return duplication_reads, deletion_reads


def _collect_reads_from_region(
    alignment_file: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    cushion: int,
    start_region_start: int,
    start_region_end: int,
    end_region_start: int,
    end_region_end: int,
) -> tuple[
    int,
    int,
    list[int],
    list[int],
    set[str],
    list[tuple[pysam.AlignedSegment, str]],
    dict[str, list[pysam.AlignedSegment]],
]:
    """
    Collect reads from breakpoint region and classify them as supporting duplication or deletion.

    Returns
    -------
    tuple
        (duplication_reads, deletion_reads, dup_insert_sizes, del_insert_sizes,
         processed_reads, supporting_reads, supplementary_reads)
    """
    duplication_reads = 0
    deletion_reads = 0
    dup_insert_sizes: list[int] = []
    del_insert_sizes: list[int] = []
    processed_reads: set[str] = set()
    supporting_reads: list[tuple[pysam.AlignedSegment, str]] = []
    supplementary_reads: dict[str, list[pysam.AlignedSegment]] = {}

    # Process ONLY the two breakpoint regions, not the entire interval
    # This dramatically improves performance for large CNVs
    regions_to_fetch = [
        (start_region_start, start_region_end),  # Start breakpoint
        (end_region_start, end_region_end),  # End breakpoint
    ]

    try:
        for region_start, region_end in regions_to_fetch:
            for read in alignment_file.fetch(chrom, region_start, region_end):
                # Skip supplementary reads in this phase - they'll be collected later
                if read.is_supplementary:
                    continue

                if _should_skip_read(read) or read.query_name in processed_reads:
                    continue

                read_start = read.reference_start

                if not _is_read_near_breakpoint(
                    read_start, start_region_start, start_region_end, end_region_start, end_region_end
                ):
                    continue

                processed_reads.add(str(read.query_name))

                duplication_reads, deletion_reads = _process_primary_read_for_evidence(
                    read,
                    alignment_file,
                    start,
                    end,
                    cushion,
                    duplication_reads,
                    deletion_reads,
                    dup_insert_sizes,
                    del_insert_sizes,
                    supporting_reads,
                )

    except Exception as e:
        logger.warning(f"Error fetching reads for {chrom}:{start}-{end}: {e}")

    return (
        duplication_reads,
        deletion_reads,
        dup_insert_sizes,
        del_insert_sizes,
        processed_reads,
        supporting_reads,
        supplementary_reads,
    )


def _collect_supplementary_alignments_for_supporting_reads(
    alignment_file: pysam.AlignmentFile,
    chrom: str,
    supporting_reads: list[tuple[pysam.AlignedSegment, str]],
    start_region_start: int,
    end_region_end: int,
) -> dict[str, list[pysam.AlignedSegment]]:
    """
    Collect supplementary alignments ONLY for reads that are supporting CNV evidence.

    This is much more efficient than collecting all supplementary reads in the interval,
    since typically only 10-100 reads support CNV out of 1M+ total reads.

    Parameters
    ----------
    alignment_file : pysam.AlignmentFile
        Open BAM/CRAM file
    chrom : str
        Chromosome name
    supporting_reads : list[tuple[pysam.AlignedSegment, str]]
        List of (read, read_group) tuples for supporting reads
    start_region_start : int
        Start of search region
    end_region_end : int
        End of search region

    Returns
    -------
    dict[str, list[pysam.AlignedSegment]]
        Dictionary mapping query_name to list of supplementary alignments
    """
    supplementary_reads: dict[str, list[pysam.AlignedSegment]] = {}

    # Create set of query names we need supplementary alignments for
    supporting_query_names = {str(read.query_name) for read, _ in supporting_reads}

    if not supporting_query_names:
        return supplementary_reads

    try:
        # Fetch supplementary alignments from the breakpoint regions
        for read in alignment_file.fetch(chrom, start_region_start, end_region_end):
            if not read.is_supplementary:
                continue

            query_name = str(read.query_name)
            if query_name in supporting_query_names:
                if query_name not in supplementary_reads:
                    supplementary_reads[query_name] = []
                supplementary_reads[query_name].append(read)

    except Exception as e:
        logger.warning(f"Error fetching supplementary reads: {e}")

    return supplementary_reads


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
    # Calculate breakpoint regions
    regions = _calculate_breakpoint_regions(start, end, cushion)
    start_region_start, start_region_end, end_region_start, end_region_end = regions

    # PHASE 1: Collect primary reads from the two breakpoint regions
    (
        duplication_reads,
        deletion_reads,
        dup_insert_sizes,
        del_insert_sizes,
        processed_reads,
        supporting_reads,
        _,  # supplementary_reads not populated in phase 1
    ) = _collect_reads_from_region(
        alignment_file,
        chrom,
        start,
        end,
        cushion,
        start_region_start,
        start_region_end,
        end_region_start,
        end_region_end,
    )

    # PHASE 2: Collect supplementary alignments ONLY for supporting reads
    # This is dramatically more efficient than collecting all supplementary reads
    supplementary_reads = _collect_supplementary_alignments_for_supporting_reads(
        alignment_file,
        chrom,
        supporting_reads,
        start_region_start,
        end_region_end,
    )

    return BreakpointEvidence(
        chrom=chrom,
        start=start,
        end=end,
        duplication_reads=duplication_reads,
        deletion_reads=deletion_reads,
        total_reads=len(processed_reads),
        dup_insert_sizes=dup_insert_sizes,
        del_insert_sizes=del_insert_sizes,
        supporting_reads=supporting_reads,
        supplementary_reads=supplementary_reads,
    )


def _write_supporting_reads_to_bam(
    bam_out: pysam.AlignmentFile,
    evidence: BreakpointEvidence,
) -> None:
    """
    Write supporting reads and their supplementary alignments to output BAM.

    Parameters
    ----------
    bam_out : pysam.AlignmentFile
        Output BAM file
    evidence : BreakpointEvidence
        Evidence containing supporting reads and supplementary alignments
    """
    for read, read_group in evidence.supporting_reads:
        # Set read group tag on the primary read
        read.set_tag("RG", read_group, value_type="Z")
        bam_out.write(read)

        # Write corresponding supplementary alignments with same read group
        for supp_read in evidence.supplementary_reads.get(str(read.query_name), []):
            supp_read.set_tag("RG", read_group, value_type="Z")
            bam_out.write(supp_read)


def _process_variants(
    vcf_in: pysam.VariantFile,
    vcf_out: pysam.VariantFile,
    alignment_file: pysam.AlignmentFile,
    cushion: int,
    bam_out: pysam.AlignmentFile | None,
) -> int:
    """
    Process all variants in input VCF and write annotated results.

    Parameters
    ----------
    vcf_in : pysam.VariantFile
        Input VCF file
    vcf_out : pysam.VariantFile
        Output VCF file
    alignment_file : pysam.AlignmentFile
        Alignment file for reading reads
    cushion : int
        Number of bases to extend search around breakpoints
    bam_out : pysam.AlignmentFile | None
        Optional output BAM file for supporting reads

    Returns
    -------
    int
        Number of variants processed
    """
    variant_count = 0
    for record in vcf_in:
        variant_count += 1
        if variant_count % 100 == 0:
            logger.info(f"Processing variant {variant_count}: {record.chrom}:{record.start}-{record.stop}")

        # Analyze breakpoints for this variant
        evidence = analyze_interval_breakpoints(alignment_file, record.chrom, record.start, record.stop, cushion)

        # Annotate VCF record with evidence
        _annotate_vcf_record_with_evidence(record, evidence)

        # Write annotated record
        vcf_out.write(record)

        # Write supporting reads to BAM if requested
        if bam_out:
            _write_supporting_reads_to_bam(bam_out, evidence)

    return variant_count


def analyze_cnv_breakpoints(
    bam_file: str,
    vcf_file: str,
    reference_fasta: str,
    cushion: int = 100,
    output_file: str | None = None,
    output_bam: str | None = None,
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
    reference_fasta : str
        Path to reference FASTA file (required for CRAM files)
    output_bam : str, optional
        Path to output BAM file with split reads supporting CNV calls (default: None, no BAM output)
    """
    alignment_file = pysam.AlignmentFile(bam_file, "r", reference_filename=reference_fasta)

    # Open input VCF and add new INFO fields to header
    with pysam.VariantFile(vcf_file) as vcf_in:
        hdr = vcf_in.header
        hdr.info.add("CNV_DUP_READS", "1", "Integer", "Number of reads supporting duplication at breakpoints")
        hdr.info.add("CNV_DEL_READS", "1", "Integer", "Number of reads supporting deletion at breakpoints")
        hdr.info.add("CNV_TOTAL_READS", "1", "Integer", "Total reads analyzed at breakpoints")
        hdr.info.add("CNV_DUP_FRAC", "1", "Float", "Fraction of reads supporting duplication")
        hdr.info.add("CNV_DEL_FRAC", "1", "Float", "Fraction of reads supporting deletion")
        hdr.info.add("DUP_READS_MEDIAN_INSERT_SIZE", "1", "Float", "Median insert size of duplication-supporting reads")
        hdr.info.add("DEL_READS_MEDIAN_INSERT_SIZE", "1", "Float", "Median insert size of deletion-supporting reads")

        # Open output VCF with modified header
        if output_file:
            vcf_out = pysam.VariantFile(output_file, "w", header=hdr)
        else:
            vcf_out = pysam.VariantFile("-", "w", header=hdr)

        # Open output BAM if requested
        bam_out = None
        if output_bam:
            bam_header = create_bam_header(alignment_file.header)
            bam_out = pysam.AlignmentFile(output_bam, "wb", header=bam_header)

        # Process all variants
        variant_count = _process_variants(vcf_in, vcf_out, alignment_file, cushion, bam_out)

        vcf_out.close()
        if bam_out:
            bam_out.close()

    # Close alignment file
    alignment_file.close()

    logger.info(f"Processed {variant_count} variants")
    if output_file:
        logger.info(f"Annotated VCF written to {output_file}")
    if output_bam:
        logger.info(f"Split reads BAM written to {output_bam}")


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
        required=True,
        help="Path to reference FASTA file",
    )
    parser.add_argument(
        "--output-bam",
        default=None,
        help="Path to output BAM file with split reads supporting CNV calls (default: None, no BAM output)",
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
            reference_fasta=args.reference_fasta,
            cushion=args.cushion,
            output_file=args.output_file,
            output_bam=args.output_bam,
        )
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
