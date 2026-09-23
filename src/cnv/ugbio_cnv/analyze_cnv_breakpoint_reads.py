"""
Analyze reads at CNV breakpoints for duplication and deletion evidence.

This script analyzes reads at CNV breakpoints to identify supporting evidence for
duplications and deletions based on read orientation and position patterns. It takes
a VCF file as input and outputs an annotated VCF with breakpoint evidence in INFO fields.

With --paired-end, discordant read pairs are counted too, folded into the same INFO fields: one vote
per fragment, split outranking pair. Pair mates go to the evidence BAM under read group PAIR.
"""

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from statistics import median

import pysam
from ugbio_cnv.jalign import create_bam_header
from ugbio_core.dna_sequence_utils import CIGAR_OPS, get_reference_alignment_end, parse_cigar_string
from ugbio_core.logger import logger

# CIGAR operation constants
CIGAR_SOFT_CLIP = CIGAR_OPS["S"]

# Flag bits identifying which mate of a pair an alignment is (FREAD1 | FREAD2); 0 for unpaired reads
MATE_FLAG_MASK = 0xC0
# Shortest span (template length) a discordant pair may have
MIN_PAIR_SPAN = 800
MIN_PAIR_MAPPING_QUALITY = 20
# Evidence BAM read group for discordant pair mates
PAIR_READ_GROUP = "PAIR"


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
    supplementary_reads: dict[tuple[str, int], list[pysam.AlignedSegment]] = field(default_factory=dict)

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


@dataclass(frozen=True)
class PairedEndConfig:
    """Configuration for paired-end (discordant read pair) breakpoint evidence."""

    enabled: bool = False
    min_pair_span: int = MIN_PAIR_SPAN
    min_pair_mapping_quality: int = MIN_PAIR_MAPPING_QUALITY

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PairedEndConfig":
        """Build a config from parsed CLI arguments."""
        return cls(
            enabled=args.paired_end,
            min_pair_span=args.min_pair_span,
            min_pair_mapping_quality=args.min_pair_mapping_quality,
        )


@dataclass
class _EvidenceVotes:
    """
    DUP/DEL votes cast by one fragment's alignments from one kind of evidence (split or pair).

    Attributes
    ----------
    labels : set[str]
        CNV types voted for ("DUP"/"DEL"); the fragment votes only if exactly one
    sizes : list[int]
        Event lengths measured by the voting alignments: breakpoint distance for split reads, span for pairs
    alignments : list[pysam.AlignedSegment]
        The voting alignments, written to the evidence BAM if this evidence wins the fragment's vote
    """

    labels: set[str] = field(default_factory=set)
    sizes: list[int] = field(default_factory=list)
    alignments: list[pysam.AlignedSegment] = field(default_factory=list)

    def add(self, read: pysam.AlignedSegment, label: str, size: int | None) -> None:
        """Record one alignment's vote and, if positive, its event length."""
        self.labels.add(label)
        if size is not None and size > 0:
            self.sizes.append(size)
        self.alignments.append(read)


@dataclass
class _FragmentEvidence:
    """Accumulated DUP/DEL votes for a single sequencing fragment."""

    split: _EvidenceVotes = field(default_factory=_EvidenceVotes)
    pair: _EvidenceVotes = field(default_factory=_EvidenceVotes)


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
    read: pysam.AlignedSegment,
) -> list[tuple[str, int, int, bool, bool, bool]]:
    """
    Get supplementary alignments for a read from the SA tag.

    Parameters
    ----------
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


def _get_read_group(read: pysam.AlignedSegment) -> str:
    """Return the read's RG tag value, or "UNKNOWN" when the tag is absent."""
    return read.get_tag("RG") if read.has_tag("RG") else "UNKNOWN"


def _mate_aware_key(read: pysam.AlignedSegment) -> tuple[str, int]:
    """
    Build a (query_name, mate flag bits) key identifying one mate of one fragment.

    Both mates share a query name, so the flag bits are needed to tell them apart. They are zero
    for unpaired reads, making the key equivalent to the query name alone.
    """
    return (str(read.query_name), read.flag & MATE_FLAG_MASK)


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
    supplementary_alns = get_supplementary_alignments(read)
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


def _pair_is_analyzable(read: pysam.AlignedSegment, pe_config: PairedEndConfig) -> bool:
    """
    Check whether a read's pair can be used as breakpoint evidence.

    is_proper_pair is not required, since discordant pairs are never proper. The mapping quality floor applies
    to this alignment only: the mate's quality would need the MQ tag, which Ultima pipelines do not emit.
    """
    return (
        read.is_paired
        and not read.mate_is_unmapped
        and read.next_reference_id == read.reference_id
        and read.template_length != 0
        and read.reference_start != read.next_reference_start  # mates must be orderable to read orientation
        and read.mapping_quality >= pe_config.min_pair_mapping_quality
    )


def _pair_brackets_breakpoints(
    read: pysam.AlignedSegment,
    interval_start: int,
    interval_end: int,
    cushion: int,
) -> bool:
    """
    Check that the pair starts in the start window and its implied right end lands in the end window.

    This confines the span to interval_length +/- 2*cushion, so the span floor only bites on short intervals.
    It constrains the rightmost mate's *end* (from TLEN), not its start.
    """
    left_start = min(read.reference_start, read.next_reference_start)
    right_end = left_start + abs(read.template_length)
    start_lo, start_hi, end_lo, end_hi = _calculate_breakpoint_regions(interval_start, interval_end, cushion)

    return start_lo <= left_start <= start_hi and end_lo <= right_end <= end_hi


def _pair_mates_flank_deleted_segment(
    read: pysam.AlignedSegment,
    interval_start: int,
    interval_end: int,
    cushion: int,
) -> bool:
    """
    Check that both mates lie outside the deleted segment, whose bases are absent from the sample.

    Bracketing pins the rightmost mate's end, not its start. Starts only: mate ends need the absent
    MC tag. Deletion-only - an everted pair has both mates inside the duplication by construction.
    """
    interior_start = interval_start + cushion
    interior_end = interval_end - cushion
    if interior_start >= interior_end:  # Windows cover the interval; no interior to be inside of
        return True

    # Non-strict: a mate starting exactly on a window edge is still inside that window
    return (
        min(read.reference_start, read.next_reference_start) <= interior_start
        and max(read.reference_start, read.next_reference_start) >= interior_end
    )


def check_pair_cnv_consistency(
    read: pysam.AlignedSegment,
    interval_start: int,
    interval_end: int,
    cushion: int,
    pe_config: PairedEndConfig,
) -> tuple[bool, bool, int | None]:
    """
    Check if a read's pair is discordant consistently with duplication or deletion.

    It must bracket both breakpoints and span >= min_pair_span. Deletions read FR with both mates outside the
    deleted segment; duplications read RF ("everted"). Returns (is_dup, is_del, span).
    """
    if not _pair_is_analyzable(read, pe_config):
        return False, False, None

    # Same-strand mates are inversion-like, not DUP/DEL, mirroring the split-read path
    if read.is_reverse == read.mate_is_reverse:
        return False, False, None

    # One span floor for both classes. For FR pairs it sits above the span tail of ordinary pairs; for
    # everted pairs it is only a smallest believable duplication, since background eversions are
    # 0.04% of pairs on production data with spans in the hundreds of kb
    span = abs(read.template_length)
    if span < pe_config.min_pair_span or not _pair_brackets_breakpoints(read, interval_start, interval_end, cushion):
        return False, False, None

    leftmost_is_reverse = read.is_reverse if read.reference_start < read.next_reference_start else read.mate_is_reverse
    if leftmost_is_reverse:
        return True, False, span

    if not _pair_mates_flank_deleted_segment(read, interval_start, interval_end, cushion):
        return False, False, None
    # The span overstates the deleted length by about one insert size; split estimates take precedence
    return False, True, span


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
    set[tuple[str, str]],
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
    processed_reads: set[tuple[str, str]] = set()  # Track (read_name, RG) pairs
    supporting_reads: list[tuple[pysam.AlignedSegment, str]] = []
    supplementary_reads: dict[str, list[pysam.AlignedSegment]] = {}

    try:
        for read in itertools.chain(
            alignment_file.fetch(chrom, start_region_start, start_region_end, multiple_iterators=True),
            alignment_file.fetch(chrom, end_region_start, end_region_end, multiple_iterators=True),
        ):
            # Supplementary reads are skipped in this phase - they're collected later
            if _should_skip_read(read):
                continue

            # Get RG tag BEFORE deduplication check
            rg = _get_read_group(read)

            # Track (read_name, RG) pair for proper deduplication
            read_rg_key = (read.query_name, rg)
            if read_rg_key in processed_reads:
                continue
            processed_reads.add(read_rg_key)

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


def _record_fragment_evidence(
    read: pysam.AlignedSegment,
    fragment: _FragmentEvidence,
    alignment_file: pysam.AlignmentFile,
    interval: tuple[int, int, int],
    pe_config: PairedEndConfig,
) -> None:
    """Add one alignment's split-read and discordant-pair evidence to its fragment's bucket."""
    start, end, cushion = interval

    is_dup, is_del, size = _process_read_for_cnv_evidence(read, alignment_file, start, end, cushion)
    votes = fragment.split
    if not (is_dup or is_del):  # Split evidence outranks pair evidence, so the pair is tested only without it
        is_dup, is_del, size = check_pair_cnv_consistency(read, start, end, cushion, pe_config)
        votes = fragment.pair
    if is_dup or is_del:
        votes.add(read, "DUP" if is_dup else "DEL", size)


def _summarize_fragments(
    fragments: dict[tuple[str, str], _FragmentEvidence],
    interval: tuple[str, int, int],
) -> BreakpointEvidence:
    """
    Reduce per-fragment votes (keyed on (query_name, RG)) to interval-level evidence counts.

    Each fragment casts at most one DUP/DEL vote, split outranking pair; one whose winning votes disagree
    counts toward the total only. The winning votes' alignments become the supporting reads.
    """
    chrom, start, end = interval
    counts = {"DUP": 0, "DEL": 0}
    split_sizes: dict[str, list[int]] = {"DUP": [], "DEL": []}
    pair_sizes: dict[str, list[int]] = {"DUP": [], "DEL": []}
    supporting_reads: list[tuple[pysam.AlignedSegment, str]] = []

    for fragment in fragments.values():
        from_split = bool(fragment.split.labels)
        votes = fragment.split if from_split else fragment.pair
        if len(votes.labels) != 1:
            continue
        (label,) = votes.labels

        counts[label] += 1
        if votes.sizes:
            # One observation per fragment, so a multi-vote fragment cannot outweigh a single-vote one
            (split_sizes if from_split else pair_sizes)[label].append(round(median(votes.sizes)))
        read_group = label if from_split else PAIR_READ_GROUP
        supporting_reads.extend((read, read_group) for read in votes.alignments)

    return BreakpointEvidence(
        chrom=chrom,
        start=start,
        end=end,
        duplication_reads=counts["DUP"],
        deletion_reads=counts["DEL"],
        total_reads=len(fragments),
        # Split values are base-pair exact, so pair values are used only where there are none
        dup_insert_sizes=split_sizes["DUP"] or pair_sizes["DUP"],
        del_insert_sizes=split_sizes["DEL"] or pair_sizes["DEL"],
        supporting_reads=supporting_reads,
    )


def _collect_fragments_from_region(
    alignment_file: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    cushion: int,
    regions: tuple[int, int, int, int],
    pe_config: PairedEndConfig,
) -> BreakpointEvidence:
    """
    Collect fragment-level CNV evidence from the two breakpoint regions given by `regions`.

    Unlike _collect_reads_from_region, both mates are evaluated (deduplication is per alignment);
    _summarize_fragments then reduces the votes to one per fragment.
    """
    start_region_start, start_region_end, end_region_start, end_region_end = regions
    fragments: dict[tuple[str, str], _FragmentEvidence] = {}
    processed_alignments: set[tuple[str, str, int]] = set()

    try:
        for read in itertools.chain(
            alignment_file.fetch(chrom, start_region_start, start_region_end, multiple_iterators=True),
            alignment_file.fetch(chrom, end_region_start, end_region_end, multiple_iterators=True),
        ):
            if _should_skip_read(read):
                continue

            rg = _get_read_group(read)

            # Deduplicate per alignment, not per fragment: the breakpoint windows overlap
            # for intervals shorter than 2*cushion, so the same alignment can be fetched
            # twice, but the two mates of a fragment must both be evaluated.
            alignment_key = (str(read.query_name), rg, read.flag)
            if alignment_key in processed_alignments:
                continue
            processed_alignments.add(alignment_key)

            fragment = fragments.setdefault((str(read.query_name), rg), _FragmentEvidence())
            _record_fragment_evidence(read, fragment, alignment_file, (start, end, cushion), pe_config)

    except Exception as e:
        logger.warning(f"Error fetching reads for {chrom}:{start}-{end}: {e}")

    return _summarize_fragments(fragments, (chrom, start, end))


def _collect_supplementary_alignments_for_supporting_reads(
    alignment_file: pysam.AlignmentFile,
    chrom: str,
    supporting_reads: list[tuple[pysam.AlignedSegment, str]],
    start_region_start: int,
    start_region_end: int,
    end_region_start: int,
    end_region_end: int,
) -> dict[tuple[str, int], list[pysam.AlignedSegment]]:
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
    start_region_end: int
        End of start breakpoint search region
    end_region_start : int
        Start of end breakpoint search region
    end_region_end : int
        End of search region

    Returns
    -------
    dict[tuple[str, int], list[pysam.AlignedSegment]]
        Dictionary mapping (query_name, mate flag bits) to list of supplementary alignments
    """
    supplementary_reads: dict[tuple[str, int], list[pysam.AlignedSegment]] = {}

    # Create set of keys we need supplementary alignments for. The key carries the mate flag
    # bits so that a paired-end fragment's two mates do not share a bucket.
    supporting_keys = {_mate_aware_key(read) for read, _ in supporting_reads}

    if not supporting_keys:
        return supplementary_reads

    try:
        # Fetch supplementary alignments from the breakpoint regions
        for read in itertools.chain(
            alignment_file.fetch(chrom, start_region_start, start_region_end, multiple_iterators=True),
            alignment_file.fetch(chrom, end_region_start, end_region_end, multiple_iterators=True),
        ):
            if not read.is_supplementary:
                continue

            key = _mate_aware_key(read)
            if key in supporting_keys:
                if key not in supplementary_reads:
                    supplementary_reads[key] = []
                supplementary_reads[key].append(read)

    except Exception as e:
        logger.warning(f"Error fetching supplementary reads: {e}")

    return supplementary_reads


def analyze_interval_breakpoints(
    alignment_file: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    cushion: int,
    pe_config: PairedEndConfig | None = None,
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
    pe_config : PairedEndConfig, optional
        Paired-end configuration. When None or disabled, only split-read evidence is used
        and the single-end code path runs unchanged.

    Returns
    -------
    BreakpointEvidence
        Counts of reads supporting duplication and deletion
    """
    # Calculate breakpoint regions
    regions = _calculate_breakpoint_regions(start, end, cushion)
    start_region_start, start_region_end, end_region_start, end_region_end = regions

    # PHASE 1: Collect primary reads from the two breakpoint regions
    if pe_config is not None and pe_config.enabled:
        evidence = _collect_fragments_from_region(alignment_file, chrom, start, end, cushion, regions, pe_config)
    else:
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
        evidence = BreakpointEvidence(
            chrom=chrom,
            start=start,
            end=end,
            duplication_reads=duplication_reads,
            deletion_reads=deletion_reads,
            total_reads=len(processed_reads),
            dup_insert_sizes=dup_insert_sizes,
            del_insert_sizes=del_insert_sizes,
            supporting_reads=supporting_reads,
        )

    # PHASE 2: Collect supplementary alignments ONLY for supporting reads
    # This is dramatically more efficient than collecting all supplementary reads
    evidence.supplementary_reads = _collect_supplementary_alignments_for_supporting_reads(
        alignment_file,
        chrom,
        evidence.supporting_reads,
        start_region_start,
        start_region_end,
        end_region_start,
        end_region_end,
    )

    return evidence


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
        for supp_read in evidence.supplementary_reads.get(_mate_aware_key(read), []):
            supp_read.set_tag("RG", read_group, value_type="Z")
            bam_out.write(supp_read)


def _process_variants(
    vcf_in: pysam.VariantFile,
    vcf_out: pysam.VariantFile,
    alignment_file: pysam.AlignmentFile,
    cushion: int,
    bam_out: pysam.AlignmentFile | None,
    pe_config: PairedEndConfig | None = None,
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
    pe_config : PairedEndConfig, optional
        Paired-end configuration (default: None, single-end behavior)

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
        evidence = analyze_interval_breakpoints(
            alignment_file, record.chrom, record.start, record.stop, cushion, pe_config
        )

        # Annotate VCF record with evidence
        _annotate_vcf_record_with_evidence(record, evidence)

        # Write annotated record
        vcf_out.write(record)

        # Write supporting reads to BAM if requested
        if bam_out:
            _write_supporting_reads_to_bam(bam_out, evidence)

    return variant_count


def _add_pair_read_group(header: pysam.AlignmentHeader) -> pysam.AlignmentHeader:
    """Return a copy of an evidence BAM header with the PAIR_READ_GROUP read group added."""
    header_dict = header.to_dict()
    read_groups = header_dict.setdefault("RG", [])
    if PAIR_READ_GROUP not in {rg["ID"] for rg in read_groups}:
        read_groups.append({"ID": PAIR_READ_GROUP, "SM": "SAMPLE", "PL": "ULTIMA"})
    return pysam.AlignmentHeader.from_dict(header_dict)


def analyze_cnv_breakpoints(
    bam_file: str,
    vcf_file: str,
    reference_fasta: str,
    cushion: int = 100,
    output_file: str | None = None,
    output_bam: str | None = None,
    *,
    paired_end_config: PairedEndConfig | None = None,
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
        Path to output BAM file with reads supporting CNV calls: split reads under read group DUP/DEL
        and, with paired-end evidence enabled, discordant pair mates under PAIR (default: None, no BAM output)
    paired_end_config : PairedEndConfig, optional
        Paired-end configuration. When None or disabled, only split-read evidence is used
        (default: None)
    """
    # --paired-end is trusted as given; unpaired reads never qualify as pair evidence anyway
    pe_config = paired_end_config if paired_end_config is not None else PairedEndConfig()
    if pe_config.enabled:
        logger.info(f"Paired-end mode: minimum pair span {pe_config.min_pair_span}")
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
            if pe_config.enabled:
                bam_header = _add_pair_read_group(bam_header)
            bam_out = pysam.AlignmentFile(output_bam, "wb", header=bam_header)

        # Process all variants
        variant_count = _process_variants(vcf_in, vcf_out, alignment_file, cushion, bam_out, pe_config)

        vcf_out.close()
        if bam_out:
            bam_out.close()

    # Close alignment file
    alignment_file.close()

    logger.info(f"Processed {variant_count} variants")
    if output_file:
        logger.info(f"Annotated VCF written to {output_file}")
    if output_bam:
        logger.info(f"Supporting reads BAM written to {output_bam}")


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
        help="Path to BAM or CRAM file",
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
        help="Path to output BAM file with reads supporting CNV calls: split reads under read group DUP/DEL "
        "and, with --paired-end, discordant pair mates under read group PAIR (default: None, no BAM output)",
    )

    pe_group = parser.add_argument_group("paired-end options")
    pe_group.add_argument(
        "--paired-end",
        action="store_true",
        default=False,
        help="Input contains paired-end reads: additionally count discordant read pairs as "
        "breakpoint evidence, folded into the existing CNV_*_READS/CNV_*_FRAC INFO fields "
        "(default: False, single-end behavior)",
    )
    pe_group.add_argument(
        "--min-pair-span",
        type=int,
        default=MIN_PAIR_SPAN,
        help="Minimum span (template length), in bases, of a discordant read pair for it to count as "
        f"duplication or deletion evidence. Only used with --paired-end (default: {MIN_PAIR_SPAN})",
    )
    pe_group.add_argument(
        "--min-pair-mapping-quality",
        type=int,
        default=MIN_PAIR_MAPPING_QUALITY,
        help="Minimum mapping quality for a read pair to contribute discordant pair evidence. "
        f"Only used with --paired-end (default: {MIN_PAIR_MAPPING_QUALITY})",
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
            paired_end_config=PairedEndConfig.from_args(args),
        )
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
