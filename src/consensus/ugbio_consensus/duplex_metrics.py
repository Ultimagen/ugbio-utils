"""
Duplex / consensus family metrics from a ReadFuserAlignSort output CRAM.

The consensus tool (``read_fuser``) writes, on each consensus read, an ``rs:B:i``
tag holding two integers::

    rs = [n_forward_strand_reads, n_reverse_strand_reads]

i.e. the number of original reads on the forward (+) and reverse (-) strand that
were fused into that consensus read. This lets us classify every consensus read
and measure family size *directly*, without re-grouping reads by their ``MI``
(molecular identifier) tag:

* **both-strands duplex** - both entries > 0 (the molecule was observed on both
  strands); family size = ``n_forward + n_reverse``.
* **single-strand duplicate** - exactly one entry is 0; family size = the
  non-zero entry.
* **singleton / pass-through** - no ``rs`` tag (a read the consensus step passed
  through unchanged); family size = 1.

``sum(rs)`` equals the length of the ``rn`` (fused read-name list) tag, so the
two encodings agree on family size.

The historical MI-tag approach (grouping reads by ``MI`` and inspecting the set
of strands) is kept in :func:`collect_family_metrics_from_mi_tags` for CRAMs
produced without the ``rs`` tag.
"""

from __future__ import annotations

import bisect
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam

# Family categories, in a fixed order for stable table/column layout.
DUPLEX = "duplex"
SINGLE_STRAND = "single_strand"
SINGLETON = "singleton"
CATEGORIES = (DUPLEX, SINGLE_STRAND, SINGLETON)


def parse_rs_tag(read: pysam.AlignedSegment) -> tuple[int, int] | None:
    """Return ``(n_forward, n_reverse)`` from a consensus read's ``rs`` tag.

    Parameters
    ----------
    read : pysam.AlignedSegment
        A single alignment record.

    Returns
    -------
    tuple[int, int] | None
        ``(n_forward, n_reverse)`` if the read carries a well-formed 2-element
        ``rs`` tag, otherwise ``None`` (no tag, or a malformed/scalar value).
    """
    if not read.has_tag("rs"):
        return None
    rs = read.get_tag("rs")
    try:
        n_fwd, n_rev = int(rs[0]), int(rs[1])
    except (TypeError, IndexError, ValueError):
        # Rare scalar / malformed rs value - caller counts these as unclassified.
        return None
    return n_fwd, n_rev


def classify_family(n_fwd: int, n_rev: int) -> str:
    """Classify a consensus family from its per-strand read counts.

    Parameters
    ----------
    n_fwd : int
        Number of forward-strand reads fused into the consensus read.
    n_rev : int
        Number of reverse-strand reads fused into the consensus read.

    Returns
    -------
    str
        :data:`DUPLEX` if both strands contributed, :data:`SINGLE_STRAND` if only
        one strand did, :data:`SINGLETON` if the total is a single read.
    """
    total = n_fwd + n_rev
    if total <= 1:
        return SINGLETON
    if n_fwd > 0 and n_rev > 0:
        return DUPLEX
    return SINGLE_STRAND


def _merge_intervals(intervals: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Merge overlapping/adjacent ``(chrom, start, end)`` intervals.

    Merging makes them disjoint so a base is never counted twice when summing
    per-category coverage across intervals.
    """
    merged: list[tuple[str, int, int]] = []
    for chrom, start, end in sorted(intervals):
        if merged and merged[-1][0] == chrom and start <= merged[-1][2]:
            prev_chrom, prev_start, prev_end = merged[-1]
            merged[-1] = (prev_chrom, prev_start, max(prev_end, end))
        else:
            merged.append((chrom, start, end))
    return merged


def _overlap_with_intervals(read, starts: list[int], intervals: list[tuple[str, int, int]]) -> int:
    """Return the read's aligned bases that fall inside the merged intervals.

    ``starts`` is the sorted list of interval start coordinates (for the read's
    chromosome) enabling a bisect lookup; ``intervals`` are the matching disjoint
    ``(chrom, start, end)`` tuples. A read may straddle interval boundaries, so we
    sum the clipped overlap against the interval at/just-before the read start and
    the following one.
    """
    total = 0
    idx = bisect.bisect_right(starts, read.reference_end) - 1
    # Walk back over the (few) intervals that could overlap this read.
    while idx >= 0:
        _chrom, i_start, i_end = intervals[idx]
        if i_end <= read.reference_start:
            break
        total += max(0, min(read.reference_end, i_end) - max(read.reference_start, i_start))
        idx -= 1
    return total


def _iter_primary_reads_by_chrom(samfile, merged, max_reads):
    """Yield ``(chrom_intervals, read)`` for primary reads, one linear fetch per chromosome.

    Rather than a separate ``fetch`` per interval (thousands of tiny CRAM reads
    thrash container decoding), we fetch each chromosome once over its covered
    span and hand back the chromosome's merged intervals for in-memory overlap
    testing. Skips secondary/supplementary/unmapped reads and stops after
    ``max_reads`` primary reads (``None`` = no cap).
    """
    scanned = 0
    by_chrom: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for chrom, start, end in merged:
        by_chrom[chrom].append((chrom, start, end))

    for chrom, chrom_intervals in by_chrom.items():
        span_start = chrom_intervals[0][1]
        span_end = chrom_intervals[-1][2]
        for read in samfile.fetch(chrom, span_start, span_end):
            if read.is_secondary or read.is_supplementary or read.is_unmapped:
                continue
            scanned += 1
            yield chrom_intervals, read
            if max_reads is not None and scanned >= max_reads:
                return


def _classify_read(read: pysam.AlignedSegment) -> tuple[str, int] | None:
    """Return ``(category, family_size)`` for a read, or ``None`` if unclassifiable.

    ``None`` means the read carries an ``rs`` tag we could not parse (malformed /
    scalar); a read with no ``rs`` tag is treated as a singleton.
    """
    rs = parse_rs_tag(read)
    if rs is None:
        if read.has_tag("rs"):
            return None  # malformed/scalar rs
        return SINGLETON, 1
    n_fwd, n_rev = rs
    return classify_family(n_fwd, n_rev), max(n_fwd + n_rev, 1)


def collect_family_metrics_from_rs_tags(
    cram_path: str,
    intervals: list[tuple[str, int, int]],
    reference: str,
    *,
    index_path: str | None = None,
    max_reads: int | None = None,
) -> dict:
    """Classify consensus reads and measure family size and coverage over ``intervals``.

    For each primary read overlapping the (merged) intervals we read its ``rs``
    tag, classify the family, record family size, and add its aligned span
    (clipped to the interval) to that category's covered-base total. Family-size
    statistics count each read once, at the interval containing its start, so
    reads spanning an interval boundary are not double-counted.

    Parameters
    ----------
    cram_path : str
        Path or ``s3://`` URI of the consensus CRAM.
    intervals : list[tuple[str, int, int | None]]
        ``(chrom, start, end)`` regions to analyse (e.g. target-BED rows, or a
        whole chromosome). An ``end`` of ``None`` means "to the end of the
        contig" and is resolved from the CRAM header.
    reference : str
        Path to the reference FASTA (required to decode a CRAM).
    index_path : str | None, optional
        Explicit ``.crai`` path (useful when reading from S3 with a local index).
    max_reads : int | None, optional
        Stop after this many primary reads (for quick sampling). ``None`` = all.

    Returns
    -------
    dict
        Keys: ``per_category`` (DataFrame indexed by category with columns
        ``n_reads``, ``avg_family_size``, ``median_family_size``,
        ``covered_bases``, ``coverage``), ``total_interval_bp``,
        ``n_reads_scanned``, ``n_unclassified`` (reads with a malformed ``rs``),
        and ``family_sizes`` (dict category -> np.ndarray of sizes).
    """
    family_sizes: dict[str, list[int]] = {c: [] for c in CATEGORIES}
    covered_bases: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    n_reads_scanned = 0
    n_unclassified = 0
    total_bp = 0

    open_kwargs = {"reference_filename": reference}
    if index_path is not None:
        open_kwargs["index_filename"] = index_path

    with pysam.AlignmentFile(cram_path, "rc", **open_kwargs) as samfile:
        # Resolve open-ended intervals ("to end of contig") from the header, then merge.
        resolved = [
            (chrom, start, samfile.get_reference_length(chrom) if end is None else end)
            for chrom, start, end in intervals
        ]
        merged = _merge_intervals(resolved)
        total_bp = sum(end - start for _, start, end in merged)
        # Per-chromosome sorted interval starts for the bisect overlap lookup.
        starts_by_chrom: dict[str, list[int]] = defaultdict(list)
        for chrom, start, _end in merged:
            starts_by_chrom[chrom].append(start)

        for chrom_intervals, read in _iter_primary_reads_by_chrom(samfile, merged, max_reads):
            n_reads_scanned += 1

            # Coverage: aligned bases inside the (disjoint) intervals on this chromosome.
            starts = starts_by_chrom[chrom_intervals[0][0]]
            overlap = _overlap_with_intervals(read, starts, chrom_intervals)
            if overlap <= 0:
                continue

            classified = _classify_read(read)
            if classified is None:
                # Malformed/scalar rs - excluded from family stats.
                n_unclassified += 1
                continue
            category, size = classified

            covered_bases[category] += overlap
            # Count each family once, at the read start (avoids double-counting straddlers).
            family_sizes[category].append(size)

    rows = {}
    for category in CATEGORIES:
        sizes = np.array(family_sizes[category], dtype=float)
        rows[category] = {
            "n_reads": len(sizes),
            "avg_family_size": float(np.mean(sizes)) if sizes.size else np.nan,
            "median_family_size": float(np.median(sizes)) if sizes.size else np.nan,
            "covered_bases": covered_bases[category],
            "coverage": covered_bases[category] / total_bp if total_bp else np.nan,
        }
    per_category = pd.DataFrame.from_dict(rows, orient="index")
    per_category.index.name = "category"

    return {
        "per_category": per_category,
        "total_interval_bp": total_bp,
        "n_reads_scanned": n_reads_scanned,
        "n_unclassified": n_unclassified,
        "family_sizes": {c: np.array(family_sizes[c]) for c in CATEGORIES},
    }


def collect_family_metrics_from_mi_tags(
    cram_path: str,
    intervals: list[tuple[str, int, int]],
    reference: str,
    *,
    index_path: str | None = None,
) -> dict:
    """Fallback: classify families by grouping reads on the ``MI`` tag.

    Use only for consensus/duplicate-marked CRAMs that lack the ``rs`` tag. A
    family is *duplex* if its reads span both strands, *single_strand* if all
    reads share one strand, and *singleton* if it has a single read. This mirrors
    the historical MI-tag analysis; :func:`collect_family_metrics_from_rs_tags`
    is preferred (exact, per-read, and yields coverage).

    Parameters
    ----------
    cram_path : str
        Path or ``s3://`` URI of the CRAM.
    intervals : list[tuple[str, int, int]]
        ``(chrom, start, end)`` regions to analyse.
    reference : str
        Path to the reference FASTA.
    index_path : str | None, optional
        Explicit ``.crai`` path.

    Returns
    -------
    dict
        Keys: ``per_category`` (DataFrame with ``n_families`` and
        ``avg_family_size`` per category) and ``n_families``.
    """
    merged = _merge_intervals(intervals)
    mi_strands: dict[object, list[str]] = defaultdict(list)

    open_kwargs = {"reference_filename": reference}
    if index_path is not None:
        open_kwargs["index_filename"] = index_path

    with pysam.AlignmentFile(cram_path, "rc", **open_kwargs) as samfile:
        for chrom, start, end in merged:
            for read in samfile.fetch(chrom, start, end):
                if read.is_secondary or read.is_supplementary or read.is_unmapped:
                    continue
                if not read.has_tag("MI"):
                    continue
                mi_strands[read.get_tag("MI")].append("-" if read.is_reverse else "+")

    sizes: dict[str, list[int]] = {c: [] for c in CATEGORIES}
    for strands in mi_strands.values():
        if len(strands) == 1:
            sizes[SINGLETON].append(1)
        elif len(set(strands)) == 1:
            sizes[SINGLE_STRAND].append(len(strands))
        else:
            sizes[DUPLEX].append(len(strands))

    rows = {}
    for category in CATEGORIES:
        arr = np.array(sizes[category], dtype=float)
        rows[category] = {
            "n_families": len(arr),
            "avg_family_size": float(np.mean(arr)) if arr.size else np.nan,
        }
    per_category = pd.DataFrame.from_dict(rows, orient="index")
    per_category.index.name = "category"

    return {"per_category": per_category, "n_families": len(mi_strands)}
