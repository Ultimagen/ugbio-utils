"""
On-target rate and target coverage for a ReadFuserAlignSort run.

Given a per-sample coverage bedGraph (the local ``bedgraph_mapq0`` output) and an
optional *targets* BED (e.g. an exome capture BED), compute:

* ``on_target_rate`` - fraction of coverage-weighted aligned bases that fall
  inside the targets, i.e. ``sum((end-start)*depth)`` over the bedGraph
  intersected with the targets, divided by the same sum over the whole bedGraph.
* ``target_mean_cvg`` - mean depth over the target territory
  (on-target weighted bases / target size).
* ``genome_mean_cvg`` - mean depth over the callable genome.

If no targets BED is supplied the on-target metrics are skipped and only
genome-wide coverage is reported.

The heavy lifting is a single streamed pass over the (large, gzipped) bedGraph,
piped through ``bedtools intersect`` - the same approach as the reference
notebook, generalised to an arbitrary BED and made target-agnostic.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class OnTargetResult:
    """Coverage summary for one sample.

    Attributes
    ----------
    total_bases_seq : int
        Coverage-weighted aligned bases genome-wide (``sum((end-start)*depth)``).
    on_target_bases_seq : int | None
        Coverage-weighted aligned bases inside the targets (``None`` if no BED).
    genome_size : int
        Callable genome size (bp) used for ``genome_mean_cvg``.
    target_size : int | None
        Target territory (bp) used for ``target_mean_cvg`` (``None`` if no BED).
    """

    total_bases_seq: int
    on_target_bases_seq: int | None
    genome_size: int
    target_size: int | None

    @property
    def genome_mean_cvg(self) -> float:
        return self.total_bases_seq / self.genome_size if self.genome_size else float("nan")

    @property
    def target_mean_cvg(self) -> float | None:
        if self.on_target_bases_seq is None or not self.target_size:
            return None
        return self.on_target_bases_seq / self.target_size

    @property
    def on_target_rate(self) -> float | None:
        if self.on_target_bases_seq is None or not self.total_bases_seq:
            return None
        return self.on_target_bases_seq / self.total_bases_seq


def _run(cmd: str) -> str:
    """Run a shell pipeline with ``pipefail`` and return its stdout.

    ``pipefail`` matters: these pipelines end in ``awk``, which exits 0 even when an
    upstream ``zcat`` or ``bedtools`` has died, so without it a truncated stream
    would be reported as a valid, silently-too-small sum.
    """
    completed = subprocess.run(  # noqa: S603
        ["bash", "-o", "pipefail", "-c", cmd],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def bed_covered_size(bed_path: str) -> int:
    """Return the number of bp covered by a BED, merging overlaps.

    Parameters
    ----------
    bed_path : str
        Path to a BED file.

    Returns
    -------
    int
        Sum of merged interval lengths (non-overlapping).
    """
    cmd = f"sort -k1,1 -k2,2n {bed_path} | bedtools merge -i - | awk '{{s+=$3-$2}} END{{print s+0}}'"
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)  # noqa: S607
    return int(out.stdout.strip() or 0)


def sorted_bed(bed_path: str, output_path: str) -> str:
    """Write a coordinate-sorted copy of ``bed_path`` (needed for ``intersect -sorted``).

    Parameters
    ----------
    bed_path : str
        Input BED.
    output_path : str
        Where to write the sorted BED.

    Returns
    -------
    str
        ``output_path``.
    """
    subprocess.run(f"sort -k1,1 -k2,2n {bed_path} > {output_path}", shell=True, check=True)  # noqa: S602
    return output_path


def compute_coverage_from_bedgraph(
    bedgraph: str,
    genome_size: int,
    *,
    targets_bed_sorted: str | None = None,
    target_size: int | None = None,
) -> OnTargetResult:
    """Stream a local coverage bedGraph once and sum genome-wide (and optional on-target) depth.

    When ``targets_bed_sorted`` is given, the stream is ``tee``'d: one branch sums
    all coverage-weighted bases, the other intersects with the targets and sums
    the on-target subset.

    Parameters
    ----------
    bedgraph : str
        Local path of the coverage bedGraph (``bedgraph_mapq0``), plain or gzipped.
    genome_size : int
        Callable genome size (bp) for ``genome_mean_cvg`` (e.g. from the sorter
        JSON ``base_coverage["Genome"]`` histogram length-weighted sum).
    targets_bed_sorted : str | None, optional
        Coordinate-sorted targets BED. If ``None``, only genome-wide totals are
        computed.
    target_size : int | None, optional
        Merged target size in bp (required with ``targets_bed_sorted``).

    Returns
    -------
    OnTargetResult
        Coverage summary for the sample.
    """
    source = f"zcat '{bedgraph}'" if bedgraph.endswith(".gz") else f"cat '{bedgraph}'"
    # %.0f, not `print`: the sums reach ~2.5e10 and awk's default OFMT would render
    # them in scientific notation, which int() cannot parse.
    sum_weighted = "awk '{s+=($3-$2)*$4} END{printf \"%.0f\\n\", s+0}'"

    total_bases = int(_run(f"{source} | {sum_weighted}") or 0)
    if targets_bed_sorted is None:
        return OnTargetResult(total_bases, None, genome_size, None)

    if target_size is None:
        raise ValueError("target_size is required when targets_bed_sorted is given")
    # A second streamed pass, deliberately not a `tee` into a process substitution:
    # the shell does not wait for a process substitution, so its result was racy,
    # and any early exit downstream of `tee` truncated it via EPIPE.
    #
    # `bedtools intersect -sorted` is NOT usable here. It requires both files in the
    # same chromosome order, but the bedGraph is in reference/CRAM-header order while
    # this BED is `sort -k1,1` (lexicographic), so bedtools aborts at chr10 - which,
    # with stderr discarded and its non-zero status swallowed mid-pipeline, silently
    # truncated both sums to their chr1 prefix. Without -sorted the BED is loaded
    # into memory and the result is order-independent.
    on_target_bases = int(_run(f"{source} | bedtools intersect -a - -b {targets_bed_sorted} | {sum_weighted}") or 0)
    return OnTargetResult(total_bases, on_target_bases, genome_size, target_size)
