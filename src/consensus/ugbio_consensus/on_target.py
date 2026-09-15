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

    if targets_bed_sorted is not None:
        if target_size is None:
            raise ValueError("target_size is required when targets_bed_sorted is given")
        # tee: one branch totals all bases, the other totals the on-target subset.
        cmd = (
            f"{source} "
            f"| tee >(awk '{{t+=($3-$2)*$4}} END{{print t+0}}' > /tmp/_consensus_tot_$$) "
            f"| bedtools intersect -a - -b {targets_bed_sorted} -sorted 2>/dev/null "
            f"| awk '{{o+=($3-$2)*$4}} END{{print o+0}}'; "
            f"cat /tmp/_consensus_tot_$$; rm -f /tmp/_consensus_tot_$$"
        )
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout.split()  # noqa: S607
        on_target_bases, total_bases = int(out[0]), int(out[1])
        return OnTargetResult(total_bases, on_target_bases, genome_size, target_size)

    cmd = f"{source} | awk '{{t+=($3-$2)*$4}} END{{print t+0}}'"
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S607
    return OnTargetResult(int(out or 0), None, genome_size, None)
