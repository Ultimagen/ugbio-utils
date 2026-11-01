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
piped through ``bedtools intersect`` (see :meth:`ugbio_core.bed_utils.BedUtils.sum_bedgraph_weighted_bases`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ugbio_core.bed_utils import BedUtils


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


def compute_coverage_from_bedgraph(
    bedgraph: str,
    genome_size: int,
    *,
    targets_bed_sorted: str | None = None,
    target_size: int | None = None,
) -> OnTargetResult:
    """Stream a local coverage bedGraph and sum genome-wide (and optional on-target) depth.

    Parameters
    ----------
    bedgraph : str
        Local path of the coverage bedGraph (``bedgraph_mapq0``), plain or gzipped.
    genome_size : int
        Callable genome size (bp) for ``genome_mean_cvg`` (e.g. from the sorter
        JSON ``base_coverage["Genome"]`` histogram length-weighted sum).
    targets_bed_sorted : str | None, optional
        Non-overlapping targets BED (as written by
        :meth:`ugbio_core.bed_utils.BedUtils.sort_and_merge_bed`). If ``None``, only
        genome-wide totals are computed.
    target_size : int | None, optional
        Merged target size in bp (required with ``targets_bed_sorted``).

    Returns
    -------
    OnTargetResult
        Coverage summary for the sample.
    """
    bed_utils = BedUtils()
    total_bases = bed_utils.sum_bedgraph_weighted_bases(bedgraph)
    if targets_bed_sorted is None:
        return OnTargetResult(total_bases, None, genome_size, None)

    if target_size is None:
        raise ValueError("target_size is required when targets_bed_sorted is given")
    on_target_bases = bed_utils.sum_bedgraph_weighted_bases(bedgraph, restrict_to_bed=targets_bed_sorted)
    return OnTargetResult(total_bases, on_target_bases, genome_size, target_size)
