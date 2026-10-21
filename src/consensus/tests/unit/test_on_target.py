"""On-target / coverage computation from a small synthetic bedGraph + BED."""

import gzip
import subprocess

import pytest
from ugbio_consensus.on_target import (
    bed_covered_size,
    compute_coverage_from_bedgraph,
    sorted_bed,
)

# bedGraph: chrom start end depth
_BEDGRAPH = "chr1\t0\t100\t10\nchr1\t100\t200\t20\nchr1\t200\t300\t5\n"
# targets: chr1:100-200 -> the middle (depth-20) block, fully on target
_TARGETS = "chr1\t100\t200\n"


@pytest.fixture
def bedgraph(tmp_path):
    p = tmp_path / "cov.bedGraph.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(_BEDGRAPH)
    return str(p)


@pytest.fixture
def targets_sorted(tmp_path):
    raw = tmp_path / "targets.bed"
    raw.write_text(_TARGETS, encoding="utf-8")
    return sorted_bed(str(raw), str(tmp_path / "targets.sorted.bed")), str(raw)


def test_bed_covered_size(targets_sorted):
    _, raw = targets_sorted
    assert bed_covered_size(raw) == 100


def test_genome_wide_only(bedgraph):
    # total weighted bases = 100*10 + 100*20 + 100*5 = 3500
    res = compute_coverage_from_bedgraph(bedgraph, genome_size=1000)
    assert res.total_bases_seq == 3500
    assert res.on_target_bases_seq is None
    assert res.on_target_rate is None
    assert res.genome_mean_cvg == pytest.approx(3.5)


def test_on_target(bedgraph, targets_sorted):
    sorted_path, _ = targets_sorted
    # on-target weighted bases = 100*20 = 2000; total = 3500
    res = compute_coverage_from_bedgraph(bedgraph, genome_size=1000, targets_bed_sorted=sorted_path, target_size=100)
    assert res.on_target_bases_seq == 2000
    assert res.on_target_rate == pytest.approx(2000 / 3500)
    assert res.target_mean_cvg == pytest.approx(20.0)


def test_target_size_required(bedgraph, targets_sorted):
    sorted_path, _ = targets_sorted
    with pytest.raises(ValueError, match="target_size is required"):
        compute_coverage_from_bedgraph(bedgraph, genome_size=1000, targets_bed_sorted=sorted_path)


# A bedGraph is emitted in reference/CRAM-header order (chr1, chr2, ... chr9, chr10),
# whereas `sorted_bed` sorts lexicographically (chr1, chr10, chr2). The two orders
# disagree from the second contig onwards, which is what made
# `bedtools intersect -sorted` abort partway through and silently truncate both sums.
# chr10 must be present: it is the contig whose lexicographic position (2nd) differs
# from its reference position (10th), and the one bedtools aborted on in production.
_REF_ORDER_CONTIGS = [f"chr{i}" for i in range(1, 11)]
_BEDGRAPH_REF_ORDER = "".join(f"{c}\t0\t100\t{i}\n" for i, c in enumerate(_REF_ORDER_CONTIGS, start=1))
_TARGETS_MULTI = "".join(f"{c}\t0\t100\n" for c in _REF_ORDER_CONTIGS)
# total weighted bases = 100 * (1+2+...+10) = 5500, all of it on target
_REF_ORDER_TOTAL = 5500


@pytest.fixture
def bedgraph_ref_order(tmp_path):
    p = tmp_path / "cov_ref_order.bedGraph.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(_BEDGRAPH_REF_ORDER)
    return str(p)


@pytest.fixture
def targets_multi_sorted(tmp_path):
    raw = tmp_path / "targets_multi.bed"
    raw.write_text(_TARGETS_MULTI, encoding="utf-8")
    return sorted_bed(str(raw), str(tmp_path / "targets_multi.sorted.bed"))


def test_bedgraph_in_reference_order_is_fully_counted(bedgraph_ref_order, targets_multi_sorted):
    """Every contig must be counted even though the bedGraph and BED orders differ.

    Regression: the sums used to stop at the first order disagreement (chr10 in
    production), reporting a truncated prefix (0 here, bedtools dying before its first flush) while
    still exiting 0.
    """
    res = compute_coverage_from_bedgraph(
        bedgraph_ref_order, genome_size=1000, targets_bed_sorted=targets_multi_sorted, target_size=1000
    )
    assert res.total_bases_seq == _REF_ORDER_TOTAL
    assert res.on_target_bases_seq == _REF_ORDER_TOTAL
    assert res.on_target_rate == pytest.approx(1.0)
    assert res.target_mean_cvg == pytest.approx(_REF_ORDER_TOTAL / 1000)


def test_missing_bedgraph_raises(tmp_path, targets_multi_sorted):
    """A dead upstream process must fail loudly, not return a truncated sum."""
    with pytest.raises(subprocess.CalledProcessError):
        compute_coverage_from_bedgraph(
            str(tmp_path / "nope.bedGraph.gz"),
            genome_size=1000,
            targets_bed_sorted=targets_multi_sorted,
            target_size=300,
        )


# --- Overlapping targets must not be double counted --------------------------------
# Depth 10 over chr1:[0,100); targets [0,100) and [50,100) overlap on [50,100).
_BEDGRAPH_FLAT = "chr1\t0\t100\t10\n"
_TARGETS_OVERLAPPING = "chr1\t0\t100\nchr1\t50\t100\n"


def _write_gz(path, text):
    with gzip.open(path, "wt") as fh:
        fh.write(text)
    return str(path)


def test_sorted_bed_merges_overlapping_targets(tmp_path):
    raw = tmp_path / "overlap.bed"
    raw.write_text(_TARGETS_OVERLAPPING, encoding="utf-8")
    out = sorted_bed(str(raw), str(tmp_path / "overlap.sorted.bed"))
    with open(out, encoding="utf-8") as fh:
        lines = fh.read().split("\n")[:-1]
    assert lines == ["chr1\t0\t100"]


def test_overlapping_targets_not_double_counted(tmp_path):
    bg = _write_gz(tmp_path / "flat.bedGraph.gz", _BEDGRAPH_FLAT)
    raw = tmp_path / "overlap.bed"
    raw.write_text(_TARGETS_OVERLAPPING, encoding="utf-8")
    targets = sorted_bed(str(raw), str(tmp_path / "overlap.sorted.bed"))
    size = bed_covered_size(targets)
    assert size == 100
    res = compute_coverage_from_bedgraph(bg, genome_size=100, targets_bed_sorted=targets, target_size=size)
    assert res.on_target_bases_seq == 1000
    assert res.on_target_rate == pytest.approx(1.0)
    assert res.target_mean_cvg == pytest.approx(10.0)


def test_unmerged_overlapping_targets_rejected(tmp_path):
    """A direct call with an overlapping (unmerged) BED fails instead of inflating the rate."""
    bg = _write_gz(tmp_path / "flat.bedGraph.gz", _BEDGRAPH_FLAT)
    raw = tmp_path / "overlap.bed"
    raw.write_text(_TARGETS_OVERLAPPING, encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        compute_coverage_from_bedgraph(bg, genome_size=100, targets_bed_sorted=str(raw), target_size=100)


def test_bed_covered_size_propagates_upstream_failure(tmp_path):
    """pipefail: a missing BED must raise, not report a target size of 0."""
    with pytest.raises(subprocess.CalledProcessError):
        bed_covered_size(str(tmp_path / "missing.bed"))


def test_paths_with_spaces_and_quotes(tmp_path):
    d = tmp_path / "dir with space's"
    d.mkdir()
    bg = _write_gz(d / "cov file.bedGraph.gz", _BEDGRAPH)
    raw = d / "targets file.bed"
    raw.write_text(_TARGETS, encoding="utf-8")
    targets = sorted_bed(str(raw), str(d / "targets sorted.bed"))
    size = bed_covered_size(str(raw))
    assert size == 100
    res = compute_coverage_from_bedgraph(bg, genome_size=1000, targets_bed_sorted=targets, target_size=size)
    assert res.total_bases_seq == 3500
    assert res.on_target_bases_seq == 2000


def test_check_disjoint_accepts_unsorted_disjoint_bed(tmp_path):
    """Overlap detection does not depend on row order: unsorted but disjoint targets are accepted."""
    bg = _write_gz(tmp_path / "cov.bedGraph.gz", _BEDGRAPH)
    raw = tmp_path / "unsorted.bed"
    raw.write_text("chr1\t200\t250\nchr1\t100\t150\n", encoding="utf-8")
    res = compute_coverage_from_bedgraph(bg, genome_size=1000, targets_bed_sorted=str(raw), target_size=100)
    assert res.on_target_bases_seq == 50 * 20 + 50 * 5
