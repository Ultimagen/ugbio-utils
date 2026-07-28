"""On-target / coverage computation from a small synthetic bedGraph + BED."""

import gzip

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
