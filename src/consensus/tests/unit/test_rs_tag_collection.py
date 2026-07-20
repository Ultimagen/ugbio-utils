"""Build a tiny in-memory BAM with rs/MI tags and check family classification & coverage."""

import numpy as np
import pysam
import pytest
from ugbio_consensus import duplex_metrics

CHROM = "chr1"
CHROM_LEN = 1000
READ_LEN = 100


def _make_read(header, name, pos, rs, *, reverse=False, mi=None):
    a = pysam.AlignedSegment(header)
    a.query_name = name
    a.query_sequence = "A" * READ_LEN
    a.flag = 16 if reverse else 0
    a.reference_id = 0
    a.reference_start = pos
    a.mapping_quality = 60
    a.cigartuples = [(0, READ_LEN)]  # 100M
    a.query_qualities = pysam.qualitystring_to_array("I" * READ_LEN)
    tags = []
    if rs is not None:
        tags.append(("rs", list(rs), "i"))  # array of ints -> rs:B:i
    if mi is not None:
        tags.append(("MI", mi))
    a.set_tags(tags)
    return a


@pytest.fixture
def bam_path(tmp_path):
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": CHROM, "LN": CHROM_LEN}]}
    path = tmp_path / "consensus.bam"
    # 2 duplex (sizes 8, 4), 2 single-strand (sizes 5, 3), 1 singleton (no rs tag)
    reads = [
        _make_read(pysam.AlignmentHeader.from_dict(header), "duplex1", 100, (4, 4), mi=1),
        _make_read(pysam.AlignmentHeader.from_dict(header), "duplex2", 120, (1, 3), mi=2),
        _make_read(pysam.AlignmentHeader.from_dict(header), "ss1", 140, (5, 0), reverse=False, mi=3),
        _make_read(pysam.AlignmentHeader.from_dict(header), "ss2", 160, (0, 3), reverse=True, mi=4),
        _make_read(pysam.AlignmentHeader.from_dict(header), "single1", 180, None),
    ]
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for r in reads:
            out.write(r)
    pysam.index(str(path))
    return str(path)


def test_family_classification_counts(bam_path):
    res = duplex_metrics.collect_family_metrics_from_rs_tags(bam_path, [(CHROM, 0, CHROM_LEN)], reference=None)
    per = res["per_category"]
    assert per.loc[duplex_metrics.DUPLEX, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLE_STRAND, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLETON, "n_reads"] == 1


def test_family_sizes(bam_path):
    res = duplex_metrics.collect_family_metrics_from_rs_tags(bam_path, [(CHROM, 0, CHROM_LEN)], reference=None)
    per = res["per_category"]
    # duplex sizes 8 and 4 -> avg 6; single-strand 5 and 3 -> avg 4; singleton -> 1
    assert per.loc[duplex_metrics.DUPLEX, "avg_family_size"] == pytest.approx(6.0)
    assert per.loc[duplex_metrics.SINGLE_STRAND, "avg_family_size"] == pytest.approx(4.0)
    assert per.loc[duplex_metrics.SINGLETON, "avg_family_size"] == pytest.approx(1.0)


def test_coverage_sums(bam_path):
    res = duplex_metrics.collect_family_metrics_from_rs_tags(bam_path, [(CHROM, 0, CHROM_LEN)], reference=None)
    per = res["per_category"]
    # each category's reads each cover READ_LEN bases over the CHROM_LEN interval
    assert per.loc[duplex_metrics.DUPLEX, "coverage"] == pytest.approx(2 * READ_LEN / CHROM_LEN)
    assert per.loc[duplex_metrics.SINGLE_STRAND, "coverage"] == pytest.approx(2 * READ_LEN / CHROM_LEN)
    assert res["total_interval_bp"] == CHROM_LEN


def test_whole_chromosome_end_none(bam_path):
    # end=None means "to the end of the contig"; resolved from the header (CHROM_LEN).
    res = duplex_metrics.collect_family_metrics_from_rs_tags(bam_path, [(CHROM, 0, None)], reference=None)
    per = res["per_category"]
    assert res["total_interval_bp"] == CHROM_LEN
    assert per.loc[duplex_metrics.DUPLEX, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLE_STRAND, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLETON, "n_reads"] == 1


def test_mi_fallback_matches(bam_path):
    res = duplex_metrics.collect_family_metrics_from_mi_tags(bam_path, [(CHROM, 0, CHROM_LEN)], reference=None)
    # MI grouping sees 4 tagged reads, each its own MI, all singletons by MI membership
    # (one read per MI) -> all classified as singleton. This documents that the rs path
    # is the accurate one for consensus reads; MI fallback needs multi-read MI groups.
    assert res["n_families"] == 4
    assert np.isnan(res["per_category"].loc[duplex_metrics.DUPLEX, "avg_family_size"])
