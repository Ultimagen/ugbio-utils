"""Build a tiny in-memory BAM with rn/nf/nr/MI tags and check family classification & coverage."""

import numpy as np
import pysam
import pytest
from ugbio_consensus import duplex_metrics

CHROM = "chr1"
CHROM_LEN = 1000
READ_LEN = 100


def _make_read(header, name, pos, strands, *, reverse=False, mi=None, extra_tags=None, rn=None):
    """Build a read with the consensus ``rn:Z`` + ``nf:i``/``nr:i`` tags.

    ``strands`` is ``(n_forward, n_reverse)``, or ``None`` for a pass-through read
    (no consensus tags at all). ``rn`` overrides the generated read-name list, to
    exercise the nf+nr vs rn cardinality cross-check. ``extra_tags`` adds raw tags
    (used to plant a trimmer ``rs:i`` on a pass-through read).
    """
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
    if strands is not None:
        n_fwd, n_rev = strands
        # rn is the consensus discriminator: a comma-separated list of the fused
        # query names, so its cardinality is nf + nr.
        names = rn if rn is not None else ",".join(f"{name}_src{i}" for i in range(n_fwd + n_rev))
        tags.append(("rn", names, "Z"))
        tags.append(("nf", n_fwd, "i"))
        tags.append(("nr", n_rev, "i"))
    if mi is not None:
        tags.append(("MI", mi))
    if extra_tags:
        tags.extend(extra_tags)
    a.set_tags(tags)
    return a


@pytest.fixture
def bam_path(tmp_path):
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": CHROM, "LN": CHROM_LEN}]}
    path = tmp_path / "consensus.bam"
    # 2 duplex (sizes 8, 4), 2 single-strand (sizes 5, 3), 1 singleton (no rn tag)
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
    res = duplex_metrics.collect_family_metrics_from_strand_tags(bam_path, [(CHROM, 0, CHROM_LEN)], reference=None)
    per = res["per_category"]
    assert per.loc[duplex_metrics.DUPLEX, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLE_STRAND, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLETON, "n_reads"] == 1


def test_family_sizes(bam_path):
    res = duplex_metrics.collect_family_metrics_from_strand_tags(bam_path, [(CHROM, 0, CHROM_LEN)], reference=None)
    per = res["per_category"]
    # duplex sizes 8 and 4 -> avg 6; single-strand 5 and 3 -> avg 4; singleton -> 1
    assert per.loc[duplex_metrics.DUPLEX, "avg_family_size"] == pytest.approx(6.0)
    assert per.loc[duplex_metrics.SINGLE_STRAND, "avg_family_size"] == pytest.approx(4.0)
    assert per.loc[duplex_metrics.SINGLETON, "avg_family_size"] == pytest.approx(1.0)
    # rn cardinality agrees with nf + nr on every consensus read.
    assert res["n_size_mismatch"] == 0
    assert res["n_unclassified"] == 0


def test_coverage_sums(bam_path):
    res = duplex_metrics.collect_family_metrics_from_strand_tags(bam_path, [(CHROM, 0, CHROM_LEN)], reference=None)
    per = res["per_category"]
    # each category's reads each cover READ_LEN bases over the CHROM_LEN interval
    assert per.loc[duplex_metrics.DUPLEX, "coverage"] == pytest.approx(2 * READ_LEN / CHROM_LEN)
    assert per.loc[duplex_metrics.SINGLE_STRAND, "coverage"] == pytest.approx(2 * READ_LEN / CHROM_LEN)
    assert res["total_interval_bp"] == CHROM_LEN


def test_whole_chromosome_end_none(bam_path):
    # end=None means "to the end of the contig"; resolved from the header (CHROM_LEN).
    res = duplex_metrics.collect_family_metrics_from_strand_tags(bam_path, [(CHROM, 0, None)], reference=None)
    per = res["per_category"]
    assert res["total_interval_bp"] == CHROM_LEN
    assert per.loc[duplex_metrics.DUPLEX, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLE_STRAND, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLETON, "n_reads"] == 1


def test_trimmer_rs_tag_is_not_mistaken_for_a_consensus_read(tmp_path):
    """A pass-through read carrying only trimmer's ``rs:i`` must stay a singleton.

    Regression test for the tag collision that forced the ``fs``/``rs`` -> ``nf``/``nr``
    rename: trimmer writes ``rs:i`` ("start position in input of segment ...") on
    the *input* reads, which survives onto the reads the consensus step passes
    through unchanged. Keying the consensus/singleton decision off a strand tag
    made every such read look like a consensus read. The discriminator is ``rn``.
    """
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": CHROM, "LN": CHROM_LEN}]}
    path = tmp_path / "with_trimmer_tags.bam"
    hdr = pysam.AlignmentHeader.from_dict(header)
    reads = [
        # Pass-through reads: trimmer rs (and the legacy fs) present, but no rn.
        _make_read(hdr, "passthrough_rs", 100, None, extra_tags=[("rs", 1, "i")]),
        _make_read(hdr, "passthrough_fs_rs", 120, None, extra_tags=[("fs", 2, "i"), ("rs", 1, "i")]),
        # A genuine duplex consensus read, for contrast.
        _make_read(hdr, "duplex1", 140, (2, 2)),
    ]
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for r in reads:
            out.write(r)
    pysam.index(str(path))

    res = duplex_metrics.collect_family_metrics_from_strand_tags(str(path), [(CHROM, 0, CHROM_LEN)], reference=None)
    per = res["per_category"]
    assert per.loc[duplex_metrics.SINGLETON, "n_reads"] == 2
    assert per.loc[duplex_metrics.SINGLETON, "avg_family_size"] == pytest.approx(1.0)
    assert per.loc[duplex_metrics.DUPLEX, "n_reads"] == 1
    assert per.loc[duplex_metrics.SINGLE_STRAND, "n_reads"] == 0
    assert res["n_unclassified"] == 0


def test_rn_cardinality_mismatch_is_counted(tmp_path):
    """nf + nr disagreeing with the rn read-name count is reported, not silently used."""
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": CHROM, "LN": CHROM_LEN}]}
    path = tmp_path / "mismatch.bam"
    hdr = pysam.AlignmentHeader.from_dict(header)
    reads = [
        # nf+nr = 4 but rn lists only 2 names.
        _make_read(hdr, "bad1", 100, (2, 2), rn="a,b"),
        # Consistent, for contrast.
        _make_read(hdr, "good1", 140, (2, 2)),
    ]
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for r in reads:
            out.write(r)
    pysam.index(str(path))

    res = duplex_metrics.collect_family_metrics_from_strand_tags(str(path), [(CHROM, 0, CHROM_LEN)], reference=None)
    assert res["n_size_mismatch"] == 1
    # Both still classify as duplex; the mismatch is reported, not dropped.
    assert res["per_category"].loc[duplex_metrics.DUPLEX, "n_reads"] == 2


def test_consensus_read_without_strand_counts_is_unclassified(tmp_path):
    """An rn tag with no usable nf/nr is malformed, not a singleton."""
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": CHROM, "LN": CHROM_LEN}]}
    path = tmp_path / "malformed.bam"
    hdr = pysam.AlignmentHeader.from_dict(header)
    read = _make_read(hdr, "no_counts", 100, None, extra_tags=[("rn", "a,b", "Z")])
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        out.write(read)
    pysam.index(str(path))

    res = duplex_metrics.collect_family_metrics_from_strand_tags(str(path), [(CHROM, 0, CHROM_LEN)], reference=None)
    assert res["n_unclassified"] == 1
    assert res["per_category"].loc[duplex_metrics.SINGLETON, "n_reads"] == 0


def test_mi_fallback_matches(bam_path):
    res = duplex_metrics.collect_family_metrics_from_mi_tags(bam_path, [(CHROM, 0, CHROM_LEN)], reference=None)
    # MI grouping sees 4 tagged reads, each its own MI, all singletons by MI membership
    # (one read per MI) -> all classified as singleton. This documents that the strand-tag
    # path is the accurate one for consensus reads; MI fallback needs multi-read MI groups.
    assert res["n_families"] == 4
    assert np.isnan(res["per_category"].loc[duplex_metrics.DUPLEX, "avg_family_size"])
