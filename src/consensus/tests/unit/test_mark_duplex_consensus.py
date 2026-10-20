#!/usr/bin/env python3
"""
Tests for mark_duplex_consensus.py on small synthetic consensus BAMs.

Every record is a single-end 50 bp consensus read with a CS tag and a po (pair orientation) tag, which is
what ReadFuserAlignSort leaves on a consensus read. Positions are 0-based.

Run with:  pytest src/consensus/tests/unit/test_mark_duplex_consensus.py
"""

import pysam
import pytest
from ugbio_consensus import mark_duplex_consensus as mdc

READ_LEN = 50
CONTIG_LEN = 50_000

HEADER = pysam.AlignmentHeader.from_dict(
    {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": CONTIG_LEN}, {"SN": "chr2", "LN": CONTIG_LEN}],
    }
)


def read(name, chrom, pos, *, cs=None, po=None, flag=0, tags=None):
    seg = pysam.AlignedSegment(HEADER)
    seg.query_name = name
    seg.flag = flag
    seg.query_sequence = "A" * READ_LEN
    seg.query_qualities = pysam.qualitystring_to_array("I" * READ_LEN)
    if chrom is None:
        seg.reference_id = -1
        seg.reference_start = -1
    else:
        seg.reference_id = HEADER.get_tid(chrom)
        seg.reference_start = pos
        seg.mapping_quality = 60
        seg.cigartuples = [(0, READ_LEN)]
    for tag, value in {"CS": cs, "po": po, **(tags or {})}.items():
        if value is None:
            continue
        seg.set_tag(tag, value, value_type="i" if isinstance(value, int) else "Z")
    return seg


def run(tmp_path, records, *, pad=2_000, **kwargs):
    inp, out = tmp_path / "in.bam", tmp_path / "out.bam"

    def order(rec):
        return (rec.reference_id if rec.reference_id >= 0 else 1 << 30, rec.reference_start)

    with pysam.AlignmentFile(str(inp), "wb", header=HEADER) as fh:
        for rec in sorted(records, key=order):
            fh.write(rec)
    counters = mdc.mark_duplex(str(inp), str(out), {}, pad=pad, **kwargs)
    return out, counters


def dx(path):
    with pysam.AlignmentFile(str(path), "rb") as bam:
        return {r.query_name: (r.get_tag("dx") if r.has_tag("dx") else None) for r in bam}


def test_both_orientations_make_every_read_of_the_family_duplex(tmp_path):
    recs = [
        read("a_f1r2_left", "chr1", 1000, cs="C1", po="F1R2"),
        read("a_f1r2_right", "chr1", 1150, cs="C1", po="F1R2", flag=16),
        read("a_f2r1_left", "chr1", 1000, cs="C1", po="F2R1"),
        read("a_f2r1_right", "chr1", 1150, cs="C1", po="F2R1", flag=16),
    ]
    out, counters = run(tmp_path, recs)
    assert set(dx(out).values()) == {1}
    assert (counters["cs_families"], counters["duplex_families"], counters["reads_dx1"]) == (1, 1, 4)


def test_two_consensus_reads_of_one_orientation_are_not_duplex(tmp_path):
    """What DS=2 cannot tell apart: a second read of the same orientation."""
    recs = [
        read("first", "chr1", 1000, cs="C1", po="F1R2"),
        read("second", "chr1", 1000, cs="C1", po="F1R2", flag=16),
    ]
    out, counters = run(tmp_path, recs)
    assert dx(out) == {"first": None, "second": None}
    assert counters["duplex_families"] == 0


def test_a_symmetric_umi_family_with_ds_two_is_not_duplex_when_it_has_one_orientation(tmp_path):
    recs = [
        read("x", "chr1", 1000, cs="C1", po="F2R1", tags={"DS": 2}),
        read("y", "chr1", 1000, cs="C1", po="F2R1", flag=16, tags={"DS": 2}),
    ]
    out, _ = run(tmp_path, recs)
    assert dx(out) == {"x": None, "y": None}


def test_families_are_judged_separately(tmp_path):
    recs = [
        read("d1", "chr1", 1000, cs="DUP", po="F1R2"),
        read("d2", "chr1", 1010, cs="DUP", po="F2R1"),
        read("s1", "chr1", 1000, cs="SIM", po="F1R2"),
        read("s2", "chr1", 1010, cs="SIM2", po="F2R1"),
    ]
    out, _ = run(tmp_path, recs)
    assert dx(out) == {"d1": 1, "d2": 1, "s1": None, "s2": None}


def test_write_simplex_writes_zero_on_the_other_families(tmp_path):
    recs = [
        read("d1", "chr1", 1000, cs="DUP", po="F1R2"),
        read("d2", "chr1", 1010, cs="DUP", po="F2R1"),
        read("s1", "chr1", 1000, cs="SIM", po="F1R2"),
        read("nocs", "chr1", 1000),
    ]
    out, _ = run(tmp_path, recs, write_simplex=True)
    assert dx(out) == {"d1": 1, "d2": 1, "s1": 0, "nocs": None}


def test_a_stale_dx_is_replaced_not_trusted(tmp_path):
    recs = [
        read("lone", "chr1", 1000, cs="C1", po="F1R2", tags={"dx": 1}),
        read("nocs", "chr1", 2000, tags={"dx": 1}),
    ]
    out, _ = run(tmp_path, recs)
    assert dx(out) == {"lone": None, "nocs": None}


def test_the_orientations_may_be_up_to_pad_apart_and_no_further(tmp_path):
    near = [read("n1", "chr1", 1000, cs="NEAR", po="F1R2"), read("n2", "chr1", 2900, cs="NEAR", po="F2R1")]
    far = [read("f1", "chr1", 10_000, cs="FAR", po="F1R2"), read("f2", "chr1", 12_100, cs="FAR", po="F2R1")]
    out, _ = run(tmp_path, near + far, pad=2_000)
    got = dx(out)
    assert (got["n1"], got["n2"]) == (1, 1)
    assert (got["f1"], got["f2"]) == (None, None)  # the early read was written before its partner appeared


def test_a_partner_on_another_contig_does_not_count(tmp_path):
    recs = [read("a", "chr1", 1000, cs="C1", po="F1R2"), read("b", "chr2", 1000, cs="C1", po="F2R1")]
    out, _ = run(tmp_path, recs)
    assert dx(out) == {"a": None, "b": None}


def test_ff_and_rr_pairs_do_not_make_a_family_duplex(tmp_path):
    recs = [read("a", "chr1", 1000, cs="C1", po="FF"), read("b", "chr1", 1000, cs="C1", po="RR")]
    out, _ = run(tmp_path, recs)
    assert dx(out) == {"a": None, "b": None}


def test_a_read_without_po_takes_the_status_of_its_family_but_adds_no_orientation(tmp_path):
    recs = [
        read("a", "chr1", 1000, cs="C1", po="F1R2"),
        read("b", "chr1", 1000, cs="C1", po="F2R1"),
        read("c", "chr1", 1000, cs="C1"),
        read("d", "chr1", 5000, cs="C2"),
    ]
    out, _ = run(tmp_path, recs)
    assert dx(out) == {"a": 1, "b": 1, "c": 1, "d": None}


def test_secondary_and_supplementary_reads_are_left_alone(tmp_path):
    recs = [
        read("a", "chr1", 1000, cs="C1", po="F1R2"),
        read("b", "chr1", 1000, cs="C1", po="F2R1"),
        read("sec", "chr1", 1000, cs="C1", po="F1R2", flag=256),
        read("sup", "chr1", 1000, cs="C1", po="F2R1", flag=2048),
    ]
    out, counters = run(tmp_path, recs)
    assert dx(out) == {"a": 1, "b": 1, "sec": None, "sup": None}
    assert counters["reads_with_cs"] == 2


def test_unplaced_unmapped_reads_pass_through(tmp_path):
    recs = [
        read("a", "chr1", 1000, cs="C1", po="F1R2"),
        read("b", "chr1", 1000, cs="C1", po="F2R1"),
        read("un", None, 0, flag=4, cs="C9", po="F1R2"),
    ]
    out, counters = run(tmp_path, recs)
    assert dx(out) == {"a": 1, "b": 1, "un": None}
    assert counters["reads"] == 3


def test_order_and_every_other_tag_are_kept_and_dx_is_an_integer(tmp_path):
    recs = [
        read("a", "chr1", 1000, cs="C1", po="F1R2", tags={"nf": 4, "nr": 0}),
        read("b", "chr1", 1000, cs="C1", po="F2R1", tags={"nf": 0, "nr": 3}),
        read("c", "chr1", 3000, cs="C2", po="F1R2"),
    ]
    out, _ = run(tmp_path, recs)
    with pysam.AlignmentFile(str(out), "rb") as bam:
        got = list(bam)
    assert [r.query_name for r in got] == ["a", "b", "c"]
    assert got[0].get_tag("nf") == 4
    assert got[1].get_tag("nr") == 3
    assert isinstance(got[0].get_tag("dx"), int)


def test_custom_tag_names(tmp_path):
    recs = [
        read("a", "chr1", 1000, cs="C1", po="F1R2"),
        read("b", "chr1", 1000, cs="C1", po="F2R1"),
    ]
    out, _ = run(tmp_path, recs, tag="dd")
    with pysam.AlignmentFile(str(out), "rb") as bam:
        assert [r.get_tag("dd") for r in bam] == [1, 1]


@pytest.mark.parametrize("pad", [0, 150, 2_000])
def test_a_family_in_one_place_is_found_at_any_pad(tmp_path, pad):
    recs = [read("a", "chr1", 1000, cs="C1", po="F1R2"), read("b", "chr1", 1000, cs="C1", po="F2R1")]
    out, _ = run(tmp_path, recs, pad=pad)
    assert dx(out) == {"a": 1, "b": 1}


def test_cli_writes_an_indexed_output_and_logs_a_summary(tmp_path):
    inp = tmp_path / "in.bam"
    with pysam.AlignmentFile(str(inp), "wb", header=HEADER) as fh:
        fh.write(read("a", "chr1", 1000, cs="C1", po="F1R2"))
        fh.write(read("b", "chr1", 1000, cs="C1", po="F2R1"))
    out = tmp_path / "out.bam"
    mdc.main([str(inp), str(out)])
    with pysam.AlignmentFile(str(out), "rb") as bam:
        assert bam.has_index()
        assert [r.get_tag("dx") for r in bam.fetch("chr1")] == [1, 1]
