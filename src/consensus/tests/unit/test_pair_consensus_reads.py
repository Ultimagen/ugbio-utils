#!/usr/bin/env python3
"""
Tests for pair_consensus_reads.py on small synthetic unaligned consensus BAMs.

Run with:  pytest src/consensus/tests/unit/test_pair_consensus_reads.py
"""

import pysam
import pytest
from ugbio_consensus import pair_consensus_reads as pcr

HEADER = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "RG": [{"ID": "Z1", "SM": "s", "tp": "reference"}]})
FWD_SEQ = "AACCGGTTAC"
REV_SEQ = "GGGATTTCCA"
FWD_QUAL = "ABCDEFGHIJ"
REV_QUAL = "0123456789"


def comp_rev(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def read(name, seq, qual, *, cs="C1", po="F1R2", flag=4, tags=None):
    seg = pysam.AlignedSegment(HEADER)
    seg.query_name = name
    seg.flag = flag
    seg.query_sequence = seq
    seg.query_qualities = pysam.qualitystring_to_array(qual)
    for tag, value in {"RG": "Z1", "CS": cs, "po": po, "nf": 3, "tp": None, **(tags or {})}.items():
        if value is not None:
            seg.set_tag(tag, value)
    return seg


def strand_pair(stem, **kwargs):
    return [
        read(stem + "0", FWD_SEQ, FWD_QUAL, **kwargs),
        read(stem + "1", REV_SEQ, REV_QUAL, **kwargs),
    ]


def run(tmp_path, records, **kwargs):
    inp, out = tmp_path / "in.bam", tmp_path / "out.bam"
    with pysam.AlignmentFile(str(inp), "wb", header=HEADER) as fh:
        for rec in records:
            fh.write(rec)
    counters = pcr.pair_reads(str(inp), str(out), **kwargs)
    with pysam.AlignmentFile(str(out), "rb", check_sq=False) as bam:
        return list(bam), counters


def test_f1r2_pair_is_r1_forward_and_r2_reverse_complemented(tmp_path):
    got, counters = run(tmp_path, strand_pair("p1x", po="F1R2"), buckets=4)
    assert [r.query_name for r in got] == ["p1x", "p1x"]
    r1, r2 = got
    assert (r1.flag, r2.flag) == (77, 141)
    assert r1.query_sequence == FWD_SEQ
    assert r2.query_sequence == comp_rev(REV_SEQ)
    assert pysam.array_to_qualitystring(r2.query_qualities) == REV_QUAL[::-1]
    assert counters["paired"] == 1


def test_f2r1_pair_is_r1_reverse_complemented_and_r2_forward(tmp_path):
    got, _ = run(tmp_path, strand_pair("p2x", po="F2R1"), buckets=4)
    r1, r2 = got
    assert (r1.flag, r2.flag) == (77, 141)
    assert r1.query_sequence == comp_rev(REV_SEQ)
    assert r2.query_sequence == FWD_SEQ


def test_tags_header_and_mate_fields_are_kept(tmp_path):
    got, _ = run(tmp_path, strand_pair("p3x", tags={"nf": 7}), buckets=2)
    for rec in got:
        assert (rec.get_tag("CS"), rec.get_tag("po"), rec.get_tag("nf"), rec.get_tag("RG")) == ("C1", "F1R2", 7, "Z1")
        assert rec.is_unmapped and rec.mate_is_unmapped
        assert rec.next_reference_id == -1
    with pysam.AlignmentFile(str(tmp_path / "out.bam"), "rb", check_sq=False) as bam:
        assert bam.header.to_dict()["RG"][0]["tp"] == "reference"


def test_a_read_without_a_partner_is_written_as_it_was(tmp_path):
    lone = read("lone0", FWD_SEQ, FWD_QUAL)
    got, counters = run(tmp_path, [lone], buckets=3)
    assert [(r.query_name, r.flag, r.query_sequence) for r in got] == [("lone0", 4, FWD_SEQ)]
    assert counters["no_partner"] == 1


@pytest.mark.parametrize("po", ["FF", "RR"])
def test_other_orientations_stay_single_end(tmp_path, po):
    got, counters = run(tmp_path, strand_pair("q", po=po), buckets=3)
    assert sorted(r.query_name for r in got) == ["q0", "q1"]
    assert {r.flag for r in got} == {4}
    assert counters["other_orientation"] == 2


def test_partners_that_disagree_on_po_or_cs_stay_single_end(tmp_path):
    a = [read("a0", FWD_SEQ, FWD_QUAL, po="F1R2"), read("a1", REV_SEQ, REV_QUAL, po="F2R1")]
    b = [read("b0", FWD_SEQ, FWD_QUAL, cs="X"), read("b1", REV_SEQ, REV_QUAL, cs="Y")]
    got, counters = run(tmp_path, a + b, buckets=3)
    assert {r.flag for r in got} == {4}
    assert sorted(r.query_name for r in got) == ["a0", "a1", "b0", "b1"]
    assert counters["partners_disagree"] == 4


def test_two_reads_with_one_digit_are_ambiguous_and_left_alone(tmp_path):
    got, counters = run(tmp_path, [read("d0", FWD_SEQ, FWD_QUAL), read("d0", FWD_SEQ, FWD_QUAL)], buckets=2)
    assert len(got) == 2 and counters["ambiguous"] == 2


def test_stems_that_extend_one_another_are_not_mixed_up(tmp_path):
    """Stem 'x1' (reads x10, x11) is not stem 'x' (reads x0, x1)."""
    recs = strand_pair("x") + strand_pair("x1")
    got, counters = run(tmp_path, recs, buckets=1)
    assert sorted(r.query_name for r in got) == ["x", "x", "x1", "x1"]
    assert counters["paired"] == 2


def test_the_two_mates_come_out_adjacent_whatever_the_input_order_or_bucket_count(tmp_path):
    recs = []
    for i in range(30):
        recs += strand_pair(f"n{i}-", po="F1R2" if i % 2 else "F2R1")
    recs = recs[::2] + recs[1::2]
    one, _ = run(tmp_path, recs, buckets=1)
    many, counters = run(tmp_path, recs, buckets=7)
    for got in (one, many):
        assert [got[i].query_name for i in range(0, 60, 2)] == [got[i + 1].query_name for i in range(0, 60, 2)]
        assert [got[i].flag for i in range(0, 60, 2)] == [77] * 30
    assert sorted(r.query_name + str(r.flag) for r in one) == sorted(r.query_name + str(r.flag) for r in many)
    assert counters["paired"] == 30


def test_cli_writes_a_counters_file(tmp_path):
    inp, out, cnt = tmp_path / "in.bam", tmp_path / "out.bam", tmp_path / "c.tsv"
    with pysam.AlignmentFile(str(inp), "wb", header=HEADER) as fh:
        for rec in strand_pair("z") + [read("solo0", FWD_SEQ, FWD_QUAL)]:
            fh.write(rec)
    pcr.main([str(inp), str(out), "--counters", str(cnt), "--buckets", "3"])
    lines = dict(line.split("\t") for line in cnt.read_text().split("\n") if line)
    assert (lines["reads"], lines["paired"], lines["no_partner"]) == ("3", "1", "1")
