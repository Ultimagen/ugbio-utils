#!/usr/bin/env python3
"""
Tests for set_consensus_strand.py on small synthetic consensus BAMs.

Run with:  pytest src/consensus/tests/unit/test_set_consensus_strand.py
"""

import pysam
import pytest
from ugbio_consensus import set_consensus_strand as scs

HEADER = pysam.AlignmentHeader.from_dict(
    {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 100_000}],
        "PG": [{"ID": "ua", "PN": "ua"}],
        "CO": ["existing comment"],
    }
)


def read(name, pos, flag, tags):
    seg = pysam.AlignedSegment(HEADER)
    seg.query_name = name
    seg.flag = flag
    seg.query_sequence = "ACGT"
    seg.query_qualities = pysam.qualitystring_to_array("IIII")
    seg.reference_id = 0
    seg.reference_start = pos
    seg.mapping_quality = 60
    seg.cigartuples = [(0, 4)]
    seg.set_tags(tags)
    return seg


def write_bam(path, reads):
    with pysam.AlignmentFile(str(path), "wb", header=HEADER) as out:
        for r in reads:
            out.write(r)
    return str(path)


# A0/A1 reverse and forward majority; B0 already reverse; B1 tie; C0 consensus without CS;
# D pass-through singleton (no nf/nr); E supplementary; F paired record; G secondary
READS = [
    read("A0", 100, 0, [("nf", 1), ("nr", 5), ("CS", "fam1")]),
    read("A1", 110, 0, [("nf", 5), ("nr", 1), ("CS", "fam1")]),
    read("B0", 120, 16, [("nf", 0), ("nr", 3), ("CS", "fam2")]),
    read("B1", 130, 0, [("nf", 2), ("nr", 2), ("CS", "fam2")]),
    read("C0", 140, 0, [("nf", 1), ("nr", 4)]),
    read("D", 150, 0, [("MI", "fam3")]),
    read("E", 160, 2048, [("nf", 1), ("nr", 9)]),
    read("F", 170, 1 | 2 | 64, [("nf", 1), ("nr", 9), ("CS", "fam4")]),
    read("G", 180, 256, [("nf", 1), ("nr", 9)]),
]


@pytest.fixture
def result(tmp_path):
    inp = write_bam(tmp_path / "in.bam", READS)
    out = str(tmp_path / "out.bam")
    counters = scs.set_consensus_strand(inp, out)
    with pysam.AlignmentFile(out) as fin:
        records = {r.query_name: r for r in fin}
        header = str(fin.header)
    return records, counters, header


@pytest.mark.parametrize(
    ("name", "flag"),
    [
        ("A0", 16),  # reverse majority sets 0x10
        ("A1", 0),  # forward majority leaves it clear
        ("B0", 16),  # already reverse stays reverse
        ("B1", 0),  # a tie is forward
        ("C0", 16),  # a missing CS does not stop the flag
        ("D", 0),  # singleton untouched
        ("E", 2048),  # supplementary untouched
        ("F", 1 | 2 | 64),  # paired untouched
        ("G", 256),  # secondary untouched
    ],
)
def test_flags(result, name, flag):
    assert result[0][name].flag == flag


def test_assigns_instead_of_toggling(tmp_path):
    inp = write_bam(tmp_path / "in.bam", [read("R", 100, 16, [("nf", 5), ("nr", 1), ("CS", "f")])])
    scs.set_consensus_strand(inp, str(tmp_path / "out.bam"))
    with pysam.AlignmentFile(str(tmp_path / "out.bam")) as fin:
        assert next(iter(fin)).flag == 0


def test_only_flag_changes(result):
    original = {r.query_name: r for r in READS}
    for name, rec in result[0].items():
        before = original[name]
        assert rec.query_name == before.query_name
        assert rec.reference_start == before.reference_start
        assert rec.cigarstring == before.cigarstring
        assert rec.query_sequence == before.query_sequence
        assert rec.get_tags() == before.get_tags()


def test_counters(result):
    counters = result[1]
    assert counters["total_records"] == 9
    assert counters["supplementary_or_secondary"] == 2
    assert counters["paired_left_alone"] == 1
    assert counters["no_nf_nr_tags"] == 1
    assert counters["no_cs_tag"] == 1
    assert counters["majority_forward"] == 1
    assert counters["majority_reverse"] == 3
    assert counters["majority_tie"] == 1
    assert counters["flag_reverse_set"] == 2
    assert counters["flag_reverse_cleared"] == 0
    assert counters["flag_unchanged"] == 3


def test_header_keeps_order_and_adds_one_comment(result):
    lines = result[2].splitlines()
    assert [line.split("\t")[0] for line in lines] == ["@HD", "@SQ", "@PG", "@CO", "@CO"]
    assert lines[3] == "@CO\texisting comment"
    assert lines[4] == scs.CO_LINE


def test_empty_input_adds_no_comment(tmp_path):
    inp = write_bam(tmp_path / "in.bam", [])
    out = str(tmp_path / "out.bam")
    assert not scs.set_consensus_strand(inp, out)
    with pysam.AlignmentFile(out) as fin:
        assert "set from the nf/nr majority" not in str(fin.header)


def test_failure_conditions():
    ok = scs.Counter(total_records=3)
    assert scs.check_counters(ok, require_cs_tag=True) == []
    assert len(scs.check_counters(scs.Counter(no_nf_nr_tags=1), require_cs_tag=False)) == 1
    assert scs.check_counters(scs.Counter(no_cs_tag=2), require_cs_tag=False) == []
    assert len(scs.check_counters(scs.Counter(no_cs_tag=2), require_cs_tag=True)) == 1


def test_main_writes_counters_and_fails_without_cs(tmp_path):
    inp = write_bam(tmp_path / "in.bam", READS)
    counters = tmp_path / "counters.txt"
    base = [inp, str(tmp_path / "out.bam"), "--counters", str(counters)]
    assert scs.main(base + ["--require-cs-tag"]) == 1
    assert "no_cs_tag\t1" in counters.read_text().splitlines()
    # the same input without the CS requirement, and with the CS-less read fixed
    assert scs.main(base) == 1  # D still has no nf/nr tag
    clean = write_bam(tmp_path / "clean.bam", [r for r in READS if r.query_name not in {"D", "C0"}])
    assert scs.main([clean, str(tmp_path / "out2.bam"), "--counters", str(counters), "--require-cs-tag"]) == 0


def test_paired_records_are_checked_for_nf_nr_and_cs(tmp_path):
    paired = [
        read("P1", 100, 1 | 64, [("nf", 2), ("nr", 1)]),  # no CS
        read("P2", 110, 1 | 128, [("CS", "f")]),  # no nf/nr
    ]
    inp = write_bam(tmp_path / "in.bam", paired)
    counters = scs.set_consensus_strand(inp, str(tmp_path / "out.bam"))
    assert (counters["no_cs_tag"], counters["no_nf_nr_tags"], counters["paired_left_alone"]) == (1, 1, 1)
    assert len(scs.check_counters(counters, require_cs_tag=True)) == 2
    with pysam.AlignmentFile(str(tmp_path / "out.bam")) as fin:
        assert [r.flag for r in fin] == [1 | 64, 1 | 128]
