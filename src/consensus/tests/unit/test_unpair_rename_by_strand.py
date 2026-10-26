#!/usr/bin/env python3
"""
Tests for unpair_rename_by_strand.py on small synthetic paired-end BAMs.

Run with:  pytest src/consensus/tests/unit/test_unpair_rename_by_strand.py
"""

import array
import random

import pysam
import pytest
from ugbio_consensus import unpair_rename_by_strand as ubs

HEADER = pysam.AlignmentHeader.from_dict(
    {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 100_000}, {"SN": "chr2", "LN": 50_000}],
    }
)


def read(name, chrom, pos, flag, tags=(), mate=None):
    """A 4 bp read; ``pos`` is 1-based. ``mate`` is (chrom, pos, tlen) to give the record mate fields."""
    seg = pysam.AlignedSegment(HEADER)
    seg.query_name = name
    seg.flag = flag
    seg.query_sequence = "ACGT"
    seg.query_qualities = pysam.qualitystring_to_array("IIII")
    if chrom is None:
        seg.reference_id = -1
        seg.reference_start = -1
    else:
        seg.reference_id = HEADER.get_tid(chrom)
        seg.reference_start = pos - 1
        seg.mapping_quality = 60
        seg.cigartuples = [(0, 4)]
    if mate:
        seg.next_reference_id = HEADER.get_tid(mate[0])
        seg.next_reference_start = mate[1] - 1
        seg.template_length = mate[2]
    seg.set_tags(list(tags))
    return seg


def write_indexed_bam(path, reads):
    with pysam.AlignmentFile(str(path), "wb", header=HEADER) as out:
        for r in reads:
            out.write(r)
    pysam.index(str(path))
    return str(path)


def tag_pairs(reads, **kwargs):
    """Run only the pair orientation step over ``reads`` and return the records it writes."""
    tagger = ubs.PairOrientationTagger(
        "po", kwargs.get("window", 2000), kwargs.get("max_queue", 4_194_304), lambda *_: True
    )
    return list(tagger.process(iter(reads))), tagger.counts


def po(rec):
    return rec.get_tag("po") if rec.has_tag("po") else None


PAIRS = [
    read("A", "chr1", 100, 99, [("MI", "fam1"), ("DS", 2), ("CS", "fam1")], ("chr1", 200, 104)),
    read("A", "chr1", 200, 147, [("MI", "fam1"), ("DS", 2), ("CS", "fam1")], ("chr1", 100, -104)),
    read("B", "chr1", 250, 163, [("MI", "fam2"), ("DS", 2), ("CS", "fam1")], ("chr1", 300, 54)),
    read("B", "chr1", 300, 83, [("MI", "fam2"), ("DS", 2), ("CS", "fam1")], ("chr1", 250, -54)),
    read("C", "chr1", 400, 2145, [("MI", "fam3")]),
]


@pytest.fixture
def unpaired(tmp_path):
    inp = write_indexed_bam(tmp_path / "in.bam", PAIRS)
    out = tmp_path / "out.bam"
    counters = ubs.unpair_by_strand(inp, str(out), workdir=str(tmp_path))
    return list(pysam.AlignmentFile(str(out))), counters, str(out)


def test_names_get_the_strand_digit(unpaired):
    assert [r.query_name for r in unpaired[0]] == ["A0", "A1", "B0", "B1", "C"]


def test_mi_is_suffixed_and_cs_is_not(unpaired):
    recs = {r.query_name: r for r in unpaired[0]}
    assert recs["A0"].get_tag("MI") == "fam10"
    assert recs["A1"].get_tag("MI") == "fam11"
    assert {recs[n].get_tag("CS") for n in ("A0", "A1", "B0", "B1")} == {"fam1"}
    assert recs["C"].get_tag("MI") == "fam3"


def test_pair_bits_and_mate_fields_are_cleared(unpaired):
    recs = {r.query_name: r for r in unpaired[0]}
    assert [recs[n].flag for n in ("A0", "A1", "B0", "B1")] == [0, 16, 0, 16]
    for name in ("A0", "A1", "B0", "B1"):
        assert recs[name].next_reference_id == -1
        assert recs[name].next_reference_start == -1
        assert recs[name].template_length == 0
    assert recs["C"].flag == 2145  # supplementary is left alone


def test_counters(unpaired):
    assert unpaired[1] == {
        "total_records": 5,
        "supp_or_secondary": 1,
        "renamed_forward": 2,
        "renamed_reverse": 2,
        "cs_tagged": 4,
    }


def test_mi_keeps_its_place_among_the_tags(tmp_path):
    tags = [("tp", array.array("b", [0, 1, -1])), ("MI", "fam"), ("DS", 2), ("CS", "c"), ("po", "F1R2"), ("xx", "last")]
    seg = read("A", "chr1", 100, 99, tags, ("chr1", 200, 104))
    inp = write_indexed_bam(tmp_path / "in.bam", [seg])
    out = tmp_path / "out.bam"
    ubs.unpair_by_strand(inp, str(out), workdir=str(tmp_path))
    (rec,) = pysam.AlignmentFile(str(out))
    assert [t[0] for t in rec.get_tags()] == ["tp", "MI", "DS", "CS", "po", "xx"]
    assert list(rec.get_tag("tp")) == [0, 1, -1]
    assert rec.get_tag("MI") == "fam0"
    assert rec.get_tag("DS") == 2


def test_header_comment_added_once(unpaired):
    with pysam.AlignmentFile(unpaired[2]) as fin:
        assert str(fin.header).count(ubs.CO_LINE) == 1


PO_READS = [
    read("P1", "chr1", 100, 65, [("MI", "f1")]),
    read("P1", "chr1", 150, 2113),
    read("P1", "chr1", 200, 145, [("MI", "f1")]),
    read("P2", "chr1", 250, 129),
    read("P2", "chr1", 300, 81, [("po", "STALE")]),
    read("P3", "chr1", 400, 65),
    read("P3", "chr1", 450, 129),
    read("P4", "chr1", 500, 145, [("po", "STALE")]),
    read("FAR", "chr1", 1000, 65),
    read("FAR", "chr1", 9000, 145),
    read("X", "chr1", 9500, 65),
    read("X", "chr2", 100, 145),
    read("U", "chr2", 200, 0),
]


@pytest.fixture
def tagged():
    reads, counts = tag_pairs(PO_READS)
    return reads, counts


def test_orientation_is_written_to_both_mates(tagged):
    recs = tagged[0]
    assert [po(recs[i]) for i in (0, 2)] == ["F1R2", "F1R2"]  # P1
    assert [po(recs[i]) for i in (3, 4)] == ["F2R1", "F2R1"]  # P2, the R2 mate comes first
    assert [po(recs[i]) for i in (5, 6)] == ["FF", "FF"]  # P3


def test_orientation_cases_without_a_tag(tagged):
    recs = tagged[0]
    assert po(recs[1]) is None  # supplementary
    assert po(recs[7]) is None  # mate absent, stale tag dropped
    assert po(recs[8]) is None and po(recs[9]) is None  # mate beyond the window
    assert po(recs[10]) is None and po(recs[11]) is None  # mate on another contig
    assert po(recs[12]) is None  # unpaired read


def test_stale_tag_is_replaced_not_duplicated(tagged):
    assert [k for k, _ in tagged[0][4].get_tags()].count("po") == 1


def test_orientation_keeps_the_input_order(tagged):
    assert [(r.query_name, r.flag) for r in tagged[0]] == [(r.query_name, r.flag) for r in PO_READS]


def test_orientation_counters(tagged):
    assert dict(tagged[1]) == {"F1R2": 2, "F2R1": 2, "FF": 2, "none": 6}


def test_tiny_queue_still_writes_every_read_in_order():
    reads, _ = tag_pairs(PO_READS, max_queue=3)
    assert [(r.query_name, r.flag) for r in reads] == [(r.query_name, r.flag) for r in PO_READS]


def test_same_mate_twice_does_not_pair():
    reads, _ = tag_pairs([read("D", "chr1", 100, 65), read("D", "chr1", 110, 65), read("D", "chr1", 120, 145)])
    assert [po(r) for r in reads] == [None, "F1R2", "F1R2"]


def random_reads(seed, n_pairs=300):
    rng = random.Random(seed)
    reads = []
    for i in range(n_pairs):
        chrom = rng.choice(["chr1", "chr2"])
        p1 = rng.randint(1, 4_000)
        p2 = p1 + rng.randint(-300, 600) if rng.random() > 0.1 else p1 + rng.randint(3_000, 9_000)
        p2 = max(1, p2)
        rev1, rev2 = rng.random() < 0.5, rng.random() < 0.5
        tags = [("MI", f"m{i}"), ("CS", f"c{i // 2}")]
        reads.append(read(f"r{i}", chrom, p1, 1 | 2 | 64 | (16 if rev1 else 32), tags))
        reads.append(read(f"r{i}", chrom, p2, 1 | 2 | 128 | (16 if rev2 else 32), tags))
        if rng.random() < 0.05:
            reads.append(read(f"r{i}", chrom, p1 + 5, 2048 | 1 | 64, tags))
    for i in range(10):  # unplaced unmapped reads at the end of the file
        reads.append(read(f"u{i}", None, 0, 4))
    reads.sort(key=lambda r: (r.reference_id if r.reference_id >= 0 else 1 << 30, r.reference_start))
    return reads


def records(path):
    with pysam.AlignmentFile(str(path)) as fin:
        return [r.to_string() for r in fin]


@pytest.mark.parametrize("tag", [None, "po"])
def test_output_does_not_depend_on_the_shard_size(tmp_path, tag):
    inp = write_indexed_bam(tmp_path / "in.bam", random_reads(1))
    whole = tmp_path / "whole.bam"
    sharded = tmp_path / "sharded.bam"
    kwargs = {"pair_orientation_tag": tag, "workdir": str(tmp_path)}
    c_whole = ubs.unpair_by_strand(inp, str(whole), shard_size=10_000_000, **kwargs)
    c_sharded = ubs.unpair_by_strand(inp, str(sharded), shard_size=700, threads=2, **kwargs)
    assert records(whole) == records(sharded)
    assert c_whole == c_sharded
    assert c_whole["total_records"] == len(random_reads(1))
    if tag:
        assert c_whole["pair_orientation_F1R2"] + c_whole["pair_orientation_F2R1"] > 0
        assert c_whole["pair_orientation_none"] > 0


def test_unplaced_reads_are_kept(tmp_path):
    inp = write_indexed_bam(tmp_path / "in.bam", random_reads(2))
    out = tmp_path / "out.bam"
    ubs.unpair_by_strand(inp, str(out), shard_size=1000, workdir=str(tmp_path))
    names = [r.query_name for r in pysam.AlignmentFile(str(out)) if r.is_unmapped]
    assert names == [f"u{i}0" for i in range(10)]


def test_missing_index_is_built(tmp_path):
    inp = write_indexed_bam(tmp_path / "in.bam", PAIRS)
    (tmp_path / "in.bam.bai").unlink()
    out = tmp_path / "out.bam"
    counters = ubs.unpair_by_strand(inp, str(out), workdir=str(tmp_path))
    assert counters["total_records"] == 5
    assert not (tmp_path / "in.bam.bai").exists()


def test_wrong_record_count_fails(tmp_path, monkeypatch):
    inp = write_indexed_bam(tmp_path / "in.bam", PAIRS)
    original = ubs.make_windows

    def fewer(fin, shard_size):
        windows, total = original(fin, shard_size)
        return windows, total + 1

    monkeypatch.setattr(ubs, "make_windows", fewer)
    with pytest.raises(RuntimeError, match="input holds"):
        ubs.unpair_by_strand(inp, str(tmp_path / "out.bam"), workdir=str(tmp_path))


def test_main_rejects_a_bad_tag(tmp_path):
    inp = write_indexed_bam(tmp_path / "in.bam", PAIRS)
    assert ubs.main([inp, str(tmp_path / "out.bam"), "--pair-orientation-tag", "p"]) == 1


def test_main_writes_the_counters(tmp_path):
    inp = write_indexed_bam(tmp_path / "in.bam", PAIRS)
    counters = tmp_path / "counters.txt"
    assert ubs.main([inp, str(tmp_path / "out.bam"), "--counters", str(counters), "--pair-orientation-tag", "po"]) == 0
    lines = counters.read_text().splitlines()
    assert lines[:2] == ["total_records\t5", "supp_or_secondary\t1"]
    assert lines[-1].startswith("pair_orientation_none\t")


def test_cram_input_with_crai_is_processed_in_full(tmp_path):
    """CRAI stores no per-contig counts, so the windows and the record check must not rely on them."""
    rng = random.Random(7)
    fasta = tmp_path / "ref.fa"
    with open(fasta, "w") as fh:
        for sq in HEADER.to_dict()["SQ"]:
            fh.write(f">{sq['SN']}\n{''.join(rng.choice('ACGT') for _ in range(sq['LN']))}\n")
    pysam.faidx(str(fasta))
    reads = random_reads(3)
    cram = tmp_path / "in.cram"
    with pysam.AlignmentFile(str(cram), "wc", header=HEADER, reference_filename=str(fasta)) as out:
        for r in reads:
            out.write(r)
    pysam.index(str(cram))
    bam = write_indexed_bam(tmp_path / "in.bam", reads)
    kwargs = {"shard_size": 1000, "workdir": str(tmp_path), "pair_orientation_tag": "po"}
    c_cram = ubs.unpair_by_strand(str(cram), str(tmp_path / "from_cram.bam"), reference=str(fasta), **kwargs)
    c_bam = ubs.unpair_by_strand(bam, str(tmp_path / "from_bam.bam"), **kwargs)
    assert c_cram["total_records"] == len(reads)
    assert c_cram == c_bam
    assert [r.split("\t")[:9] for r in records(tmp_path / "from_cram.bam")] == [
        r.split("\t")[:9] for r in records(tmp_path / "from_bam.bam")
    ]
