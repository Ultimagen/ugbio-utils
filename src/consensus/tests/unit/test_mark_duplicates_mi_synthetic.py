#!/usr/bin/env python3
"""
End-to-end and unit tests for mark_duplicates_mi.py on small synthetic BAMs.

test_mark_duplicates_mi.py pins the tool against one real duplex molecule. This file builds
its inputs from scratch, so every dedup-key corner (boundary drift, UMI order, strand,
dovetailing, trans-contig mates, ...) is exercised on purpose, in whole-file and sharded
mode alike. Every pair is 2 x 50 bp, UMIs are 8 bp, and positions are 0-based.

Run with:  pytest src/consensus/tests/unit/test_mark_duplicates_mi_synthetic.py
"""

import argparse
import random
import shutil
import subprocess
import sys

import pysam
import pytest
from ugbio_consensus import mark_duplicates_mi as mdmi

READ_LEN = 50
UMI_A, UMI_B, UMI_C = "AAAAAAAA", "CCCCCCCC", "GGGGGGGG"
CONTIG_LEN = 20_000

needs_samtools = pytest.mark.skipif(shutil.which("samtools") is None, reason="sharded mode needs the samtools CLI")

HEADER = pysam.AlignmentHeader.from_dict(
    {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": CONTIG_LEN}, {"SN": "chr2", "LN": CONTIG_LEN}],
    }
)


# --- builders ---------------------------------------------------------------------------


def segment(name, chrom, pos, *, flag, length=READ_LEN, tags=None):
    """One mapped (or, with chrom=None, unplaced unmapped) record."""
    seg = pysam.AlignedSegment(HEADER)
    seg.query_name = name
    seg.flag = flag
    seg.query_sequence = "A" * length
    seg.query_qualities = pysam.qualitystring_to_array("I" * length)
    if chrom is None:
        seg.reference_id = -1
        seg.reference_start = -1
    else:
        seg.reference_id = HEADER.get_tid(chrom)
        seg.reference_start = pos
        seg.mapping_quality = 60
        seg.cigartuples = [(0, length)]
    for tag, value in (tags or {}).items():
        seg.set_tag(tag, value, value_type="Z")
    return seg


def pair(name, chrom, left, right, u5, u3, *, kind="F1R2", mate_chrom=None, tags=None):
    """
    A read pair whose mates start at `left` and `right`. `kind` says which strands the
    mates are on and which of them is R1: F1R2 has R1 forward at `left`, F2R1 has R1
    reverse at `right`; FF and RR keep R1 at `left`. u5 sits on R1 and u3 on R2.
    """
    r1_reverse = kind in ("F2R1", "RR")
    r2_reverse = kind in ("F1R2", "RR")
    r1_pos, r2_pos = (right, left) if kind == "F2R1" else (left, right)
    r1_flag = 1 | 64 | (16 if r1_reverse else 0)
    r2_flag = 1 | 128 | (16 if r2_reverse else 0)
    r1 = segment(name, chrom, r1_pos, flag=r1_flag, tags={"u5": u5, **(tags or {})})
    r2 = segment(name, mate_chrom or chrom, r2_pos, flag=r2_flag, tags={"u3": u3, **(tags or {})})
    return [r1, r2]


def flatten(*groups):
    return [rec for group in groups for rec in (group if isinstance(group, list) else [group])]


def build_bam(path, records, *, index=True):
    def order(rec):
        return (rec.reference_id if rec.reference_id >= 0 else 1 << 30, rec.reference_start)

    with pysam.AlignmentFile(str(path), "wb", header=HEADER) as out:
        for rec in sorted(flatten(records), key=order):
            out.write(rec)
    if index:
        pysam.index(str(path))
    return path


def cli(inp, *args, out=None, check=True):
    cmd = [sys.executable, "-m", "ugbio_consensus.mark_duplicates_mi", str(inp)]
    if out is not None:
        cmd.append(str(out))
    return subprocess.run([*cmd, *map(str, args)], check=check, capture_output=True, text=True)


def run(tmp_path, records, *args, name="out.bam"):
    inp = build_bam(tmp_path / "in.bam", records)
    cli(inp, *args, out=tmp_path / name)
    return tmp_path / name


def marked(path):
    """name -> {MI, DS, CS, po}; asserts that both mates of a pair carry identical tags."""
    out = {}
    with pysam.AlignmentFile(str(path), "rb") as bam:
        for read in bam:
            tags = {t: read.get_tag(t) if read.has_tag(t) else None for t in ("MI", "DS", "CS", "po")}
            assert out.setdefault(read.query_name, tags) == tags, f"mates of {read.query_name} disagree"
    return out


def snapshot(path):
    """Every record in file order: (name, flag, start, end, MI, DS, CS, po)."""
    with pysam.AlignmentFile(str(path), "rb") as bam:
        return [
            (
                r.query_name,
                r.flag,
                r.reference_name,
                r.reference_start,
                r.reference_end,
                *(r.get_tag(t) if r.has_tag(t) else None for t in ("MI", "DS", "CS", "po")),
            )
            for r in bam
        ]


def aux_order(path):
    """Tag names in emitted order, for the first record: a raw view of `samtools view`."""
    with pysam.AlignmentFile(str(path), "rb") as bam:
        return [t for t, _ in next(iter(bam)).get_tags()]


# --- whole-file: MI / DS ------------------------------------------------------------------


def test_duplicates_share_mi_ds_and_a_singleton_gets_neither_mi_nor_ds_above_one(tmp_path):
    recs = flatten(
        pair("c", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("a", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("b", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("lone", "chr1", 5000, 5200, UMI_A, UMI_B),
    )
    got = marked(run(tmp_path, recs))
    assert {n: (got[n]["MI"], got[n]["DS"]) for n in "abc"} == {n: ("a", 3) for n in "abc"}
    assert (got["lone"]["MI"], got["lone"]["DS"]) == (None, 1)


def test_mi_is_the_lexicographically_first_name_regardless_of_file_order(tmp_path):
    recs = flatten(
        pair("zz", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("mm", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("aa", "chr1", 1000, 1200, UMI_A, UMI_B),
    )
    assert {v["MI"] for v in marked(run(tmp_path, recs)).values()} == {"aa"}


@pytest.mark.parametrize(
    ("left2", "right2"),
    [(1001, 1200), (1000, 1201), (1100, 1200), (1000, 1300), (2000, 2200)],
    ids=["left-shifted", "right-shifted", "left-far", "right-far", "elsewhere"],
)
def test_pairs_with_a_different_boundary_are_not_duplicates(tmp_path, left2, right2):
    recs = flatten(pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B), pair("p2", "chr1", left2, right2, UMI_A, UMI_B))
    got = marked(run(tmp_path, recs))
    assert [got[n]["DS"] for n in ("p1", "p2")] == [1, 1]
    assert [got[n]["MI"] for n in ("p1", "p2")] == [None, None]


def test_mates_in_either_order_in_the_file_give_the_same_key(tmp_path):
    """R2 first in the file, R1 first in the file: the key is symmetric in the mates."""
    p1 = pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B)
    p2 = pair("p2", "chr1", 1000, 1200, UMI_A, UMI_B)
    p2.reverse()
    got = marked(run(tmp_path, flatten(p1, p2)))
    assert (got["p1"]["MI"], got["p2"]["MI"], got["p1"]["DS"]) == ("p1", "p1", 2)


def test_different_umis_split_a_position_into_separate_families_with_use_umi(tmp_path):
    recs = flatten(
        pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("p2", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("q1", "chr1", 1000, 1200, UMI_C, UMI_B),
        pair("q2", "chr1", 1000, 1200, UMI_C, UMI_B),
    )
    got = marked(run(tmp_path, recs, "--use-umi"))
    assert {n: (v["MI"], v["DS"]) for n, v in got.items()} == {
        "p1": ("p1", 2),
        "p2": ("p1", 2),
        "q1": ("q1", 2),
        "q2": ("q1", 2),
    }


def test_without_use_umi_the_umi_is_ignored(tmp_path):
    recs = flatten(
        pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("q1", "chr1", 1000, 1200, UMI_C, UMI_C),
    )
    got = marked(run(tmp_path, recs))
    assert (got["p1"]["MI"], got["q1"]["MI"], got["p1"]["DS"]) == ("p1", "p1", 2)


def test_u5_is_read_from_r1_and_u3_from_r2_only(tmp_path):
    """A stray u3 on R1 or u5 on R2 must not enter the key."""
    p1 = pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B)
    p2 = pair("p2", "chr1", 1000, 1200, UMI_A, UMI_B)
    p2[0].set_tag("u3", UMI_C, value_type="Z")
    p2[1].set_tag("u5", UMI_C, value_type="Z")
    got = marked(run(tmp_path, flatten(p1, p2), "--use-umi"))
    assert (got["p1"]["MI"], got["p2"]["MI"]) == ("p1", "p1")


def test_reads_without_umi_tags_cluster_together_with_use_umi(tmp_path):
    p1 = pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B)
    p2 = pair("p2", "chr1", 1000, 1200, UMI_A, UMI_B)
    for rec in flatten(p1, p2):
        for tag in ("u5", "u3"):
            if rec.has_tag(tag):
                rec.set_tag(tag, None)
    got = marked(run(tmp_path, flatten(p1, p2), "--use-umi"))
    assert (got["p1"]["MI"], got["p2"]["DS"]) == ("p1", 2)


def test_slash_suffix_on_mate_names_is_stripped_when_pairing(tmp_path):
    recs = []
    for stem in ("a", "b"):
        r1, r2 = pair(stem, "chr1", 1000, 1200, UMI_A, UMI_B)
        r1.query_name, r2.query_name = f"{stem}/1", f"{stem}/2"
        recs += [r1, r2]
    got = marked(run(tmp_path, recs))
    assert {v["DS"] for v in got.values()} == {2}
    assert {v["MI"] for v in got.values()} == {"a"}


def test_strip_run_id_drops_the_prefix_from_mi_and_cs(tmp_path):
    recs = flatten(
        pair("606174-bbb", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("606174-aaa", "chr1", 1000, 1200, UMI_A, UMI_B),
    )
    got = marked(run(tmp_path, recs, "--use-umi", "--strip-run-id"))
    assert {(v["MI"], v["CS"]) for v in got.values()} == {("aaa", "aaa")}


def test_strip_run_id_leaves_a_name_without_a_dash_alone(tmp_path):
    recs = flatten(pair("bbb", "chr1", 1000, 1200, UMI_A, UMI_B), pair("aaa", "chr1", 1000, 1200, UMI_A, UMI_B))
    got = marked(run(tmp_path, recs, "--use-umi", "--strip-run-id"))
    assert {(v["MI"], v["CS"]) for v in got.values()} == {("aaa", "aaa")}


def test_ds_is_written_as_an_integer_tag(tmp_path):
    out = run(tmp_path, pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B))
    with pysam.AlignmentFile(str(out), "rb") as bam:
        read = next(iter(bam))
        assert read.get_tag("DS", with_value_type=True)[1] in "cCsSiI"


def test_aux_order_is_mi_ds_cs(tmp_path):
    """The emitted aux order is part of the contract; only a raw `samtools view` diff sees it."""
    recs = flatten(pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B), pair("p2", "chr1", 1000, 1200, UMI_A, UMI_B))
    tags = aux_order(run(tmp_path, recs, "--use-umi"))
    assert [t for t in tags if t in ("MI", "DS", "CS")] == ["MI", "DS", "CS"]


def test_aux_order_with_pair_orientation_tag_appends_it_last(tmp_path):
    recs = flatten(pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B), pair("p2", "chr1", 1000, 1200, UMI_A, UMI_B))
    tags = aux_order(run(tmp_path, recs, "--use-umi", "--pair-orientation-tag", "po"))
    assert [t for t in tags if t in ("MI", "DS", "CS", "po")] == ["MI", "DS", "CS", "po"]


def test_output_is_indexed_and_keeps_record_order(tmp_path):
    recs = flatten(*(pair(f"p{i}", "chr1", 100 + 10 * i, 300 + 10 * i, UMI_A, UMI_B) for i in range(20)))
    inp = build_bam(tmp_path / "in.bam", recs)
    cli(inp, out=tmp_path / "out.bam")
    with pysam.AlignmentFile(str(tmp_path / "out.bam"), "rb") as bam:
        assert bam.has_index()
    assert [s[:5] for s in snapshot(tmp_path / "out.bam")] == [s[:5] for s in snapshot(inp)]


def test_stale_mi_and_ds_are_recomputed_not_trusted(tmp_path):
    p1 = pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B, tags={"MI": "stale"})
    for rec in p1:
        rec.set_tag("DS", 99, value_type="i")
    got = marked(run(tmp_path, p1))
    assert (got["p1"]["MI"], got["p1"]["DS"]) == (None, 1)


# --- whole-file: records that never cluster -------------------------------------------------


def test_unpaired_unmapped_secondary_and_transcontig_reads_are_singletons(tmp_path):
    recs = flatten(
        segment("unpaired", "chr1", 3000, flag=0, tags={"MI": "stale", "CS": "stale"}),
        segment("unplaced", None, 0, flag=4, tags={"MI": "stale", "CS": "stale"}),
        pair("trans", "chr1", 4000, 4000, UMI_A, UMI_B, mate_chrom="chr2"),
        pair("dup1", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("dup2", "chr1", 1000, 1200, UMI_A, UMI_B),
    )
    got = marked(run(tmp_path, recs, "--use-umi"))
    for name in ("unpaired", "unplaced", "trans"):
        assert got[name] == {"MI": None, "DS": 1, "CS": None, "po": None}, name
    assert got["dup1"]["DS"] == 2


def test_pair_with_one_mate_missing_is_a_singleton(tmp_path):
    only_r1 = pair("half", "chr1", 1000, 1200, UMI_A, UMI_B)[:1]
    recs = flatten(only_r1, pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B))
    got = marked(run(tmp_path, recs, "--use-umi"))
    assert got["half"] == {"MI": None, "DS": 1, "CS": None, "po": None}
    assert got["p1"]["DS"] == 1


def test_secondary_and_supplementary_records_do_not_form_pairs(tmp_path):
    """A secondary R2 sharing a primary pair's name must not pair with, or duplicate, anything."""
    p1 = pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B)
    p2 = pair("p2", "chr1", 1000, 1200, UMI_A, UMI_B)
    extra = segment("p1", "chr1", 1000, flag=1 | 128 | 256, tags={"u3": UMI_B})
    supp = segment("p2", "chr1", 1000, flag=1 | 128 | 2048, tags={"u3": UMI_B})
    got = marked(run(tmp_path, flatten(p1, p2, extra, supp), "--use-umi"))
    assert (got["p1"]["MI"], got["p1"]["DS"]) == ("p1", 2)


# --- whole-file: position tolerance and cross-strand ------------------------------------------


def _duplex(tmp_path, *args, shift=0, partner_kind="F2R1"):
    """One F1R2 pair (A,B) and its opposite-strand partner keyed (B,A), `shift` bp off on the right."""
    recs = flatten(
        pair("fwd", "chr1", 1000, 1200, UMI_A, UMI_B, kind="F1R2"),
        pair("rev", "chr1", 1000, 1200 + shift, UMI_B, UMI_A, kind=partner_kind),
    )
    return marked(run(tmp_path, recs, "--use-umi", *args))


def test_default_keeps_the_two_strands_in_separate_families_linked_by_cs(tmp_path):
    got = _duplex(tmp_path)
    assert (got["fwd"]["MI"], got["rev"]["MI"]) == (None, None)
    assert (got["fwd"]["DS"], got["rev"]["DS"]) == (1, 1)
    assert got["fwd"]["CS"] == got["rev"]["CS"] == "fwd"


def test_cross_strand_merges_the_two_strands_into_one_family(tmp_path):
    got = _duplex(tmp_path, "--cross-strand")
    assert (got["fwd"]["MI"], got["rev"]["MI"]) == ("fwd", "fwd")
    assert (got["fwd"]["DS"], got["rev"]["DS"]) == (2, 2)
    assert got["fwd"]["CS"] == got["rev"]["CS"] == "fwd"


def test_cross_strand_alone_does_not_bridge_a_boundary_shift(tmp_path):
    got = _duplex(tmp_path, "--cross-strand", shift=2)
    assert (got["fwd"]["DS"], got["rev"]["DS"]) == (1, 1)
    assert got["fwd"]["CS"] != got["rev"]["CS"]


def test_cross_strand_with_tolerance_bridges_the_shift(tmp_path):
    got = _duplex(tmp_path, "--cross-strand", "--position-tolerance", "2", shift=2)
    assert (got["fwd"]["MI"], got["rev"]["MI"], got["fwd"]["DS"]) == ("fwd", "fwd", 2)


def test_tolerance_without_cross_strand_never_merges_across_strands(tmp_path):
    got = _duplex(tmp_path, "--position-tolerance", "5", shift=2)
    assert (got["fwd"]["DS"], got["rev"]["DS"]) == (1, 1)


def test_tolerance_that_is_too_small_does_not_bridge(tmp_path):
    got = _duplex(tmp_path, "--cross-strand", "--position-tolerance", "1", shift=2)
    assert (got["fwd"]["DS"], got["rev"]["DS"]) == (1, 1)


def test_same_strand_umi_collision_is_not_a_duplex_pair(tmp_path):
    """(A,B) and (B,A) on the *same* strand: neither CS-linked nor merged, even with --cross-strand."""
    got = _duplex(tmp_path, "--cross-strand", partner_kind="F1R2")
    assert (got["fwd"]["DS"], got["rev"]["DS"]) == (1, 1)
    assert got["fwd"]["CS"] == "fwd"
    assert got["rev"]["CS"] == "rev"


def test_same_orientation_partners_merge_under_tolerance_alone(tmp_path):
    recs = flatten(
        pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("p2", "chr1", 1001, 1202, UMI_A, UMI_B),
        pair("p3", "chr1", 5000, 5200, UMI_A, UMI_B),
    )
    got = marked(run(tmp_path, recs, "--use-umi", "--position-tolerance", "2"))
    assert [(got[n]["MI"], got[n]["DS"]) for n in ("p1", "p2")] == [("p1", 2), ("p1", 2)]
    assert (got["p3"]["MI"], got["p3"]["DS"]) == (None, 1)


@pytest.mark.parametrize(("tolerance", "expected_ds"), [(0, 1), (1, 3), (2, 3), (5, 3)])
def test_tolerance_chains_transitively(tmp_path, tolerance, expected_ds):
    """Boundaries 1 bp apart in a row: the first and the third are 2 bp apart but join through the second."""
    recs = flatten(*(pair(f"p{i}", "chr1", 1000 + i, 1200 + i, UMI_A, UMI_B) for i in range(3)))
    got = marked(run(tmp_path, recs, "--use-umi", "--position-tolerance", tolerance))
    assert {v["DS"] for v in got.values()} == {expected_ds}
    if tolerance:
        assert {v["MI"] for v in got.values()} == {"p0"}


def test_tolerance_does_not_merge_different_umis(tmp_path):
    recs = flatten(pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B), pair("p2", "chr1", 1001, 1201, UMI_C, UMI_B))
    got = marked(run(tmp_path, recs, "--use-umi", "--position-tolerance", "3"))
    assert (got["p1"]["DS"], got["p2"]["DS"]) == (1, 1)


def test_tolerance_does_not_merge_across_contigs(tmp_path):
    recs = flatten(pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B), pair("p2", "chr2", 1000, 1200, UMI_A, UMI_B))
    got = marked(run(tmp_path, recs, "--use-umi", "--position-tolerance", "3"))
    assert (got["p1"]["DS"], got["p2"]["DS"]) == (1, 1)


def test_tolerance_and_cross_strand_are_ignored_without_use_umi(tmp_path):
    recs = flatten(pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B), pair("p2", "chr1", 1001, 1202, UMI_A, UMI_B))
    base = run(tmp_path, recs, name="base.bam")
    flagged = run(tmp_path, recs, "--position-tolerance", "5", "--cross-strand", name="flagged.bam")
    assert snapshot(base) == snapshot(flagged)
    assert {v["CS"] for v in marked(flagged).values()} == {None}


def test_defaults_are_unchanged_by_the_new_options(tmp_path):
    recs = flatten(
        pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("p2", "chr1", 1001, 1202, UMI_A, UMI_B),
        pair("p3", "chr1", 1000, 1200, UMI_B, UMI_A, kind="F2R1"),
    )
    default = run(tmp_path, recs, "--use-umi", name="default.bam")
    explicit = run(tmp_path, recs, "--use-umi", "--position-tolerance", "0", name="explicit.bam")
    assert snapshot(default) == snapshot(explicit)
    assert mdmi.DEFAULT_POSITION_TOLERANCE == 0


def test_merged_cluster_with_both_strands_has_cs_equal_to_mi(tmp_path):
    recs = flatten(
        pair("a1", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("a2", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("b1", "chr1", 1000, 1200, UMI_B, UMI_A, kind="F2R1"),
    )
    got = marked(run(tmp_path, recs, "--use-umi", "--cross-strand"))
    assert {(v["MI"], v["DS"], v["CS"]) for v in got.values()} == {("a1", 3, "a1")}


# --- CS: dovetailing and the FLAG --------------------------------------------------------------


def test_cs_reads_orientation_from_the_flag_for_dovetailed_pairs(tmp_path):
    """
    Fragment shorter than the read: the reverse mate starts at or before the forward one.
    Orientation by position would call the F1R2 pair here F2R1 and miss the link.
    """
    f1r2 = pair("fwd", "chr1", 1000, 1010, UMI_A, UMI_B, kind="F1R2")
    f2r1 = pair("rev", "chr1", 1000, 1010, UMI_B, UMI_A, kind="F2R1")
    # dovetail: R1 (forward) starts to the right of R2 (reverse)
    f1r2[0].reference_start, f1r2[1].reference_start = 1010, 1000
    got = marked(run(tmp_path, flatten(f1r2, f2r1), "--use-umi", "--pair-orientation-tag", "po"))
    assert got["fwd"]["CS"] == got["rev"]["CS"] == "fwd"
    assert (got["fwd"]["po"], got["rev"]["po"]) == ("F1R2", "F2R1")


def test_cs_is_not_set_for_reads_that_never_paired(tmp_path):
    recs = flatten(segment("solo", "chr1", 1000, flag=0), pair("p1", "chr1", 2000, 2200, UMI_A, UMI_B))
    got = marked(run(tmp_path, recs, "--use-umi"))
    assert got["solo"]["CS"] is None
    assert got["p1"]["CS"] == "p1"


def test_remarking_an_already_merged_file_gives_the_same_result(tmp_path):
    recs = flatten(
        pair("a1", "chr1", 1000, 1200, UMI_A, UMI_B),
        pair("b1", "chr1", 1001, 1202, UMI_B, UMI_A, kind="F2R1"),
        pair("c1", "chr1", 3000, 3200, UMI_C, UMI_C),
    )
    args = ("--use-umi", "--cross-strand", "--position-tolerance", "2", "--pair-orientation-tag", "po")
    first = run(tmp_path, recs, *args, name="first.bam")
    second = tmp_path / "second.bam"
    cli(first, *args, out=second)
    assert snapshot(second) == snapshot(first)


# --- --pair-orientation-tag ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["F1R2", "F2R1", "FF", "RR"])
def test_pair_orientation_value_for_each_strand_combination(tmp_path, kind):
    got = marked(run(tmp_path, pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B, kind=kind), "--pair-orientation-tag", "po"))
    assert got["p1"]["po"] == kind


def test_pair_orientation_works_for_a_dovetailed_pair(tmp_path):
    """The FLAG decides, not which mate starts further left."""
    rec = pair("p1", "chr1", 1000, 1010, UMI_A, UMI_B, kind="F2R1")
    rec[0].reference_start, rec[1].reference_start = 1000, 1010  # R1 reverse starts left of R2 forward
    assert marked(run(tmp_path, rec, "--pair-orientation-tag", "po"))["p1"]["po"] == "F2R1"


def test_pair_orientation_tag_name_is_configurable(tmp_path):
    out = run(tmp_path, pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B), "--pair-orientation-tag", "XO")
    with pysam.AlignmentFile(str(out), "rb") as bam:
        records = list(bam)
    assert {r.get_tag("XO") for r in records} == {"F1R2"}
    assert not any(r.has_tag("po") for r in records)


def test_pair_orientation_tag_is_a_string_tag(tmp_path):
    out = run(tmp_path, pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B), "--pair-orientation-tag", "po")
    with pysam.AlignmentFile(str(out), "rb") as bam:
        assert {r.get_tag("po", with_value_type=True)[1] for r in bam} == {"Z"}


def test_pair_orientation_is_independent_of_dedup_options(tmp_path):
    recs = flatten(
        pair("a", "chr1", 1000, 1200, UMI_A, UMI_B, kind="F1R2"),
        pair("b", "chr1", 1001, 1202, UMI_B, UMI_A, kind="F2R1"),
        pair("c", "chr1", 4000, 4200, UMI_C, UMI_C, kind="FF"),
    )
    plain = marked(run(tmp_path, recs, "--pair-orientation-tag", "po", name="plain.bam"))
    umi = marked(
        run(
            tmp_path,
            recs,
            "--use-umi",
            "--cross-strand",
            "--position-tolerance",
            "3",
            "--pair-orientation-tag",
            "po",
            name="umi.bam",
        )
    )
    assert {n: v["po"] for n, v in plain.items()} == {n: v["po"] for n, v in umi.items()}
    assert {n: v["po"] for n, v in umi.items()} == {"a": "F1R2", "b": "F2R1", "c": "FF"}


def test_pair_orientation_is_absent_for_unpaired_and_trans_contig_reads(tmp_path):
    recs = flatten(
        segment("solo", "chr1", 100, flag=0, tags={"po": "STALE"}),
        segment("unplaced", None, 0, flag=4, tags={"po": "STALE"}),
        pair("trans", "chr1", 4000, 4000, UMI_A, UMI_B, mate_chrom="chr2", tags={"po": "STALE"}),
    )
    got = marked(run(tmp_path, recs, "--pair-orientation-tag", "po"))
    assert {n: v["po"] for n, v in got.items()} == {"solo": None, "unplaced": None, "trans": None}


def test_pair_orientation_secondary_records_keep_whatever_they_carry(tmp_path):
    p1 = pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B)
    secondary = segment("p1", "chr1", 2000, flag=1 | 128 | 256, tags={"po": "KEEP"})
    out = run(tmp_path, flatten(p1, secondary), "--pair-orientation-tag", "po")
    with pysam.AlignmentFile(str(out), "rb") as bam:
        by_flag = {r.flag & 256: r.get_tag("po") for r in bam}
    assert by_flag == {0: "F1R2", 256: "KEEP"}


def test_pair_orientation_only_the_named_tag_is_managed(tmp_path):
    """Only the named tag is managed: a differently-named stale tag is not the tool's to touch."""
    out = run(
        tmp_path, pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B, tags={"po": "OLD"}), "--pair-orientation-tag", "XO"
    )
    with pysam.AlignmentFile(str(out), "rb") as bam:
        assert {(r.get_tag("po"), r.get_tag("XO")) for r in bam} == {("OLD", "F1R2")}


@pytest.mark.parametrize("bad", ["p", "pop", "", "1o", "p!", "p o"])
def test_pair_orientation_rejects_malformed_tag_names(tmp_path, bad):
    inp = build_bam(tmp_path / "in.bam", pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B))
    result = cli(inp, "--pair-orientation-tag", bad, out=tmp_path / "out.bam", check=False)
    assert result.returncode != 0
    assert "two-character SAM tag" in result.stderr or "expected one argument" in result.stderr


def test_sam_tag_validator():
    assert mdmi._sam_tag("po") == "po"
    assert mdmi._sam_tag("X1") == "X1"
    for bad in ("p", "pop", "1p", "p-", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            mdmi._sam_tag(bad)


def _unit_read(flag=0, **tags):
    read = pysam.AlignedSegment()
    read.query_name = "x"
    read.flag = flag
    for tag, value in tags.items():
        read.set_tag(tag, value, value_type="Z")
    return read


def test_set_or_strip_po_stamps_overwrites_and_strips():
    stamped = _unit_read()
    mdmi._set_or_strip_po(stamped, "FF", "po")
    assert stamped.get_tag("po") == "FF"

    overwritten = _unit_read(po="STALE")
    mdmi._set_or_strip_po(overwritten, "RR", "po")
    assert overwritten.get_tag("po") == "RR"

    stripped = _unit_read(po="STALE", XX="keep")
    mdmi._set_or_strip_po(stripped, None, "po")
    assert not stripped.has_tag("po")
    assert stripped.get_tag("XX") == "keep"


@pytest.mark.parametrize("flag", [256, 2048])
def test_set_or_strip_po_ignores_secondary_and_supplementary(flag):
    read = _unit_read(flag, po="KEEP")
    mdmi._set_or_strip_po(read, "FF", "po")
    mdmi._set_or_strip_po(read, None, "po")
    assert read.get_tag("po") == "KEEP"


def test_set_or_strip_cs_stamps_strips_and_removes_run_id():
    read = _unit_read()
    mdmi._set_or_strip_cs(read, "x", {"x": "606-abc"}, strip_run_id=False)
    assert read.get_tag("CS") == "606-abc"
    mdmi._set_or_strip_cs(read, "x", {"x": "606-abc"}, strip_run_id=True)
    assert read.get_tag("CS") == "abc"
    mdmi._set_or_strip_cs(read, "x", {}, strip_run_id=False)
    assert not read.has_tag("CS")
    stale = _unit_read(CS="old")
    mdmi._set_or_strip_cs(stale, "x", None, strip_run_id=False)
    assert not stale.has_tag("CS")


# --- unit: clustering helpers --------------------------------------------------------------------


def test_assign_mi_unit():
    mi, ds = mdmi.assign_mi({("c", 1, 2): ["b", "a"], ("c", 3, 4): ["z"]})
    assert mi == {"a": "a", "b": "a"}
    assert ds == {"a": 2, "b": 2, "z": 1}


def test_merge_never_unions_clusters_in_different_groups():
    clusters = {
        ("chr1", 100, 300, UMI_A, UMI_B): ["n1"],
        ("chr1", 101, 301, UMI_A, UMI_C): ["n2"],
        ("chr2", 101, 301, UMI_A, UMI_B): ["n3"],
    }
    assert mdmi.merge_duplex_partners(clusters, {}, 5, cross_strand=True) == 0
    assert len(clusters) == 3


def test_merge_keeps_the_lowest_key_and_all_names():
    k_low, k_mid, k_high = (("chr1", 100 + i, 300, UMI_A, UMI_B) for i in (0, 1, 2))
    clusters = {k_high: ["h"], k_low: ["l"], k_mid: ["m"]}
    assert mdmi.merge_duplex_partners(clusters, {}, 1, cross_strand=False) == 1
    assert set(clusters) == {k_low}
    assert sorted(clusters[k_low]) == ["h", "l", "m"]


def test_merge_counts_each_merged_group_once():
    group1 = [("chr1", 100 + i, 300, UMI_A, UMI_B) for i in (0, 1)]
    group2 = [("chr1", 100 + i, 300, UMI_C, UMI_B) for i in (0, 1)]
    clusters = {k: [str(i)] for i, k in enumerate(group1 + group2)}
    assert mdmi.merge_duplex_partners(clusters, {}, 1, cross_strand=False) == 2
    assert len(clusters) == 2


def test_merge_requires_the_second_boundary_to_be_in_tolerance_too():
    clusters = {("chr1", 100, 300, UMI_A, UMI_B): ["n1"], ("chr1", 100, 304, UMI_A, UMI_B): ["n2"]}
    assert mdmi.merge_duplex_partners(clusters, {}, 3, cross_strand=False) == 0
    assert mdmi.merge_duplex_partners(clusters, {}, 4, cross_strand=False) == 1


def test_merge_cross_strand_uses_the_per_cluster_orientation_both_ways():
    fwd = ("chr1", 100, 300, UMI_A, UMI_B)
    rev = ("chr1", 100, 300, UMI_B, UMI_A)
    clusters = {fwd: ["f"], rev: ["r"]}
    # both orientations flipped is still opposite strands
    assert mdmi.merge_duplex_partners(dict(clusters), {fwd: True, rev: False}, 0, cross_strand=True) == 1


def test_build_mi_map_without_orient_never_merges():
    mi, ds, clusters = mdmi.build_mi_map({"n1": ("chr1", 100, 300), "n2": ("chr1", 101, 301), "n3": None})
    assert (mi, ds, len(clusters)) == ({}, {"n1": 1, "n2": 1}, 2)


def test_build_mi_map_with_orient_but_zero_tolerance_changes_nothing():
    keys = {"n1": ("chr1", 100, 300, UMI_A, UMI_B), "n2": ("chr1", 101, 301, UMI_A, UMI_B)}
    plain = mdmi.build_mi_map(dict(keys))
    with_orient = mdmi.build_mi_map(dict(keys), {}, 0, cross_strand=False)
    assert plain[:2] == with_orient[:2]


@pytest.mark.parametrize(
    ("r1_reverse", "r2_reverse", "expected"),
    [(False, True, "F1R2"), (True, False, "F2R1"), (False, False, "FF"), (True, True, "RR")],
)
def test_pair_orientation_is_symmetric_in_position(r1_reverse, r2_reverse, expected):
    """Where the mates sit does not matter, only the strands."""
    for r1_start, r2_start in ((100, 300), (300, 100), (100, 100)):
        r1 = ("chr1", r1_start, r1_start + 50, True, r1_reverse, "", "")
        r2 = ("chr1", r2_start, r2_start + 50, False, r2_reverse, "", "")
        assert mdmi._pair_orientation(r1, r2) == expected


# --- unit: regions and shards ----------------------------------------------------------------------


def test_load_regions_merges_sorts_pads_and_skips_headers(tmp_path):
    bed = tmp_path / "t.bed"
    bed.write_text(
        "# comment\ntrack name=x\nbrowser position chr1\n\n"
        "chr1\t300\t400\nchr1\t100\t200\nchr1\t190\t250\nchr1\t400\t450\nchr2\t5\t10\n"
    )
    regions = mdmi.load_regions(str(bed))
    assert regions["chr1"] == ([100, 300], [250, 450])
    assert regions["chr2"] == ([5], [10])
    padded = mdmi.load_regions(str(bed), padding=60)
    assert padded["chr1"] == ([40], [510])  # 100-60 .. 250+60 now touches 300-60 .. 450+60
    assert padded["chr2"] == ([0], [70])  # clamped at zero


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [(0, 100, False), (0, 101, True), (150, 160, True), (200, 300, False), (199, 300, True), (50, 500, True)],
)
def test_overlaps_is_half_open(start, end, expected):
    assert mdmi._overlaps(([100], [200]), start, end) is expected


def test_overlaps_without_regions_is_always_true():
    assert mdmi._overlaps(None, 0, 1)


def test_overlaps_with_several_intervals():
    regions = ([100, 300, 500], [200, 400, 600])
    assert [mdmi._overlaps(regions, s, s + 10) for s in (50, 195, 250, 395, 450, 595, 700)] == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_build_shards_on_a_grid(tmp_path):
    bam = build_bam(tmp_path / "in.bam", pair("p", "chr1", 100, 300, UMI_A, UMI_B))
    shards = mdmi.build_shards(str(bam), {}, 8_000, None)
    assert shards[:3] == [("chr1", 0, 8_000), ("chr1", 8_000, 16_000), ("chr1", 16_000, CONTIG_LEN)]
    assert [s for s in shards if s[0] == "chr2"][0] == ("chr2", 0, 8_000)
    assert len(shards) == 6


def test_build_shards_follow_the_regions_and_split_wide_ones(tmp_path):
    bam = build_bam(tmp_path / "in.bam", pair("p", "chr1", 100, 300, UMI_A, UMI_B))
    regions = {"chr1": ([1000, 5000], [1250, 5700]), "chrX": ([0], [10])}
    shards = mdmi.build_shards(str(bam), {}, 10_000, regions, max_shard_span=300)
    # 700 bp at max span 300 -> ceil(700/300)=3 parts of ceil(700/3)=234 bp, the last one clipped
    assert shards == [("chr1", 1000, 1250), ("chr1", 5000, 5234), ("chr1", 5234, 5468), ("chr1", 5468, 5700)]


def test_build_shards_clips_regions_to_the_contig(tmp_path):
    bam = build_bam(tmp_path / "in.bam", pair("p", "chr1", 100, 300, UMI_A, UMI_B))
    shards = mdmi.build_shards(str(bam), {}, 10_000, {"chr1": ([CONTIG_LEN - 100], [CONTIG_LEN + 500])})
    assert shards == [("chr1", CONTIG_LEN - 100, CONTIG_LEN)]


# --- CLI behaviour -------------------------------------------------------------------------------


def test_output_is_required_unless_stats_only(tmp_path):
    inp = build_bam(tmp_path / "in.bam", pair("p", "chr1", 100, 300, UMI_A, UMI_B))
    result = cli(inp, check=False)
    assert result.returncode != 0
    assert "output is required" in result.stderr


def test_sharded_mode_requires_an_index(tmp_path):
    inp = build_bam(tmp_path / "in.bam", pair("p", "chr1", 100, 300, UMI_A, UMI_B), index=False)
    result = cli(inp, "--jobs", "2", out=tmp_path / "out.bam", check=False)
    assert result.returncode != 0
    assert "must be indexed" in result.stderr


def _family_table(stdout):
    rows = [line.split("\t") for line in stdout.strip().splitlines() if "\t" in line]
    assert rows[0] == ["family_size", "families", "reads", "pct_reads"]
    return {int(r[0]): (int(r[1]), int(r[2]), r[3]) for r in rows[1:]}


def test_family_size_table_is_printed_to_stdout(tmp_path):
    recs = flatten(
        *(pair(f"d{i}", "chr1", 1000, 1200, UMI_A, UMI_B) for i in range(3)),
        pair("s1", "chr1", 3000, 3200, UMI_A, UMI_B),
        pair("s2", "chr1", 4000, 4200, UMI_A, UMI_B),
    )
    inp = build_bam(tmp_path / "in.bam", recs)
    result = cli(inp, out=tmp_path / "out.bam")
    assert _family_table(result.stdout) == {2: (2, 4, "40.0%"), 6: (1, 6, "60.0%")}


def test_stats_only_reads_the_tags_of_an_already_marked_file(tmp_path):
    recs = flatten(
        *(pair(f"d{i}", "chr1", 1000, 1200, UMI_A, UMI_B) for i in range(3)),
        pair("s1", "chr1", 3000, 3200, UMI_A, UMI_B),
    )
    out = run(tmp_path, recs)
    stats = cli(out, "--stats-only")
    assert _family_table(stats.stdout) == {2: (1, 2, "25.0%"), 6: (1, 6, "75.0%")}
    assert not (tmp_path / "stats").exists()


def test_stats_only_ignores_secondary_and_supplementary_records(tmp_path):
    recs = flatten(
        pair("p1", "chr1", 1000, 1200, UMI_A, UMI_B),
        segment("p1", "chr1", 2000, flag=1 | 128 | 256),
        segment("p1", "chr1", 2500, flag=1 | 128 | 2048),
    )
    out = run(tmp_path, recs)
    assert _family_table(cli(out, "--stats-only").stdout) == {2: (1, 2, "100.0%")}


def test_limit_stops_the_whole_file_pass(tmp_path):
    recs = flatten(*(pair(f"p{i:02d}", "chr1", 100 + 300 * i, 300 + 300 * i, UMI_A, UMI_B) for i in range(10)))
    inp = build_bam(tmp_path / "in.bam", recs)
    cli(inp, "--limit", "6", out=tmp_path / "out.bam")
    with pysam.AlignmentFile(str(tmp_path / "out.bam"), "rb") as bam:
        assert sum(1 for _ in bam.fetch(until_eof=True)) == 6


# --- sharded == whole-file ---------------------------------------------------------------------------


def random_dataset(seed=7, n_pairs=260):
    """
    Pairs piled on a few hot positions, so there are real duplicate clusters, boundary
    drift, UMI collisions, both strands, and clusters that straddle a 3 kb shard edge.
    """
    rng = random.Random(seed)
    lefts = [500, 1500, 2890, 2990, 3010, 6000, 8990, 9010, 14000, 19500]
    umis = [UMI_A, UMI_B, UMI_C]
    recs = []
    for i in range(n_pairs):
        chrom = rng.choice(["chr1", "chr2"])
        left = rng.choice(lefts) + rng.choice([0, 0, 0, 0, 1, 2, 3])
        span = rng.choice([100, 150, 200]) + rng.choice([0, 0, 0, 1, 2])
        u5, u3 = rng.choice(umis), rng.choice(umis)
        kind = rng.choice(["F1R2", "F1R2", "F2R1", "F2R1", "FF", "RR"])
        mate_chrom = ("chr2" if chrom == "chr1" else "chr1") if i % 41 == 0 else None
        recs += pair(f"p{i:04d}", chrom, left, left + span, u5, u3, kind=kind, mate_chrom=mate_chrom)
    for i in range(12):
        recs.append(segment(f"solo{i}", rng.choice(["chr1", "chr2"]), rng.randrange(0, 19_000), flag=0))
    for i in range(5):
        recs.append(segment(f"unplaced{i}", None, 0, flag=4, tags={"MI": "stale", "CS": "stale"}))
    # a half pair: its mate is simply not in the file
    recs += pair("half", "chr1", 2000, 2200, UMI_A, UMI_B)[:1]
    return recs


@pytest.fixture(scope="module")
def dataset_bam(tmp_path_factory):
    return build_bam(tmp_path_factory.mktemp("dataset") / "in.bam", random_dataset())


OPTION_SETS = [
    pytest.param((), id="plain"),
    pytest.param(("--use-umi",), id="umi"),
    pytest.param(("--use-umi", "--position-tolerance", "2"), id="umi-tol2"),
    pytest.param(("--use-umi", "--cross-strand"), id="umi-cross"),
    pytest.param(("--use-umi", "--cross-strand", "--position-tolerance", "3"), id="umi-cross-tol3"),
    pytest.param(("--use-umi", "--pair-orientation-tag", "po"), id="umi-po"),
    pytest.param(
        ("--use-umi", "--cross-strand", "--position-tolerance", "2", "--pair-orientation-tag", "po", "--strip-run-id"),
        id="everything",
    ),
]


def _overlaps_bed(rec_span, regions):
    chrom, start, end = rec_span
    return mdmi._overlaps(regions.get(chrom), start, end)


@needs_samtools
@pytest.mark.parametrize("options", OPTION_SETS)
@pytest.mark.parametrize("shard_size", [3_000, 7_777], ids=["grid3000", "grid7777"])
def test_sharded_grid_matches_whole_file(tmp_path, dataset_bam, options, shard_size):
    whole = tmp_path / "whole.bam"
    sharded = tmp_path / "sharded.bam"
    cli(dataset_bam, *options, out=whole)
    cli(dataset_bam, *options, "--jobs", "2", "--shard-size", shard_size, out=sharded)
    assert snapshot(sharded) == snapshot(whole)


@needs_samtools
@pytest.mark.parametrize("options", OPTION_SETS)
def test_sharded_regions_match_whole_file_restricted_to_the_regions(tmp_path, dataset_bam, options):
    bed = tmp_path / "panel.bed"
    bed.write_text("chr1\t520\t1620\nchr1\t2950\t3050\nchr1\t8900\t9100\nchr2\t5990\t6100\nchr2\t14000\t14100\n")
    regions = mdmi.load_regions(str(bed))
    whole = tmp_path / "whole.bam"
    sharded = tmp_path / "sharded.bam"
    cli(dataset_bam, *options, out=whole)
    cli(dataset_bam, *options, "--jobs", "3", "--regions", bed, "--max-shard-span", "70", out=sharded)
    expected = [s for s in snapshot(whole) if s[2] and _overlaps_bed((s[2], s[3], s[4] or s[3] + 1), regions)]
    assert expected, "the BED selected nothing"
    assert snapshot(sharded) == expected


@needs_samtools
def test_sharded_regions_with_padding_match_whole_file(tmp_path, dataset_bam):
    bed = tmp_path / "panel.bed"
    bed.write_text("chr1\t1000\t1100\nchr2\t8000\t8100\n")
    padded = mdmi.load_regions(str(bed), 600)
    whole, sharded = tmp_path / "whole.bam", tmp_path / "sharded.bam"
    cli(dataset_bam, "--use-umi", "--pair-orientation-tag", "po", out=whole)
    cli(
        dataset_bam,
        "--use-umi",
        "--pair-orientation-tag",
        "po",
        "--jobs",
        "2",
        "--regions",
        bed,
        "--region-padding",
        "600",
        out=sharded,
    )
    expected = [s for s in snapshot(whole) if s[2] and _overlaps_bed((s[2], s[3], s[4] or s[3] + 1), padded)]
    assert expected
    assert snapshot(sharded) == expected


@needs_samtools
def test_sharded_output_is_sorted_indexed_and_has_unplaced_reads_last(tmp_path, dataset_bam):
    out = tmp_path / "out.bam"
    cli(dataset_bam, "--jobs", "2", "--shard-size", "3000", out=out)
    with pysam.AlignmentFile(str(out), "rb") as bam:
        assert bam.has_index()
        keys = [(r.reference_id if r.reference_id >= 0 else 1 << 30, r.reference_start) for r in bam]
    assert keys == sorted(keys)
    assert sum(1 for k in keys if k[0] == 1 << 30) == 5


@needs_samtools
def test_sharded_run_leaves_no_temporary_files_behind(tmp_path, dataset_bam):
    out_dir = tmp_path / "o"
    out_dir.mkdir()
    cli(dataset_bam, "--jobs", "2", "--shard-size", "9000", out=out_dir / "out.bam")
    assert sorted(p.name for p in out_dir.iterdir()) == ["out.bam", "out.bam.bai"]


@needs_samtools
def test_sharded_tmp_dir_is_honoured(tmp_path, dataset_bam):
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    cli(dataset_bam, "--jobs", "2", "--shard-size", "9000", "--tmp-dir", tmp, out=tmp_path / "out.bam")
    assert list(tmp.iterdir()) == []


@needs_samtools
def test_sharded_limit_is_ignored(tmp_path, dataset_bam):
    a, b = tmp_path / "a.bam", tmp_path / "b.bam"
    cli(dataset_bam, "--jobs", "2", "--shard-size", "9000", out=a)
    result = cli(dataset_bam, "--jobs", "2", "--shard-size", "9000", "--limit", "3", out=b)
    assert "--limit is ignored" in result.stderr
    assert snapshot(a) == snapshot(b)


# --- sharded: the pad, and the emission rule -----------------------------------------------------------


@needs_samtools
def test_pad_bounds_how_far_apart_two_mates_may_be(tmp_path):
    """
    Mates 5 kb apart land in different 3 kb shards. With a 1 kb pad they never meet and the
    pair degrades to a singleton; with a 6 kb pad it matches the whole-file answer.
    """
    recs = flatten(
        pair("far1", "chr1", 1000, 6100, UMI_A, UMI_B),
        pair("far2", "chr1", 1000, 6100, UMI_A, UMI_B),
    )
    inp = build_bam(tmp_path / "in.bam", recs)
    whole, narrow, wide = (tmp_path / f"{n}.bam" for n in ("whole", "narrow", "wide"))
    cli(inp, "--pair-orientation-tag", "po", out=whole)
    cli(inp, "--pair-orientation-tag", "po", "--shard-size", "3000", "--pad", "1000", "--jobs", "2", out=narrow)
    cli(inp, "--pair-orientation-tag", "po", "--shard-size", "3000", "--pad", "6000", "--jobs", "2", out=wide)
    assert marked(whole)["far1"] == {"MI": "far1", "DS": 2, "CS": None, "po": "F1R2"}
    assert marked(narrow)["far1"] == {"MI": None, "DS": 1, "CS": None, "po": None}
    assert snapshot(wide) == snapshot(whole)


@needs_samtools
def test_sharded_run_reports_reads_left_without_a_mate(tmp_path):
    recs = flatten(pair("far", "chr1", 1000, 6100, UMI_A, UMI_B), pair("near", "chr1", 1000, 1200, UMI_A, UMI_B))
    inp = build_bam(tmp_path / "in.bam", recs)
    result = cli(inp, "--shard-size", "3000", "--pad", "1000", "--jobs", "2", out=tmp_path / "out.bam")
    assert "2 reads left without a mate" in result.stderr


def _single_reads(tmp_path, specs, bed_lines, *extra, name="out.bam"):
    """Unpaired reads (name, start, length) on chr1; return the output names in order."""
    recs = [segment(n, "chr1", s, flag=0, length=ln) for n, s, ln in specs]
    inp = build_bam(tmp_path / "in.bam", recs)
    bed = tmp_path / "t.bed"
    bed.write_text("".join(f"chr1\t{a}\t{b}\n" for a, b in bed_lines))
    cli(inp, "--regions", bed, "--jobs", "2", *extra, out=tmp_path / name)
    with pysam.AlignmentFile(str(tmp_path / name), "rb") as bam:
        return [r.query_name for r in bam]


def test_a_read_reaching_into_an_interval_from_upstream_is_emitted(tmp_path):
    """The behaviour of `samtools view -L`: an overlap test, not a start-in-interval test."""
    names = _single_reads(
        tmp_path,
        [("upstream", 980, 50), ("inside", 1100, 50), ("gap", 1700, 50), ("downstream", 1480, 50)],
        [(1000, 1500)],
    )
    assert names == ["upstream", "inside", "downstream"]


def test_a_read_in_the_gap_between_intervals_is_not_emitted(tmp_path):
    names = _single_reads(
        tmp_path,
        [("a", 1100, 50), ("gap", 1700, 50), ("b", 2100, 50)],
        [(1000, 1500), (2000, 2500)],
    )
    assert names == ["a", "b"]


def test_a_read_reaching_into_the_second_interval_is_emitted_once_and_in_order(tmp_path):
    names = _single_reads(
        tmp_path,
        [("a", 1100, 50), ("reach", 1980, 50), ("b", 2100, 50)],
        [(1000, 1500), (2000, 2500)],
    )
    assert names == ["a", "reach", "b"]


def test_a_read_spanning_two_intervals_is_emitted_exactly_once(tmp_path):
    names = _single_reads(
        tmp_path,
        [("a", 1100, 50), ("long", 1400, 1200), ("b", 2100, 50)],
        [(1000, 1500), (2000, 2500)],
    )
    assert names == ["a", "long", "b"]


def test_a_read_spanning_many_split_shards_is_emitted_exactly_once(tmp_path):
    names = _single_reads(
        tmp_path,
        [("long", 1010, 900), ("a", 1100, 50), ("b", 1800, 50)],
        [(1000, 2000)],
        "--max-shard-span",
        "100",
    )
    assert sorted(names) == ["a", "b", "long"]
    assert names == sorted(names, key=lambda n: {"long": 1010, "a": 1100, "b": 1800}[n])


def test_a_read_exactly_touching_an_interval_edge_is_not_emitted(tmp_path):
    """Intervals are half-open: a read ending at the interval start does not overlap it."""
    names = _single_reads(tmp_path, [("touch", 950, 50), ("in", 1000, 50), ("after", 1500, 50)], [(1000, 1500)])
    assert names == ["in"]


def test_reads_on_contigs_absent_from_the_bed_are_dropped(tmp_path):
    recs = [segment("on", "chr1", 1100, flag=0), segment("off", "chr2", 1100, flag=0)]
    inp = build_bam(tmp_path / "in.bam", recs)
    bed = tmp_path / "t.bed"
    bed.write_text("chr1\t1000\t1500\n")
    cli(inp, "--regions", bed, "--jobs", "2", out=tmp_path / "out.bam")
    with pysam.AlignmentFile(str(tmp_path / "out.bam"), "rb") as bam:
        assert [r.query_name for r in bam] == ["on"]


@needs_samtools
def test_family_is_sized_from_reads_outside_the_interval(tmp_path):
    """
    Two duplicate pairs whose R1s lie upstream of the interval and whose R2s are inside it.
    Only the R2s are emitted, but the padded window still sees both mates, so DS is 2.
    """
    recs = flatten(
        pair("a", "chr1", 900, 1100, UMI_A, UMI_B),
        pair("b", "chr1", 900, 1100, UMI_A, UMI_B),
    )
    inp = build_bam(tmp_path / "in.bam", recs)
    bed = tmp_path / "t.bed"
    bed.write_text("chr1\t1000\t1400\n")
    cli(inp, "--regions", bed, "--jobs", "2", "--pair-orientation-tag", "po", out=tmp_path / "out.bam")
    snap = snapshot(tmp_path / "out.bam")
    assert [s[0] for s in snap] == ["a", "b"]
    assert {(s[5], s[6], s[8]) for s in snap} == {("a", 2, "F1R2")}


@needs_samtools
def test_a_cluster_straddling_a_split_boundary_agrees_on_mi(tmp_path):
    recs = flatten(*(pair(f"p{i}", "chr1", 1290, 1490, UMI_A, UMI_B) for i in range(4)))
    inp = build_bam(tmp_path / "in.bam", recs)
    bed = tmp_path / "t.bed"
    bed.write_text("chr1\t1000\t2000\n")
    whole, sharded = tmp_path / "whole.bam", tmp_path / "sharded.bam"
    cli(inp, "--use-umi", out=whole)
    cli(inp, "--use-umi", "--regions", bed, "--max-shard-span", "300", "--jobs", "2", out=sharded)
    assert snapshot(sharded) == snapshot(whole)
    assert {v["MI"] for v in marked(sharded).values()} == {"p0"}


@needs_samtools
def test_cross_strand_merge_survives_a_shard_boundary(tmp_path):
    """A tolerance-merged duplex pair whose two strands start either side of a shard edge."""
    recs = flatten(
        pair("fwd", "chr1", 2995, 3195, UMI_A, UMI_B, kind="F1R2"),
        pair("rev", "chr1", 3000, 3196, UMI_B, UMI_A, kind="F2R1"),
    )
    inp = build_bam(tmp_path / "in.bam", recs)
    args = ("--use-umi", "--cross-strand", "--position-tolerance", "5", "--pair-orientation-tag", "po")
    whole, sharded = tmp_path / "whole.bam", tmp_path / "sharded.bam"
    cli(inp, *args, out=whole)
    cli(inp, *args, "--shard-size", "3000", "--jobs", "2", out=sharded)
    assert snapshot(sharded) == snapshot(whole)
    got = marked(sharded)
    assert (got["fwd"]["MI"], got["rev"]["MI"], got["fwd"]["DS"]) == ("fwd", "fwd", 2)
    assert (got["fwd"]["po"], got["rev"]["po"]) == ("F1R2", "F2R1")


@needs_samtools
def test_sharded_unplaced_reads_get_ds_and_lose_stale_tags(tmp_path):
    recs = flatten(
        pair("p", "chr1", 1000, 1200, UMI_A, UMI_B),
        segment("un", None, 0, flag=4, tags={"MI": "stale", "CS": "stale", "po": "stale"}),
    )
    inp = build_bam(tmp_path / "in.bam", recs)
    out = tmp_path / "out.bam"
    cli(inp, "--use-umi", "--pair-orientation-tag", "po", "--shard-size", "9000", "--jobs", "2", out=out)
    assert marked(out)["un"] == {"MI": None, "DS": 1, "CS": None, "po": None}


@needs_samtools
def test_sharded_without_unplaced_reads_omits_the_unplaced_block(tmp_path):
    inp = build_bam(tmp_path / "in.bam", pair("p", "chr1", 1000, 1200, UMI_A, UMI_B))
    out = tmp_path / "out.bam"
    cli(inp, "--shard-size", "9000", "--jobs", "2", out=out)
    assert set(marked(out)) == {"p"}


@needs_samtools
@pytest.mark.parametrize("jobs", [1, 2, 4])
def test_result_does_not_depend_on_the_number_of_jobs(tmp_path, dataset_bam, jobs):
    ref, out = tmp_path / "ref.bam", tmp_path / "out.bam"
    cli(dataset_bam, "--use-umi", "--cross-strand", "--position-tolerance", "2", out=ref)
    cli(
        dataset_bam,
        "--use-umi",
        "--cross-strand",
        "--position-tolerance",
        "2",
        "--shard-size",
        "5000",
        "--jobs",
        jobs,
        out=out,
    )
    assert snapshot(out) == snapshot(ref)


# --- concat_shards --------------------------------------------------------------------------------------


@needs_samtools
def test_concat_shards_to_bam_preserves_order(tmp_path):
    parts = []
    for i, start in enumerate((100, 500, 900)):
        parts.append(build_bam(tmp_path / f"{i}.bam", segment(f"r{i}", "chr1", start, flag=0), index=False))
    out = tmp_path / "out.bam"
    mdmi.concat_shards([str(p) for p in parts], str(out), {}, str(tmp_path), threads=2)
    with pysam.AlignmentFile(str(out), "rb", check_sq=False) as bam:
        assert [r.query_name for r in bam] == ["r0", "r1", "r2"]


@needs_samtools
def test_concat_shards_raises_when_samtools_fails(tmp_path):
    bad = tmp_path / "not_a_bam.bam"
    bad.write_text("garbage")
    with pytest.raises(subprocess.CalledProcessError):
        mdmi.concat_shards([str(bad)], str(tmp_path / "out.bam"), {}, str(tmp_path))


def test_print_family_sizes_skips_empty_rows_and_survives_no_reads(capsys):
    mdmi.print_family_sizes({2: 0})
    assert capsys.readouterr().out == "family_size\tfamilies\treads\tpct_reads\n"
    mdmi.print_family_sizes({2: 0, 4: 1, 6: 1})
    assert capsys.readouterr().out.splitlines()[1:] == ["4\t1\t4\t40.0%", "6\t1\t6\t60.0%"]
