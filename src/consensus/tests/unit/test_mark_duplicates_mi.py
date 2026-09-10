#!/usr/bin/env python3
"""
Tests for mark_duplicates_mi.py, focused on the CS cross-strand duplex tag (BIOIN-3068).

The fixture is a pair of files hand-picked from 606174 / L15806 / Z0229 (xGen v2 PE):

    resources/cs_family_input.bam  -- input, already MI/DS-marked upstream by sorter, no CS
    resources/cs_family.bam        -- the expected output of `--use-umi`

They hold one duplex molecule at chr1:6181255, 64 pairs / 128 records, split by the
UMI part of the dedup key into two per-strand MI families whose (u5, u3) pairs are exact
mutual reverses -- the signature CS exists to normalise:

    MI 606174-0567298634  45 pairs  F1R2  u5=CGGCTAAT u3=TTGCAGAC  DS=45
    MI 606174-0083386786  19 pairs  F2R1  u5=TTGCAGAC u3=CGGCTAAT  DS=19

and CS = 606174-0083386786 (the lexicographically smaller MI) on all 128 records.

Run with:  pytest src/consensus/tests/unit
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pysam
import pytest
from ugbio_consensus import mark_duplicates_mi as mdmi

DATA = Path(__file__).resolve().parents[1] / "resources"
INPUT_BAM = DATA / "cs_family_input.bam"
EXPECTED_BAM = DATA / "cs_family.bam"

TAGS = ("MI", "DS", "CS")

# Sharded mode concatenates its per-shard BAMs with `samtools cat`. The CLI is present in
# ugbio_base, but ugbio-utils-bioinfo-CI runs a bare `uv run pytest` with no such guarantee.
needs_samtools = pytest.mark.skipif(
    shutil.which("samtools") is None, reason="sharded mode needs the samtools CLI"
)


def read_tags(path):
    """(qname, flag) -> (MI, DS, CS, rname, pos) for every record."""
    out = {}
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        for read in bam:
            key = (read.query_name, read.flag)
            assert key not in out, f"duplicate (qname, flag) {key} in {path}"
            out[key] = tuple(read.get_tag(t) if read.has_tag(t) else None for t in TAGS) + (
                read.reference_name,
                read.reference_start,
            )
    return out


def run(inp, out, *extra):
    subprocess.run(
        [sys.executable, "-m", "ugbio_consensus.mark_duplicates_mi", str(inp), str(out), *extra],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


@pytest.fixture(scope="module")
def expected():
    return read_tags(EXPECTED_BAM)


def test_whole_file_reproduces_expected_output(tmp_path_factory, expected):
    """--use-umi on the fixture input must reproduce cs_family.bam exactly."""
    out = run(INPUT_BAM, tmp_path_factory.mktemp("wf") / "out.bam", "--use-umi")
    assert read_tags(out) == expected


@needs_samtools
@pytest.mark.parametrize(
    "extra",
    [
        # production shape: --regions with the panel intervals as the shards
        ("--jobs", "2", "--regions", "REGIONS"),
        # a shard boundary deliberately placed inside the molecule's span (6181000+300)
        ("--jobs", "2", "--regions", "REGIONS", "--max-shard-span", "300"),
    ],
    ids=["regions", "regions-split-molecule"],
)
def test_sharded_agrees_with_whole_file(tmp_path, expected, extra):
    """
    Sharded mode is a separate writer, and it is what the WDL task actually runs
    (--jobs ~{cpu} --regions targets_bed). CS must not depend on how the genome is cut.

    Only --regions is covered: the fixture carries a full hg38 header, so any
    --shard-size produces one shard per contig (>3300) and `samtools cat` of that many
    files dominates the runtime. --regions + --max-shard-span exercises the same writer
    and additionally splits the molecule across two shards.
    """
    bed = tmp_path / "panel.bed"
    bed.write_text("chr1\t6181000\t6182000\n")
    extra = tuple(str(bed) if a == "REGIONS" else a for a in extra)
    out = run(INPUT_BAM, tmp_path / "out.bam", "--use-umi", *extra)
    assert read_tags(out) == expected


def test_without_use_umi_there_is_no_cs(tmp_path):
    """
    Without the UMI part of the key both strands already share one cluster, so CS would
    carry no information beyond MI and is not emitted. The two orientations merge into a
    single 64-pair family whose MI is the same name CS picks with --use-umi.
    """
    out = run(INPUT_BAM, tmp_path / "out.bam")
    tags = read_tags(out)
    assert len(tags) == 128
    assert {t[TAGS.index("CS")] for t in tags.values()} == {None}
    assert {t[TAGS.index("MI")] for t in tags.values()} == {"606174-0083386786"}
    assert {t[TAGS.index("DS")] for t in tags.values()} == {64}


def test_remarking_is_idempotent(tmp_path, expected):
    """Feeding an already-CS-tagged file back in must recompute, not double up."""
    out = run(EXPECTED_BAM, tmp_path / "out.bam", "--use-umi")
    assert read_tags(out) == expected


def test_stale_cs_is_stripped(tmp_path):
    """
    Re-marking without --use-umi must remove a CS the input carried. A stale CS is worse
    than no CS: it claims a duplex link that the current key does not support.
    """
    out = run(EXPECTED_BAM, tmp_path / "out.bam")
    tags = read_tags(out)
    assert {t[TAGS.index("CS")] for t in tags.values()} == {None}


def test_cs_is_total_over_clustered_pairs(expected):
    """Every record in the fixture carries a CS, not only a chosen representative."""
    for key, vals in expected.items():
        assert vals[TAGS.index("CS")] is not None, f"{key} has MI={vals[0]} but no CS"


def test_cluster_with_no_partner_gets_cs_equal_to_its_own_mi(tmp_path):
    """
    Keep only one of the two strands, so the surviving cluster has no cross-strand
    partner. CS must still be emitted, pointing at the cluster's own MI.

    This is the case the fixture cannot show, and it is the one that matters downstream:
    `demux --umi=CS` on the consensus branch treats a missing tag as an empty value, so
    every CS-less read at a position would collapse into a single duplicate family.
    """
    one_strand = tmp_path / "one_strand.bam"
    kept = 0
    with (
        pysam.AlignmentFile(str(INPUT_BAM), "rb", check_sq=False) as src,
        pysam.AlignmentFile(str(one_strand), "wb", header=src.header) as dst,
    ):
        for read in src:
            if read.has_tag("MI") and read.get_tag("MI") == "606174-0567298634":
                dst.write(read)
                kept += 1
    assert kept == 90, kept  # the 45-pair F1R2 cluster
    pysam.index(str(one_strand))

    out = run(one_strand, tmp_path / "out.bam", "--use-umi")
    tags = read_tags(out)
    assert len(tags) == 90
    for key, vals in tags.items():
        mi, ds, cs = vals[0], vals[1], vals[2]
        assert ds == 45
        assert cs == mi == "606174-0567298634", f"{key}: MI={mi} CS={cs}"


# --- assign_cs unit tests: the cross-strand rule itself -----------------------------
#
# The two clusters below share a position and carry mutually reversed UMI pairs, which is
# what makes them candidate strands of one molecule. Whether they are actually linked
# turns on orientation, and that is the part worth pinning: on a real 12.7 M-record CRAM,
# dropping the orientation term linked 508 further cluster pairs, 416 of which had both
# clusters on the same strand -- impossible for a duplex molecule, and common because a
# library draws UMIs from a fixed set of ~32.

FWD, REV = False, True  # orient[key] = "R1 is the reverse-strand mate" (F2R1)
KEY_AB = ("chr1", 100, 300, "AAAAAAAA", "BBBBBBBB")
KEY_BA = ("chr1", 100, 300, "BBBBBBBB", "AAAAAAAA")


def test_assign_cs_links_opposite_strands():
    """F1R2 keyed (A, B) and F2R1 keyed (B, A) are one molecule: same CS."""
    cs = mdmi.assign_cs({KEY_AB: ["n_fwd"], KEY_BA: ["n_rev"]}, {KEY_AB: FWD, KEY_BA: REV})
    assert cs["n_fwd"] == cs["n_rev"] == "n_fwd"


def test_assign_cs_does_not_link_same_strand():
    """
    Two forward clusters whose UMI pairs happen to be reverses of each other are a UMI
    collision, not a duplex pair. Each must keep a CS of its own.
    """
    cs = mdmi.assign_cs({KEY_AB: ["n_one"], KEY_BA: ["n_two"]}, {KEY_AB: FWD, KEY_BA: FWD})
    assert cs["n_one"] == "n_one"
    assert cs["n_two"] == "n_two"


def test_assign_cs_is_total_and_self_referential_when_alone():
    """A cluster with no candidate partner still gets a CS, pointing at its own MI."""
    cs = mdmi.assign_cs({KEY_AB: ["n_b", "n_a"]}, {KEY_AB: FWD})
    assert cs == {"n_a": "n_a", "n_b": "n_a"}  # min(names) is the representative


def test_assign_cs_is_empty_without_the_umi_part():
    """Without --use-umi the key has no UMI part, so CS would only restate MI."""
    assert mdmi.assign_cs({("chr1", 100, 300): ["n1", "n2"]}, {}) == {}


def test_assign_cs_orientation_defaults_to_forward():
    """
    A key missing from `orient` must not raise. It cannot happen via the two callers,
    but assign_cs is the one place the invariant would be silently violated.
    """
    cs = mdmi.assign_cs({KEY_AB: ["n1"], KEY_BA: ["n2"]}, {})
    assert cs["n1"] == "n1" and cs["n2"] == "n2"


def _rec(start, end, *, is_read1, is_reverse):
    return ("chr1", start, end, is_read1, is_reverse, "AAAAAAAA", "BBBBBBBB")


@pytest.mark.parametrize(
    ("r1", "r2", "expected_is_reverse"),
    [
        # textbook F1R2 / F2R1: the FLAG and the mates' order agree
        (_rec(100, 200, is_read1=True, is_reverse=False), _rec(250, 300, is_read1=False, is_reverse=True), False),
        (_rec(250, 300, is_read1=True, is_reverse=True), _rec(100, 200, is_read1=False, is_reverse=False), True),
        # dovetailed: the fragment is shorter than the read, so the reverse mate starts
        # at or before the forward one. 10.5 % of pairs on this library, and the reason
        # orientation must come off the FLAG rather than the start positions.
        (_rec(100, 200, is_read1=True, is_reverse=True), _rec(105, 205, is_read1=False, is_reverse=False), True),
        (_rec(100, 200, is_read1=True, is_reverse=True), _rec(100, 200, is_read1=False, is_reverse=False), True),
        (_rec(100, 200, is_read1=True, is_reverse=False), _rec(100, 200, is_read1=False, is_reverse=True), False),
    ],
    ids=["f1r2", "f2r1", "dovetail-r1-left", "same-start-r1-rev", "same-start-r1-fwd"],
)
def test_r1_is_reverse_reads_the_flag_not_the_positions(r1, r2, expected_is_reverse):
    assert mdmi._r1_is_reverse(r1, r2) is expected_is_reverse
