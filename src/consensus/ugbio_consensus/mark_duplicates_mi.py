#!/usr/bin/env python3
"""
Duplicate detection for paired-end reads.

Deduplication key: (chromosome, left_read_start, right_read_end)
Optionally include UMI tags u3 and u5 in the key.

MI tag is set to the lexicographically first query name within each duplicate cluster.
Singletons (unique pairs, unmapped, or unpaired reads) get MI = their own query name.

With --use-umi a second tag, CS, links the two strands of one original duplex molecule,
which the UMI part of the key necessarily splits into two MI families. See assign_cs().

--orientation-in-key (on by default with --use-umi, off with --cross-strand or --no-orientation-in-key)
adds the pair orientation (F1R2, F2R1, FF, RR) to the key, for libraries whose u5 and u3 can be equal.
A molecule keys as (A, B) on one strand and (B, A) on the other, which
is what keeps the strands apart; with A == B both strands key as (A, A) and, without the
orientation, land in one MI family. CS still links them. See _pair_key().

Two opt-in refinements, both off by default:
--position-tolerance merges same-UMI clusters whose fragment boundaries differ by a few bp
(read1/read2 use asymmetric adapters, so one molecule's copies can map 1-3 bp apart), and
--cross-strand additionally merges F1R2/F2R1 partners into one MI family. See
merge_duplex_partners().

Two execution modes:

* whole-file (default): one streaming pass to pair reads by name, holding one dict
  entry per read name. Exact, but memory scales with the file -- ~4 GB per 14 M reads.
* sharded (--jobs / --shard-size / --regions): the genome is cut into windows that are
  processed independently and concatenated. Memory is bounded by the window, and the
  windows run in parallel. See "Sharded execution" below for why this gives the same
  answer as the whole-file mode.

Usage:
    python mark_duplicates_mi.py input.cram output.cram -T ref.fa
    python mark_duplicates_mi.py s3://bucket/in.cram out.cram -T ref.fa --use-umi
    python mark_duplicates_mi.py in.cram out.bam --use-umi --jobs 16 --regions panel.bed
"""

# ruff: noqa: C901, PLR0912, PLR0915, FBT001, FBT002, FBT003
#
# The suppressions above are deliberate, and are about provenance rather than style.
# This module is verified byte-for-byte -- MI, DS and CS on all 12,670,749 records of
# 606174-L15806-Z0229, plus 55,884 CS links with 0 gained and 0 lost -- against the blob
# that produced the four validated Omics arms of BIOIN-3068. Splitting process_shard()
# to satisfy C901/PLR0912/PLR0915, or turning the `use_umi` / `strip_run_id` positionals
# into keyword-only arguments to satisfy FBT, is a rewrite of code whose *output* is the
# thing under warranty. If you do refactor it, re-run /data/Runs/BIOIN-3068/cmp_stream.py
# (MI/DS vs the pre-CS baseline) and cs_links.py (linkage vs the validated blob) first,
# and note that only a raw `samtools view` diff sees the emitted aux order MI, DS, CS.
#
# Since that validation, output changed in five places (and --position-tolerance became whole-file
# only). Four are outside the default --use-umi path on primary records: FF/RR families with
# --orientation-in-key get their own CS; CS links survive --position-tolerance merges; sharded mode
# never pairs mates further apart than --pad; and a secondary/supplementary record far from its pair
# (another contig, or beyond --pad) no longer takes its pair's MI/DS/CS in whole-file mode (see
# _takes_family_tags). The fifth does touch the default path: mates with equal starts are ordered by
# end in _pair_key, so a dovetailed pair keys the same whichever mate is R1 and its strands share CS.

import argparse
import bisect
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict  # still used in build_mi_map
from concurrent.futures import ProcessPoolExecutor

import pysam

# A duplicate cluster needs at least two pairs; a lone pair is a singleton.
MIN_CLUSTER_SIZE = 2

# Length of a UMI-bearing dedup key: (chrom, left_start, right_end, u5, u3).
# A shorter key was built without --use-umi, so there is no cross-strand link to make.
UMI_KEY_LEN = 5
POSITION_KEY_LEN = 3

# Default is 0 (exact boundary match) so that --use-umi output is unchanged unless asked.
DEFAULT_POSITION_TOLERANCE = 0

# Default mate-search window for sharded mode. Measured on 606174-L15806-Z0229
# (xGen pan-cancer, PE): 99.05% of pairs span <400 bp, 99.81% <500 bp, 99.90% <20 kb.
# 2 kb is well past the real fragment distribution; the remaining 0.1% are chimeric
# pairs, which the padded window cannot pair up (see "Sharded execution").
MAX_MATE_DISTANCE = 2_000
DEFAULT_SHARD_SIZE = 10_000_000
# Safety valve only: with --regions the shards are the BED intervals themselves, and a
# panel's targets are a few kb wide. This caps a pathologically wide merged interval.
DEFAULT_MAX_SHARD_SPAN = 50_000


def _sam_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]", value):
        raise argparse.ArgumentTypeError(f"{value!r} is not a two-character SAM tag")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Detect duplicate paired-end reads and set MI tag")
    parser.add_argument("input", help="Input BAM/CRAM (local path or s3://)")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output BAM/CRAM (omit with --stats-only)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Read MI tags from an already-marked file and print family size distribution",
    )
    parser.add_argument(
        "--use-umi",
        action="store_true",
        help="Include u3 and u5 UMI tags in the deduplication key",
    )
    parser.add_argument(
        "--pair-orientation-tag",
        type=_sam_tag,
        default=None,
        metavar="TAG",
        help="Write each primary read's pair orientation (F1R2, F2R1, FF or RR) to this "
        "two-character tag, and drop any stale value. Reads whose mate is on another contig, "
        "absent, or (in sharded mode) further than --pad away get no tag.",
    )
    parser.add_argument(
        "--position-tolerance",
        type=int,
        default=DEFAULT_POSITION_TOLERANCE,
        metavar="BP",
        help="With --use-umi, merge same-UMI clusters whose fragment boundaries differ by at most "
        f"this many bp (default {DEFAULT_POSITION_TOLERANCE}, i.e. exact match). "
        "Independent of --cross-strand.",
    )
    parser.add_argument(
        "--cross-strand",
        action="store_true",
        help="With --use-umi, merge F1R2 and F2R1 duplex partners of the same molecule into one "
        "MI family instead of only linking them via CS. The partners' boundaries must match "
        "within --position-tolerance.",
    )
    parser.add_argument(
        "--orientation-in-key",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add the pair orientation (F1R2, F2R1, FF, RR) to the deduplication key, so that pairs of "
        "different orientation never share an MI. Needed when u5 and u3 can be equal (a symmetric UMI): "
        "the two strands of a molecule then key identically and would otherwise be one family. CS still "
        "links the strands. Default: on with --use-umi, off with --cross-strand or without --use-umi. "
        "Cannot be forced on together with --cross-strand, which merges the strands on purpose.",
    )
    parser.add_argument(
        "--reference",
        "-T",
        help="Reference FASTA (required for CRAM I/O)",
    )
    parser.add_argument(
        "--strip-run-id",
        action="store_true",
        help="Strip the run-id prefix (everything before the first '-') from MI tag values",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N primary paired reads (for testing)",
    )
    sharded = parser.add_argument_group(
        "sharded execution",
        "Bounded-memory parallel mode. Any of these options switches it on.",
    )
    sharded.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help="Number of shards to process in parallel (default 1)",
    )
    sharded.add_argument(
        "--shard-size",
        type=int,
        default=None,
        metavar="BP",
        help=f"Shard width in bp when --regions is not given (default {DEFAULT_SHARD_SIZE})",
    )
    sharded.add_argument(
        "--regions",
        metavar="BED",
        help="Restrict the output to reads overlapping these intervals (e.g. a panel BED), "
        "and use the intervals themselves as the shards. Unplaced unmapped reads are "
        "dropped when this is given.",
    )
    sharded.add_argument(
        "--region-padding",
        type=int,
        default=0,
        metavar="BP",
        help="Grow every --regions interval by this much on each side before merging "
        "(default 0, i.e. take the BED as given)",
    )
    sharded.add_argument(
        "--max-shard-span",
        type=int,
        default=DEFAULT_MAX_SHARD_SPAN,
        metavar="BP",
        help=f"Split any --regions interval wider than this (default {DEFAULT_MAX_SHARD_SPAN})",
    )
    sharded.add_argument(
        "--pad",
        type=int,
        default=MAX_MATE_DISTANCE,
        metavar="BP",
        help=f"How far past a shard to look for mates (default {MAX_MATE_DISTANCE}). "
        "Pairs whose two mates are further apart than this are treated as "
        "singletons and counted in the 'without a mate in their window' total.",
    )
    sharded.add_argument(
        "--tmp-dir",
        metavar="DIR",
        help="Directory for per-shard BAMs (default: alongside the output)",
    )
    args = parser.parse_args()
    if args.orientation_in_key and args.cross_strand:
        parser.error("--orientation-in-key and --cross-strand contradict each other")
    if args.orientation_in_key is None:
        args.orientation_in_key = args.use_umi and not args.cross_strand
    if args.position_tolerance > 0 and (args.jobs > 1 or args.shard_size or args.regions):
        # Tolerance merges chain transitively, so a merged family has no bounded extent and two shards can
        # see different parts of it, which would tag the two mates of one pair differently.
        parser.error("--position-tolerance needs whole-file mode; drop --jobs, --shard-size and --regions")
    return args


# ---------------------------------------------------------------------------
# Pass 1: collect pair keys using a sliding mate buffer
# ---------------------------------------------------------------------------

# Records are tuples rather than dicts because one is held per unmatched read name, and
# at panel depth a shard window can hold millions of them. UMI strings are interned
# through `umis` for the same reason: a library draws them from a fixed set, so the
# distinct values number in the thousands however many reads there are.
_CHROM, _START, _END, _IS_READ1, _IS_REVERSE, _U5, _U3 = range(7)


def _read_rec(read, use_umi: bool, umis: dict[str, str] | None = None) -> tuple:
    u5 = u3 = ""
    if use_umi:
        u5 = read.get_tag("u5") if read.has_tag("u5") else ""
        u3 = read.get_tag("u3") if read.has_tag("u3") else ""
        if umis is not None:
            u5 = umis.setdefault(u5, u5)
            u3 = umis.setdefault(u3, u3)
    return (read.reference_name, read.reference_start, read.reference_end, read.is_read1, read.is_reverse, u5, u3)


def _pair_key(r1: tuple, r2: tuple, use_umi: bool, orientation_in_key: bool = False) -> tuple:
    # Ties on start are broken by end, so a pair keys the same whichever mate is R1: the two strands of a
    # dovetailed molecule swap R1 and R2, and must still share the positional part of the key for CS.
    left, right = (r1, r2) if (r1[_START], r1[_END]) <= (r2[_START], r2[_END]) else (r2, r1)
    key: tuple = (left[_CHROM], left[_START], right[_END])
    if use_umi:
        # u5 is always on R1 and u3 on R2 by library design, regardless of strand
        key += (r1[_U5], r2[_U3])
    if orientation_in_key:
        # Last, so key[:UMI_KEY_LEN] stays the positional part plus the UMI pair that assign_cs reads.
        key += (_pair_orientation(r1, r2),)
    return key


def _r1_is_reverse(r1: tuple, r2: tuple) -> bool:
    """
    True when the pair is F2R1 rather than F1R2, i.e. R1 is the strand read in reverse.

    This is the FLAG bit, not "R1 starts to the right of R2", which looks equivalent and
    is not: on this library the two disagree for 10.5 % of pairs (94,030 of 899,185 in the
    first 2 M records of 606174-L15806-Z0229), 77,591 of them F2R1 pairs, and 65,199 with
    both mates starting at the *same* position. Fragments shorter than the read length
    dovetail, so the reverse mate can start at or before the forward one. A variant of
    assign_cs that used the positional proxy was measured, and lost 5,815 of the 55,884
    cross-strand links this version finds.

    Only assign_cs uses this. It is a property of the *pair*, not of the cluster: two
    pairs can share a dedup key and disagree here, because the key records
    (R1.u5, R2.u3) without recording which mate R1 was. assign_cs therefore takes the
    first value seen per cluster, which is arbitrary for those clusters -- see there.
    """
    return r1[_IS_REVERSE]


def _pair_orientation(r1: tuple, r2: tuple) -> str:
    """F1R2, F2R1, FF or RR, from the strand of R1 and R2."""
    if r1[_IS_REVERSE] == r2[_IS_REVERSE]:
        return "RR" if r1[_IS_REVERSE] else "FF"
    return "F2R1" if r1[_IS_REVERSE] else "F1R2"


def _set_or_strip_po(read, orientation: str | None, tag: str) -> None:
    """Stamp the pair orientation, or remove a stale one. Secondary and supplementary records are left alone."""
    if read.is_secondary or read.is_supplementary:
        return
    if orientation is not None:
        read.set_tag(tag, orientation, value_type="Z")
    elif read.has_tag(tag):
        read.tags = [(k, v) for k, v in read.tags if k != tag]


def collect_pair_keys(
    bam_path: str,
    use_umi: bool,
    open_kwargs: dict,
    limit: int | None = None,
    orientations: dict[str, str] | None = None,
    orientation_in_key: bool = False,
    unmatched: set[str] | None = None,
) -> tuple[dict[str, tuple | None], dict[tuple, bool]]:
    """
    Stream the coordinate-sorted file once, matching mates via a sliding buffer.

    The buffer holds reads whose mate has not yet been seen. For coordinate-sorted
    input it stays small (bounded by reads within one insert-size window). Reads
    left in the buffer after EOF had no mate → treated as singletons.

    Returns:
      read_name -> dedup_key (or None for singletons / trans-chrom pairs), and
      dedup_key -> R1-is-the-rightmost-mate, one entry per cluster, for assign_cs.
      When `orientations` is given it is also filled, name -> pair orientation.
      When `unmatched` is given it is filled with the names seen as a single primary record.
    """
    # pending: name -> first-seen rec (waiting for its mate)
    pending: dict[str, tuple] = {}
    pair_keys: dict[str, tuple | None] = {}
    orient: dict[tuple, bool] = {}
    umis: dict[str, str] = {}
    n_seen = 0

    with pysam.AlignmentFile(bam_path, "r", **open_kwargs) as bam:
        for read in bam:
            if read.is_secondary or read.is_supplementary:
                continue
            if not read.is_paired or read.is_unmapped:
                continue

            n_seen += 1
            if n_seen % 100_000 == 0:
                logging.info("  pass1: %d reads ingested, %d pending", n_seen, len(pending))
            if limit and n_seen >= limit:
                break

            name = read.query_name
            if name.endswith(("/1", "/2")):
                name = name[:-2]
            rec = _read_rec(read, use_umi, umis)

            if name in pending:
                mate = pending.pop(name)
                if mate[_CHROM] != rec[_CHROM]:
                    pair_keys[name] = None
                else:
                    # ensure r1/r2 order for UMI extraction
                    r1, r2 = (rec, mate) if rec[_IS_READ1] else (mate, rec)
                    key = _pair_key(r1, r2, use_umi, orientation_in_key)
                    pair_keys[name] = key
                    orient.setdefault(key, _r1_is_reverse(r1, r2))
                    if orientations is not None and r1[_IS_READ1] != r2[_IS_READ1]:
                        orientations[name] = _pair_orientation(r1, r2)
            else:
                pending[name] = rec

    # reads still in pending had no mate in the file
    for name in pending:
        pair_keys[name] = None
    if unmatched is not None:
        unmatched.update(pending)

    logging.info("Pass 1 done: %d reads ingested, %d left without mate", n_seen, len(pending))
    return pair_keys, orient


# ---------------------------------------------------------------------------
# Build MI map from pair keys
# ---------------------------------------------------------------------------


def assign_mi(clusters: dict[tuple, list[str]]) -> tuple[dict[str, str], dict[str, int]]:
    """
    Turn dedup-key clusters into MI/DS maps. MI = lexicographically first name in the
    cluster; a cluster of one is a unique pair, so it gets DS=1 and no MI at all.
    """
    mi_map: dict[str, str] = {}
    ds_map: dict[str, int] = {}
    for names in clusters.values():
        cluster_size = len(names)
        if cluster_size < MIN_CLUSTER_SIZE:  # unique pair — DS=1, no MI
            ds_map[names[0]] = 1
            continue
        representative = min(names)
        for name in names:
            mi_map[name] = representative
            ds_map[name] = cluster_size
    return mi_map, ds_map


def assign_cs(
    clusters: dict[tuple, list[str]],
    orient: dict[tuple, bool],
    aliases: dict[tuple, tuple] | None = None,
) -> dict[str, str]:
    """
    Link the two strands of one original duplex molecule with a shared CS tag.

    With --use-umi the dedup key ends in (R1.u5, R2.u3). R1 is the leftmost mate of an
    F1R2 pair and the rightmost of an F2R1 pair, so one molecule keys as (A, B) on one
    strand and (B, A) on the other, and its two strands land in *different* MI families.
    That split is wanted -- each strand is consensus-called on its own -- but it loses
    the duplex relationship, and read_fuser groups strictly on MI, so nothing downstream
    can pair the two strand consensus reads back up again.

    Two clusters are the two strands of one molecule iff they share the positional part
    of the key, their UMI pairs are mutual reverses, *and* they sit on opposite strands.
    Putting the UMI pair back into F1R2 order normalises the key across strands:

        (chrom, left_start, right_end, (u5, u3) as the F1R2 strand would have keyed it)

    The orientation term is not redundant, and measuring is what settled it. Dropping it
    -- i.e. grouping on `tuple(sorted((u5, u3)))`, which is simpler and needs no `orient`
    argument -- linked 508 further pairs of clusters on a 12.7 M-record 606174 CRAM, and
    416 of those 508 had *both* clusters on the same strand (278 FF, 138 RR), against a
    100.0 % FR composition (55,884 / 55,884) for the links the orientation-aware rule
    finds. A duplex molecule's two strands cannot share an orientation, so those are
    coincidences, not molecules -- and they are common because a library draws its UMIs
    from a fixed set of ~32, which makes an (A, B) / (B, A) collision at one position
    unremarkable.

    `orient` holds one bool per *cluster*, which is what keeps the normalisation cheap:
    the reviewed version carried a second name -> normalised-key dict alongside
    `pair_keys`, and that per-read dict is the entire memory regression it introduced
    (peak RSS 1858 MiB against 1021 unpatched; one bool per cluster costs 1199 MiB). The
    price is that a cluster whose members disagree about which mate R1 was -- 2,816 of
    412,965 families, because the key records (R1.u5, R2.u3) without recording R1's
    orientation -- takes whichever value was seen first. That is arbitrary, but it is
    exactly as arbitrary as the reviewed version's `duplex_key_of[names[0]]`, which
    likewise samples one member, and it is confined to 0.7 % of families.

    CS is the lexicographically smallest MI in the group -- which is exactly the MI the
    molecule would have been given had the UMIs been left out of the key altogether.

    CS is *total* over every clustered pair: a cluster with no cross-strand partner gets
    CS = its own representative. So CS always names a duplex group and CS == MI reads as
    "no partner found". Leaving the tag off in that case instead would make a missing CS
    ambiguous for any consumer that keys on it -- notably `demux --umi=CS`, which would
    collapse every CS-less read at a position into a single family.

    Note the tag name: CS:Z is a *predefined* SAM tag ("Color read sequence", a SOLiD
    legacy, SAMtags section 1.2). Nothing in the UG stack emits it -- not the trimmer
    formats, not any @CO tag definition -- so it is free in practice, and BIOIN-3068
    chose to keep it. Compare read_fuser/README.md "## Tags", where fs/rs had to be
    renamed for exactly this reason.

    Without UMIs, CS links by position alone. If the key carries the pair orientation
    (--orientation-in-key) the two strands are separate MI families, and every family at one
    (chrom, left_start, right_end) shares a CS: the strands of a molecule cannot be told apart
    from independent molecules at the same place, so this over-links on a deep panel. Without the
    orientation both strands share one key, CS would carry no information beyond MI, and no
    CS is written.

    When the key carries the orientation, only F1R2 and F2R1 families are linked: an FF or RR
    family cannot be one strand of a duplex molecule, so it keeps a CS of its own instead of
    sharing one (and with it dx downstream) with the FR families at its position.

    `aliases` (old key -> surviving key, from merge_duplex_partners) lets a tolerance-merged
    cluster link through every key its members had before the merge; otherwise a strand whose
    partner was merged under a neighbouring key would lose its CS link.

    Returns read_name -> CS, empty if the key carries nothing to link.
    """
    if any(len(key) == POSITION_KEY_LEN for key in clusters):  # no UMI, no orientation: nothing to link
        return {}

    parent: dict[tuple, tuple] = {}

    def find(x: tuple) -> tuple:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for old, new in (aliases or {}).items():
        if new in clusters:
            ra, rb = find(_cs_link_key(new, orient)), find(_cs_link_key(old, orient))
            if ra != rb:
                parent[ra] = rb

    groups: dict[tuple, list[tuple[str, list[str]]]] = defaultdict(list)
    for key, names in clusters.items():
        link = _cs_link_key(key, orient)
        groups[find(link) if parent else link].append((min(names), names))

    cs_map: dict[str, str] = {}
    for subclusters in groups.values():
        cs = min(rep for rep, _ in subclusters)
        for _, names in subclusters:
            for name in names:
                cs_map[name] = cs
    return cs_map


def _cs_link_key(key: tuple, orient: dict[tuple, bool]) -> tuple:
    """The part of a dedup key that the two strands of one molecule share; see assign_cs."""
    if len(key) == POSITION_KEY_LEN + 1:  # no UMI, orientation last: link by position
        link, orientation = key[:POSITION_KEY_LEN], key[-1]
    else:
        umi = key[3:5]
        if orient.get(key):  # F2R1: R1 was the rightmost mate, so undo the swap
            umi = umi[::-1]
        link = key[:3] + umi
        orientation = key[UMI_KEY_LEN] if len(key) > UMI_KEY_LEN else None
    if orientation in ("FF", "RR"):
        link += (orientation,)
    return link


def _set_or_strip_cs(read, name: str, cs_map: dict[str, str] | None, strip_run_id: bool) -> None:
    """
    Stamp CS, or remove any CS the input already carried.

    Stripping matters as much as stamping: re-marking an already-marked file must not
    leave a stale CS behind, the same way MI is recomputed rather than trusted. Without
    --use-umi cs_map is empty, so every CS in the input is removed.
    """
    cs = cs_map.get(name) if cs_map else None
    if cs is not None:
        if strip_run_id and "-" in cs:
            cs = cs.split("-", 1)[1]
        read.set_tag("CS", cs)
    elif read.has_tag("CS"):
        read.tags = [(k, v) for k, v in read.tags if k != "CS"]


def merge_duplex_partners(
    clusters: dict[tuple, list[str]],
    orient: dict[tuple, bool],
    tolerance: int,
    cross_strand: bool,
    aliases: dict[tuple, tuple] | None = None,
) -> int:
    """
    Merge clusters that are the same molecule but landed in separate exact-position clusters.

    Read1/read2 use asymmetric adapters, so copies of one molecule can map a few bp apart
    on one fragment boundary; `tolerance` is how much drift is allowed. With `cross_strand`
    the F1R2 and F2R1 strands of a duplex molecule become eligible too, which reunites them
    into one MI instead of only linking them via CS (see assign_cs).

    Candidates are grouped by (chrom, UMI pair). Without `cross_strand` the UMI pair is the
    literal recorded one, so only same-orientation variants merge. With it, the pair is
    first put back into F1R2 order using `orient` -- the same normalisation as assign_cs,
    and for the same reason: grouping on an unordered UMI pair would also merge two
    same-strand clusters whose UMIs merely happen to be mutual reverses.

    Drifts chain transitively (union-find). Mutates `clusters`, keeping each merged group
    under its lowest key; `orient` is left as is, so a merged cluster keeps that key's value.
    If `aliases` is given, every key merged away is recorded there as key -> surviving key,
    so assign_cs can still link the strands through the keys the clusters had before merging.
    Returns the number of merged groups.
    """
    if tolerance <= 0 and not cross_strand:
        return 0  # the exact cluster key already captures this

    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for key in clusters:
        if len(key) < UMI_KEY_LEN:
            return 0  # no UMI part: nothing to group on
        umi = key[3:5]
        if cross_strand and orient.get(key):  # F2R1: undo the swap, as in assign_cs
            umi = umi[::-1]
        # Without cross_strand the orientation (if it is in the key) must match as well, so a tolerance
        # merge never joins two orientations.
        groups[(key[0], umi, () if cross_strand else key[UMI_KEY_LEN:])].append(key)

    parent: dict[tuple, tuple] = {}

    def find(x: tuple) -> tuple:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group_keys in groups.values():
        by_left = sorted(group_keys, key=lambda k: k[1])
        for i, k1 in enumerate(by_left):
            for j in range(i + 1, len(by_left)):
                k2 = by_left[j]
                if k2[1] - k1[1] > tolerance:
                    break  # sorted by left_start -- nothing further can qualify
                if abs(k2[2] - k1[2]) <= tolerance:
                    ra, rb = find(k1), find(k2)
                    if ra != rb:
                        parent[ra] = rb

    merged: dict[tuple, list[tuple]] = defaultdict(list)
    for key in list(clusters):
        merged[find(key)].append(key)

    n_merges = 0
    for members in merged.values():
        if len(members) < MIN_CLUSTER_SIZE:
            continue
        n_merges += 1
        target = min(members)
        for key in members:
            if key != target:
                clusters[target].extend(clusters.pop(key))
                if aliases is not None:
                    aliases[key] = target
    return n_merges


def build_mi_map(
    pair_keys: dict[str, tuple | None],
    orient: dict[tuple, bool] | None = None,
    position_tolerance: int = 0,
    cross_strand: bool = False,
    aliases: dict[tuple, tuple] | None = None,
) -> tuple[dict[str, str], dict[str, int], dict[tuple, list[str]]]:
    """
    Group read names by dedup key and assign MI values.
    Singletons (key is None) are left out of both maps and default to DS=1, no MI.

    If `orient` is given (i.e. --use-umi), nearby clusters are merged first when
    `position_tolerance` > 0 and/or `cross_strand` is True -- see `merge_duplex_partners`,
    which records the merged keys in `aliases` if it is given.

    Returns (mi_map, ds_map, clusters) where ds_map[name] = cluster size in pairs.
    """
    clusters: dict[tuple, list[str]] = defaultdict(list)
    for name, key in pair_keys.items():
        if key is not None:
            clusters[key].append(name)

    if orient is not None:
        n_merges = merge_duplex_partners(clusters, orient, position_tolerance, cross_strand, aliases)
        logging.info(
            "Merged %d cluster group(s) (tolerance=%d bp, cross_strand=%s)",
            n_merges,
            position_tolerance,
            cross_strand,
        )

    n_dup_clusters = sum(1 for v in clusters.values() if len(v) > 1)
    n_dup_reads = sum(len(v) for v in clusters.values() if len(v) > 1)
    logging.info(
        "%d unique pair positions; %d duplicate clusters (%d read names)",
        len(clusters),
        n_dup_clusters,
        n_dup_reads,
    )

    mi_map, ds_map = assign_mi(clusters)
    return mi_map, ds_map, clusters


def print_family_size_distribution(mi_map: dict[str, str], total_pairs: int = 0, n_unmatched: int = 0) -> None:
    """
    `total_pairs` counts every primary read name; `n_unmatched` of them were seen as a single
    record (their mate absent or unmapped) and are size-1 families, the rest are pairs.
    """
    # mi_map only contains reads in duplicate clusters; multiply by 2 for individual reads
    from collections import Counter

    family_sizes = Counter(n * 2 for n in Counter(mi_map.values()).values())
    # unique pairs (not in any duplicate cluster) → size-2 families
    n_singleton_pairs = total_pairs - len(mi_map) - n_unmatched
    if n_singleton_pairs > 0:
        family_sizes[2] = family_sizes.get(2, 0) + n_singleton_pairs
    if n_unmatched > 0:
        family_sizes[1] = family_sizes.get(1, 0) + n_unmatched

    print_family_sizes(family_sizes)


def stats_from_mi_tags(bam_path: str, open_kwargs: dict) -> None:
    """Read MI tags from an existing tagged file and print family size distribution."""
    from collections import Counter

    mi_counts: Counter = Counter()
    no_mi_names: Counter = Counter()
    with pysam.AlignmentFile(bam_path, "r", **open_kwargs) as bam:
        for read in bam:
            if read.is_secondary or read.is_supplementary:
                continue
            if read.has_tag("MI"):
                mi_counts[read.get_tag("MI")] += 1
            else:
                no_mi_names[read.query_name] += 1

    family_sizes = Counter(mi_counts.values())
    # group no-MI reads by query name (each pair → size-2 family)
    for n in no_mi_names.values():
        family_sizes[n] += 1
    logging.info("%d reads without MI tag (%d pairs)", sum(no_mi_names.values()), len(no_mi_names))

    print_family_sizes(family_sizes)


# ---------------------------------------------------------------------------
# Pass 2: copy reads to output with MI tag added/updated
# ---------------------------------------------------------------------------


def _takes_family_tags(read, key: tuple | None, pad: int) -> bool:
    """
    Whether a record gets its pair's MI/DS/CS. Primary records always do. A secondary or supplementary
    record does only if it lies on its pair's contig, the pair's fragment spans at most `pad`, and the
    fragment lies within `pad` of the record's start. That is the condition under which a sharded run is
    guaranteed to have seen the pair, so both execution modes tag non-primary records the same way.
    """
    if not (read.is_secondary or read.is_supplementary):
        return True
    if key is None or read.reference_name != key[0]:
        return False
    return key[2] - key[1] <= pad and key[1] >= read.reference_start - pad and key[2] <= read.reference_start + pad


def _strip_family_tags(read) -> None:
    if read.has_tag("MI") or read.has_tag("CS"):
        read.tags = [(k, v) for k, v in read.tags if k not in ("MI", "CS")]
    read.set_tag("DS", 1, value_type="i")


def write_with_mi(  # noqa: PLR0913
    bam_path: str,
    output_path: str,
    mi_map: dict[str, str],
    ds_map: dict[str, int],
    open_kwargs: dict,
    limit: int | None = None,
    strip_run_id: bool = False,
    cs_map: dict[str, str] | None = None,
    po_tag: str | None = None,
    orientations: dict[str, str] | None = None,
    pair_keys: dict[str, tuple | None] | None = None,
    pad: int = MAX_MATE_DISTANCE,
) -> None:
    """Pass 2 of whole-file mode. With `pair_keys`, non-primary records follow _takes_family_tags."""
    write_mode = "wc" if output_path.endswith(".cram") else "wb"
    n_written = 0

    with pysam.AlignmentFile(bam_path, "r", **open_kwargs) as bam:
        write_kwargs = dict(open_kwargs) if write_mode == "wc" else {}
        with pysam.AlignmentFile(output_path, write_mode, header=bam.header, **write_kwargs) as out:
            for read in bam:
                if limit and n_written >= limit:
                    break
                n_written += 1
                if n_written % 100_000 == 0:
                    logging.info("  pass2: %d reads written", n_written)
                name = read.query_name
                if name.endswith(("/1", "/2")):
                    name = name[:-2]
                if pair_keys is not None and not _takes_family_tags(read, pair_keys.get(name), pad):
                    _strip_family_tags(read)
                    out.write(read)
                    continue
                if name in mi_map:
                    mi = mi_map[name]
                    if strip_run_id and "-" in mi:
                        mi = mi.split("-", 1)[1]
                    read.set_tag("MI", mi)
                elif read.has_tag("MI"):
                    read.tags = [(k, v) for k, v in read.tags if k != "MI"]
                # DS = duplicate set size in pairs (1 for singletons)
                read.set_tag("DS", ds_map.get(name, 1), value_type="i")
                _set_or_strip_cs(read, name, cs_map, strip_run_id)
                if po_tag:
                    _set_or_strip_po(read, (orientations or {}).get(name), po_tag)
                out.write(read)


# ---------------------------------------------------------------------------
# Sharded execution
# ---------------------------------------------------------------------------
#
# The dedup key is (chrom, left_start, right_end), so every member of a duplicate
# cluster shares the same left_start, and both mates of a pair are always on the same
# chromosome (a trans-chromosomal pair is a singleton by definition). A window
# [shard_start - pad, shard_end + pad) therefore sees every member of every cluster
# whose left_start falls anywhere in [shard_start - pad, shard_end), as long as the two
# mates are no further apart than `pad`. Two adjacent shards that both cover a given
# left_start see the same member set and so pick the same representative name, which is
# why no cross-shard bookkeeping is needed.
#
# The one behavioural difference from whole-file mode: a pair whose mates are further
# apart than `pad` is not paired up, so it becomes a singleton (DS=1, no MI) instead of
# joining a cluster. This holds even when one padded window happens to contain both mates,
# because the shard that emits the other mate may not, and the two mates of a pair must
# never be tagged differently. That only changes the output if two such far-apart pairs share a
# key, since a lone pair is a DS=1 singleton in whole-file mode too. Measured on
# 606174-L15806-Z0229 against whole-file output over the same reads: at pad=20 kb, 4
# clusters / 58 of 13,886,714 reads (0.0004%); at the pad=2 kb default with the panel
# BED as shards, 32 clusters / 416 of 12,670,749 reads (0.003%). Every one of those 208
# pairs spans 6.3-6.6 kb, i.e. far outside the fragment distribution -- chimeric pairs
# that the consensus caller would not fuse anyway.
#
# A second difference when --regions is given: shard k emits the reads that overlap it and
# start at or after shard k-1's end, so a read reaching into an interval from upstream is
# kept (as with `samtools view -L`), each read is emitted once, and the concatenated output
# stays sorted. On a fixed grid this is the plain start-in-shard rule.


def load_regions(bed_path: str, padding: int = 0) -> dict[str, tuple[list[int], list[int]]]:
    """Read a BED into per-chromosome sorted, merged (starts, ends) lists."""
    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(bed_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            raw[fields[0]].append((max(0, int(fields[1]) - padding), int(fields[2]) + padding))

    regions: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, intervals in raw.items():
        starts: list[int] = []
        ends: list[int] = []
        for start, end in sorted(intervals):
            if starts and start <= ends[-1]:
                ends[-1] = max(ends[-1], end)
            else:
                starts.append(start)
                ends.append(end)
        regions[chrom] = (starts, ends)
    return regions


def _overlaps(regions: tuple[list[int], list[int]] | None, start: int, end: int) -> bool:
    """Does [start, end) hit any interval? Intervals are sorted and disjoint."""
    if regions is None:
        return True
    starts, ends = regions
    i = bisect.bisect_right(starts, end - 1) - 1
    return i >= 0 and ends[i] > start


def build_shards(
    bam_path: str,
    open_kwargs: dict,
    shard_size: int,
    regions: dict[str, tuple[list[int], list[int]]] | None,
    max_shard_span: int = DEFAULT_MAX_SHARD_SPAN,
) -> list[tuple[str, int, int]]:
    """
    Cut the reference into (chrom, start, end) shards, in header order.

    With --regions the shards ARE the target intervals. Slicing a targeted panel on a
    fixed bp grid instead is hopeless: coverage spans four orders of magnitude between
    on- and off-target, so on 606174 a single 2 Mb shard held 12.67 M of 13.9 M reads
    and the parallel run was no faster than the serial one. One shard per target gives
    ~1000 tasks whose cost tracks their read count, which a dynamically scheduled pool
    can actually balance.
    """
    shards: list[tuple[str, int, int]] = []
    with pysam.AlignmentFile(bam_path, "r", **open_kwargs) as bam:
        lengths = dict(zip(bam.references, bam.lengths, strict=False))
        for chrom in bam.references:
            if regions is None:
                for start in range(0, lengths[chrom], shard_size):
                    shards.append((chrom, start, min(start + shard_size, lengths[chrom])))
                continue
            chrom_regions = regions.get(chrom)
            if chrom_regions is None:
                continue
            for start, region_end in zip(*chrom_regions, strict=False):
                end = min(region_end, lengths[chrom])
                n_parts = max(1, -(-(end - start) // max_shard_span))  # ceil
                span = -(-(end - start) // n_parts)
                for part_start in range(start, end, span):
                    shards.append((chrom, part_start, min(part_start + span, end)))
    return shards


def process_shard(job: tuple) -> dict:
    """
    Mark one shard and write its reads to their own BAM.

    Reads are emitted by this shard iff they overlap [start, end) and start at or after
    `prev_end` (the previous shard's end on this chromosome, 0 if none) -- that partitions
    the output exactly once across shards and keeps the concatenation coordinate-sorted.
    """
    (
        bam_path,
        chrom,
        start,
        end,
        pad,
        use_umi,
        position_tolerance,
        cross_strand,
        open_kwargs,
        chrom_regions,
        out_path,
        strip_run_id,
        prev_end,
        po_tag,
        orientation_in_key,
    ) = job

    # Cluster straight off the stream: at panel depth an intermediate name -> key dict
    # would double the peak footprint of the pass for no benefit.
    clusters: dict[tuple, list[str]] = defaultdict(list)
    orient: dict[tuple, bool] = {}
    orientations: dict[str, str] = {}
    pending: dict[str, tuple] = {}
    umis: dict[str, str] = {}
    n_far = 0
    nonprimary_names: set[str] = set()

    with pysam.AlignmentFile(bam_path, "r", **open_kwargs) as bam:
        length = bam.get_reference_length(chrom)
        # The window has to reach `pad` before the first read this shard emits, which with --regions can
        # start upstream of the shard. Fetch is start-sorted, so the first read at or after prev_end is it.
        first_emitted = start
        for read in bam.fetch(chrom, start, end):
            if read.reference_start >= prev_end:
                first_emitted = min(start, read.reference_start)
                break
        for read in bam.fetch(chrom, max(0, first_emitted - pad), min(end + pad, length)):
            if read.is_secondary or read.is_supplementary:
                nonprimary_names.add(read.query_name.removesuffix("/1").removesuffix("/2"))
                continue
            if not read.is_paired or read.is_unmapped:
                continue
            name = read.query_name
            if name.endswith(("/1", "/2")):
                name = name[:-2]
            rec = _read_rec(read, use_umi, umis)
            if name in pending:
                mate = pending.pop(name)
                if rec[_START] - mate[_START] > pad:
                    # Mates further apart than pad stay unpaired even when this window holds both: the
                    # shard that emits the other mate may not see this one, and both mates need the same tags.
                    n_far += (start <= rec[_START] < end) + (start <= mate[_START] < end)
                    continue
                r1, r2 = (rec, mate) if rec[_IS_READ1] else (mate, rec)
                key = _pair_key(r1, r2, use_umi, orientation_in_key)
                clusters[key].append(name)
                orient.setdefault(key, _r1_is_reverse(r1, r2))
                if po_tag and r1[_IS_READ1] != r2[_IS_READ1]:
                    orientations[name] = _pair_orientation(r1, r2)
            else:
                pending[name] = rec

        # Names still pending had no mate inside the window: either the mate is
        # genuinely absent (unmapped/unpaired) or the pair spans more than `pad`.
        # Only those this shard owns are counted, or shards sharing a padded window
        # would report the same orphan twice.
        n_orphans = n_far + sum(1 for rec in pending.values() if start <= rec[_START] < end)
        pending.clear()

        # The pair key of every name with a non-primary record here, for _takes_family_tags. Taken before
        # any merge, so it is the pair's own key, as in whole-file mode.
        nonprimary_keys: dict[str, tuple] = {}
        if nonprimary_names:
            for key, names in clusters.items():
                for n in names:
                    if n in nonprimary_names:
                        nonprimary_keys[n] = key
        aliases: dict[tuple, tuple] = {}
        if use_umi:
            merge_duplex_partners(clusters, orient, position_tolerance, cross_strand, aliases)

        mi_map, ds_map = assign_mi(clusters)
        # Cross-strand partners share the positional part of the key, so both strands are
        # always inside the same padded window -- CS needs no cross-shard bookkeeping for
        # the same reason MI does not (see "Sharded execution").
        cs_map = assign_cs(clusters, orient, aliases)

        # Statistics count only the clusters this shard owns (left_start inside the
        # shard proper), for the same reason.
        family_sizes: dict[int, int] = defaultdict(int)
        n_pairs = 0
        for key, names in clusters.items():
            if start <= key[1] < end:
                family_sizes[2 * len(names)] += 1
                n_pairs += len(names)
        # n_orphans counts records, not pairs: each is a one-read family.
        family_sizes[1] += n_orphans

        n_written = 0
        with pysam.AlignmentFile(out_path, "wb", header=bam.header) as out:
            for read in bam.fetch(chrom, start, end):
                if not (prev_end <= read.reference_start < end):
                    continue
                if not _overlaps(chrom_regions, read.reference_start, read.reference_end or read.reference_start + 1):
                    continue
                name = read.query_name
                if name.endswith(("/1", "/2")):
                    name = name[:-2]
                if not _takes_family_tags(read, nonprimary_keys.get(name), pad):
                    _strip_family_tags(read)
                    out.write(read)
                    n_written += 1
                    continue
                if name in mi_map:
                    mi = mi_map[name]
                    if strip_run_id and "-" in mi:
                        mi = mi.split("-", 1)[1]
                    read.set_tag("MI", mi)
                elif read.has_tag("MI"):
                    read.tags = [(k, v) for k, v in read.tags if k != "MI"]
                read.set_tag("DS", ds_map.get(name, 1), value_type="i")
                _set_or_strip_cs(read, name, cs_map, strip_run_id)
                if po_tag:
                    _set_or_strip_po(read, orientations.get(name), po_tag)
                out.write(read)
                n_written += 1

    return {
        "shard": f"{chrom}:{start}-{end}",
        "out_path": out_path,
        "n_written": n_written,
        "n_pairs": n_pairs,
        "n_orphans": n_orphans,
        "family_sizes": dict(family_sizes),
    }


def process_unplaced(job: tuple) -> dict:
    """Copy the unplaced unmapped reads (the '*' block) through with DS=1."""
    bam_path, open_kwargs, out_path, po_tag = job
    n_written = 0
    with pysam.AlignmentFile(bam_path, "r", **open_kwargs) as bam:
        with pysam.AlignmentFile(out_path, "wb", header=bam.header) as out:
            for read in bam.fetch("*"):
                if read.has_tag("MI"):
                    read.tags = [(k, v) for k, v in read.tags if k != "MI"]
                # unplaced reads are never in a cluster, so any CS here is stale
                _set_or_strip_cs(read, read.query_name, None, False)
                if po_tag:
                    _set_or_strip_po(read, None, po_tag)
                read.set_tag("DS", 1, value_type="i")
                out.write(read)
                n_written += 1
    return {
        "shard": "*",
        "out_path": out_path,
        "n_written": n_written,
        "n_pairs": 0,
        "n_orphans": 0,
        "family_sizes": {},
    }


def run_sharded(args, open_kwargs: dict) -> None:
    # Shards are random-access region queries, so an index is mandatory here.
    with pysam.AlignmentFile(args.input, "r", **open_kwargs) as bam:
        if not bam.has_index():
            raise SystemExit(f"error: {args.input} must be indexed for sharded mode")
    if args.limit:
        logging.warning("--limit is ignored in sharded mode")

    regions = load_regions(args.regions, args.region_padding) if args.regions else None
    shard_size = args.shard_size or DEFAULT_SHARD_SIZE
    shards = build_shards(args.input, open_kwargs, shard_size, regions, args.max_shard_span)
    if regions is not None:
        n_bp = sum(e - s for starts, ends in regions.values() for s, e in zip(starts, ends, strict=False))
        logging.info(
            "Restricted to %s (+/-%d bp): %d merged intervals, %d bp",
            args.regions,
            args.region_padding,
            sum(len(starts) for starts, _ in regions.values()),
            n_bp,
        )
    logging.info(
        "Sharded mode: %d shards, pad %d bp, %d parallel job(s)",
        len(shards),
        args.pad,
        args.jobs,
    )

    tmp_dir = args.tmp_dir or os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".", "")
    tmp_dir = tempfile.mkdtemp(prefix="mark_dups_mi_shards_", dir=tmp_dir or None)
    logging.info("Per-shard BAMs in %s", tmp_dir)

    jobs = [
        (
            args.input,
            chrom,
            start,
            end,
            args.pad,
            args.use_umi,
            args.position_tolerance,
            args.cross_strand,
            open_kwargs,
            regions.get(chrom) if regions is not None else None,
            os.path.join(tmp_dir, f"{i:06d}.bam"),
            args.strip_run_id,
            shards[i - 1][2] if i > 0 and shards[i - 1][0] == chrom else 0,
            args.pair_orientation_tag,
            args.orientation_in_key,
        )
        for i, (chrom, start, end) in enumerate(shards)
    ]

    results: list[dict] = [None] * len(jobs)  # keep coordinate order for the concat
    n_done = 0
    # chunksize=1: shard cost spans orders of magnitude on a panel (one hot target can
    # hold more reads than a thousand cold ones), so hand them out one at a time.
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, result in enumerate(pool.map(process_shard, jobs, chunksize=1)):
            results[i] = result
            n_done += 1
            if n_done % 50 == 0 or n_done == len(jobs):
                logging.info("  %d/%d shards done", n_done, len(jobs))
            logging.debug(
                "  shard %s: %d reads written, %d pairs, %d unmatched",
                result["shard"],
                result["n_written"],
                result["n_pairs"],
                result["n_orphans"],
            )

    # Unplaced unmapped reads have no position to shard on, and are dropped outright
    # when the run is restricted to regions.
    if regions is None:
        unplaced = os.path.join(tmp_dir, "unplaced.bam")
        result = process_unplaced((args.input, open_kwargs, unplaced, args.pair_orientation_tag))
        logging.info("  unplaced reads: %d written", result["n_written"])
        if result["n_written"]:
            results.append(result)
        else:
            os.remove(unplaced)

    concat_shards([r["out_path"] for r in results], args.output, open_kwargs, tmp_dir, threads=args.jobs)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Threaded: at panel scale the concatenated output is several hundred GB, and a
    # single-threaded index over it takes longer than the whole marking pass.
    logging.info("Indexing %s", args.output)
    pysam.index("-@", str(args.jobs), args.output)

    total_written = sum(r["n_written"] for r in results)
    total_pairs = sum(r["n_pairs"] for r in results)
    total_orphans = sum(r["n_orphans"] for r in results)
    logging.info(
        "Done: %d reads written, %d pairs matched, %d reads left without a mate in their window",
        total_written,
        total_pairs,
        total_orphans,
    )

    family_sizes: dict[int, int] = defaultdict(int)
    for result in results:
        for size, count in result["family_sizes"].items():
            family_sizes[size] += count
    print_family_sizes(family_sizes)


def concat_shards(shard_paths: list[str], output_path: str, open_kwargs: dict, tmp_dir: str, threads: int = 1) -> None:
    """Concatenate the per-shard BAMs (already in coordinate order) into the output."""
    fofn = os.path.join(tmp_dir, "shards.fofn")
    with open(fofn, "w") as fh:
        fh.write("\n".join(shard_paths) + "\n")

    if output_path.endswith(".cram"):
        # Piped, not staged through a merged BAM: at panel scale that intermediate is
        # itself several hundred GB, on the same disk as the shards and the output.
        logging.info("Concatenating %d shards into %s", len(shard_paths), output_path)
        # S607: samtools is resolved from PATH on purpose. ugbio_base builds and ships
        # the CLI (ugbio_base/Dockerfile), and the same code has to run in a developer's
        # conda/uv environment, where the absolute path differs. Both argument lists are
        # fully literal apart from paths this process was given, so there is nothing to
        # inject through.
        cat = subprocess.Popen(
            ["samtools", "cat", "-b", fofn, "-o", "-"],  # noqa: S607
            stdout=subprocess.PIPE,
        )
        ref = open_kwargs["reference_filename"]
        view = subprocess.Popen(
            ["samtools", "view", "-C", "-@", str(threads), "-T", ref, "-o", output_path, "-"],  # noqa: S607
            stdin=cat.stdout,
        )
        cat.stdout.close()  # let samtools cat see EOF/SIGPIPE if view dies
        view_rc, cat_rc = view.wait(), cat.wait()
        for name, rc in (("samtools view", view_rc), ("samtools cat", cat_rc)):
            if rc != 0:
                raise subprocess.CalledProcessError(rc, name)
    else:
        logging.info("Concatenating %d shards into %s", len(shard_paths), output_path)
        subprocess.run(["samtools", "cat", "-b", fofn, "-o", output_path], check=True)  # noqa: S607


def print_family_sizes(family_sizes: dict[int, int]) -> None:
    total_reads = sum(size * n for size, n in family_sizes.items())
    print("family_size\tfamilies\treads\tpct_reads")
    for size in sorted(family_sizes):
        n_fam = family_sizes[size]
        if not n_fam:  # sharded runs seed size 1 with the orphan count, which may be 0
            continue
        n_reads = size * n_fam
        print(f"{size}\t{n_fam}\t{n_reads}\t{100 * n_reads / total_reads:.1f}%")


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    open_kwargs: dict = {}
    if args.reference:
        open_kwargs["reference_filename"] = args.reference

    if args.stats_only:
        logging.info("Stats-only mode — reading MI tags from %s", args.input)
        stats_from_mi_tags(args.input, open_kwargs)
        return

    if not args.output:
        sys.exit("error: output is required unless --stats-only is set")

    if args.jobs > 1 or args.shard_size or args.regions:
        run_sharded(args, open_kwargs)
        return

    logging.info("Pass 1 — scanning %s", args.input)
    orientations: dict[str, str] | None = {} if args.pair_orientation_tag else None
    unmatched: set[str] = set()
    pair_keys, orient = collect_pair_keys(
        args.input, args.use_umi, open_kwargs, args.limit, orientations, args.orientation_in_key, unmatched
    )
    logging.info("Collected info for %d primary paired read names", len(pair_keys))

    aliases: dict[tuple, tuple] = {}
    mi_map, ds_map, clusters = build_mi_map(
        pair_keys, orient if args.use_umi else None, args.position_tolerance, args.cross_strand, aliases
    )
    logging.info("MI map built (%d entries)", len(mi_map))

    cs_map = assign_cs(clusters, orient, aliases)
    if cs_map:
        members: dict[str, set[str]] = defaultdict(set)
        for name, cs in cs_map.items():
            members[cs].add(mi_map.get(name, name))
        n_linked = sum(1 for name, cs in cs_map.items() if len(members[cs]) > 1)
        logging.info(
            "CS map built (%d pairs in %d duplex groups; %d pairs in a group that has a " "cross-strand partner)",
            len(cs_map),
            len(members),
            n_linked,
        )

    logging.info("Pass 2 — writing %s", args.output)
    write_with_mi(
        args.input,
        args.output,
        mi_map,
        ds_map,
        open_kwargs,
        args.limit,
        args.strip_run_id,
        cs_map,
        args.pair_orientation_tag,
        orientations,
        pair_keys,
        args.pad,
    )

    logging.info("Indexing %s", args.output)
    pysam.index(args.output)

    logging.info("Done")
    print_family_size_distribution(mi_map, total_pairs=len(pair_keys), n_unmatched=len(unmatched))


if __name__ == "__main__":
    main()
