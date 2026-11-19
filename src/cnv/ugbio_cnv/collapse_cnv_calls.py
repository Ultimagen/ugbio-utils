"""
Collapse overlapping PASS CNV calls of the same type into one call, and drop the non-PASS calls inside PASS calls.

After filtering, one event can still have several overlapping PASS calls (e.g. a split-read candidate and a depth
call on the same deletion). Overlapping PASS calls of the same type (truvari collapse --chain, no size or distance
requirement) become one call: the highest-QUAL call is kept, with
  - start: among the group's starts within breakpoint_window of the leftmost start, the one with the tightest CIPOS;
    end: the same with CIEND, around the rightmost end (so a precise breakpoint wins over a nearby imprecise one);
  - CIPOS / CIEND of the chosen start / end, SVLEN of the new interval;
  - CN_MEAN: the length-weighted mean of the group;
  - CNV_SOURCE: the sources of the group; CNV_ID: the IDs of the calls merged into it.
A DEL and a DUP are never merged. So after the collapse no two PASS calls of the same type overlap, and every base
is covered by at most two PASS calls (a DEL and a DUP, one per haplotype).

Calls that are not PASS are not merged. One that lies at least filtered_inside_frac inside a single PASS call, of
either type, is dropped (the PASS call reports the region); the others are written unchanged. Each one is compared
with the PASS calls directly, never through a chain of overlaps, so every dropped call is covered by a call that
stays.

filter_nested_cnv_candidates (before filtering, after breakpoint refinement):
  Drop split-read, pair and cn.mops candidates that are shorter than 500 bp, >= 50% inside a same-type depth call (a
  second call on an event that is already called), or with >= 50% reciprocal overlap with a same-type candidate kept
  before them (refinement often moves several candidates of one deletion to the same breakpoints). Split-read
  candidates are kept first, then pair candidates, each by the most reads in their cluster (CLUSTER_SUPPORT), then
  cn.mops candidates.
"""

import argparse
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pysam
from ugbio_cnv.cnv_vcf_consts import (
    CNV_SOURCE_CNMOPS,
    CNV_SOURCE_DEPTH,
    CNV_SOURCE_PAIR,
    CNV_SOURCE_SPLIT_READ,
    INFO_TAG_REGISTRY,
)
from ugbio_cnv.depth_cnv_caller import genotype_for
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils

COLLAPSE_TAGS = ("CollapseId", "NumCollapsed", "NumConsolidated", "MatchId")


def _ci_width(ci) -> int:
    """Width of a confidence interval (CIPOS / CIEND); a missing one counts as infinitely wide."""
    return ci[1] - ci[0] if ci and None not in ci else sys.maxsize


def _member(rec: pysam.VariantRecord) -> dict:
    return {
        "id": rec.id,
        "start": rec.start,
        "end": rec.stop,
        "cipos": rec.info.get("CIPOS"),
        "ciend": rec.info.get("CIEND"),
        "cn_mean": rec.info.get("CN_MEAN"),
        "source": rec.info.get("CNV_SOURCE", ()),
    }


def merged_breakpoints(members: list[dict], window: int) -> tuple[int, int, tuple, tuple]:
    """
    Start, end, CIPOS and CIEND of a collapsed group.

    The start is the tightest-CIPOS start within window of the leftmost start (ties: the leftmost); the end is the
    tightest-CIEND end within window of the rightmost end (ties: the rightmost), among the ends after that start.
    """
    left = min(m["start"] for m in members)
    starts = [m for m in members if m["start"] - left <= window]
    s = min(starts, key=lambda m: (_ci_width(m["cipos"]), m["start"]))
    right = max(m["end"] for m in members)
    ends = [m for m in members if right - m["end"] <= window and m["end"] > s["start"]]  # never before the start
    e = min(ends, key=lambda m: (_ci_width(m["ciend"]), -m["end"]))
    return s["start"], e["end"], s["cipos"], e["ciend"]


def _weighted_cn(members: list[dict]) -> float | None:
    known = [(m["cn_mean"], m["end"] - m["start"]) for m in members if m["cn_mean"] is not None]
    total = sum(length for _, length in known)
    return round(sum(cn * length for cn, length in known) / total, 3) if total else None


def _apply_group(rec: pysam.VariantRecord, members: list[dict], window: int) -> None:
    start, end, cipos, ciend = merged_breakpoints(members, window)
    rec.start, rec.stop = start, end
    rec.info["SVLEN"] = end - start
    if cipos is not None:
        rec.info["CIPOS"] = cipos
    if ciend is not None:
        rec.info["CIEND"] = ciend
    if (cn := _weighted_cn(members)) is not None:
        rec.info["CN_MEAN"] = cn
        for sample in rec.samples:  # the genotype follows the merged copy number (homozygous deletion below CN 0.5)
            rec.samples[sample]["GT"] = genotype_for(rec.info["SVTYPE"], cn)
    rec.info["CNV_SOURCE"] = sorted({s for m in members for s in (m["source"] or ())})
    rec.info["CNV_ID"] = [m["id"] for m in members if m["id"] != rec.id]


def inside_fraction(start: int, end: int, starts: np.ndarray, ends: np.ndarray) -> float:
    """Largest fraction of [start, end) that lies inside one of the intervals [starts, ends)."""
    if end <= start or not len(starts):
        return 0.0
    overlap = np.minimum(ends, end) - np.maximum(starts, start)
    return float(max(overlap.max(), 0) / (end - start))


def _drop_redundant_filtered(records: list[pysam.VariantRecord], min_frac: float) -> list[pysam.VariantRecord]:
    """Drop the non-PASS records that lie >= min_frac inside one PASS record (DEL or DUP) on the same contig."""
    passing = defaultdict(list)
    for rec in records:
        if "PASS" in rec.filter:
            passing[rec.chrom].append((rec.start, rec.stop))
    intervals = {k: (np.array([s for s, _ in v]), np.array([e for _, e in v])) for k, v in passing.items()}
    kept = []
    for rec in records:
        if "PASS" not in rec.filter and rec.chrom in intervals:
            if inside_fraction(rec.start, rec.stop, *intervals[rec.chrom]) >= min_frac:
                continue
        kept.append(rec)
    return kept


def collapse_cnv_calls(
    input_vcf: str, output_vcf: str, breakpoint_window: int = 2500, filtered_inside_frac: float = 0.5
) -> int:
    """
    Collapse overlapping PASS calls of the same type in input_vcf; writes output_vcf (bgzipped, indexed).

    Parameters
    ----------
    input_vcf : str
        Filtered CNV calls (FilterVCF output) with unique IDs.
    output_vcf : str
        Output .vcf.gz.
    breakpoint_window : int
        A start (end) within this distance of the group's leftmost start (rightmost end) may be chosen for its
        tighter CIPOS (CIEND).
    filtered_inside_frac : float
        A non-PASS call at least this fraction inside a PASS call (DEL or DUP) is dropped; 0 keeps every non-PASS call.

    Returns
    -------
    int
        Number of calls merged into others.
    """
    vu = VcfUtils()
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(output_vcf))) as tmp:
        kept_vcf = os.path.join(tmp, "kept.vcf.gz")
        removed_vcf = vu.collapse_vcf(
            vcf=input_vcf,
            output_vcf=kept_vcf,
            refdist=0,
            pctseq=0.0,
            pctsize=0.0,
            maxsize=-1,
            ignore_filter=False,  # PASS calls only; the others are written unchanged
            ignore_sv_type=False,  # a DEL and a DUP are never merged
            pick_best=True,  # keep the highest QUAL
            erase_removed=False,
        )
        groups: dict[float, list[dict]] = defaultdict(list)
        with pysam.VariantFile(removed_vcf) as removed:
            for rec in removed:
                groups[float(rec.info["MatchId"][0])].append(_member(rec))
        n_merged = sum(len(g) for g in groups.values())
        unsorted = os.path.join(tmp, "collapsed.unsorted.vcf.gz")
        with pysam.VariantFile(kept_vcf) as vin:
            header = vin.header.copy()
            if "CNV_ID" not in header.info:
                header.info.add(*INFO_TAG_REGISTRY["CNV_ID"][:-1])
            records = []
            for rec in vin:
                rec.translate(header)
                cid = rec.info.get("CollapseId")
                if cid is not None and (members := groups.get(float(cid))):
                    _apply_group(rec, [_member(rec), *members], breakpoint_window)
                for tag in COLLAPSE_TAGS:
                    if tag in rec.info:
                        del rec.info[tag]
                records.append(rec)
            kept = _drop_redundant_filtered(records, filtered_inside_frac) if filtered_inside_frac > 0 else records
            n_dropped = len(records) - len(kept)
            with pysam.VariantFile(unsorted, "wz", header=header) as vout:
                for rec in kept:
                    vout.write(rec)
        vu.sort_vcf(unsorted, output_vcf)
        vu.index_vcf(output_vcf)
        if os.path.exists(removed_vcf):
            os.unlink(removed_vcf)
    logger.info(
        f"{output_vcf}: {n_merged} PASS calls merged into {len(groups)} calls of the same type; "
        f"{n_dropped} non-PASS calls dropped inside a PASS call"
    )
    return n_merged


# candidates kept first in the duplicate rule of filter_nested_cnv_candidates
CANDIDATE_KEEP_ORDER = (CNV_SOURCE_SPLIT_READ, CNV_SOURCE_PAIR, CNV_SOURCE_CNMOPS)


def _candidate_rank(r: pysam.VariantRecord) -> tuple:
    """Order in which candidates are kept: split-read, pair, then cn.mops; then the most reads, then position."""
    source = r.info.get("CNV_SOURCE", ())
    rank = next((k for k, s in enumerate(CANDIDATE_KEEP_ORDER) if s in source), 0)
    # cn.mops candidates have no read count; the tag is not defined when no candidate has one
    reads = (r.info.get("CLUSTER_SUPPORT") if "CLUSTER_SUPPORT" in r.header.info else None) or 0
    return (rank, -reads, r.start, r.stop)


def filter_nested_cnv_candidates(
    input_vcf: str, output_vcf: str, min_len: int = 500, min_inside: float = 0.5, min_overlap: float = 0.5
) -> dict[str, int]:
    """
    Drop split-read, pair and cn.mops candidates that are short, nested in a depth call or duplicates; returns counts.

    A candidate is dropped when shorter than min_len ("short"), >= min_inside inside a same-type depth call ("nested"),
    or with >= min_overlap reciprocal overlap with a same-type candidate kept before it ("duplicate"; split-read
    candidates are kept first, then pair candidates, each by the most reads in their cluster, then cn.mops).
    """
    with pysam.VariantFile(input_vcf) as vin:
        records = list(vin)
        header = vin.header.copy()

    def source(r):
        return tuple(r.info.get("CNV_SOURCE", ()))

    candidates = [
        r for r in records if source(r) in ((CNV_SOURCE_SPLIT_READ,), (CNV_SOURCE_PAIR,), (CNV_SOURCE_CNMOPS,))
    ]
    depth_calls: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r in records:
        if CNV_SOURCE_DEPTH in source(r):
            depth_calls.setdefault((r.chrom, r.info["SVTYPE"]), []).append((r.start, r.stop))
    depth = {k: np.array(v) for k, v in depth_calls.items()}
    kept_candidates: dict[tuple[str, str], list[tuple[int, int]]] = {}
    dropped_ids, dropped = set(), {"nested": 0, "short": 0, "duplicate": 0}
    for r in sorted(candidates, key=_candidate_rank):
        length, key = r.stop - r.start, (r.chrom, r.info["SVTYPE"])
        reason = None
        if length < min_len:
            reason = "short"
        elif key in depth and inside_fraction(r.start, r.stop, depth[key][:, 0], depth[key][:, 1]) >= min_inside:
            reason = "nested"
        elif any(
            min(e, r.stop) - max(s, r.start) >= min_overlap * max(e - s, length)
            for s, e in kept_candidates.get(key, [])
        ):
            reason = "duplicate"
        if reason:
            dropped[reason] += 1
            dropped_ids.add(id(r))
        else:
            kept_candidates.setdefault(key, []).append((r.start, r.stop))
    kept = [r for r in records if id(r) not in dropped_ids]
    with pysam.VariantFile(output_vcf, "wz", header=header) as vout:
        for r in kept:
            vout.write(r)
    pysam.tabix_index(output_vcf, preset="vcf", force=True)
    logger.info(f"{output_vcf}: kept {len(kept)} of {len(records)} (dropped {dropped})")
    return dropped


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collapse_cnv_calls",
        description="Collapse overlapping PASS CNV calls of the same type into one call, and drop the non-PASS calls "
        "inside PASS calls.",
    )
    parser.add_argument("--input-vcf", required=True, help="Filtered CNV calls (FilterVCF output)")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument(
        "--breakpoint-window",
        type=int,
        default=2500,
        help="Breakpoints within this distance of the group's outer edges may be chosen for a tighter CI "
        "(default: 2500)",
    )
    parser.add_argument(
        "--filtered-inside-frac",
        type=float,
        default=0.5,
        help="Drop a non-PASS call that lies at least this fraction inside a PASS call (DEL or DUP); 0 keeps them all "
        "(default: 0.5)",
    )
    return parser.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv[1:])
    collapse_cnv_calls(args.input_vcf, args.output_vcf, args.breakpoint_window, args.filtered_inside_frac)


def main() -> None:
    run(sys.argv)


def parse_args_nested(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="filter_nested_cnv_candidates",
        description="Drop split-read, pair and cn.mops candidates that are < 500 bp, nested or duplicates",
    )
    parser.add_argument("--input-vcf", required=True, help="Refined calls")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument("--min-len", type=int, default=500, help="Minimal candidate length after refinement")
    return parser.parse_args(argv)


def run_nested(argv: list[str]):
    args = parse_args_nested(argv[1:])
    filter_nested_cnv_candidates(args.input_vcf, args.output_vcf, args.min_len)


def main_nested():
    run_nested(sys.argv)


if __name__ == "__main__":
    main()
