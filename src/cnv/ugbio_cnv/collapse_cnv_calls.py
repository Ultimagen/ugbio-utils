"""
Collapse overlapping PASS CNV calls of the same type into one call.

After filtering, one event can still have several overlapping PASS calls (e.g. a split-read candidate and a depth
call on the same deletion). Overlapping PASS calls of the same type (truvari collapse --chain, no size or distance
requirement) become one call: the highest-QUAL call is kept, with
  - start: among the group's starts within breakpoint_window of the leftmost start, the one with the tightest CIPOS;
    end: the same with CIEND, around the rightmost end (so a precise breakpoint wins over a nearby imprecise one);
  - CIPOS / CIEND of the chosen start / end, SVLEN of the new interval;
  - CN_MEAN: the length-weighted mean of the group;
  - CNV_SOURCE: the sources of the group; CNV_ID: the IDs of the calls merged into it.
A DEL and a DUP are never merged. Calls that are not PASS are written unchanged and never removed.
"""

import argparse
import os
import sys
import tempfile
from collections import defaultdict

import pysam
from ugbio_cnv.cnv_vcf_consts import INFO_TAG_REGISTRY
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
    tightest-CIEND end within window of the rightmost end (ties: the rightmost).
    """
    left = min(m["start"] for m in members)
    starts = [m for m in members if m["start"] - left <= window]
    s = min(starts, key=lambda m: (_ci_width(m["cipos"]), m["start"]))
    right = max(m["end"] for m in members)
    ends = [m for m in members if right - m["end"] <= window]
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
    rec.info["CNV_SOURCE"] = sorted({s for m in members for s in (m["source"] or ())})
    rec.info["CNV_ID"] = [m["id"] for m in members if m["id"] != rec.id]


def collapse_cnv_calls(input_vcf: str, output_vcf: str, breakpoint_window: int = 2500) -> int:
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
            with pysam.VariantFile(unsorted, "wz", header=header) as vout:
                for rec in vin:
                    rec.translate(header)
                    cid = rec.info.get("CollapseId")
                    if cid is not None and (members := groups.get(float(cid))):
                        _apply_group(rec, [_member(rec), *members], breakpoint_window)
                    for tag in COLLAPSE_TAGS:
                        if tag in rec.info:
                            del rec.info[tag]
                    vout.write(rec)
        vu.sort_vcf(unsorted, output_vcf)
        vu.index_vcf(output_vcf)
        if os.path.exists(removed_vcf):
            os.unlink(removed_vcf)
    logger.info(f"{output_vcf}: {n_merged} PASS calls merged into {len(groups)} calls of the same type")
    return n_merged


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collapse_cnv_calls", description="Collapse overlapping PASS CNV calls of the same type into one call."
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
    return parser.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv[1:])
    collapse_cnv_calls(args.input_vcf, args.output_vcf, args.breakpoint_window)


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
