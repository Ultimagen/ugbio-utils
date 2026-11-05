"""
Snap CNV breakpoints to split-read junctions, or shrink deletions with discordant pairs.

For each DEL/DUP call, primary reads (MAPQ >= 20) within +-window of either edge that carry an SA tag on the same
chromosome and strand are split into their two aligned pieces, ordered along the read:
  DEL: piece 1 ends at the left breakpoint L, piece 2 starts at the right breakpoint R (R > L)
  DUP (tandem): piece 1 ends at the duplication end E, piece 2 starts at the duplication start S (S < E)
Junctions are grouped by similar position (+-100 bp) and size (+-10%); the largest group with >= 2 reads replaces the
call's breakpoints (SNAP=SR). Calls > 3 kb only accept junctions at least half their size, so a small indel near a
large event cannot shrink it. Without a junction, DEL calls are shrunk to the interval allowed by FR discordant pairs
spanning both edges (SNAP=PE). Otherwise the call is unchanged (SNAP=none). The depth breakpoints are kept in
DEPTH_POS / DEPTH_END.
"""

import argparse
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pysam
from ugbio_cnv.cnv_vcf_consts import DEPTH_CNV_INFO_TAGS
from ugbio_core.logger import logger

CIGAR_PATTERN = re.compile(r"(\d+)([MIDNSHP=X])")


@dataclass
class SnapParams:
    window: int = 1500  # search reads within +-window of each call edge
    min_reads: int = 2  # split reads (or pairs) needed to move a breakpoint
    min_mapq: int = 20  # primary read MAPQ
    min_sa_mapq: int = 1  # supplementary alignment MAPQ (from the SA tag)
    pos_tol: int = 100  # junctions in one group: start within +-pos_tol ...
    size_tol: float = 0.10  # ... and size within +-size_tol
    min_junction: int = 100  # shorter junctions are small indels
    small_call: int = 3000  # longer calls need a junction of >= 0.5 x their size
    min_pe_len: int = 50  # a pair-shrunk deletion must keep at least this length


def read_length_from_cigar(cigar: str) -> int:
    """Read length (incl. soft and hard clips) from a CIGAR string."""
    return sum(int(n) for n, op in CIGAR_PATTERN.findall(cigar) if op in "MIS=XH")


def alignment_piece(pos0: int, strand: str, cigar: str, read_len: int) -> tuple[int, int, int, int]:
    """
    One aligned piece of a read.

    Returns
    -------
    tuple[int, int, int, int]
        (query_start, query_end) in read orientation, then (ref_start, ref_end) 0-based half-open.
    """
    ops = [(int(n), op) for n, op in CIGAR_PATTERN.findall(cigar)]
    lead = ops[0][0] if ops[0][1] in "SH" else 0
    qlen = sum(n for n, op in ops if op in "MI=X")
    rlen = sum(n for n, op in ops if op in "MDN=X")
    qs, qe = lead, lead + qlen
    if strand == "-":
        qs, qe = read_len - qe, read_len - qs
    return qs, qe, pos0, pos0 + rlen


def split_read_junctions(
    chrom: str,
    pos0: int,
    cigar: str,
    sa_tag: str,
    *,
    is_reverse: bool,
    min_sa_mapq: int = 1,
) -> list[tuple[int, int, str]]:
    """
    Deletion / tandem-duplication junctions implied by one split read.

    Each SA piece on the same chromosome and strand is ordered with the primary piece along the read. If the second
    piece starts after the first ends on the reference, the junction is a deletion (L, R); if it starts before, a
    tandem duplication (S, E).

    Returns
    -------
    list[tuple[int, int, str]]
        (start, end, svtype) per junction, 0-based.
    """
    strand = "-" if is_reverse else "+"
    read_len = read_length_from_cigar(cigar)
    primary = alignment_piece(pos0, strand, cigar, read_len)
    out = []
    for sa in sa_tag.rstrip(";").split(";"):
        sa_chrom, sa_pos, sa_strand, sa_cigar, sa_mapq, _ = sa.split(",")
        if sa_chrom != chrom or sa_strand != strand or int(sa_mapq) < min_sa_mapq:
            continue
        other = alignment_piece(int(sa_pos) - 1, sa_strand, sa_cigar, read_len)
        first, second = sorted((primary, other), key=lambda x: x[0])  # along the read
        if strand == "-":  # on the minus strand, read order runs right to left on the reference
            first, second = second, first
        if second[2] > first[3]:
            out.append((first[3], second[2], "DEL"))
        elif second[2] < first[3]:
            out.append((second[2], first[3], "DUP"))
    return out


def edge_junctions(
    aln: pysam.AlignmentFile, chrom: str, start: int, end: int, svtype: str, p: SnapParams
) -> list[tuple[int, int]]:
    """Split-read junctions of the call's type from reads near either edge of [start, end)."""
    out, seen = [], set()
    for a, b in ((start - p.window, start + p.window), (end - p.window, end + p.window)):
        for r in aln.fetch(chrom, max(a, 0), b):
            if r.is_secondary or r.is_supplementary or r.mapping_quality < p.min_mapq or not r.has_tag("SA"):
                continue
            if r.query_name in seen:
                continue
            seen.add(r.query_name)
            for js, je, jt in split_read_junctions(
                chrom,
                r.reference_start,
                r.cigarstring,
                r.get_tag("SA"),
                is_reverse=r.is_reverse,
                min_sa_mapq=p.min_sa_mapq,
            ):
                if jt == svtype:
                    out.append((js, je))
    return out


def best_junction_cluster(
    junctions: list[tuple[int, int]], call_len: int, p: SnapParams
) -> tuple[int, int, int] | None:
    """Largest group of junctions with similar position and size; breakpoints are the group medians."""
    # depth cannot resolve events below ~1-2 kb, so only short calls may shrink to a much smaller junction
    min_size = p.min_junction if call_len <= p.small_call else 0.5 * call_len
    js = [(a, b) for a, b in junctions if min_size <= b - a <= 3 * call_len + 2 * p.window]
    if len(js) < p.min_reads:
        return None
    js = np.array(js)
    sizes = js[:, 1] - js[:, 0]
    best = None
    for a, b in js:
        size = b - a
        sel = js[(np.abs(js[:, 0] - a) <= p.pos_tol) & (np.abs(sizes - size) <= p.size_tol * size)]
        if best is None or len(sel) > len(best):
            best = sel
    if len(best) < p.min_reads:
        return None
    return int(np.median(best[:, 0])), int(np.median(best[:, 1])), len(best)


def pair_bounds(
    aln: pysam.AlignmentFile, chrom: str, start: int, end: int, p: SnapParams
) -> tuple[int, int, int] | None:
    """Interval allowed by FR pairs with one read near the left edge and its mate near the right edge."""
    left_end, right_start = [], []
    for r in aln.fetch(chrom, max(start - p.window, 0), start + p.window):
        if (
            r.is_secondary
            or r.is_supplementary
            or not r.is_paired
            or r.mate_is_unmapped
            or r.is_reverse
            or not r.mate_is_reverse
            or r.next_reference_name != chrom
            or r.mapping_quality < p.min_mapq
        ):
            continue
        if end - p.window <= r.next_reference_start <= end + p.window and r.next_reference_start > r.reference_end:
            left_end.append(r.reference_end)
            right_start.append(r.next_reference_start)
    if len(left_end) < p.min_reads:
        return None
    return int(np.percentile(left_end, 90)), int(np.percentile(right_start, 10)), len(left_end)


def _snap_contig(args: tuple) -> dict:
    vcf, aln_path, reference, chrom, p = args
    res = {}
    with pysam.AlignmentFile(aln_path, reference_filename=reference) as aln, pysam.VariantFile(vcf) as v:
        if chrom not in aln.references:
            return res
        for rec in v.fetch(chrom):
            svtype, s, e = rec.info["SVTYPE"], rec.start, rec.stop
            hit = best_junction_cluster(edge_junctions(aln, chrom, s, e, svtype, p), e - s, p)
            if hit:
                res[rec.id] = ("SR", hit[0], hit[1], hit[2])
                continue
            if svtype == "DEL":
                pb = pair_bounds(aln, chrom, s, e, p)
                if pb and pb[1] > pb[0]:
                    ns, ne = max(s, pb[0]), min(e, pb[1])
                    if ne - ns >= p.min_pe_len and (ns, ne) != (s, e):
                        res[rec.id] = ("PE", ns, ne, pb[2])
    return res


def snap_cnv_breakpoints(
    input_vcf: str, alignment: str, reference: str, output_vcf: str, threads: int = 8, params: SnapParams | None = None
) -> Counter:
    """
    Snap the breakpoints of every call in input_vcf and write output_vcf (bgzipped, indexed).

    Parameters
    ----------
    input_vcf : str
        CNV calls with unique IDs.
    alignment : str
        BAM / CRAM of the sample.
    reference : str
        Reference FASTA (for CRAM).
    output_vcf : str
        Output .vcf.gz.
    threads : int
        Contigs processed in parallel.
    params : SnapParams, optional
        Snapping parameters.

    Returns
    -------
    Counter
        Number of calls per snap source (SR, PE).
    """
    p = params or SnapParams()
    if not any(os.path.exists(input_vcf + ext) for ext in (".tbi", ".csi")):
        pysam.tabix_index(input_vcf, preset="vcf", force=True)  # contigs are fetched in parallel
    with pysam.VariantFile(input_vcf) as v:
        contigs = list(dict.fromkeys(r.chrom for r in v))
    snaps = {}
    with Pool(threads) as pool:
        for r in pool.imap_unordered(_snap_contig, [(input_vcf, alignment, reference, c, p) for c in contigs]):
            snaps.update(r)
    with pysam.VariantFile(input_vcf) as vin:
        header = vin.header.copy()
        for tag in ("SNAP", "SNAP_N", "DEPTH_POS", "DEPTH_END", "CIEND"):
            if tag not in header.info:
                header.info.add(*DEPTH_CNV_INFO_TAGS[tag][:-1])
        records = []
        for rec in vin:
            source, ns, ne, n_support = snaps.get(rec.id, ("none", rec.start, rec.stop, 0))
            new = rec.copy()
            new.translate(header)
            new.info["DEPTH_POS"], new.info["DEPTH_END"] = rec.pos, rec.stop
            new.info["SNAP"], new.info["SNAP_N"] = source, n_support
            if source != "none":
                new.start, new.stop = ns, ne
                new.info["SVLEN"] = ne - ns
                ci = (-10, 10) if source == "SR" else (-200, 200)
                new.info["CIPOS"], new.info["CIEND"] = ci, ci
            records.append(new)
    order = {c: i for i, c in enumerate(header.contigs)}
    records.sort(key=lambda r: (order[r.chrom], r.start))
    with pysam.VariantFile(output_vcf, "wz", header=header) as vout:
        for r in records:
            vout.write(r)
    pysam.tabix_index(output_vcf, preset="vcf", force=True)
    counts = Counter(s[0] for s in snaps.values())
    logger.info(f"{output_vcf}: {sum(counts.values())} of {len(records)} calls snapped ({dict(counts)})")
    return counts


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="snap_cnv_breakpoints", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-vcf", required=True, help="CNV calls (unique IDs)")
    parser.add_argument("--alignment", required=True, help="Sample BAM / CRAM")
    parser.add_argument("--reference-fasta", required=True, help="Reference FASTA")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument("--threads", type=int, default=8, help="Contigs processed in parallel")
    return parser.parse_args(argv)


def run(argv: list[str]):
    args = parse_args(argv[1:])
    snap_cnv_breakpoints(args.input_vcf, args.alignment, args.reference_fasta, args.output_vcf, args.threads)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
