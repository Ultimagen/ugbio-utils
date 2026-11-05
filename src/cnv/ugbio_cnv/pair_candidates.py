"""
Discordant-pair CNV candidates for the depth CNV caller (paired-end data).

pair_cnv_candidates:
  1. Per contig, in parallel, `samtools view -q 20 -F 0xF0C -f 0x1` streams the reads (no tags decoded) and the
     leftmost read of each pair with both mates mapped on the same chromosome and on opposite strands is kept.
     Forward read left, reverse mate right (FR) supports a deletion when the span (TLEN) is >= MIN_PAIR_SPAN, the
     span floor of the breakpoint evidence (analyze_cnv_breakpoint_reads). Reverse read left, forward mate right
     (RF, everted) supports a tandem duplication when the mate starts > min_rf_gap (50 bp) after the read: overlapping
     short fragments also read RF, with the mates within ~50 bp. An everted pair lies inside the duplicated copy, so
     its span is at most the duplication length and a span floor of 800 would miss the duplications below ~1 kb.
  2. Pairs are clustered per chromosome and type when both the left reads' ends and the mates' starts agree within
     cluster_tol; clusters with >= min_pairs pairs and up to 1 Mb are kept. A deletion is the gap between the left
     reads' ends (90th percentile) and the mates' starts (10th percentile), the interval breakpoint snapping allows
     (breakpoint_snapping.pair_bounds); it is kept from min_del_gap (300 bp): the gap can come out shorter than the
     deletion, and the 500 bp minimum is applied after snapping (filter_nested_cnv_candidates), when split reads have
     moved a real deletion to its breakpoints. A duplication contains the pairs: from the left reads' first start to
     the mates' last end (left start + TLEN), so the evidence step brackets the same pairs; kept from 500 bp.
  3. The same depth check as the split-read candidates (split_read_candidates.score_with_depth), and candidates with a
     same-type call of the input VCF overlapping >= 50% reciprocally are dropped (already called). The rest are added
     with CNV_SOURCE=pair, PAIR_READS and CIPOS / CIEND = +-ci (as snapping gives pair-shrunk calls).

Paired-end data only. Run after split_read_cnv_candidates; the nested / short filter (filter_nested_cnv_candidates)
after snapping treats pair candidates like split-read candidates.
"""

import argparse
import sys
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ugbio_cnv.analyze_cnv_breakpoint_reads import MIN_PAIR_SPAN
from ugbio_cnv.breakpoint_snapping import alignment_piece
from ugbio_cnv.cnv_vcf_consts import CNV_SOURCE_PAIR
from ugbio_cnv.depth_cnv_caller import read_sex_chromosomes
from ugbio_cnv.split_read_candidates import (
    PAIR_FIELDS,
    CandidateParams,
    add_candidates_to_calls,
    samtools_lines,
    score_with_depth,
)
from ugbio_core.logger import logger


@dataclass
class PairParams:
    min_mapq: int = 20
    min_span: int = MIN_PAIR_SPAN  # FR pairs (deletions): TLEN floor
    min_rf_gap: int = 50  # RF pairs (duplications): mate start - read start; overlapping short fragments are below it
    min_del_gap: int = 300  # shortest deletion candidate (the gap); 500 bp is required after snapping
    cluster_tol: int = 600  # pairs in one cluster: left ends and mate starts within +-cluster_tol
    min_pairs: int = 3
    ci: int = 200  # CIPOS / CIEND = +-ci


def _contig_pairs(args: tuple) -> list[tuple]:
    """Discordant pairs of one contig (leftmost read), selected in Python: no tags and no samtools -e expression."""
    alignment, reference, contig, p = args
    rows = []
    view = ["-q", str(p.min_mapq), "-F", "0xF0C", "-f", "0x1"]
    for line in samtools_lines(alignment, reference, contig, view, PAIR_FIELDS):
        f = line.split("\t", 9)
        tlen = int(f[8])
        if tlen <= 0 or f[6] != "=":  # leftmost read, mate on the same contig
            continue
        flag, pos0, mpos0 = int(f[1]), int(f[3]) - 1, int(f[7]) - 1
        rev, mate_rev = bool(flag & 16), bool(flag & 32)
        if not rev and mate_rev and tlen >= p.min_span:
            svtype = "DEL"  # FR
        elif rev and not mate_rev and mpos0 > pos0 + p.min_rf_gap:
            svtype = "DUP"  # RF, everted
        else:
            continue
        left_end = alignment_piece(pos0, "+", f[5], 0)[3]
        rows.append((contig, pos0, left_end, mpos0, pos0 + tlen, svtype))
    return rows


def stream_pairs(alignment: str, reference: str, contigs: list[str], p: PairParams, threads: int = 8) -> pd.DataFrame:
    """Discordant pairs (leftmost read): chrom, left_start, left_end, right_start, right_end, svtype.

    One process per contig (threads in parallel). Mates on opposite strands only: same-strand pairs are inversion-like.
    """
    with Pool(max(1, threads)) as pool:
        parts = pool.map(_contig_pairs, [(alignment, reference, c, p) for c in contigs])
    rows = [r for part in parts for r in part]
    return pd.DataFrame(rows, columns=["chrom", "left_start", "left_end", "right_start", "right_end", "svtype"])


def cluster_pairs(pairs: pd.DataFrame, p: PairParams, cp: CandidateParams) -> pd.DataFrame:
    """Candidates (chrom, start, end, svtype, n_reads) from clusters of pairs whose two ends agree."""
    rows = []
    for (chrom, svtype), g in pairs.sort_values(["left_end", "right_start"], kind="stable").groupby(
        ["chrom", "svtype"], sort=False
    ):
        ls, le, rs, re_ = (g[c].to_numpy() for c in ("left_start", "left_end", "right_start", "right_end"))
        used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            m = (~used) & (np.abs(le - le[i]) <= p.cluster_tol) & (np.abs(rs - rs[i]) <= p.cluster_tol)
            used |= m
            if m.sum() < p.min_pairs:
                continue
            if svtype == "DEL":  # the rule of breakpoint_snapping.pair_bounds
                start, end = int(np.percentile(le[m], 90)), int(np.percentile(rs[m], 10))
            else:
                start, end = int(ls[m].min()), int(re_[m].max())
            min_len = p.min_del_gap if svtype == "DEL" else cp.min_len
            if min_len <= end - start <= cp.max_len:
                rows.append((chrom, start, end, svtype, int(m.sum())))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "svtype", "n_reads"])


def pair_cnv_candidates(  # noqa: PLR0913
    alignment: str,
    reference: str,
    input_vcf: str,
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    output_vcf: str,
    contigs: list[str] | None = None,
    threads: int = 8,
    params: PairParams | None = None,
) -> int:
    """
    Add discordant-pair candidates to the calls of input_vcf; returns the number of candidates added.

    The sex chromosome setup (haploid / not called contigs) is the one call_depth_cnvs recorded in the VCF header.
    """
    p = params or PairParams()
    sex = read_sex_chromosomes(input_vcf)
    if not contigs:
        contigs = pd.read_parquet(annotation, columns=["chrom"])["chrom"].astype(str).unique().tolist()
    contigs = [c for c in contigs if c not in sex.skipped]
    cp = CandidateParams()
    pairs = stream_pairs(alignment, reference, contigs, p, threads)
    clusters = cluster_pairs(pairs, p, cp) if len(pairs) else pd.DataFrame()
    scored = score_with_depth(clusters, depth_mq0, depth_mq20, annotation, sex.haploid, cp) if len(clusters) else None
    if scored is None or not len(scored):
        scored = pd.DataFrame(columns=["chrom", "start", "end", "svtype"])
    n = add_candidates_to_calls(
        input_vcf,
        scored,
        output_vcf,
        cp,
        source=CNV_SOURCE_PAIR,
        id_prefix="PAIR",
        count_tag="PAIR_READS",
        ci=(-p.ci, p.ci),
    )
    logger.info(
        f"{output_vcf}: pairs with span >= {p.min_span} {len(pairs)}, clusters {len(clusters)}, "
        f"with depth support {len(scored)}, added {n}"
    )
    return n


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pair_cnv_candidates", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--alignment", required=True, help="Sample BAM / CRAM (paired-end)")
    parser.add_argument("--reference-fasta", required=True, help="Reference FASTA")
    parser.add_argument("--input-vcf", required=True, help="Calls so far (split_read_cnv_candidates output)")
    parser.add_argument("--depth-mq0", required=True, help="mosdepth regions.bed.gz, all reads (with .csi)")
    parser.add_argument("--depth-mq20", required=True, help="mosdepth regions.bed.gz, MAPQ>=20 (with .csi)")
    parser.add_argument("--annotation", required=True, help="Reference annotation parquet")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument("--contigs", nargs="+", default=None, help="Contigs (default: the contigs of the annotation)")
    parser.add_argument("--threads", type=int, default=8, help="Contigs processed in parallel")
    return parser.parse_args(argv)


def run(argv: list[str]):
    args = parse_args(argv[1:])
    pair_cnv_candidates(
        args.alignment,
        args.reference_fasta,
        args.input_vcf,
        args.depth_mq0,
        args.depth_mq20,
        args.annotation,
        args.output_vcf,
        args.contigs,
        args.threads,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
