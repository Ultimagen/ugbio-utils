"""
Split-read CNV candidates for the depth CNV caller.

split_read_cnv_candidates:
  1. Per contig, in parallel: the supplementary records give each split read's other pieces (in place of the SA tag,
     so that no tags are decoded), then the primary records (MAPQ >= 20) of those reads are streamed; each read's two
     pieces on the same chromosome and strand give a deletion or tandem-duplication junction
     (breakpoint_refinement.split_read_junctions).
  2. Junctions are clustered (position +-100 bp, size +-10%); clusters with >= 2 reads and 500 bp - 1 Mb are kept,
     and clusters with >= 50% reciprocal overlap are collapsed to the one with most reads.
  3. Each cluster is scored with depth (all reads, 100 bp): mean inside vs median of the 5 kb flanks, over the
     callable bins only (no N, GC in range). Candidates overlapping an assembly gap, and those without a depth change
     in the right direction (DEL ratio < 0.85, DUP > 1.15), are dropped.
  4. Candidates with >= 50% reciprocal overlap with a same-type depth call are dropped (already called); the rest are
     added to the depth calls with CNV_SOURCE=split_read.
"""

import argparse
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ugbio_cnv.breakpoint_refinement import split_read_junctions
from ugbio_cnv.cnv_candidate_utils import (
    SPLIT_READ_FIELDS,
    CandidateParams,
    add_candidates_to_calls,
    dedup_clusters,
    samtools_lines,
    score_with_depth,
)
from ugbio_cnv.depth_cnv_caller import read_sex_chromosomes
from ugbio_core.logger import logger


def _contig_junctions(args: tuple) -> list[tuple]:
    """
    Junctions of one contig without decoding tags: the supplementary records stand in for the SA tag.

    Pass 1 collects the supplementary alignments (not secondary / duplicate) per read name and mate; pass 2 streams
    the primary records (MAPQ >= min_mapq, not secondary / supplementary / duplicate) of those read names and pairs
    each with its supplementary pieces on the contig. Pieces on other contigs never give a junction.
    """
    alignment, reference, contig, p = args
    pieces: dict[tuple[str, int], list[str]] = {}
    for line in samtools_lines(alignment, reference, contig, ["-f", "0x800", "-F", "0x504"], SPLIT_READ_FIELDS):
        f = line.split("\t", 6)
        strand = "-" if int(f[1]) & 16 else "+"
        pieces.setdefault((f[0], int(f[1]) & 0xC0), []).append(f"{f[2]},{f[3]},{strand},{f[5]},{f[4]},0")
    if not pieces:
        return []
    out = []
    with tempfile.NamedTemporaryFile("w", suffix=".names.txt") as names:
        names.write("\n".join(sorted({q for q, _ in pieces})) + "\n")
        names.flush()
        args = ["-q", str(p.min_mapq), "-F", "0xD04", "-N", names.name]
        for line in samtools_lines(alignment, reference, contig, args, SPLIT_READ_FIELDS):
            f = line.split("\t", 6)
            flag = int(f[1])
            sa = pieces.get((f[0], flag & 0xC0))
            if sa is None:
                continue
            out.extend(
                (contig, *j)
                for j in split_read_junctions(
                    contig, int(f[3]) - 1, f[5], ";".join(sa), is_reverse=bool(flag & 16), min_sa_mapq=p.min_sa_mapq
                )
            )
    return out


def stream_junctions(
    alignment: str,
    reference: str,
    contigs: list[str],
    p: CandidateParams,
    threads: int = 8,
) -> pd.DataFrame:
    """Junctions (chrom, start, end, svtype, len) from all primary split reads, within [min_len, max_len].

    One process per contig (threads in parallel).
    """
    with Pool(max(1, threads)) as pool:
        parts = pool.map(_contig_junctions, [(alignment, reference, c, p) for c in contigs])
    j = pd.DataFrame([r for part in parts for r in part], columns=["chrom", "start", "end", "svtype"])
    j["len"] = j["end"] - j["start"]
    return j[(j["len"] >= p.min_len) & (j["len"] <= p.max_len)].reset_index(drop=True)


def cluster_junctions(j: pd.DataFrame, p: CandidateParams) -> pd.DataFrame:
    """Greedy clustering per chromosome and type, by start; breakpoints are the cluster medians."""
    rows = []
    for (chrom, svtype), g in j.sort_values(["start", "end"], kind="stable").groupby(["chrom", "svtype"], sort=False):
        s, e = g["start"].to_numpy(), g["end"].to_numpy()
        used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            size = e[i] - s[i]
            m = (~used) & (np.abs(s - s[i]) <= p.pos_tol) & (np.abs((e - s) - size) <= p.size_tol * size)
            used |= m
            if m.sum() >= p.min_reads:
                rows.append((chrom, int(np.median(s[m])), int(np.median(e[m])), svtype, int(m.sum())))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "svtype", "n_reads"])


def split_read_cnv_candidates(
    alignment: str,
    reference: str,
    depth_vcf: str,
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    output_vcf: str,
    contigs: list[str] | None = None,
    threads: int = 8,
    params: CandidateParams | None = None,
) -> int:
    """
    Add split-read candidates to the depth calls; returns the number of candidates added.

    The sex chromosome setup (haploid / not called contigs, male PARs) is the one call_depth_cnvs recorded in the depth
    VCF.
    """
    p = params or CandidateParams()
    sex = read_sex_chromosomes(depth_vcf)
    skipped = sex.skipped
    if not contigs:  # the contigs the depth caller annotates, in the reference's naming (chr1.. or 1..)
        contigs = pd.read_parquet(annotation, columns=["chrom"])["chrom"].astype(str).unique().tolist()
    contigs = [c for c in contigs if c not in skipped]
    j = stream_junctions(alignment, reference, contigs, p, threads)
    c = cluster_junctions(j, p)
    c = dedup_clusters(c, p) if len(c) else c
    s = score_with_depth(c, depth_mq0, depth_mq20, annotation, sex, p) if len(c) else pd.DataFrame()
    n = (
        add_candidates_to_calls(depth_vcf, s, output_vcf, p)
        if len(s)
        else add_candidates_to_calls(
            depth_vcf, pd.DataFrame(columns=["chrom", "start", "end", "svtype"]), output_vcf, p
        )
    )
    logger.info(f"{output_vcf}: junctions {len(j)}, clusters {len(c)}, with depth support {len(s)}, added {n}")
    return n


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="split_read_cnv_candidates", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--alignment", required=True, help="Sample BAM / CRAM")
    parser.add_argument("--reference-fasta", required=True, help="Reference FASTA")
    parser.add_argument("--depth-vcf", required=True, help="Depth calls (call_depth_cnvs)")
    parser.add_argument("--depth-mq0", required=True, help="mosdepth regions.bed.gz, all reads (with .csi)")
    parser.add_argument("--depth-mq20", required=True, help="mosdepth regions.bed.gz, MAPQ>=20 (with .csi)")
    parser.add_argument("--annotation", required=True, help="Reference annotation parquet")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument("--contigs", nargs="+", default=None, help="Contigs (default: the contigs of the annotation)")
    parser.add_argument("--threads", type=int, default=8, help="Contigs processed in parallel")
    return parser.parse_args(argv)


def run(argv: list[str]):
    args = parse_args(argv[1:])
    split_read_cnv_candidates(
        args.alignment,
        args.reference_fasta,
        args.depth_vcf,
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
