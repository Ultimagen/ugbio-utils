"""
Split-read CNV candidates for the depth CNV caller.

split_read_cnv_candidates:
  1. Per contig, in parallel: the supplementary records give each split read's other pieces (in place of the SA tag,
     so that no tags are decoded), then the primary records (MAPQ >= 20) of those reads are streamed; each read's two
     pieces on the same chromosome and strand give a deletion or tandem-duplication junction
     (breakpoint_snapping.split_read_junctions).
  2. Junctions are clustered (position +-100 bp, size +-10%); clusters with >= 2 reads and 500 bp - 1 Mb are kept,
     and clusters with >= 50% reciprocal overlap are collapsed to the one with most reads.
  3. Each cluster is scored with depth (all reads, 100 bp): mean inside vs median of the 5 kb flanks. Candidates
     without a depth change in the right direction (DEL ratio < 0.85, DUP > 1.15) are dropped.
  4. Candidates with >= 50% reciprocal overlap with a same-type depth call are dropped (already called); the rest are
     added to the depth calls with CNV_SOURCE=split_read.

cnmops_cnv_candidates (optional, with the cn.mops calls of the sample):
  cn.mops calls without a same-type call with >= 50% reciprocal overlap are added with CNV_SOURCE=cn.mops and the
  depth features of the other candidates, without the depth-change check (cn.mops finds losses in low-mappability
  sequence where the all-read depth shows none).

filter_nested_cnv_candidates (after breakpoint snapping):
  Drop split-read, pair and cn.mops candidates that are shorter than 500 bp, >= 50% inside a same-type depth call (a
  second call on an event that is already called), or with >= 50% reciprocal overlap with a same-type candidate kept
  before them (snapping often moves several candidates of one deletion to the same breakpoints). Split-read
  candidates are kept first, each by the most reads in their cluster (SR_READS), then pair candidates (PAIR_READS),
  then cn.mops candidates.
"""

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pysam
from ugbio_cnv.breakpoint_snapping import split_read_junctions
from ugbio_cnv.cnv_vcf_consts import (
    CNV_SOURCE_CNMOPS,
    CNV_SOURCE_DEPTH,
    CNV_SOURCE_PAIR,
    CNV_SOURCE_SPLIT_READ,
    DEPTH_CNV_INFO_TAGS,
)
from ugbio_cnv.depth_annotation import BIN_SIZE
from ugbio_cnv.depth_cnv_caller import genotype_for, read_sex_chromosomes
from ugbio_core.logger import logger


@dataclass
class CandidateParams:
    min_reads: int = 2
    pos_tol: int = 100
    size_tol: float = 0.10
    min_len: int = 500
    max_len: int = 1_000_000
    flank: int = 5000  # depth flanks on either side of the candidate
    flank_gap: int = 200  # bp skipped next to the breakpoints
    min_flank_bins: int = 10
    max_del_ratio: float = 0.85
    min_dup_ratio: float = 1.15
    min_overlap: float = 0.5  # reciprocal overlap for dedup and for "already called"
    min_mapq: int = 20
    min_sa_mapq: int = 1


# CRAM fields decoded (htslib required_fields): QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, and RNEXT, PNEXT, TLEN for
# pairs. Not decoding SEQ, QUAL and the tags is what makes the scans fast: on Ultima CRAMs the per-read tags are
# most of the data, and reading any tag (SA, or a samtools -e filter) decodes them all.
SPLIT_READ_FIELDS = "0x3F"
PAIR_FIELDS = "0x1FE"


def samtools_lines(alignment: str, reference: str, region: str, args: list[str], fields: str):
    """SAM lines of `samtools view <args> <alignment> <region>`, decoding only `fields` (CRAM)."""
    cmd = ["samtools", "view", "-@", "1", "-T", reference, "--input-fmt-option", f"required_fields={fields}"]
    cmd += [*args, alignment, region]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20) as proc:  # noqa: S603
        yield from proc.stdout
    if proc.returncode:
        raise RuntimeError(f"samtools view failed ({proc.returncode}): {' '.join(cmd)}")


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


def dedup_clusters(c: pd.DataFrame, p: CandidateParams) -> pd.DataFrame:
    """Collapse same-type clusters with >= min_overlap reciprocal overlap, keeping the one with the most reads."""
    keep = []
    ranked = c.sort_values(["n_reads", "start", "end"], ascending=[False, True, True], kind="stable")
    for _, g in ranked.groupby(["chrom", "svtype"], sort=False):
        kept = []
        for r in g.itertuples(index=False):
            if not any(
                min(r.end, k.end) - max(r.start, k.start) >= p.min_overlap * max(r.end - r.start, k.end - k.start)
                for k in kept
            ):
                kept.append(r)
        keep += kept
    return pd.DataFrame(keep, columns=c.columns).reset_index(drop=True)


def _depth_values(tbx: pysam.TabixFile, chrom: str, start: int, end: int) -> np.ndarray:
    start, end = max(start, 0), max(end, 0)
    if end <= start:
        return np.array([])
    try:
        return np.array([float(x.split("\t")[3]) for x in tbx.fetch(chrom, start, end)])
    except ValueError:  # contig absent from the depth file
        return np.array([])


def score_with_depth(  # noqa: PLR0913
    cand: pd.DataFrame,
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    haploid: set[str],
    p: CandidateParams,
    *,
    require_change: bool = True,
) -> pd.DataFrame:
    """Depth ratio, copy number and z per candidate; with require_change, those without the expected change drop."""
    t0 = pysam.TabixFile(depth_mq0, index=depth_mq0 + ".csi")
    t20 = pysam.TabixFile(depth_mq20, index=depth_mq20 + ".csi")
    ann = pd.read_parquet(annotation)
    ann["chrom"] = ann["chrom"].astype(str)
    ann = ann.set_index(["chrom", "start"])
    out = []
    for r in cand.itertuples(index=False):
        inside = _depth_values(t0, r.chrom, r.start, r.end)
        flanks = np.concatenate(
            [
                _depth_values(t0, r.chrom, r.start - p.flank, r.start - p.flank_gap),
                _depth_values(t0, r.chrom, r.end + p.flank_gap, r.end + p.flank),
            ]
        )
        if len(inside) == 0 or len(flanks) < p.min_flank_bins or np.median(flanks) <= 0:
            continue
        base = 1.0 if r.chrom in haploid else 2.0
        ratio = inside.mean() / np.median(flanks)
        sd = np.std(flanks) / np.median(flanks)
        z = (ratio - 1) / (sd / np.sqrt(len(inside))) if sd > 0 else 0.0
        no_change = (r.svtype == "DEL" and ratio >= p.max_del_ratio) or (r.svtype == "DUP" and ratio <= p.min_dup_ratio)
        if require_change and no_change:
            continue
        uniq = _depth_values(t20, r.chrom, r.start, r.end)
        a = ann.reindex([(r.chrom, b) for b in range(r.start // BIN_SIZE * BIN_SIZE, r.end, BIN_SIZE)])
        out.append(
            {
                **r._asdict(),
                "cn_mean": base * ratio,
                "baseline": base,
                "z": z,
                "segdup": float(np.nanmean(a["segdup"])),
                "uniq_frac": float(uniq.mean() / inside.mean()) if inside.mean() > 0 and len(uniq) else np.nan,
            }
        )
    return pd.DataFrame(out)


def add_candidates_to_calls(  # noqa: PLR0913
    depth_vcf: str,
    cand: pd.DataFrame,
    output_vcf: str,
    p: CandidateParams,
    *,
    source: str = CNV_SOURCE_SPLIT_READ,
    id_prefix: str = "SR",
    count_tag: str | None = "SR_READS",
    ci: tuple[int, int] = (-10, 10),
) -> int:
    """
    Write the calls of depth_vcf plus the candidates not already called (>= min_overlap reciprocal); returns # added.

    The candidates get CNV_SOURCE=source, IDs <id_prefix>_<k>, their read count in count_tag (if any) and
    CIPOS / CIEND = ci.
    """
    with pysam.VariantFile(depth_vcf) as vin:
        header = vin.header.copy()
        if count_tag and count_tag not in header.info:
            header.info.add(*DEPTH_CNV_INFO_TAGS[count_tag][:-1])
        records = list(vin)
    sample = list(header.samples)[0]
    dep = pd.DataFrame(
        [(r.chrom, r.start, r.stop, r.info["SVTYPE"]) for r in records], columns=["chrom", "start", "end", "svtype"]
    )
    new = []
    for r in cand.itertuples(index=False):
        d = dep[(dep.chrom == r.chrom) & (dep.svtype == r.svtype) & (dep.start < r.end) & (dep.end > r.start)]
        ov = (np.minimum(d.end, r.end) - np.maximum(d.start, r.start)) / np.maximum(d.end - d.start, r.end - r.start)
        if (ov >= p.min_overlap).any():
            continue
        rec = header.new_record(
            contig=r.chrom, start=r.start, stop=r.end, alleles=("N", f"<{r.svtype}>"), id=f"{id_prefix}_{len(new)}"
        )
        rec.filter.add("PASS")
        rec.info["SVTYPE"], rec.info["SVLEN"] = r.svtype, r.end - r.start
        rec.info["CIPOS"], rec.info["CIEND"] = ci, ci
        rec.info["CNV_SOURCE"] = (source,)
        rec.info["CN_MEAN"], rec.info["BASE_CN"] = round(float(r.cn_mean), 3), int(r.baseline)
        rec.info["Z"], rec.info["N_MERGED"] = round(float(r.z), 2), 1
        rec.info["SEGDUP"] = round(float(r.segdup), 3)
        if not np.isnan(r.uniq_frac):
            rec.info["UNIQ_FRAC"] = round(float(r.uniq_frac), 3)
        if count_tag:
            rec.info[count_tag] = int(r.n_reads)
        rec.samples[sample]["GT"] = genotype_for(r.svtype)
        new.append(rec)
    order = {c: k for k, c in enumerate(header.contigs)}
    with pysam.VariantFile(output_vcf, "wz", header=header) as vout:
        for rec in sorted([*records, *new], key=lambda x: (order[x.chrom], x.start)):
            rec.translate(header)
            vout.write(rec)
    pysam.tabix_index(output_vcf, preset="vcf", force=True)
    return len(new)


def split_read_cnv_candidates(  # noqa: PLR0913
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

    The sex chromosome setup (haploid / not called contigs) is the one call_depth_cnvs recorded in the depth VCF.
    """
    p = params or CandidateParams()
    sex = read_sex_chromosomes(depth_vcf)
    haploid, skipped = sex.haploid, sex.skipped
    if not contigs:  # the contigs the depth caller annotates, in the reference's naming (chr1.. or 1..)
        contigs = pd.read_parquet(annotation, columns=["chrom"])["chrom"].astype(str).unique().tolist()
    contigs = [c for c in contigs if c not in skipped]
    j = stream_junctions(alignment, reference, contigs, p, threads)
    c = cluster_junctions(j, p)
    c = dedup_clusters(c, p) if len(c) else c
    s = score_with_depth(c, depth_mq0, depth_mq20, annotation, haploid, p) if len(c) else pd.DataFrame()
    n = (
        add_candidates_to_calls(depth_vcf, s, output_vcf, p)
        if len(s)
        else add_candidates_to_calls(
            depth_vcf, pd.DataFrame(columns=["chrom", "start", "end", "svtype"]), output_vcf, p
        )
    )
    logger.info(f"{output_vcf}: junctions {len(j)}, clusters {len(c)}, with depth support {len(s)}, added {n}")
    return n


# candidates kept first in the duplicate rule of filter_nested_cnv_candidates
CANDIDATE_KEEP_ORDER = (CNV_SOURCE_SPLIT_READ, CNV_SOURCE_PAIR, CNV_SOURCE_CNMOPS)


def _candidate_rank(r: pysam.VariantRecord) -> tuple:
    """Order in which candidates are kept: split-read, pair, then cn.mops; then the most reads, then position."""
    source = r.info.get("CNV_SOURCE", ())
    rank = next((k for k, s in enumerate(CANDIDATE_KEEP_ORDER) if s in source), 0)
    tag = {0: "SR_READS", 1: "PAIR_READS"}.get(rank)  # cn.mops candidates have no read count
    reads = (r.info.get(tag) if tag and tag in r.header.info else None) or 0  # the tag is not always defined
    return (rank, -reads, r.start, r.stop)


def cnmops_cnv_candidates(  # noqa: PLR0913
    input_vcf: str,
    cnmops_vcf: str,
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    output_vcf: str,
    params: CandidateParams | None = None,
) -> int:
    """
    Add the cn.mops calls of cnmops_vcf that input_vcf does not have (>= min_overlap reciprocal, same type) as
    CNV_SOURCE=cn.mops candidates; returns the number added. Every DEL / DUP >= min_len is taken whatever its FILTER;
    calls on sex chromosomes the depth caller does not call, and calls without depth flanks, are skipped.
    """
    p = params or CandidateParams()
    sex = read_sex_chromosomes(input_vcf)
    with pysam.VariantFile(cnmops_vcf) as v:
        cand = pd.DataFrame(
            [
                (r.chrom, r.start, r.stop, r.info["SVTYPE"], 0)
                for r in v
                if r.info.get("SVTYPE") in ("DEL", "DUP")
                and r.chrom not in sex.skipped
                and r.stop - r.start >= p.min_len
            ],
            columns=["chrom", "start", "end", "svtype", "n_reads"],
        )
    scored = (
        score_with_depth(cand, depth_mq0, depth_mq20, annotation, sex.haploid, p, require_change=False)
        if len(cand)
        else pd.DataFrame(columns=["chrom", "start", "end", "svtype"])
    )
    n = add_candidates_to_calls(
        input_vcf, scored, output_vcf, p, source=CNV_SOURCE_CNMOPS, id_prefix="CNMOPS", count_tag=None, ci=(-500, 500)
    )
    logger.info(f"{output_vcf}: cn.mops calls {len(cand)}, added {n}")
    return n


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
        elif key in depth and (
            (np.minimum(depth[key][:, 1], r.stop) - np.maximum(depth[key][:, 0], r.start)).clip(min=0).max()
            >= min_inside * length
        ):
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


def parse_args_nested(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="filter_nested_cnv_candidates",
        description="Drop split-read, pair and cn.mops candidates that are < 500 bp, nested or duplicates",
    )
    parser.add_argument("--input-vcf", required=True, help="Snapped calls")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument("--min-len", type=int, default=500, help="Minimal candidate length after snapping")
    return parser.parse_args(argv)


def parse_args_cnmops(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cnmops_cnv_candidates", description="Add the cn.mops calls the depth caller does not have as candidates"
    )
    parser.add_argument("--input-vcf", required=True, help="Calls so far (split_read / pair_cnv_candidates output)")
    parser.add_argument("--cnmops-vcf", required=True, help="cn.mops calls of the sample (process_cnvs output)")
    parser.add_argument("--depth-mq0", required=True, help="mosdepth regions.bed.gz, all reads (with .csi)")
    parser.add_argument("--depth-mq20", required=True, help="mosdepth regions.bed.gz, MAPQ>=20 (with .csi)")
    parser.add_argument("--annotation", required=True, help="Reference annotation parquet")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    return parser.parse_args(argv)


def run_cnmops(argv: list[str]):
    a = parse_args_cnmops(argv[1:])
    cnmops_cnv_candidates(a.input_vcf, a.cnmops_vcf, a.depth_mq0, a.depth_mq20, a.annotation, a.output_vcf)


def run_nested(argv: list[str]):
    args = parse_args_nested(argv[1:])
    filter_nested_cnv_candidates(args.input_vcf, args.output_vcf, args.min_len)


def main():
    run(sys.argv)


def main_nested():
    run_nested(sys.argv)


def main_cnmops():
    run_cnmops(sys.argv)


if __name__ == "__main__":
    main()
