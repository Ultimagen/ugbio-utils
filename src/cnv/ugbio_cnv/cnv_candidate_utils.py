"""
Shared pieces of the depth CNV caller's candidate steps (split_read_candidates, pair_candidates, cnmops_candidates).

- CandidateParams: clustering, depth-check and overlap parameters.
- samtools_lines: SAM lines of a region, decoding only the needed CRAM fields.
- dedup_clusters: collapse same-type read clusters with >= 50% reciprocal overlap to the one with the most reads.
- score_with_depth: depth features of each candidate (mean inside vs median of the flanks) and the depth-change check.
- add_candidates_to_calls: add the candidates that are not already called to the depth calls.
"""

import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pysam
from ugbio_cnv.cnv_vcf_consts import CNV_INFO_TAGS, CNV_SOURCE_SPLIT_READ
from ugbio_cnv.depth_annotation import BIN_SIZE
from ugbio_cnv.depth_cnv_caller import SexChromosomes, genotype_for


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
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20) as proc:
        yield from proc.stdout
    if proc.returncode:
        raise RuntimeError(f"samtools view failed ({proc.returncode}): {' '.join(cmd)}")


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


def _depth_bins(tbx: pysam.TabixFile, chrom: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Start and mean depth of the depth bins overlapping [start, end)."""
    start, end = max(start, 0), max(end, 0)
    if end <= start:
        return np.array([], dtype=np.int64), np.array([])
    try:
        rows = [x.split("\t") for x in tbx.fetch(chrom, start, end)]
    except ValueError:  # contig absent from the depth file
        return np.array([], dtype=np.int64), np.array([])
    return np.array([int(r[1]) for r in rows], dtype=np.int64), np.array([float(r[3]) for r in rows])


def _depth_values(tbx: pysam.TabixFile, chrom: str, start: int, end: int) -> np.ndarray:
    return _depth_bins(tbx, chrom, start, end)[1]


def _depth_per_copy(
    tbx: pysam.TabixFile, sex: SexChromosomes, chrom: str, start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Depth per expected copy of the bins in [start, end), and their baseline (2, 1, or 2 in a male PAR)."""
    starts, depth = _depth_bins(tbx, chrom, start, end)
    base = sex.baseline(chrom, starts)
    return depth / base, base


def score_with_depth(
    cand: pd.DataFrame,
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    sex: SexChromosomes,
    p: CandidateParams,
    *,
    require_change: bool = True,
) -> pd.DataFrame:
    """
    Depth ratio, copy number and z per candidate; with require_change, those without the expected change drop.

    Each depth bin is divided by its baseline (sex chromosome setup, including the male PARs), so a candidate next to
    or across a baseline change (a PAR edge) is not read as a gain or a loss. A candidate with no bin in the annotation
    (outside the regions it was built for, e.g. depth_calling_regions) is dropped: the depth caller does not call there.
    """
    t0 = pysam.TabixFile(depth_mq0, index=depth_mq0 + ".csi")
    t20 = pysam.TabixFile(depth_mq20, index=depth_mq20 + ".csi")
    ann = pd.read_parquet(annotation)
    ann["chrom"] = ann["chrom"].astype(str)
    ann = ann.set_index(["chrom", "start"])
    out = []
    for r in cand.itertuples(index=False):
        a = ann.reindex([(r.chrom, b) for b in range(r.start // BIN_SIZE * BIN_SIZE, r.end, BIN_SIZE)])
        if a["segdup"].isna().all():
            continue
        inside, inside_base = _depth_per_copy(t0, sex, r.chrom, r.start, r.end)
        flanks = np.concatenate(
            [
                _depth_per_copy(t0, sex, r.chrom, r.start - p.flank, r.start - p.flank_gap)[0],
                _depth_per_copy(t0, sex, r.chrom, r.end + p.flank_gap, r.end + p.flank)[0],
            ]
        )
        if len(inside) == 0 or len(flanks) < p.min_flank_bins or np.median(flanks) <= 0:
            continue
        base = float(inside_base.mean())  # the contig's baseline, or a mix across a PAR edge
        ratio = inside.mean() / np.median(flanks)
        sd = np.std(flanks) / np.median(flanks)
        z = (ratio - 1) / (sd / np.sqrt(len(inside))) if sd > 0 else 0.0
        no_change = (r.svtype == "DEL" and ratio >= p.max_del_ratio) or (r.svtype == "DUP" and ratio <= p.min_dup_ratio)
        if require_change and no_change:
            continue
        uniq = _depth_values(t20, r.chrom, r.start, r.end)
        raw_inside = inside * inside_base
        out.append(
            {
                **r._asdict(),
                "cn_mean": base * ratio,
                "baseline": base,
                "z": z,
                "segdup": float(np.nanmean(a["segdup"])),
                "uniq_frac": float(uniq.mean() / raw_inside.mean()) if raw_inside.mean() > 0 and len(uniq) else np.nan,
            }
        )
    return pd.DataFrame(out)


def add_candidates_to_calls(
    depth_vcf: str,
    cand: pd.DataFrame,
    output_vcf: str,
    p: CandidateParams,
    *,
    source: str = CNV_SOURCE_SPLIT_READ,
    id_prefix: str = "SR",
    count_tag: str | None = "CLUSTER_SUPPORT",
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
            header.info.add(*CNV_INFO_TAGS[count_tag][:-1])
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
        rec.info["CN_MEAN"], rec.info["BASE_CN"] = round(float(r.cn_mean), 3), round(float(r.baseline))
        rec.info["Z"], rec.info["N_MERGED"] = round(float(r.z), 2), 1
        rec.info["SEGDUP_FRAC"] = round(float(r.segdup), 3)
        if not np.isnan(r.uniq_frac):
            rec.info["UNIQ_FRAC"] = round(float(r.uniq_frac), 3)
        if count_tag:
            rec.info[count_tag] = int(r.n_reads)
        rec.samples[sample]["GT"] = genotype_for(r.svtype, float(r.cn_mean))
        new.append(rec)
    order = {c: k for k, c in enumerate(header.contigs)}
    with pysam.VariantFile(output_vcf, "wz", header=header) as vout:
        for rec in sorted([*records, *new], key=lambda x: (order[x.chrom], x.start)):
            rec.translate(header)
            vout.write(rec)
    pysam.tabix_index(output_vcf, preset="vcf", force=True)
    return len(new)
