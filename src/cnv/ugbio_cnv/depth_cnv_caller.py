"""
Cohort-free germline CNV calling from read depth.

Per sample:
  100 bp depth (mosdepth, all reads and MAPQ>=20) + static reference annotation (depth_annotation.py)
  -> static mask (N, extreme GC; never the sample's own coverage)
  -> GC correction fitted on the sample's clean autosomal bins
  -> copy-number estimate per bin (2 x ratio to the autosomal median)
  -> sex chromosome copy numbers from the clean chrX / chrY bins (or the given sex); chrX / chrY are not called
     if the depth is ambiguous
  -> units of `unit_bins` callable bins (masked bins are skipped, so units span them)
  -> weighted PELT per chromosome; segdup / low-mappability units weighted by their own (larger) noise
  -> segments called by direction (loss / gain) vs the contig baseline (2, or 1 for haploid contigs)
  -> same-direction calls merged across noise and masked gaps (one call per event)
  -> call edges refined on the 100 bp bins
"""

import argparse
import re
import sys
from dataclasses import dataclass, field

import numba
import numpy as np
import pandas as pd
import pysam
from ugbio_cnv.cnv_bed_format_utils import add_vcf_header
from ugbio_cnv.cnv_vcf_consts import (
    CNV_SOURCE_DEPTH,
    DEPTH_CNV_INFO_TAGS,
)
from ugbio_cnv.depth_annotation import BIN_SIZE
from ugbio_cnv.estimate_ploidy import _determine_karyotype
from ugbio_core.logger import logger

AUTOSOME_PATTERN = re.compile(r"(chr)?\d+")


@dataclass
class Params:
    unit_bins: int = 5  # callable 100 bp bins per unit (~500 bp of callable sequence)
    gc_min: float = 0.25
    gc_max: float = 0.70
    penalty_scale: float = 3.0  # beta = penalty_scale * log(n), data in units of their own noise
    cn_clip: float = 6.0
    min_units: int = 2  # minimal segment length in units
    min_abs_dcn: float = 0.4  # |mean CN - baseline| to call a segment
    min_z: float = 5.0  # z of the segment mean vs baseline
    difficult_frac: float = 0.5  # a unit / segment is difficult if this fraction is segdup or low-mappability
    min_abs_dcn_difficult: float = 1.0  # |dCN| for a confident call in segdup / low-mappability sequence
    merge_gap_frac: float = 0.25  # merge if gap <= frac x the smaller flanking call ...
    merge_gap_max: int = 2_000_000  # ... and gap <= this (bp)
    merge_masked_frac: float = 0.5  # or the gap is mostly uncallable (static mask)
    min_len: int = 1000
    refine_window: int = 1000  # edge refinement search window (bp) on 100 bp bins
    min_len_refined: int = 500
    merge_block_normal_bp: int = 50_000  # a gap with this much confidently normal unique sequence is a real boundary
    blip_max_bp: int = 5000  # opposite calls inside a merge are absorbed if low-confidence or shorter; else kept nested
    chunk_units: int = 10_000  # PELT is quadratic without changepoints; chunk boundaries are healed by the merge
    min_sex_bins: int = 1000  # clean chrX / chrY bins needed to infer the sex chromosome copy numbers


# VCF header keys carrying the sex chromosome setup from call_depth_cnvs to split_read_cnv_candidates
HAPLOID_HEADER_KEY = "depth_cnv_haploid_contigs"
SKIPPED_HEADER_KEY = "depth_cnv_skipped_contigs"
KARYOTYPE_HEADER_KEY = "depth_cnv_karyotype"
# depth ratios to the autosomes too close to a one- vs two-copy (chrX) or zero- vs one-copy (chrY) boundary to decide
X_RATIO_AMBIGUOUS = (0.65, 0.85)
Y_RATIO_AMBIGUOUS = (0.15, 0.35)


@dataclass
class SexChromosomes:
    haploid: set[str] = field(default_factory=set)  # contigs with baseline copy number 1
    skipped: set[str] = field(default_factory=set)  # contigs not called (no copy expected)
    karyotype: str = "."  # e.g. XY; how the setup was decided


# ---------------------------------------------------------------- inputs
def read_mosdepth_regions(path: str) -> pd.DataFrame:
    """Read a mosdepth `.regions.bed.gz` (chrom, start, end, mean depth)."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "depth"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64, "depth": np.float32},
    )


def load_bins(depth_mq0: str, depth_mq20: str, annotation: str, contigs: list[str] | None = None) -> pd.DataFrame:
    """Join all-read and MAPQ>=20 depth with the static annotation (100 bp bins)."""
    ann = pd.read_parquet(annotation)
    ann["chrom"] = ann["chrom"].astype(str)
    if contigs:
        ann = ann[ann["chrom"].isin(contigs)]
    d0 = read_mosdepth_regions(depth_mq0)
    d20 = read_mosdepth_regions(depth_mq20).rename(columns={"depth": "depth_uniq"})[["chrom", "start", "depth_uniq"]]
    bins = ann.merge(d0, on=["chrom", "start"], how="inner").merge(d20, on=["chrom", "start"], how="left")
    return bins.sort_values(["chrom", "start"], kind="stable").reset_index(drop=True)


SEXES = ("male", "female")


def sex_chromosome_setup(sex: str, chrx_name: str = "chrX", chry_name: str = "chrY") -> SexChromosomes:
    """
    Sex chromosome setup from a given sex.

    Male: chrX and chrY are haploid. Female: chrX is diploid and chrY is not called.
    """
    if sex not in SEXES:
        raise ValueError(f"sex must be one of {SEXES}, got {sex!r}")
    if sex == "male":
        return SexChromosomes({chrx_name, chry_name}, set(), "XY (given)")
    return SexChromosomes(set(), {chry_name}, "XX (given)")


def sex_chromosome_ratios(
    bins: pd.DataFrame, chrx_name: str, chry_name: str, p: Params
) -> tuple[float | None, float | None]:
    """
    Median depth ratio to the autosomes (cn / 2) of the clean (callable, unique, non-segdup) chrX and chrY bins.

    None for a chromosome with fewer than p.min_sex_bins clean bins (e.g. not among the called contigs).
    """
    clean = bins["callable"] & (bins["lowmap"] == 0) & (bins["segdup"] == 0)

    def ratio(chrom: str) -> float | None:
        cn = bins.loc[clean & (bins["chrom"] == chrom), "cn"]
        return float(cn.median()) / 2 if len(cn) >= p.min_sex_bins else None

    return ratio(chrx_name), ratio(chry_name)


def infer_sex_chromosomes(
    x_ratio: float | None, y_ratio: float | None, chrx_name: str = "chrX", chry_name: str = "chrY"
) -> SexChromosomes:
    """
    Sex chromosome setup from the chrX / chrY depth ratios, with the karyotype table of estimate_ploidy.

    One chrX copy makes chrX haploid, two or more diploid; chrY is haploid if present and not called otherwise.
    Without chrX data nothing is decided (the sex chromosomes are not being called). If the karyotype cannot be
    determined, chrX and chrY are not called.
    """
    if x_ratio is None:
        if y_ratio is not None:
            logger.warning(f"{chrx_name} has too few clean bins to infer the karyotype: {chry_name} is not called")
            return SexChromosomes(skipped={chrx_name, chry_name}, karyotype=f"UNDETERMINED (no {chrx_name})")
        return SexChromosomes(karyotype=f"not inferred (no {chrx_name})")
    ambiguous = X_RATIO_AMBIGUOUS[0] < x_ratio < X_RATIO_AMBIGUOUS[1] or (
        y_ratio is not None and Y_RATIO_AMBIGUOUS[0] < y_ratio < Y_RATIO_AMBIGUOUS[1]
    )
    if ambiguous:
        karyotype = "UNDETERMINED"
    elif y_ratio is None:  # chrY not called: chrX alone decides
        karyotype = "X" if 0.25 <= x_ratio <= 0.75 else "XX" if 0.75 < x_ratio <= 1.25 else "UNDETERMINED"  # noqa: PLR2004
    else:
        karyotype = _determine_karyotype(x_ratio, y_ratio)
    found = f"{chrx_name} {x_ratio:.2f}" + (f", {chry_name} {y_ratio:.2f}" if y_ratio is not None else "")
    if karyotype == "UNDETERMINED":
        logger.warning(
            f"Cannot infer the sex chromosome copy numbers from depth (ratio to the autosomes: {found}): "
            f"{chrx_name} and {chry_name} are not called (give the sex to call them)"
        )
        return SexChromosomes(skipped={chrx_name, chry_name}, karyotype=f"UNDETERMINED (inferred: {found})")
    n_x, has_y = karyotype.count("X"), "Y" in karyotype
    if n_x > 2 or karyotype.count("Y") > 1:  # noqa: PLR2004
        logger.warning(f"Karyotype {karyotype}: extra sex chromosome copies are called against 2 (chrX) / 1 (chrY)")
    sex = SexChromosomes(karyotype=f"{karyotype} (inferred: {found})")
    if n_x == 1:
        sex.haploid.add(chrx_name)
    if has_y:
        sex.haploid.add(chry_name)
    elif y_ratio is not None:
        sex.skipped.add(chry_name)
    return sex


def read_sex_chromosomes(vcf: str) -> SexChromosomes:
    """Sex chromosome setup recorded in a call_depth_cnvs VCF header."""
    with pysam.VariantFile(vcf) as v:
        meta = {r.key: r.value for r in v.header.records if r.type == "GENERIC"}
    if HAPLOID_HEADER_KEY not in meta:
        raise ValueError(f"{vcf}: no {HAPLOID_HEADER_KEY} header line (not a call_depth_cnvs VCF)")

    def contigs(key: str) -> set[str]:
        value = meta.get(key, ".")
        return set() if value == "." else set(value.split(","))

    return SexChromosomes(contigs(HAPLOID_HEADER_KEY), contigs(SKIPPED_HEADER_KEY), meta.get(KARYOTYPE_HEADER_KEY, "."))


# ---------------------------------------------------------------- normalization
def normalize(bins: pd.DataFrame, haploid: set[str], p: Params) -> pd.DataFrame:
    """GC-corrected copy-number estimate per bin; adds callable, cn, baseline and uniq_frac columns.

    cn is relative to the autosomes (2 = autosomal median); baseline is 1 on the haploid contigs, else 2.
    """
    bins = bins.copy()
    bins["callable"] = (bins["n_frac"] == 0) & bins["gc"].between(p.gc_min, p.gc_max)
    autosome = bins["chrom"].map(lambda c: AUTOSOME_PATTERN.fullmatch(c) is not None)
    clean = bins["callable"] & autosome & (bins["lowmap"] == 0) & (bins["segdup"] == 0)
    # GC model: median depth per 1% GC on clean autosomal bins, smoothed
    gc_bin = (bins["gc"] * 100).round()
    model = bins.loc[clean].groupby(gc_bin[clean])["depth"].median()
    model = model.rolling(5, center=True, min_periods=1).median()
    bins["ratio"] = bins["depth"] / gc_bin.map(model)
    median_ratio = bins.loc[clean, "ratio"].median()
    if not np.isfinite(median_ratio) or median_ratio <= 0:
        raise ValueError(
            "No usable coverage: the median depth of clean autosomal bins is 0 or undefined "
            "(empty or very low-coverage input, or depth and annotation contigs do not match)"
        )
    bins["cn"] = np.clip(2 * bins["ratio"] / median_ratio, 0, p.cn_clip)
    bins["baseline"] = np.where(bins["chrom"].isin(haploid), 1.0, 2.0)
    bins["uniq_frac"] = np.where(bins["depth"] > 0, bins["depth_uniq"] / bins["depth"], np.nan)
    return bins


def make_units(bins: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Groups of unit_bins consecutive callable bins per chromosome (masked bins are skipped)."""
    cb = bins[bins["callable"] & bins["cn"].notna()].copy()
    cb["unit"] = cb.groupby("chrom", sort=False).cumcount() // p.unit_bins
    units = (
        cb.groupby(["chrom", "unit"], sort=False)
        .agg(
            start=("start", "first"),
            end=("end", "last"),
            cn=("cn", "mean"),
            n=("cn", "size"),
            baseline=("baseline", "first"),
            lowmap=("lowmap", "mean"),
            segdup=("segdup", "mean"),
            uniq_frac=("uniq_frac", "mean"),
        )
        .reset_index()
    )
    return units[units["n"] == p.unit_bins].reset_index(drop=True)


# ---------------------------------------------------------------- segmentation
@numba.njit(cache=True)
def pelt_weighted_l2(x: np.ndarray, w: np.ndarray, beta: float, min_size: int) -> np.ndarray:
    """
    Exact PELT with the weighted L2 cost sum w (x - mean_w)^2.

    Parameters
    ----------
    x : np.ndarray
        Values (float64).
    w : np.ndarray
        Weights (float64, > 0).
    beta : float
        Penalty per changepoint.
    min_size : int
        Minimal segment length.

    Returns
    -------
    np.ndarray
        Segment end indices (exclusive), the last one is len(x).
    """
    n = len(x)
    sw = np.zeros(n + 1)
    swx = np.zeros(n + 1)
    swx2 = np.zeros(n + 1)
    for i in range(n):
        sw[i + 1] = sw[i] + w[i]
        swx[i + 1] = swx[i] + w[i] * x[i]
        swx2[i + 1] = swx2[i] + w[i] * x[i] * x[i]
    f = np.full(n + 1, np.inf)
    f[0] = -beta
    last = np.zeros(n + 1, dtype=np.int64)
    cand = np.zeros(n + 1, dtype=np.int64)
    n_cand = 1
    vals = np.zeros(n + 1)
    for t in range(min_size, n + 1):
        best = np.inf
        best_s = 0
        for k in range(n_cand):
            s = cand[k]
            if t - s < min_size:
                vals[k] = np.inf
                continue
            sum_w = sw[t] - sw[s]
            cost = swx2[t] - swx2[s] - (swx[t] - swx[s]) ** 2 / sum_w
            vals[k] = f[s] + cost
            if vals[k] + beta < best:
                best = vals[k] + beta
                best_s = s
        f[t] = best
        last[t] = best_s
        new_n = 0
        for k in range(n_cand):  # pruning: keep candidates that can still be optimal
            if vals[k] <= f[t] or t - cand[k] < min_size:
                cand[new_n] = cand[k]
                new_n += 1
        cand[new_n] = t
        n_cand = new_n + 1
    ends = []
    t = n
    while t > 0:
        ends.append(t)
        t = last[t]
    return np.array(ends[::-1], dtype=np.int64)


def unit_noise(units: pd.DataFrame) -> pd.Series:
    """Robust per-class noise of one unit (MAD of successive differences within each class and chromosome)."""
    sigma = {}
    for cls, u in units.groupby("difficult"):
        d = np.concatenate([np.diff(g["cn"].to_numpy()) for _, g in u.groupby("chrom", sort=False)])
        sigma[cls] = np.median(np.abs(d)) / 0.6745 / np.sqrt(2)
    return pd.Series(sigma)


def segment(units: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Segment each chromosome and call each segment as DEL, DUP or uncalled ('')."""
    units = units.copy()
    units["difficult"] = (units["segdup"] >= p.difficult_frac) | (units["lowmap"] >= p.difficult_frac)
    sigma = unit_noise(units)
    units["w"] = 1.0 / units["difficult"].map(sigma) ** 2
    rows = []
    for chrom, u in units.groupby("chrom", sort=False):
        x = u["cn"].to_numpy(dtype=np.float64)
        w = u["w"].to_numpy(dtype=np.float64)
        if len(x) < 2 * p.min_units:
            continue
        beta = p.penalty_scale * np.log(len(x))
        ends = []
        for c0 in range(0, len(x), p.chunk_units):
            c1 = min(c0 + p.chunk_units, len(x))
            if c1 - c0 < 2 * p.min_units:
                ends[-1] = c1
                continue
            ends.extend(c0 + pelt_weighted_l2(x[c0:c1], w[c0:c1], beta, p.min_units))
        s = 0
        for e in ends:
            seg = u.iloc[s:e]
            ws = w[s:e]
            mean = float((ws * x[s:e]).sum() / ws.sum())
            base = float(seg["baseline"].iloc[0])
            rows.append(
                {
                    "chrom": chrom,
                    "start": int(seg["start"].iloc[0]),
                    "end": int(seg["end"].iloc[-1]),
                    "n_units": e - s,
                    "cn_mean": mean,
                    "baseline": base,
                    "z": (mean - base) * np.sqrt(ws.sum()),
                    "difficult_frac": float(seg["difficult"].mean()),
                    "lowmap": float(seg["lowmap"].mean()),
                    "segdup": float(seg["segdup"].mean()),
                    "uniq_frac": float(seg["uniq_frac"].mean()),
                }
            )
            s = e
    segs = pd.DataFrame(rows)
    dcn = segs["cn_mean"] - segs["baseline"]
    called = (dcn.abs() >= p.min_abs_dcn) & (segs["z"].abs() >= p.min_z)
    segs["direction"] = np.where(called, np.where(dcn < 0, "DEL", "DUP"), "")
    # difficult sequence needs a full copy change to be confident
    segs["confident"] = called & ((segs["difficult_frac"] < p.difficult_frac) | (dcn.abs() >= p.min_abs_dcn_difficult))
    segs.attrs["sigma"] = sigma.to_dict()
    return segs


# ---------------------------------------------------------------- one call per event
class CallableIndex:
    """Fraction of callable bins in any interval, from a per-chromosome cumulative sum."""

    def __init__(self, bins: pd.DataFrame):
        self.idx = {}
        for chrom, b in bins.groupby("chrom", sort=False):
            self.idx[chrom] = (b["start"].to_numpy(), np.concatenate([[0], np.cumsum(b["callable"].to_numpy())]))

    def fraction(self, chrom: str, start: int, end: int) -> float:
        starts, cum = self.idx[chrom]
        i, j = np.searchsorted(starts, start), np.searchsorted(starts, end)
        return float((cum[j] - cum[i]) / (j - i)) if j > i else 0.0


class NormalIndex:
    """bp of uncalled (copy-neutral) segments in unique sequence inside any interval."""

    def __init__(self, segs: pd.DataFrame, difficult_frac: float):
        self.idx = {}
        normal = segs[(segs["direction"] == "") & (segs["difficult_frac"] < difficult_frac)]
        for chrom, g in normal.groupby("chrom", sort=False):
            ordered = g.sort_values("start")
            self.idx[chrom] = (ordered["start"].to_numpy(), ordered["end"].to_numpy())

    def normal_bp(self, chrom: str, start: int, end: int) -> int:
        if chrom not in self.idx:
            return 0
        s, e = self.idx[chrom]
        return int(np.clip(np.minimum(e, end) - np.maximum(s, start), 0, None).sum())


def merge_calls(segs: pd.DataFrame, bins: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Repeat the merge pass over its own output until the number of calls stops changing."""
    cidx, nidx = CallableIndex(bins), NormalIndex(segs, p.difficult_frac)
    calls = _merge_pass(segs, p, cidx, nidx)
    while True:
        again = _merge_pass(
            calls.assign(confident=calls["conf_units"] >= 0.5 * calls["n_units"]), p, cidx, nidx, keep=calls
        )
        if len(again) == len(calls):
            break
        calls = again
    calls["low_conf"] = calls["conf_units"] < 0.5 * calls["n_units"]
    return calls[(calls["end"] - calls["start"]) >= p.min_len].reset_index(drop=True)


def _merge_pass(  # noqa: C901
    segs: pd.DataFrame,
    p: Params,
    cidx: CallableIndex,
    nidx: NormalIndex,
    keep: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge same-direction calls on a chromosome when the stretch between them is short relative to the calls
    (or mostly uncallable), unless it holds merge_block_normal_bp of confidently normal unique sequence.
    Opposite-direction calls inside a merge are absorbed when weak or short, and kept as nested calls otherwise.
    """
    out = []
    for chrom, s in segs.groupby("chrom", sort=False):
        merged = []
        for seg in s[s["direction"] != ""].sort_values("start").to_dict("records"):
            c = seg
            if keep is None:
                c = dict(
                    c, n_merged=1, cn_sum=c["cn_mean"] * c["n_units"], conf_units=c["n_units"] if c["confident"] else 0
                )
            # earliest same-direction call within reach for which the gap rule holds
            target = None
            for j in range(len(merged) - 1, -1, -1):
                prev = merged[j]
                if prev["direction"] != c["direction"]:
                    continue
                gap = c["start"] - prev["end"]
                if gap > p.merge_gap_max:
                    break
                flank = min(prev["end"] - prev["start"], c["end"] - c["start"])
                small = gap <= p.merge_gap_frac * flank
                uncallable = gap > 0 and (1 - cidx.fraction(chrom, prev["end"], c["start"])) >= p.merge_masked_frac
                back_to_normal = nidx.normal_bp(chrom, prev["end"], c["start"]) >= p.merge_block_normal_bp
                if (small or uncallable) and not back_to_normal:
                    target = j
            if target is not None:
                prev = merged[target]
                kept = []
                for b in [*merged[target + 1 :], c]:
                    if b["direction"] == c["direction"]:  # same direction: fold into prev
                        prev["end"] = max(prev["end"], b["end"])
                        prev["n_units"] += b["n_units"]
                        prev["cn_sum"] += b["cn_sum"]
                        prev["conf_units"] += b["conf_units"]
                        prev["n_merged"] += b["n_merged"]
                        prev["z"] = max(prev["z"], b["z"], key=abs)
                    elif b["conf_units"] >= 0.5 * b["n_units"] and (b["end"] - b["start"]) >= p.blip_max_bp:
                        kept.append(b)  # strong opposite call: kept as a nested call
                merged[target + 1 :] = kept
                continue
            merged.append(c)
        out.extend(merged)
    calls = pd.DataFrame(out)
    if len(calls):
        calls["cn_mean"] = calls["cn_sum"] / calls["n_units"]
    return calls.reset_index(drop=True)


# ---------------------------------------------------------------- edge refinement (100 bp depth)
def refine_edges(calls: pd.DataFrame, bins: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    Move each call edge within +-refine_window to the 100 bp position that best fits a step between the
    baseline and the call's copy number (squared error on callable bins). Calls may shrink or grow.
    """
    calls = calls.copy()
    w = p.refine_window // BIN_SIZE
    for chrom, idx in calls.groupby("chrom").groups.items():
        b = bins[bins["chrom"] == chrom]
        cn = np.full(int(b["start"].max()) // BIN_SIZE + 1, np.nan)
        ok = b["callable"].to_numpy()
        cn[(b["start"].to_numpy() // BIN_SIZE)[ok]] = b["cn"].to_numpy()[ok]
        for i in idx:
            c = calls.loc[i]
            base, level = c["baseline"], c["cn_mean"]
            for side in ("start", "end"):
                e = int(c[side]) // BIN_SIZE
                lo, hi = max(e - w, 0), min(e + w, len(cn) - 1)
                best, best_k = np.inf, e
                for k in range(lo, hi + 1):  # step at bin k: [lo, k) one level, [k, hi) the other
                    left, right = cn[lo:k], cn[k:hi]
                    if side == "start":
                        err = np.nansum((left - base) ** 2) + np.nansum((right - level) ** 2)
                    else:
                        err = np.nansum((left - level) ** 2) + np.nansum((right - base) ** 2)
                    if err < best:
                        best, best_k = err, k
                calls.loc[i, side] = best_k * BIN_SIZE
    calls = calls[calls["end"] - calls["start"] >= BIN_SIZE]
    return calls.reset_index(drop=True)


# ---------------------------------------------------------------- output
def depth_cnv_header(sample_name: str, fasta_index: str, sex: SexChromosomes | None = None) -> pysam.VariantHeader:
    """CNV VCF header (contigs, ALTs, registry tags, GT), the depth caller's INFO tags and the sex chromosome setup."""
    header = add_vcf_header(sample_name, fasta_index)
    for tag in DEPTH_CNV_INFO_TAGS.values():
        if tag[0] not in header.info:
            header.info.add(*tag[:-1])
    sex = sex or SexChromosomes()
    header.add_meta(HAPLOID_HEADER_KEY, ",".join(sorted(sex.haploid)) or ".")
    header.add_meta(SKIPPED_HEADER_KEY, ",".join(sorted(sex.skipped)) or ".")
    header.add_meta(KARYOTYPE_HEADER_KEY, sex.karyotype)
    return header


def genotype_for(svtype: str) -> tuple:
    """Genotype from the call type: DEL 0/1, DUP ./1 (copy number gains are not genotyped)."""
    return (0, 1) if svtype == "DEL" else (None, 1)


def write_depth_calls(
    calls: pd.DataFrame, output_vcf: str, sample_name: str, fasta_index: str, sex: SexChromosomes | None = None
) -> None:
    """Write depth calls as a bgzipped, indexed VCF in reference contig order."""
    header = depth_cnv_header(sample_name, fasta_index, sex)
    order = {c: k for k, c in enumerate(header.contigs)}
    calls = calls.assign(_order=calls["chrom"].map(order)).sort_values(["_order", "start"])
    cipos = (-BIN_SIZE * 5, BIN_SIZE * 5)  # one unit
    with pysam.VariantFile(output_vcf, "wz", header=header) as out:
        for k, c in enumerate(calls.itertuples(index=False)):
            rec = out.new_record(
                contig=c.chrom, start=int(c.start), stop=int(c.end), alleles=("N", f"<{c.direction}>"), id=f"DEPTH_{k}"
            )
            rec.filter.add("PASS")
            if c.low_conf:
                rec.info["LOW_CONF"] = True
            rec.info["SVTYPE"] = c.direction
            rec.info["SVLEN"] = int(c.end - c.start)
            rec.info["CIPOS"] = cipos
            rec.info["CIEND"] = cipos
            rec.info["CNV_SOURCE"] = (CNV_SOURCE_DEPTH,)
            rec.info["CN_MEAN"] = round(float(c.cn_mean), 3)
            rec.info["BASE_CN"] = int(c.baseline)
            rec.info["Z"] = round(float(c.z), 2)
            rec.info["N_MERGED"] = int(c.n_merged)
            rec.info["SEGDUP"] = round(float(c.segdup), 3)
            if not np.isnan(c.uniq_frac):
                rec.info["UNIQ_FRAC"] = round(float(c.uniq_frac), 3)
            rec.samples[sample_name]["GT"] = genotype_for(c.direction)
            out.write(rec)
    pysam.tabix_index(output_vcf, preset="vcf", force=True)


def call_depth_cnvs(  # noqa: PLR0913
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    sample_name: str,
    fasta_index: str,
    output_vcf: str,
    sex: str | None = None,
    contigs: list[str] | None = None,
    chrx_name: str = "chrX",
    chry_name: str = "chrY",
    params: Params | None = None,
) -> pd.DataFrame:
    """
    Call germline CNVs of one sample from 100 bp depth and write them as VCF.

    Parameters
    ----------
    depth_mq0, depth_mq20 : str
        mosdepth `.regions.bed.gz` with 100 bp windows, all reads (-Q 0) and MAPQ>=20 (-Q 20).
    annotation : str
        Reference annotation parquet from build_depth_annotation.
    sample_name : str
        Sample name for the VCF (the CRAM's sample).
    fasta_index : str
        Reference .fai (VCF contigs).
    output_vcf : str
        Output .vcf.gz.
    sex : str, optional
        "male" or "female". By default the sex chromosome copy numbers are inferred from the depth of chrX and chrY;
        if the depth is ambiguous, chrX and chrY are not called.
    contigs : list[str], optional
        Contigs to call (default: all annotated contigs).
    chrx_name, chry_name : str
        Sex chromosome names in the reference.
    params : Params, optional
        Caller parameters.

    Returns
    -------
    pd.DataFrame
        The calls.
    """
    p = params or Params()
    bins = normalize(load_bins(depth_mq0, depth_mq20, annotation, contigs), set(), p)
    x_ratio, y_ratio = sex_chromosome_ratios(bins, chrx_name, chry_name, p)
    inferred = infer_sex_chromosomes(x_ratio, y_ratio, chrx_name, chry_name)
    setup = inferred
    if sex:
        setup = sex_chromosome_setup(sex, chrx_name, chry_name)
        if x_ratio is not None and (inferred.haploid, inferred.skipped) != (setup.haploid, setup.skipped):
            logger.warning(f"given sex {sex} ({setup.karyotype}), but the depth gives {inferred.karyotype}")
    logger.info(
        f"sex chromosomes: {setup.karyotype}; haploid {sorted(setup.haploid)}, not called {sorted(setup.skipped)}"
    )
    bins = bins[~bins["chrom"].isin(setup.skipped)].reset_index(drop=True)
    bins["baseline"] = np.where(bins["chrom"].isin(setup.haploid), 1.0, 2.0)
    segs = segment(make_units(bins, p), p)
    calls = merge_calls(segs, bins, p)
    calls = refine_edges(calls, bins, p)
    calls = calls[(calls["end"] - calls["start"]) >= p.min_len_refined].reset_index(drop=True)
    write_depth_calls(calls, output_vcf, sample_name, fasta_index, setup)
    logger.info(f"{output_vcf}: {len(calls)} depth calls (noise per unit: {segs.attrs['sigma']})")
    return calls


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="call_depth_cnvs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--depth-mq0", required=True, help="mosdepth regions.bed.gz, 100 bp windows, all reads")
    parser.add_argument("--depth-mq20", required=True, help="mosdepth regions.bed.gz, 100 bp windows, MAPQ>=20")
    parser.add_argument("--annotation", required=True, help="Reference annotation parquet (build_depth_annotation)")
    parser.add_argument(
        "--sex",
        choices=SEXES,
        default=None,
        help="Sample sex (default: inferred from chrX / chrY depth; chrX / chrY are not called if it is ambiguous)",
    )
    parser.add_argument("--sample-name", required=True, help="Sample name for the output VCF")
    parser.add_argument("--fasta-index", required=True, help="Reference .fai")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument("--contigs", nargs="+", default=None, help="Contigs to call (default: all annotated)")
    parser.add_argument("--chrx-name", default="chrX", help="chrX name in the reference")
    parser.add_argument("--chry-name", default="chrY", help="chrY name in the reference")
    return parser.parse_args(argv)


def run(argv: list[str]):
    args = parse_args(argv[1:])
    call_depth_cnvs(
        args.depth_mq0,
        args.depth_mq20,
        args.annotation,
        args.sample_name,
        args.fasta_index,
        args.output_vcf,
        args.sex,
        args.contigs,
        args.chrx_name,
        args.chry_name,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
