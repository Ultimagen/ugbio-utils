"""
Cohort-free germline CNV calling from read depth.

Per sample:
  100 bp depth (mosdepth, all reads and MAPQ>=20) + static reference annotation (depth_annotation.py)
  -> static mask (N, extreme GC; never the sample's own coverage)
  -> GC correction fitted on the sample's clean autosomal bins
  -> copy-number estimate per bin (2 x ratio to the autosomal median)
  -> sex chromosome copy numbers from the clean chrX / chrY bins (or the given sex); chrX / chrY are not called
     if the depth is ambiguous
  -> in a male, the pseudoautosomal regions of chrX (--par-bed) keep baseline 2 (the reference masks their chrY copy,
     so all PAR reads map to chrX)
  -> units of `unit_bins` callable bins (masked bins are skipped, so units span them)
  -> weighted PELT per chromosome, not across assembly gaps (> gap_break_bp) or PAR edges; segdup /
     low-mappability units weighted by their own (larger) noise
  -> segments called by direction (loss / gain) vs the baseline (2, or 1 on haploid contigs outside the PARs)
  -> same-direction calls merged across noise and short masked gaps (one call per event), not across assembly gaps
     or PAR edges
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
    CNV_INFO_TAGS,
    CNV_SOURCE_DEPTH,
)
from ugbio_cnv.depth_annotation import BIN_SIZE, read_annotation
from ugbio_cnv.estimate_ploidy import determine_karyotype
from ugbio_core.logger import logger

AUTOSOME_PATTERN = re.compile(r"(chr)?\d+")


@dataclass
class Params:
    """
    Parameters of call_depth_cnvs.

    Copy numbers are in copies (a diploid baseline is 2), lengths in bp unless the name says bins or units. A bin is a
    100 bp window of the annotation; a unit is unit_bins consecutive callable bins (see make_units).

    Attributes
    ----------
    unit_bins : int
        Callable bins per unit, the data point of the segmentation.
    gc_min, gc_max : float
        Bins outside this GC fraction are masked (not callable).
    penalty_scale : float
        PELT penalty per changepoint is penalty_scale * log(number of units); the units are weighted by their noise.
    cn_clip : float
        Upper clip of the per-bin copy number.
    min_units : int
        Minimal segment length (units).
    min_abs_dcn, min_z : float
        A segment is called if |mean copy number - baseline| >= min_abs_dcn and |z| >= min_z.
    difficult_frac : float
        A unit or segment is difficult if at least this fraction is segdup or low-mappability.
    min_abs_dcn_difficult : float
        |copy number change| for a confident (not LOW_CONF) call in difficult sequence.
    merge_gap_frac, merge_gap_max : float, int
        Same-direction calls merge if the gap is <= merge_gap_frac x the smaller call and <= merge_gap_max bp ...
    merge_masked_frac : float
        ... or at least this fraction of the gap is not callable.
    min_len : int
        Minimal call length before edge refinement.
    refine_window : int
        Edge refinement search window (+-bp).
    min_len_refined : int
        Minimal call length after edge refinement.
    merge_block_normal_bp : int
        A gap with this much confidently normal unique sequence is never merged across.
    blip_max_bp : int
        An opposite-direction call inside a merge is absorbed if it is low-confidence or shorter than this; otherwise
        it is kept as a nested call.
    gap_break_bp : int
        A run of N bins this long (assembly gap) ends a stretch: no segment or merge crosses it.
    chunk_units : int
        PELT runs on chunks of this many units (it is quadratic without changepoints); the merge heals chunk edges.
    min_sex_bins : int
        Clean chrX / chrY bins needed to infer the sex chromosome copy numbers.
    """

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
    gap_break_bp: int = 100_000  # segments and merges do not cross a run of N bins this long (assembly gap)
    chunk_units: int = 10_000  # PELT is quadratic without changepoints; chunk boundaries are healed by the merge
    min_sex_bins: int = 1000  # clean chrX / chrY bins needed to infer the sex chromosome copy numbers


# VCF header keys carrying the sex chromosome setup from call_depth_cnvs to the candidate steps (split-read, pair,
# cn.mops)
HAPLOID_HEADER_KEY = "depth_cnv_haploid_contigs"
SKIPPED_HEADER_KEY = "depth_cnv_skipped_contigs"
KARYOTYPE_HEADER_KEY = "depth_cnv_karyotype"
DIPLOID_HEADER_KEY = "depth_cnv_diploid_regions"  # regions of haploid contigs with baseline 2 (male PARs)
# depth ratios to the autosomes too close to a one- vs two-copy (chrX) or zero- vs one-copy (chrY) boundary to decide
X_RATIO_AMBIGUOUS = (0.65, 0.85)
Y_RATIO_AMBIGUOUS = (0.15, 0.35)


@dataclass
class SexChromosomes:
    """
    Sex chromosome setup of a sample: which contigs are haploid, which are not called, and how this was decided.

    Attributes
    ----------
    haploid : set[str]
        Contigs with baseline copy number 1.
    skipped : set[str]
        Contigs that are not called (no copy expected, or the karyotype could not be determined).
    karyotype : str
        Karyotype and how it was decided, e.g. "XY (inferred: chrX 0.50, chrY 0.49)" or "XX (given)".
    diploid : list[tuple[str, int, int]]
        Regions (chrom, start, end) of haploid contigs with baseline 2 (the male chrX PARs).
    """

    haploid: set[str] = field(default_factory=set)  # contigs with baseline copy number 1
    skipped: set[str] = field(default_factory=set)  # contigs not called (no copy expected)
    karyotype: str = "."  # e.g. XY; how the setup was decided
    diploid: list[tuple[str, int, int]] = field(default_factory=list)  # baseline 2 inside haploid contigs (male PARs)

    def baseline(self, chrom: str, starts: np.ndarray) -> np.ndarray:
        """
        Baseline copy number of bins on one contig.

        Parameters
        ----------
        chrom : str
            Contig.
        starts : np.ndarray
            Bin start positions on the contig.

        Returns
        -------
        np.ndarray
            1.0 on a haploid contig outside the diploid regions, else 2.0, per bin.
        """
        base = np.full(len(starts), 1.0 if chrom in self.haploid else 2.0)
        for c, s, e in self.diploid:
            if c == chrom:
                base[(starts >= s) & (starts < e)] = 2.0
        return base


def read_par_bed(par_bed: str, chrom: str) -> list[tuple[str, int, int]]:
    """
    Pseudoautosomal regions of one chromosome.

    Parameters
    ----------
    par_bed : str
        BED of the pseudoautosomal regions (chrom, start, end); header, track and browser lines are skipped.
    chrom : str
        Chromosome to keep.

    Returns
    -------
    list[tuple[str, int, int]]
        The regions (chrom, start, end) on chrom.
    """
    out = []
    with open(par_bed) as f:
        for line in f:
            if line.strip() and not line.startswith(("#", "track", "browser")):
                c, s, e, *_ = line.rstrip("\n").split("\t")
                if c == chrom:
                    out.append((c, int(s), int(e)))
    return out


# ---------------------------------------------------------------- inputs
def read_mosdepth_regions(path: str) -> pd.DataFrame:
    """
    Read a mosdepth regions file.

    Parameters
    ----------
    path : str
        mosdepth `.regions.bed.gz` (chrom, start, end, mean depth).

    Returns
    -------
    pd.DataFrame
        Columns chrom, start, end, depth.
    """
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "depth"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64, "depth": np.float32},
    )


def load_bins(depth_mq0: str, depth_mq20: str, annotation: str, contigs: list[str] | None = None) -> pd.DataFrame:
    """
    Join the all-read and MAPQ>=20 depth with the reference annotation, per 100 bp bin.

    Parameters
    ----------
    depth_mq0, depth_mq20 : str
        mosdepth `.regions.bed.gz` with 100 bp windows, all reads and MAPQ>=20.
    annotation : str
        Reference annotation parquet from build_depth_annotation.
    contigs : list[str], optional
        Contigs to keep (default: all annotated contigs).

    Returns
    -------
    pd.DataFrame
        The annotation columns plus depth (all reads) and depth_uniq (MAPQ>=20), sorted by chrom and start. Bins
        without all-read depth are dropped.
    """
    ann = read_annotation(annotation)
    if contigs:
        ann = ann[ann["chrom"].isin(contigs)]
    d0 = read_mosdepth_regions(depth_mq0)
    d20 = read_mosdepth_regions(depth_mq20).rename(columns={"depth": "depth_uniq"})[["chrom", "start", "depth_uniq"]]
    bins = ann.merge(d0, on=["chrom", "start"], how="inner").merge(d20, on=["chrom", "start"], how="left")
    return bins.sort_values(["chrom", "start"], kind="stable").reset_index(drop=True)


KARYOTYPES = ("XX", "XY", "XXY", "XYY", "X0", "XXX", "XXXY")  # estimate_ploidy.determine_karyotype labels


def sex_chromosome_names(fasta_index: str) -> tuple[str, str]:
    """
    chrX / chrY names of a reference: chrX, chrY (hg38 style) unless the index has X, Y (GRCh37 style) instead.

    Parameters
    ----------
    fasta_index : str
        Reference .fai.

    Returns
    -------
    tuple[str, str]
        The chrX and chrY names.
    """
    with open(fasta_index) as f:
        names = {line.split("\t", 1)[0] for line in f}
    if "chrX" not in names and "X" in names:
        return "X", "Y"
    return "chrX", "chrY"


def karyotype_copies(karyotype: str, chrx_name: str = "chrX", chry_name: str = "chrY") -> tuple[set[str], bool]:
    """
    Haploid sex chromosomes of a karyotype: chrX if it has one X, chrY if it has a Y.

    Extra copies (XXX, XYY, XXXY) are called against two chrX / one chrY copy (logged as a warning).

    Parameters
    ----------
    karyotype : str
        Karyotype label, e.g. XY, XXY or X0 (estimate_ploidy.determine_karyotype).
    chrx_name, chry_name : str
        Sex chromosome names in the reference.

    Returns
    -------
    tuple[set[str], bool]
        The haploid contigs, and whether the karyotype has a Y.
    """
    n_x, n_y = karyotype.count("X"), karyotype.count("Y")
    if n_x > 2 or n_y > 1:  # noqa: PLR2004
        logger.warning(f"Karyotype {karyotype}: extra sex chromosome copies are called against 2 (chrX) / 1 (chrY)")
    haploid = {chrx_name} if n_x == 1 else set()
    if n_y:
        haploid.add(chry_name)
    return haploid, n_y > 0


def sex_chromosome_setup(karyotype: str, chrx_name: str = "chrX", chry_name: str = "chrY") -> SexChromosomes:
    """
    Sex chromosome setup from a given karyotype, the same as for an inferred one.

    One chrX copy makes chrX haploid, two or more diploid; chrY is haploid if present and not called otherwise. E.g.
    XY: chrX and chrY haploid; XX: chrX diploid, chrY not called; XXY: chrX diploid, chrY haploid; X0: chrX haploid,
    chrY not called.

    Parameters
    ----------
    karyotype : str
        One of KARYOTYPES.
    chrx_name, chry_name : str
        Sex chromosome names in the reference.

    Returns
    -------
    SexChromosomes
        The setup, karyotype e.g. "XY (given)".

    Raises
    ------
    ValueError
        If karyotype is not one of KARYOTYPES.
    """
    if karyotype not in KARYOTYPES:
        raise ValueError(f"karyotype must be one of {KARYOTYPES}, got {karyotype!r}")
    haploid, has_y = karyotype_copies(karyotype, chrx_name, chry_name)
    return SexChromosomes(haploid, set() if has_y else {chry_name}, f"{karyotype} (given)")


def sex_chromosome_ratios(
    bins: pd.DataFrame, chrx_name: str, chry_name: str, p: Params
) -> tuple[float | None, float | None]:
    """
    Depth of chrX and chrY relative to the autosomes.

    Parameters
    ----------
    bins : pd.DataFrame
        Normalized bins (normalize).
    chrx_name, chry_name : str
        Sex chromosome names in the reference.
    p : Params
        Caller parameters (min_sex_bins).

    Returns
    -------
    tuple[float | None, float | None]
        Median cn / 2 of the clean (callable, not low-mappability, not segdup) chrX and chrY bins: 1 for two copies,
        0.5 for one. None for a chromosome with fewer than p.min_sex_bins clean bins (e.g. not among the contigs).
    """
    clean = bins["callable"] & (bins["lowmap"] == 0) & (bins["segdup"] == 0)

    def ratio(chrom: str) -> float | None:
        cn = bins.loc[clean & (bins["chrom"] == chrom), "cn"]
        return float(cn.median()) / 2 if len(cn) >= p.min_sex_bins else None

    return ratio(chrx_name), ratio(chry_name)


def infer_sex_chromosomes(
    x_ratio: float | None,
    y_ratio: float | None,
    chrx_name: str = "chrX",
    chry_name: str = "chrY",
    *,
    chry_in_bins: bool = False,
) -> SexChromosomes:
    """
    Sex chromosome setup from the chrX / chrY depth ratios, with estimate_ploidy.determine_karyotype.

    One chrX copy makes chrX haploid, two or more diploid; chrY is haploid if present and not called otherwise.
    Without chrX data nothing is decided (the sex chromosomes are not being called). If the karyotype cannot be
    determined (a ratio between two copy numbers, see X_RATIO_AMBIGUOUS / Y_RATIO_AMBIGUOUS), chrX and chrY are not
    called. A chrY that is being called but has too few clean bins for a ratio is not called either (its copy number
    is unknown).

    Parameters
    ----------
    x_ratio, y_ratio : float or None
        chrX / chrY depth ratio to the autosomes (sex_chromosome_ratios); None if not measured.
    chrx_name, chry_name : str
        Sex chromosome names in the reference.
    chry_in_bins : bool
        chrY is among the bins being called (so y_ratio None means too few clean bins, not "not called").

    Returns
    -------
    SexChromosomes
        The inferred setup.
    """
    if x_ratio is None:
        if y_ratio is not None:
            logger.warning(f"{chrx_name} has too few clean bins to infer the karyotype: {chry_name} is not called")
            return SexChromosomes(skipped={chrx_name, chry_name}, karyotype=f"UNDETERMINED (no {chrx_name})")
        return SexChromosomes(
            skipped={chry_name} if chry_in_bins else set(), karyotype=f"not inferred (no {chrx_name})"
        )
    ambiguous = X_RATIO_AMBIGUOUS[0] < x_ratio < X_RATIO_AMBIGUOUS[1] or (
        y_ratio is not None and Y_RATIO_AMBIGUOUS[0] < y_ratio < Y_RATIO_AMBIGUOUS[1]
    )
    karyotype = "UNDETERMINED" if ambiguous else determine_karyotype(x_ratio, y_ratio)  # no chrY: chrX alone
    found = f"{chrx_name} {x_ratio:.2f}" + (f", {chry_name} {y_ratio:.2f}" if y_ratio is not None else "")
    if karyotype == "UNDETERMINED":
        logger.warning(
            f"Cannot infer the sex chromosome copy numbers from depth (ratio to the autosomes: {found}): "
            f"{chrx_name} and {chry_name} are not called (give the sex to call them)"
        )
        return SexChromosomes(skipped={chrx_name, chry_name}, karyotype=f"UNDETERMINED (inferred: {found})")
    haploid, has_y = karyotype_copies(karyotype, chrx_name, chry_name)
    sex = SexChromosomes(haploid=haploid, karyotype=f"{karyotype} (inferred: {found})")
    if not has_y and (y_ratio is not None or chry_in_bins):
        if y_ratio is None:
            logger.warning(f"{chry_name} has too few clean bins for its copy number: it is not called")
        sex.skipped.add(chry_name)
    return sex


def read_sex_chromosomes(vcf: str) -> SexChromosomes:
    """
    Sex chromosome setup recorded in a call_depth_cnvs VCF header.

    Parameters
    ----------
    vcf : str
        A call_depth_cnvs VCF (or a VCF derived from it that kept its header).

    Returns
    -------
    SexChromosomes
        The setup the depth calls were made with.

    Raises
    ------
    ValueError
        If the header has no sex chromosome setup.
    """
    with pysam.VariantFile(vcf) as v:
        meta = {r.key: r.value for r in v.header.records if r.type == "GENERIC"}
    if HAPLOID_HEADER_KEY not in meta:
        raise ValueError(f"{vcf}: no {HAPLOID_HEADER_KEY} header line (not a call_depth_cnvs VCF)")

    def contigs(key: str) -> set[str]:
        value = meta.get(key, ".")
        return set() if value == "." else set(value.split(","))

    diploid = []
    if meta.get(DIPLOID_HEADER_KEY, ".") != ".":
        for region in meta[DIPLOID_HEADER_KEY].split(","):
            chrom, span = region.rsplit(":", 1)
            start, end = span.split("-")
            diploid.append((chrom, int(start), int(end)))
    return SexChromosomes(
        contigs(HAPLOID_HEADER_KEY), contigs(SKIPPED_HEADER_KEY), meta.get(KARYOTYPE_HEADER_KEY, "."), diploid
    )


# ---------------------------------------------------------------- normalization
def normalize(bins: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    GC-corrected copy number estimate per bin.

    A bin is callable if it has no N and its GC fraction is within [gc_min, gc_max] (a static mask: the sample's own
    coverage never masks a bin). The GC model is the median depth per 1% GC of the clean autosomal bins (callable, not
    low-mappability, not segdup).

    Parameters
    ----------
    bins : pd.DataFrame
        Bins from load_bins.
    p : Params
        Caller parameters (gc_min, gc_max, cn_clip).

    Returns
    -------
    pd.DataFrame
        A copy of bins with callable, ratio (depth / GC model), cn (2 x ratio / its autosomal median, clipped to
        [0, cn_clip]) and uniq_frac (MAPQ>=20 / all-read depth; NaN without reads).

    Raises
    ------
    ValueError
        If the clean autosomal bins have no coverage.
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
    bins["uniq_frac"] = np.where(bins["depth"] > 0, bins["depth_uniq"] / bins["depth"], np.nan)
    return bins


def set_baseline(bins: pd.DataFrame, sex: SexChromosomes) -> pd.DataFrame:
    """
    Baseline (expected) copy number of each bin.

    Parameters
    ----------
    bins : pd.DataFrame
        Bins with chrom and start.
    sex : SexChromosomes
        Sex chromosome setup.

    Returns
    -------
    pd.DataFrame
        A copy of bins with baseline: 1 on haploid contigs outside the diploid regions (male PARs), else 2.
    """
    bins = bins.copy()
    bins["baseline"] = 2.0
    for chrom, idx in bins.groupby("chrom", sort=False).groups.items():
        bins.loc[idx, "baseline"] = sex.baseline(chrom, bins.loc[idx, "start"].to_numpy())
    return bins


def assembly_gaps(starts: np.ndarray, n_frac: np.ndarray, min_bp: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Assembly gaps of one chromosome.

    Parameters
    ----------
    starts : np.ndarray
        Sorted bin starts of one chromosome.
    n_frac : np.ndarray
        N fraction of the same bins.
    min_bp : int
        Minimal gap length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Start and end positions of the runs of N bins (n_frac >= 0.5) of at least min_bp.
    """
    is_n = np.r_[0, (n_frac >= 0.5).astype(np.int8), 0]  # noqa: PLR2004
    edges = np.diff(is_n)
    run_start, run_end = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long_run = (run_end - run_start) * BIN_SIZE >= min_bp
    return starts[run_start[long_run]], starts[run_end[long_run] - 1] + BIN_SIZE


def assembly_gap_ends(starts: np.ndarray, n_frac: np.ndarray, min_bp: int) -> np.ndarray:
    """
    Ends of the assembly gaps of one chromosome (see assembly_gaps).

    Parameters
    ----------
    starts : np.ndarray
        Sorted bin starts of one chromosome.
    n_frac : np.ndarray
        N fraction of the same bins.
    min_bp : int
        Minimal gap length.

    Returns
    -------
    np.ndarray
        End positions of the runs of N bins (n_frac >= 0.5) of at least min_bp.
    """
    return assembly_gaps(starts, n_frac, min_bp)[1]


def make_units(bins: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    Group the callable bins into units, the data points of the segmentation.

    A unit is unit_bins (5) consecutive callable bins of one chromosome, about 500 bp of callable sequence. Bins
    that are not callable (N, or GC outside [gc_min, gc_max]) are skipped, not counted, so a unit spans any masked
    bins between its callable bins: e.g. 3 callable bins, 10 masked bins and 2 callable bins are one unit of 1.5 kb.
    Copy number is averaged over the unit's callable bins only.

    Units belong to stretches. A new stretch starts after an assembly gap (a run of N bins of at least gap_break_bp)
    and where the baseline changes (PAR edges); segments and merges never cross a stretch boundary. A unit that
    would span a boundary is split in two incomplete units, which are dropped. Shorter masked runs (short N runs,
    extreme GC) do not start a new stretch.

    Parameters
    ----------
    bins : pd.DataFrame
        Bins with callable, cn and baseline (normalize, set_baseline).
    p : Params
        Caller parameters (unit_bins, gap_break_bp).

    Returns
    -------
    pd.DataFrame
        One row per complete unit: chrom, stretch (per chromosome), unit (index of the unit on the chromosome),
        start / end (of its first / last callable bin), cn (mean), n (bins), baseline, and the mean lowmap, segdup
        and uniq_frac of its bins.
    """
    cb = bins[bins["callable"] & bins["cn"].notna()].copy()
    cb["unit"] = cb.groupby("chrom", sort=False).cumcount() // p.unit_bins
    gaps_before = pd.Series(0, index=cb.index)
    for chrom, b in bins.groupby("chrom", sort=False):
        gap_ends = assembly_gap_ends(b["start"].to_numpy(), b["n_frac"].to_numpy(), p.gap_break_bp)
        on_chrom = cb["chrom"] == chrom
        gaps_before[on_chrom] = np.searchsorted(gap_ends, cb.loc[on_chrom, "start"].to_numpy(), side="right")
    by_chrom = cb.groupby("chrom", sort=False)
    new_stretch = (gaps_before.groupby(cb["chrom"]).diff().fillna(0) != 0) | (
        by_chrom["baseline"].diff().fillna(0) != 0
    )
    cb["stretch"] = new_stretch.groupby(cb["chrom"]).cumsum()
    units = (
        cb.groupby(["chrom", "stretch", "unit"], sort=False)
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
    Exact PELT segmentation with the weighted L2 cost sum w (x - mean_w)^2.

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


def _mad_noise(d: np.ndarray) -> float:
    """
    Noise of one unit from the successive differences of units.

    Parameters
    ----------
    d : np.ndarray
        Successive copy number differences.

    Returns
    -------
    float
        MAD / 0.6745 / sqrt(2); NaN when there are no differences or they are all equal.
    """
    s = np.median(np.abs(d)) / 0.6745 / np.sqrt(2) if len(d) else np.nan
    return float(s) if np.isfinite(s) and s > 0 else np.nan


def unit_noise(units: pd.DataFrame) -> pd.Series:
    """
    Robust noise of one unit (copies) per class, difficult or not.

    The noise is the MAD of successive differences within each class and chromosome. A class without usable
    differences (e.g. at most one difficult unit per chromosome, or all equal) takes the noise of all units, so its
    weights stay finite; if that is not usable either, 1 (one copy).

    Parameters
    ----------
    units : pd.DataFrame
        Units with chrom, cn and difficult.

    Returns
    -------
    pd.Series
        Noise per class (index: difficult, False / True).
    """

    def diffs(u: pd.DataFrame) -> np.ndarray:
        parts = [np.diff(g["cn"].to_numpy()) for _, g in u.groupby("chrom", sort=False)]
        return np.concatenate(parts) if parts else np.array([])

    overall = _mad_noise(diffs(units))
    overall = overall if np.isfinite(overall) else 1.0
    sigma = {}
    for cls, u in units.groupby("difficult"):
        s = _mad_noise(diffs(u))
        sigma[cls] = s if np.isfinite(s) else overall
    return pd.Series(sigma)


def segment(units: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    Segment each chromosome and call each segment as DEL, DUP or uncalled ('').

    Units are weighted by 1 / noise^2 of their class (difficult units are noisier). Segments do not cross a stretch
    boundary (assembly gap or PAR edge, see make_units). The penalty is the chromosome's, so the boundaries do not
    change the sensitivity.

    Parameters
    ----------
    units : pd.DataFrame
        Units from make_units.
    p : Params
        Caller parameters.

    Returns
    -------
    pd.DataFrame
        One row per segment: chrom, stretch, start, end, n_units, cn_mean (weighted), baseline, z (of the mean vs the
        baseline), difficult_frac, lowmap, segdup, uniq_frac, direction (DEL / DUP / '') and confident. The noise per
        class is in attrs["sigma"].
    """
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
        cuts = np.flatnonzero(np.diff(u["stretch"].to_numpy())) + 1
        ends = []
        for r0, r1 in zip(np.r_[0, cuts], np.r_[cuts, len(x)], strict=True):  # stretches between gaps / PAR edges
            for c0 in range(r0, r1, p.chunk_units):
                c1 = min(c0 + p.chunk_units, r1)
                if c1 - c0 < 2 * p.min_units:
                    if c0 > r0:  # a short last chunk: part of the stretch's previous segment
                        ends[-1] = c1
                    else:  # a stretch too short to segment: its own segment, never joined across the boundary
                        ends.append(c1)
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
                    "stretch": int(seg["stretch"].iloc[0]),
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
    # the columns also when no chromosome is segmented
    columns = ["chrom", "stretch", "start", "end", "n_units", "cn_mean", "baseline", "z"]
    segs = pd.DataFrame(rows, columns=[*columns, "difficult_frac", "lowmap", "segdup", "uniq_frac"])
    dcn = segs["cn_mean"] - segs["baseline"]
    called = (dcn.abs() >= p.min_abs_dcn) & (segs["z"].abs() >= p.min_z)
    segs["direction"] = np.where(called, np.where(dcn < 0, "DEL", "DUP"), "")
    # difficult sequence needs a full copy change to be confident
    segs["confident"] = called & ((segs["difficult_frac"] < p.difficult_frac) | (dcn.abs() >= p.min_abs_dcn_difficult))
    segs.attrs["sigma"] = sigma.to_dict()
    return segs


# ---------------------------------------------------------------- one call per event
class CallableIndex:
    """
    Fraction of callable bins in any interval, from a per-chromosome cumulative sum.

    Parameters
    ----------
    bins : pd.DataFrame
        Bins with chrom, start and callable, sorted by start per chromosome.
    """

    def __init__(self, bins: pd.DataFrame):
        self.idx = {}
        for chrom, b in bins.groupby("chrom", sort=False):
            self.idx[chrom] = (b["start"].to_numpy(), np.concatenate([[0], np.cumsum(b["callable"].to_numpy())]))

    def fraction(self, chrom: str, start: int, end: int) -> float:
        """
        Fraction of callable bins starting in [start, end) on chrom; 0 for an interval without bins.

        Parameters
        ----------
        chrom : str
            Chromosome.
        start, end : int
            Interval.

        Returns
        -------
        float
            The callable fraction.
        """
        starts, cum = self.idx[chrom]
        i, j = np.searchsorted(starts, start), np.searchsorted(starts, end)
        return float((cum[j] - cum[i]) / (j - i)) if j > i else 0.0


class NormalIndex:
    """
    bp of uncalled (copy-neutral) segments in unique sequence inside any interval.

    Parameters
    ----------
    segs : pd.DataFrame
        Segments from segment.
    difficult_frac : float
        Segments with at least this difficult fraction are not counted as unique.
    """

    def __init__(self, segs: pd.DataFrame, difficult_frac: float):
        self.idx = {}
        normal = segs[(segs["direction"] == "") & (segs["difficult_frac"] < difficult_frac)]
        for chrom, g in normal.groupby("chrom", sort=False):
            ordered = g.sort_values("start")
            self.idx[chrom] = (ordered["start"].to_numpy(), ordered["end"].to_numpy())

    def normal_bp(self, chrom: str, start: int, end: int) -> int:
        """
        bp of normal unique segments inside [start, end) on chrom.

        Parameters
        ----------
        chrom : str
            Chromosome.
        start, end : int
            Interval.

        Returns
        -------
        int
            The overlap in bp (0 on a chromosome without such segments).
        """
        if chrom not in self.idx:
            return 0
        s, e = self.idx[chrom]
        return int(np.clip(np.minimum(e, end) - np.maximum(s, start), 0, None).sum())


# per-segment unit fractions, averaged over the units of a merged call
SEGMENT_FRACTIONS = ("difficult_frac", "lowmap", "segdup", "uniq_frac")


def merge_calls(segs: pd.DataFrame, bins: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    Merge the called segments into one call per event.

    The merge pass (_merge_pass) is repeated over its own output until the number of calls stops changing. A call is
    low-confidence (LOW_CONF) if less than half of its units come from confident segments.

    Parameters
    ----------
    segs : pd.DataFrame
        Segments from segment.
    bins : pd.DataFrame
        Bins (callable fraction of the gaps).
    p : Params
        Caller parameters.

    Returns
    -------
    pd.DataFrame
        The calls of at least min_len: the segment columns (cn_mean and the fractions averaged over all merged units,
        z the largest), n_merged, conf_units and low_conf.
    """
    cidx, nidx = CallableIndex(bins), NormalIndex(segs, p.difficult_frac)
    calls = _merge_pass(segs, p, cidx, nidx)
    while True:
        again = _merge_pass(
            calls.assign(confident=calls["conf_units"] >= 0.5 * calls["n_units"]), p, cidx, nidx, first_pass=False
        )
        if len(again) == len(calls):
            break
        calls = again
    calls["low_conf"] = calls["conf_units"] < 0.5 * calls["n_units"]
    return calls[(calls["end"] - calls["start"]) >= p.min_len].reset_index(drop=True)


def _merge_pass(  # noqa: C901, PLR0912
    segs: pd.DataFrame,
    p: Params,
    cidx: CallableIndex,
    nidx: NormalIndex,
    *,
    first_pass: bool = True,
) -> pd.DataFrame:
    """
    One merge pass over the called segments (or the calls of a previous pass).

    Same-direction calls on a chromosome are merged when the gap between them is short relative to the calls (or
    mostly uncallable), unless it holds merge_block_normal_bp of confidently normal unique sequence. They are not
    merged across a stretch boundary. Opposite-direction calls inside a merge are absorbed when weak or short, and
    kept as nested calls otherwise.

    Parameters
    ----------
    segs : pd.DataFrame
        Segments (first pass) or calls of the previous pass.
    p : Params
        Caller parameters.
    cidx : CallableIndex
        Callable fraction of the gaps.
    nidx : NormalIndex
        Normal unique sequence in the gaps.
    first_pass : bool
        segs are segments: add the running sums a merge needs.

    Returns
    -------
    pd.DataFrame
        The calls.
    """
    out = []
    for chrom, s in segs.groupby("chrom", sort=False):
        merged = []
        for seg in s[s["direction"] != ""].sort_values("start").to_dict("records"):
            c = seg
            if first_pass:
                c = dict(
                    c, n_merged=1, cn_sum=c["cn_mean"] * c["n_units"], conf_units=c["n_units"] if c["confident"] else 0
                )
                for f in SEGMENT_FRACTIONS:  # uniq_frac is NaN in a segment without reads
                    known = not np.isnan(c[f])
                    c[f"{f}_sum"], c[f"{f}_units"] = (c[f] * c["n_units"], c["n_units"]) if known else (0.0, 0)
            # earliest same-direction call within reach for which the gap rule holds
            target = None
            for j in range(len(merged) - 1, -1, -1):
                prev = merged[j]
                if prev["stretch"] != c["stretch"]:
                    break  # no merge across an assembly gap or a PAR edge
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
                        for f in SEGMENT_FRACTIONS:
                            prev[f"{f}_sum"] += b[f"{f}_sum"]
                            prev[f"{f}_units"] += b[f"{f}_units"]
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
        for f in SEGMENT_FRACTIONS:  # over all merged units, not the first segment's
            calls[f] = calls[f"{f}_sum"] / calls[f"{f}_units"].replace(0, np.nan)
    return calls.reset_index(drop=True)


# ---------------------------------------------------------------- edge refinement (100 bp depth)
def refine_edges(calls: pd.DataFrame, bins: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    Refine the call edges on the 100 bp bins.

    Each edge moves within +-refine_window to the bin position that best fits a step between the baseline and the
    call's copy number (squared error on callable bins). Calls may shrink or grow; calls shorter than one bin are
    dropped.

    Parameters
    ----------
    calls : pd.DataFrame
        Calls from merge_calls.
    bins : pd.DataFrame
        Bins with callable and cn.
    p : Params
        Caller parameters (refine_window).

    Returns
    -------
    pd.DataFrame
        The calls with refined start and end.
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
    """
    VCF header of the depth calls.

    Parameters
    ----------
    sample_name : str
        Sample name.
    fasta_index : str
        Reference .fai (contigs).
    sex : SexChromosomes, optional
        Sex chromosome setup, recorded in the header for the candidate steps.

    Returns
    -------
    pysam.VariantHeader
        The CNV header (contigs, ALTs, GT), the CNV INFO tags and the sex chromosome setup.
    """
    header = add_vcf_header(sample_name, fasta_index)
    for tag in CNV_INFO_TAGS.values():
        if tag[0] not in header.info:
            header.info.add(*tag[:-1])
    sex = sex or SexChromosomes()
    header.add_meta(HAPLOID_HEADER_KEY, ",".join(sorted(sex.haploid)) or ".")
    header.add_meta(SKIPPED_HEADER_KEY, ",".join(sorted(sex.skipped)) or ".")
    header.add_meta(KARYOTYPE_HEADER_KEY, sex.karyotype)
    header.add_meta(DIPLOID_HEADER_KEY, ",".join(f"{c}:{s}-{e}" for c, s, e in sex.diploid) or ".")
    return header


HOM_DEL_MAX_CN = 0.5  # a deletion with a lower copy number has lost every copy


def genotype_for(svtype: str, cn_mean: float | None = None) -> tuple:
    """
    Genotype of a call, from its type and copy number.

    Parameters
    ----------
    svtype : str
        DEL or DUP.
    cn_mean : float, optional
        Mean copy number of the call.

    Returns
    -------
    tuple
        DEL: (1, 1) when no copy is left (cn_mean < HOM_DEL_MAX_CN, on any contig, a male chrX included), else
        (0, 1). DUP: (None, 1), as copy number gains are not genotyped.
    """
    if svtype != "DEL":
        return (None, 1)
    return (1, 1) if cn_mean is not None and cn_mean < HOM_DEL_MAX_CN else (0, 1)


def write_depth_calls(
    calls: pd.DataFrame, output_vcf: str, sample_name: str, fasta_index: str, sex: SexChromosomes | None = None
) -> None:
    """
    Write the depth calls as a bgzipped, indexed VCF in reference contig order.

    Parameters
    ----------
    calls : pd.DataFrame
        Calls from call_depth_cnvs.
    output_vcf : str
        Output .vcf.gz (a .tbi is written next to it).
    sample_name : str
        Sample name.
    fasta_index : str
        Reference .fai (contigs and their order).
    sex : SexChromosomes, optional
        Sex chromosome setup, recorded in the header.
    """
    header = depth_cnv_header(sample_name, fasta_index, sex)
    order = {c: k for k, c in enumerate(header.contigs)}
    if len(calls):
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
            rec.info["SEGDUP_FRAC"] = round(float(c.segdup), 3)
            if not np.isnan(c.uniq_frac):
                rec.info["UNIQ_FRAC"] = round(float(c.uniq_frac), 3)
            rec.samples[sample_name]["GT"] = genotype_for(c.direction, float(c.cn_mean))
            out.write(rec)
    pysam.tabix_index(output_vcf, preset="vcf", force=True)


def call_depth_cnvs(  # noqa: PLR0913
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    sample_name: str,
    fasta_index: str,
    output_vcf: str,
    karyotype: str | None = None,
    contigs: list[str] | None = None,
    chrx_name: str | None = None,
    chry_name: str | None = None,
    par_bed: str | None = None,
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
    karyotype : str, optional
        Sex karyotype, one of KARYOTYPES (sex_chromosome_setup). By default it is inferred from the depth of chrX and
        chrY; if the depth is ambiguous, chrX and chrY are not called.
    contigs : list[str], optional
        Contigs to call (default: all annotated contigs).
    chrx_name, chry_name : str, optional
        Sex chromosome names in the reference (default: from the fasta index, sex_chromosome_names).
    par_bed : str, optional
        Pseudoautosomal regions (BED, e.g. the par_regions genome resource). With one X and a Y, the chrX PARs keep
        baseline 2 (their chrY copy is masked in the reference, so all their reads map to chrX); without a Y (X0) they
        stay haploid. Without it the whole one-X chrX is haploid.
    params : Params, optional
        Caller parameters.

    Returns
    -------
    pd.DataFrame
        The calls.
    """
    p = params or Params()
    default_x, default_y = sex_chromosome_names(fasta_index)
    chrx_name, chry_name = chrx_name or default_x, chry_name or default_y
    bins = normalize(load_bins(depth_mq0, depth_mq20, annotation, contigs), p)
    x_ratio, y_ratio = sex_chromosome_ratios(bins, chrx_name, chry_name, p)
    inferred = infer_sex_chromosomes(
        x_ratio, y_ratio, chrx_name, chry_name, chry_in_bins=bool((bins["chrom"] == chry_name).any())
    )
    setup = inferred
    if karyotype:
        setup = sex_chromosome_setup(karyotype, chrx_name, chry_name)
        # compare the copy numbers; chrY "not called" only differs when it has no data (e.g. not among the bins)
        if x_ratio is not None and (inferred.haploid, inferred.skipped - {chry_name}) != (
            setup.haploid,
            setup.skipped - {chry_name},
        ):
            logger.warning(f"given karyotype {karyotype}, but the depth gives {inferred.karyotype}")
    # one X and a Y: the chrX PARs have two copies (X + Y); every supported reference masks the chrY copy, so all
    # their reads map to chrX. Not for a karyotype without a Y (X0: chrY not called), whose PARs have one copy.
    one_x_with_y = chrx_name in setup.haploid and chry_name not in setup.skipped
    par = read_par_bed(par_bed, chrx_name) if par_bed and one_x_with_y else []
    setup.diploid = par  # in the header, so the candidate steps score the PARs against 2 copies too
    logger.info(
        f"sex chromosomes: {setup.karyotype}; haploid {sorted(setup.haploid)}, not called {sorted(setup.skipped)}"
        + (f", diploid PARs {par}" if par else "")
    )
    bins = set_baseline(bins[~bins["chrom"].isin(setup.skipped)].reset_index(drop=True), setup)
    segs = segment(make_units(bins, p), p)
    # no segment called (a small region or a clean sample): no calls, and a VCF without records
    calls = pd.DataFrame(
        columns=[
            *segs.columns,
            "n_merged",
            "cn_sum",
            "conf_units",
            "low_conf",
            *(f"{f}_{k}" for f in SEGMENT_FRACTIONS for k in ("sum", "units")),
        ]
    )
    if (segs["direction"] != "").any():
        calls = refine_edges(merge_calls(segs, bins, p), bins, p)
        calls = calls[(calls["end"] - calls["start"]) >= p.min_len_refined].reset_index(drop=True)
    write_depth_calls(calls, output_vcf, sample_name, fasta_index, setup)
    logger.info(f"{output_vcf}: {len(calls)} depth calls (noise per unit: {segs.attrs['sigma']})")
    return calls


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse the call_depth_cnvs command line.

    Parameters
    ----------
    argv : list[str]
        Arguments, without the program name.

    Returns
    -------
    argparse.Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(
        prog="call_depth_cnvs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--depth-mq0", required=True, help="mosdepth regions.bed.gz, 100 bp windows, all reads")
    parser.add_argument("--depth-mq20", required=True, help="mosdepth regions.bed.gz, 100 bp windows, MAPQ>=20")
    parser.add_argument("--annotation", required=True, help="Reference annotation parquet (build_depth_annotation)")
    parser.add_argument(
        "--karyotype",
        choices=KARYOTYPES,
        default=None,
        help="Sex karyotype of the sample (default: inferred from chrX / chrY depth; chrX / chrY are not called if it "
        "is ambiguous). One X: chrX haploid, else diploid; a Y: chrY haploid, else not called",
    )
    parser.add_argument("--sample-name", required=True, help="Sample name for the output VCF")
    parser.add_argument("--fasta-index", required=True, help="Reference .fai")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument("--contigs", nargs="+", default=None, help="Contigs to call (default: all annotated)")
    parser.add_argument(
        "--chrx-name", default=None, help="chrX name in the reference (default: chrX, or X if the index has no chrX)"
    )
    parser.add_argument(
        "--chry-name", default=None, help="chrY name in the reference (default: chrY, or Y if the index has no chrX)"
    )
    parser.add_argument(
        "--par-bed",
        default=None,
        help="Pseudoautosomal regions BED (par_regions resource): male chrX PARs keep 2 copies",
    )
    return parser.parse_args(argv)


def run(argv: list[str]):
    """
    Run call_depth_cnvs.

    Parameters
    ----------
    argv : list[str]
        Command line, the program name first.
    """
    args = parse_args(argv[1:])
    call_depth_cnvs(
        args.depth_mq0,
        args.depth_mq20,
        args.annotation,
        args.sample_name,
        args.fasta_index,
        args.output_vcf,
        args.karyotype,
        args.contigs,
        args.chrx_name,
        args.chry_name,
        args.par_bed,
    )


def main():
    """Entry point of call_depth_cnvs."""
    run(sys.argv)


if __name__ == "__main__":
    main()
