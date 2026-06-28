# Copyright 2026 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# DESCRIPTION
#    LPA Caller for Ultima Genomics WGS data.
#
#    Targeted caller for the LPA KIV-2 VNTR on a single CRAM/BAM.
#
#    Three steps:
#      1. Estimate the total LPA KIV-2 VNTR unit copy number from depth in the
#         KIV-2 reference region, normalized against autosomal stable regions
#         with GC-bias correction.
#      2. Estimate the haplotype-resolved (heterozygous) unit copy number using
#         two linked marker SNV sites (LPA:296T>G + LPA:1264C>G) whose ALT
#         alleles co-occur on the same KIV-2 repeat copy.
#      3. Call two small variants (LPA:4733G>A, LPA:4925G>A) by collapsing read
#         counts across all 6 reference repeat copies and applying a binomial
#         likelihood across all possible ALT copy-number values bounded by the
#         total KIV-2 copy number.
#
#    Outputs <prefix>.targeted.json and <prefix>.targeted.vcf(.gz) containing
#    the LPA results only.
#

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyfaidx
import pysam
from ugbio_core.logger import logger
from ugbio_core.math_utils import log_binomial, phred

# ----------------------------------------------------------------------------
# LPA caller defaults (hg38). The KIV-2 array coordinates and the autosomal
# normalization regions are overridable via CLI; the marker and small-variant
# site lists are hg38-specific and hard-coded (no CLI override).
# ----------------------------------------------------------------------------

# Reference KIV-2 array on hg38: chr6:160613491-160646997
# (33507 bp, 6 repeat units of ~5552 bp).
DEFAULT_KIV2_CHROM = "chr6"
DEFAULT_KIV2_START = 160613491  # 1-based, inclusive
DEFAULT_KIV2_END = 160646997  # 1-based, inclusive
DEFAULT_KIV2_REPEAT_COUNT = 6
DEFAULT_KIV2_UNIT_LEN = 5552

# Heterozygous-marker SNV pair (linked across every repeat copy of a haplotype).
# Per-repeat spacing is irregular (~5543-5552 bp) so explicit positions are used
# instead of uniform extrapolation.
DEFAULT_MARKER_VARIANTS = [
    {
        "hgvs": "LPA:296T>G",
        "ref": "T",
        "alt": "G",
        "positions": [160613786, 160619338, 160624884, 160630428, 160635977, 160641520],
    },
    {
        "hgvs": "LPA:1264C>G",
        "ref": "C",
        "alt": "G",
        "positions": [160614754, 160620306, 160625852, 160631396, 160636945, 160642488],
    },
]

# Small variant sites in the KIV-2 repeat (the two reported LPA variants).
# REF/ALT are genome-strand (LPA gene is on minus strand, hence the C/T pair
# for variants labeled "G>A"). Per-repeat positions are explicit because the
# small-variant region uses a slightly different per-repeat spacing than the
# marker region, and repeat 4 of LPA:4925G>A is omitted.
DEFAULT_SMALL_VARIANTS = [
    {
        "hgvs": "LPA:4925G>A",
        "ref": "C",
        "alt": "T",
        "positions": [160618484, 160624030, 160629574, 160640667, 160646214],
    },
    {
        "hgvs": "LPA:4733G>A",
        "ref": "C",
        "alt": "T",
        "positions": [160618676, 160624222, 160629766, 160635315, 160640859, 160646406],
    },
]

# Autosomal stable regions used as the diploid depth baseline.
# Picked from broadly euchromatic, copy-stable intervals (no segdup overlap,
# moderate GC) on chromosomes 1, 4, 7, 11, 14, 19. Roughly 6 x 2 Mb = 12 Mb,
# which is plenty for a robust mean-depth estimate at 30x.
DEFAULT_NORM_REGIONS = [
    ("chr1", 50_000_000, 52_000_000),
    ("chr4", 80_000_000, 82_000_000),
    ("chr7", 60_000_000, 62_000_000),
    ("chr11", 40_000_000, 42_000_000),
    ("chr14", 70_000_000, 72_000_000),
    ("chr19", 20_000_000, 22_000_000),
]

# Bin size for GC-bias correction.
GC_BIN_SIZE = 1000
# GC bucket width (fraction): 0.05 -> 20 buckets between 0 and 1.
GC_BUCKET_WIDTH = 0.05

# Variant filter thresholds for the TargetedLowQual / TargetedRepeatConflict
# tags emitted on the small-variant rows.
LOW_QUAL_THRESHOLD = 10.0
# Residual ALT-count fraction above which the integer copy partition is flagged
# as inconsistent and the TargetedRepeatConflict filter is applied.
REPEAT_CONFLICT_RESIDUAL = 0.05

# ALT-allele fraction window for calling the marker pair heterozygous. Outside
# this window the call collapses to homozygous REF (below) or homozygous ALT
# (above).
HET_MARKER_ALT_FRACTION_MIN = 0.10
HET_MARKER_ALT_FRACTION_MAX = 0.90

MIN_BASE_QUALITY = 20
MIN_MAPPING_QUALITY = 10
# KIV-2 is a 6-copy VNTR: ~90% of reads have MAPQ=0 because they multi-map
# across the reference repeat units. They must not be filtered out; doing so
# collapses observed KIV-2 depth ~10x and breaks total CN.
KIV2_MIN_MAPPING_QUALITY = 0

# Per-base noise floor for small-variant calling: combined sequencing error and
# residual cross-paralog/cross-repeat mismapping rate at KIV-2 sites. Without
# this, the binomial P(alt|alt_cn=0) collapses to ~1e-85 for any real signal
# and QUAL pins at the Phred floor.
#
# Calibrated for UG flow chemistry SNP substitution errors (~0.1-0.3% per base,
# ~0.15% on this dataset). All currently configured small variants are SNVs.
LPA_VARIANT_NOISE_FLOOR = 0.003
# Standard VCF QUAL cap.
MAX_PHRED_QUAL = 99.0

# Trim this fraction from each tail of the per-bin depth distribution when
# estimating the autosomal baseline. Drops mappability gaps (near-zero bins)
# and CN outliers. Using the median instead biased the baseline ~6-7% high on
# this dataset because the panel is bimodal (some bins ~0, most ~30x).
NORM_TRIM_FRACTION = 0.10


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


@dataclass
class MarkerSpec:
    hgvs: str
    ref: str
    alt: str
    positions: list[int]  # 1-based positions in every reference repeat copy


@dataclass
class AlleleCounts:
    ref: int = 0
    alt: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.ref + self.alt + self.other

    @property
    def informative(self) -> int:
        return self.ref + self.alt


@dataclass
class SmallVariantCall:
    hgvs: str
    qual: float
    alt_copy_number: int
    alt_copy_number_quality: float
    ref_count: int
    alt_count: int
    positions: list[int] = field(default_factory=list)
    ref_allele: str = ""
    alt_allele: str = ""


@dataclass
class LpaCall:
    kiv2_copy_number: float
    ref_marker_allele_copy_number: float | None
    alt_marker_allele_copy_number: float | None
    call_type: str
    variants: list[SmallVariantCall]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _build_marker(spec: dict, kiv2_start: int, unit_len: int, n_repeats: int) -> MarkerSpec:
    """Project a marker onto every repeat copy.

    If the spec provides an explicit ``positions`` list, use it as-is
    (small variants need this because their per-repeat spacing is not the
    uniform ``unit_len`` of the marker region). Otherwise extrapolate from
    ``first_pos`` at fixed ``unit_len`` spacing.
    """
    if "positions" in spec:
        positions = list(spec["positions"])
    else:
        offset = spec["first_pos"] - kiv2_start
        positions = [kiv2_start + offset + i * unit_len for i in range(n_repeats)]
    return MarkerSpec(hgvs=spec["hgvs"], ref=spec["ref"], alt=spec["alt"], positions=positions)


def _gc_fraction(fasta: pyfaidx.Fasta, chrom: str, start: int, end: int) -> float:
    """GC fraction in [start, end] (1-based, inclusive)."""
    seq = str(fasta[chrom][start - 1 : end]).upper()
    if not seq:
        return 0.0
    # str.count is C-level and ~30x faster than a Python generator sum.
    g = seq.count("G")
    c = seq.count("C")
    a = seq.count("A")
    t = seq.count("T")
    at_gc = a + c + g + t
    return (g + c) / at_gc if at_gc > 0 else 0.0


def _bucket_gc(gc: float) -> float:
    """Round GC fraction down to the nearest bucket midpoint."""
    # Clamp so GC exactly == 1.0 lands in the top bucket instead of one past it.
    n_buckets = int(round(1.0 / GC_BUCKET_WIDTH))
    idx = min(int(gc / GC_BUCKET_WIDTH), n_buckets - 1)
    return (idx + 0.5) * GC_BUCKET_WIDTH


def _validate_marker_positions(
    fasta: pyfaidx.Fasta,
    chrom: str,
    markers: list[MarkerSpec],
    genome_build: str,
) -> None:
    """Fail fast if the configured REF base does not match the reference FASTA.

    The marker / small-variant positions are hard-coded hg38 coordinates and
    are not overridable on the CLI. If a different reference build is supplied,
    the caller would otherwise silently call against the wrong sites.
    """
    contigs = set(fasta.keys())
    if chrom not in contigs:
        raise RuntimeError(
            f"Reference FASTA does not contain contig {chrom!r}. "
            f"The marker / small-variant positions are hard-coded {genome_build} coordinates; "
            f"supply a {genome_build} reference."
        )

    mismatches: list[str] = []
    for m in markers:
        for pos in m.positions:
            base = str(fasta[chrom][pos - 1 : pos]).upper()
            if base != m.ref.upper():
                mismatches.append(f"{m.hgvs} {chrom}:{pos} expected {m.ref} got {base or 'N/A'}")

    if mismatches:
        max_preview = 5
        preview = "; ".join(mismatches[:max_preview])
        more = f" (+{len(mismatches) - max_preview} more)" if len(mismatches) > max_preview else ""
        raise RuntimeError(
            "Marker REF bases do not match the reference FASTA — the hard-coded "
            f"positions are {genome_build} coordinates and look wrong for this reference. "
            f"Supply a {genome_build} reference. Mismatches: {preview}{more}"
        )


# ----------------------------------------------------------------------------
# Depth estimation
# ----------------------------------------------------------------------------


def _mean_depth_over_bins(
    bam: pysam.AlignmentFile,
    fasta: pyfaidx.Fasta,
    regions: Iterable[tuple[str, int, int]],
    bin_size: int = GC_BIN_SIZE,
    min_mapq: int = MIN_MAPPING_QUALITY,
) -> dict[float, list[float]]:
    """
    Compute per-bin mean read depth across the given regions, grouped by GC
    bucket. Returns {gc_bucket: [depths_per_bin]}.

    Implementation: one ``fetch`` per region, accumulate each read's reference
    span (``reference_start``..``reference_end``) into a numpy depth array, then
    bin and average. This is ~100x faster than calling ``count_coverage`` per
    bin because it avoids per-base CIGAR pileup and amortizes pysam's per-call
    overhead across the whole region. We trade exact base-resolution coverage
    (CIGAR D/N gaps are not subtracted) for binned mean depth, which is what
    GC normalization actually needs — and the gap bias is uniform across
    regions, so it cancels in the depth ratio.
    """
    by_gc: dict[float, list[float]] = {}
    for chrom, start, end in regions:
        region_len = end - start + 1
        if region_len < bin_size:
            continue
        depth = np.zeros(region_len, dtype=np.int32)
        region_start0 = start - 1  # 0-based offset within the depth array
        for read in bam.fetch(chrom, region_start0, end):
            if (
                read.is_unmapped
                or read.is_secondary
                or read.is_supplementary
                or read.is_duplicate
                or read.is_qcfail
                or read.mapping_quality < min_mapq
            ):
                continue
            rs = read.reference_start
            re = read.reference_end
            if re is None or re <= region_start0 or rs >= end:
                continue
            lo = max(rs, region_start0) - region_start0
            hi = min(re, end) - region_start0
            if hi > lo:
                depth[lo:hi] += 1

        n_bins = region_len // bin_size
        if n_bins == 0:
            continue
        # Reshape into (n_bins, bin_size) and take per-bin mean in one go.
        binned = depth[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
        for b in range(n_bins):
            bin_start = start + b * bin_size
            bin_end = bin_start + bin_size - 1
            gc = _gc_fraction(fasta, chrom, bin_start, bin_end)
            by_gc.setdefault(_bucket_gc(gc), []).append(float(binned[b]))
    return by_gc


def _trimmed_mean(values: list[float] | np.ndarray, trim: float = NORM_TRIM_FRACTION) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return 0.0
    k = int(round(arr.size * trim))
    if 2 * k >= arr.size:
        return float(arr.mean())
    return float(arr[k : arr.size - k].mean())


def _estimate_total_kiv2_copy_number(
    bam: pysam.AlignmentFile,
    fasta: pyfaidx.Fasta,
    kiv2_chrom: str,
    kiv2_start: int,
    kiv2_end: int,
    n_ref_repeats: int,
    norm_regions: list[tuple[str, int, int]],
) -> float:
    """
    Total LPA KIV-2 VNTR unit copy number, summed over the two haplotypes.

    With 6 reference repeat units and a perfectly diploid reference call, KIV-2
    bin depth equals the autosomal mean depth and the returned value is 2*6=12.
    Returned value is the per-cell (diploid) repeat-unit count.
    """
    logger.info("Estimating GC-bias profile from %d normalization regions", len(norm_regions))
    norm_by_gc = _mean_depth_over_bins(bam, fasta, norm_regions)
    # Per-GC-bucket baseline uses a trimmed mean too, for the same reason
    # the overall baseline does.
    gc_medians = {gc: _trimmed_mean(depths) for gc, depths in norm_by_gc.items() if depths}
    all_depths = [d for depths in norm_by_gc.values() for d in depths]
    overall_baseline = _trimmed_mean(all_depths)
    if overall_baseline <= 0:
        raise RuntimeError("Normalization regions have zero coverage; check input/reference")
    logger.info(
        "Autosomal trimmed-mean depth (drop %.0f%% each tail): %.2f",
        100 * NORM_TRIM_FRACTION,
        overall_baseline,
    )

    # KIV-2 binned depth, GC-corrected against the normalization profile.
    # MAPQ filter is dropped here: most KIV-2 reads multi-map (MAPQ=0).
    kiv2_by_gc = _mean_depth_over_bins(
        bam, fasta, [(kiv2_chrom, kiv2_start, kiv2_end)], min_mapq=KIV2_MIN_MAPPING_QUALITY
    )
    corrected = []
    for gc, depths in kiv2_by_gc.items():
        baseline = gc_medians.get(gc, overall_baseline)
        if baseline <= 0:
            baseline = overall_baseline
        scale = overall_baseline / baseline
        corrected.extend(d * scale for d in depths)
    if not corrected:
        raise RuntimeError("No KIV-2 depth bins computed")
    kiv2_mean = float(np.mean(corrected))
    logger.info("GC-corrected KIV-2 mean depth: %.2f", kiv2_mean)

    # Ratio * 2 (haplotype copies) * n_ref_repeats (per-allele baseline).
    copy_number = (kiv2_mean / overall_baseline) * 2.0 * n_ref_repeats
    return copy_number


# ----------------------------------------------------------------------------
# Allele counting at marker / variant positions
# ----------------------------------------------------------------------------


def _count_alleles_at_positions(
    bam: pysam.AlignmentFile,
    chrom: str,
    positions: list[int],
    ref_allele: str,
    alt_allele: str,
) -> AlleleCounts:
    """Sum REF/ALT read counts across all listed (1-based) positions."""
    counts = AlleleCounts()
    if not positions:
        return counts
    start = min(positions) - 1
    end = max(positions)
    pos_set = set(positions)
    # Normalize to uppercase so lowercase query bases (soft-masked reference,
    # some aligners) match the configured REF/ALT and don't fall into 'other'.
    ref_upper = ref_allele.upper()
    alt_upper = alt_allele.upper()
    # KIV-2 reads multi-map -> use the KIV-2 MAPQ floor, not the autosomal one.
    for column in bam.pileup(
        chrom,
        start,
        end,
        truncate=True,
        min_base_quality=MIN_BASE_QUALITY,
        min_mapping_quality=KIV2_MIN_MAPPING_QUALITY,
        ignore_overlaps=True,
        stepper="samtools",
    ):
        ref_pos_1based = column.reference_pos + 1
        if ref_pos_1based not in pos_set:
            continue
        for read in column.pileups:
            if read.is_del or read.is_refskip or read.indel != 0:
                continue
            aln = read.alignment
            # Match the depth-side read filtering. The pysam "samtools" stepper
            # already drops these by default, but make it explicit so the count
            # stays correct if the stepper / pileup options ever change.
            if aln.is_secondary or aln.is_supplementary or aln.is_duplicate or aln.is_qcfail or aln.is_unmapped:
                continue
            base = aln.query_sequence[read.query_position].upper()
            if base == ref_upper:
                counts.ref += 1
            elif base == alt_upper:
                counts.alt += 1
            else:
                counts.other += 1
    return counts


# ----------------------------------------------------------------------------
# Heterozygous marker calling
# ----------------------------------------------------------------------------


def _call_heterozygous_markers(
    bam: pysam.AlignmentFile,
    chrom: str,
    markers: list[MarkerSpec],
    kiv2_copy_number: float,
) -> tuple[float | None, float | None, str, list[AlleleCounts]]:
    """
    Use the linked marker sites to split kiv2_copy_number into REF/ALT haplotype
    unit copy numbers. Returns (ref_cn, alt_cn, call_type, per_marker_counts).
    """
    per_marker_counts = [_count_alleles_at_positions(bam, chrom, m.positions, m.ref, m.alt) for m in markers]
    total_alt = sum(c.alt for c in per_marker_counts)
    total_informative = sum(c.informative for c in per_marker_counts)
    if total_informative == 0:
        return None, None, "No coverage at marker sites", per_marker_counts

    alt_fraction = total_alt / total_informative
    # ALT-allele evidence threshold (>=10% reads supports a heterozygous call).
    if alt_fraction < HET_MARKER_ALT_FRACTION_MIN:
        return None, None, "Homozygous REF markers call", per_marker_counts
    if alt_fraction > HET_MARKER_ALT_FRACTION_MAX:
        return None, None, "Homozygous ALT markers call", per_marker_counts

    alt_cn = kiv2_copy_number * alt_fraction
    ref_cn = kiv2_copy_number - alt_cn
    return ref_cn, alt_cn, "Heterozygous markers call", per_marker_counts


# ----------------------------------------------------------------------------
# Small-variant calling (binomial across all possible ALT copy numbers)
# ----------------------------------------------------------------------------


def _call_small_variant(
    counts: AlleleCounts,
    kiv2_copy_number: float,
    hgvs: str,
    ref_allele: str,
    alt_allele: str,
    positions: list[int],
) -> SmallVariantCall:
    """
    Likelihood ratio: P(alt_cn = best) vs. P(alt_cn = 0). Both QUAL and ALT-CN
    quality are reported as Phred-scaled posteriors under a uniform prior over
    integer ALT copy numbers in [0, round(kiv2_copy_number)].
    """
    total_cn = max(1, int(round(kiv2_copy_number)))
    n = counts.informative
    if n == 0:
        return SmallVariantCall(
            hgvs=hgvs,
            qual=0.0,
            alt_copy_number=0,
            alt_copy_number_quality=0.0,
            ref_count=counts.ref,
            alt_count=counts.alt,
            positions=positions,
            ref_allele=ref_allele,
            alt_allele=alt_allele,
        )

    noise = LPA_VARIANT_NOISE_FLOOR
    log_liks = []
    for alt_cn in range(total_cn + 1):
        # Mix the ideal alt fraction with a flat noise floor so the alt_cn=0
        # hypothesis can absorb sporadic mis-mapped/error reads instead of
        # being assigned probability 0.
        ideal = alt_cn / total_cn
        p_alt = ideal * (1.0 - noise) + (1.0 - ideal) * noise
        log_liks.append(log_binomial(n, counts.alt, p_alt))

    log_liks = np.array(log_liks)
    # Posterior under uniform prior.
    m = log_liks.max()
    post = np.exp(log_liks - m)
    post = post / post.sum()
    best = int(np.argmax(post))
    best_post = float(post[best])

    # QUAL is the Phred-scaled posterior that a variant IS present, i.e.
    # -10*log10(P(alt_cn=0)). When the best call IS alt_cn=0, P(alt_cn=0)
    # ~= 1 and that formula collapses to ~0, which is misleading. Follow the
    # standard VCF convention and report 0.0 in that case.
    if best == 0:
        qual = 0.0
    else:
        # Clamp p away from 0 so phred() does not return +inf, then cap at
        # the VCF QUAL ceiling.
        p0 = min(max(float(post[0]), 1e-15), 1 - 1e-15)
        qual = min(float(phred([p0])[0]), MAX_PHRED_QUAL)

    p_resid = min(max(1.0 - best_post, 1e-15), 1 - 1e-15)
    alt_cn_quality = min(float(phred([p_resid])[0]), MAX_PHRED_QUAL)

    return SmallVariantCall(
        hgvs=hgvs,
        qual=qual,
        alt_copy_number=best,
        alt_copy_number_quality=alt_cn_quality,
        ref_count=counts.ref,
        alt_count=counts.alt,
        positions=positions,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
    )


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------


def _write_json(out_path: Path, sample_id: str, call: LpaCall, genome_build: str) -> None:
    payload = {
        "softwareVersion": "ugbio_cnv.lpa_caller",
        "sampleId": sample_id,
        "genomeBuild": genome_build,
        "lpa": {
            "kiv2CopyNumber": call.kiv2_copy_number,
            "refMarkerAlleleCopyNumber": call.ref_marker_allele_copy_number,
            "altMarkerAlleleCopyNumber": call.alt_marker_allele_copy_number,
            "type": call.call_type,
            "variants": [
                {
                    "hgvs": v.hgvs,
                    "qual": v.qual,
                    "altCopyNumber": v.alt_copy_number,
                    "altCopyNumberQuality": v.alt_copy_number_quality,
                }
                for v in call.variants
            ],
        },
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def _format_dup_record(
    chrom: str,
    start: int,
    end: int,
    ref_base: str,
    sample_id: str,
    call: LpaCall,
    n_ref_repeats: int,
) -> str:
    """Single symbolic SV record summarizing total + per-haplotype KIV-2 copy numbers.

    SVTYPE/ALT are chosen relative to the diploid reference baseline
    (``2 * n_ref_repeats``): a sample with fewer total KIV-2 units than the
    reference is emitted as ``<DEL>`` (``SVTYPE=DEL``), more units as ``<DUP>``
    (``SVTYPE=DUP``). A single symbolic ALT is used instead of duplicated ALTs
    because the duplicate-ALT form is ambiguous to common VCF parsers.
    Per-haplotype unit counts are reported via FORMAT/REPCN, total via
    FORMAT/CN and INFO/TOTAL_CN.
    """
    # start/end are 1-based inclusive (KIV-2: chr6:160613491-160646997 = 33507 bp).
    sv_len = end - start + 1
    total_cn = call.kiv2_copy_number
    diploid_baseline = 2 * n_ref_repeats
    if total_cn < diploid_baseline:
        svtype = "DEL"
        alt = "<DEL>"
    else:
        svtype = "DUP"
        alt = "<DUP>"
    if call.ref_marker_allele_copy_number is not None and call.alt_marker_allele_copy_number is not None:
        cn1 = call.ref_marker_allele_copy_number / n_ref_repeats
        cn2 = call.alt_marker_allele_copy_number / n_ref_repeats
        cn_info = f"{cn1:.6f},{cn2:.6f}"
        repcn = f"{int(round(call.ref_marker_allele_copy_number))}|{int(round(call.alt_marker_allele_copy_number))}"
    else:
        cn_info = "."
        repcn = ".|."
    info = (
        f"SVTYPE={svtype};SVCLAIM=D;END={end};SVLEN={sv_len};CN={cn_info};"
        f"EVENT=LPA:KIV2;EVENTTYPE=VNTR;TOTAL_CN={total_cn:.6f}"
    )
    fmt = "GT:CN:REPCN:PS"
    sample_field = f"0/1:{total_cn:.6f}:{repcn}:{start}"
    return f"{chrom}\t{start}\t.\t{ref_base}\t{alt}\t.\tPASS\t{info}\t{fmt}\t{sample_field}"


def _format_small_variant_record(
    chrom: str,
    v: SmallVariantCall,
    total_cn: int,
    phase_set: int,
) -> list[str]:
    """One VCF row per reference repeat copy carrying the variant."""
    lines = []
    # FILTER tags
    filters = []
    if v.qual < LOW_QUAL_THRESHOLD:
        filters.append("TargetedLowQual")
    # "Repeat conflict" is a soft signal that REF/ALT counts don't cleanly match
    # an integer copy partition. Approximate it by residual against the best fit.
    if total_cn > 0 and (v.ref_count + v.alt_count) > 0:
        expected_alt = (v.alt_copy_number / total_cn) * (v.ref_count + v.alt_count)
        residual = abs(v.alt_count - expected_alt) / max(1, v.ref_count + v.alt_count)
        if residual > REPEAT_CONFLICT_RESIDUAL:
            filters.append("TargetedRepeatConflict")
    filter_field = ";".join(filters) if filters else "PASS"

    # Genotype: total_cn alleles, alt_copy_number of which are 1.
    n_alt = v.alt_copy_number
    n_ref = max(0, total_cn - n_alt)
    gt_alleles = ["0"] * n_ref + ["1"] * n_alt
    gt = "/".join(gt_alleles) if gt_alleles else "."

    qual_field = f"{v.qual:.2f}"
    info = f"EVENT={v.hgvs};EVENTTYPE=VARIANT_IN_HOMOLOGY_REGION"

    for pos in v.positions:
        fmt = "GT:GQ:PS"
        sample_field = f"{gt}:{int(round(v.alt_copy_number_quality))}:{phase_set}"
        lines.append(
            f"{chrom}\t{pos}\t.\t{v.ref_allele}\t{v.alt_allele}\t{qual_field}\t{filter_field}\t{info}\t{fmt}\t{sample_field}"
        )
    return lines


def _write_vcf(
    out_path: Path,
    sample_id: str,
    call: LpaCall,
    chrom: str,
    kiv2_start: int,
    kiv2_end: int,
    n_ref_repeats: int,
    fasta: pyfaidx.Fasta,
    contigs: list[tuple[str, int]],
) -> None:
    ref_base = str(fasta[chrom][kiv2_start - 1 : kiv2_start]).upper() or "N"
    header = [
        "##fileformat=VCFv4.2",
        "##source=ugbio_cnv.lpa_caller",
        '##ALT=<ID=DUP,Description="Duplication relative to the reference">',
        '##ALT=<ID=DEL,Description="Deletion relative to the reference">',
        '##FILTER=<ID=TargetedLowQual,Description="Variant quality below threshold">',
        '##FILTER=<ID=TargetedRepeatConflict,Description="Read counts inconsistent with an integer copy partition">',
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End position">',
        '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">',
        '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length of structural variant">',
        '##INFO=<ID=SVCLAIM,Number=1,Type=String,Description="Claims supported for the structural variant (D=depth)">',
        '##INFO=<ID=CN,Number=.,Type=Float,Description="Per-haplotype copy number ratio vs reference">',
        '##INFO=<ID=EVENT,Number=1,Type=String,Description="Event name">',
        '##INFO=<ID=EVENTTYPE,Number=1,Type=String,Description="Event type">',
        '##INFO=<ID=TOTAL_CN,Number=1,Type=Float,Description="Total LPA KIV-2 unit copy number (diploid)">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
        '##FORMAT=<ID=CN,Number=1,Type=Float,Description="Total LPA KIV-2 unit copy number (diploid)">',
        '##FORMAT=<ID=REPCN,Number=1,Type=String,Description="Per-haplotype repeat unit copy number">',
        '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set">',
    ]
    for c_name, c_len in contigs:
        header.append(f"##contig=<ID={c_name},length={c_len}>")
    header.append(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_id}")

    rows = [_format_dup_record(chrom, kiv2_start, kiv2_end, ref_base, sample_id, call, n_ref_repeats)]
    total_cn_int = max(1, int(round(call.kiv2_copy_number)))
    small_rows = []
    for v in call.variants:
        small_rows.extend(_format_small_variant_record(chrom, v, total_cn_int, kiv2_start))
    # Per-repeat rows are emitted per-variant, not by position; sort so the
    # contig stays coordinate-ordered for tabix indexing.
    small_rows.sort(key=lambda line: int(line.split("\t", 2)[1]))
    rows.extend(small_rows)

    text = "\n".join(header + rows) + "\n"
    # .vcf.gz must be BGZF (not plain gzip) for bcftools/tabix to index/read it.
    if str(out_path).endswith(".gz"):
        with pysam.BGZFile(str(out_path), "w") as f:
            f.write(text.encode())
    else:
        with open(out_path, "w") as f:
            f.write(text)


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------


def _parse_norm_regions(text: str) -> list[tuple[str, int, int]]:
    out = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        chrom, rng = token.split(":")
        start, end = rng.split("-")
        out.append((chrom, int(start), int(end)))
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="lpa_caller", description="LPA KIV-2 VNTR copy number and variant caller")
    ap.add_argument("--cram", required=True, help="Input CRAM or BAM file (indexed)")
    ap.add_argument("--reference", required=True, help="Reference FASTA (hg38)")
    ap.add_argument("--sample-id", required=True, help="Sample identifier emitted in JSON/VCF")
    ap.add_argument("--output-prefix", required=True, help="Output prefix; writes <prefix>.targeted.{json,vcf.gz}")
    ap.add_argument("--genome-build", default="hg38")
    ap.add_argument("--kiv2-chrom", default=DEFAULT_KIV2_CHROM)
    ap.add_argument("--kiv2-start", type=int, default=DEFAULT_KIV2_START)
    ap.add_argument("--kiv2-end", type=int, default=DEFAULT_KIV2_END)
    ap.add_argument("--kiv2-repeat-count", type=int, default=DEFAULT_KIV2_REPEAT_COUNT)
    ap.add_argument("--kiv2-unit-length", type=int, default=DEFAULT_KIV2_UNIT_LEN)
    ap.add_argument(
        "--norm-regions",
        type=str,
        default=None,
        help="Comma-separated chrom:start-end regions used as the autosomal depth baseline. "
        "Default: 6 x 2 Mb stable regions hardcoded in the module.",
    )
    ap.add_argument(
        "--no-vcf",
        action="store_true",
        help="Skip writing the VCF (JSON only).",
    )
    return ap.parse_args(argv)


def run(argv: list[str]) -> None:
    args = _parse_args(argv)

    norm_regions = _parse_norm_regions(args.norm_regions) if args.norm_regions else DEFAULT_NORM_REGIONS

    mode = "rc" if args.cram.endswith(".cram") else "rb"
    bam = pysam.AlignmentFile(args.cram, mode, reference_filename=args.reference)
    fasta = pyfaidx.Fasta(args.reference)

    try:
        markers = [
            _build_marker(spec, args.kiv2_start, args.kiv2_unit_length, args.kiv2_repeat_count)
            for spec in DEFAULT_MARKER_VARIANTS
        ]
        small_variants = [
            _build_marker(spec, args.kiv2_start, args.kiv2_unit_length, args.kiv2_repeat_count)
            for spec in DEFAULT_SMALL_VARIANTS
        ]

        _validate_marker_positions(fasta, args.kiv2_chrom, markers + small_variants, args.genome_build)

        kiv2_cn = _estimate_total_kiv2_copy_number(
            bam,
            fasta,
            args.kiv2_chrom,
            args.kiv2_start,
            args.kiv2_end,
            args.kiv2_repeat_count,
            norm_regions,
        )
        logger.info("Total KIV-2 copy number (diploid): %.3f", kiv2_cn)

        ref_cn, alt_cn, call_type, _ = _call_heterozygous_markers(bam, args.kiv2_chrom, markers, kiv2_cn)
        logger.info("Marker call: %s (ref=%s, alt=%s)", call_type, ref_cn, alt_cn)

        variant_calls = []
        for var in small_variants:
            counts = _count_alleles_at_positions(bam, args.kiv2_chrom, var.positions, var.ref, var.alt)
            call = _call_small_variant(
                counts,
                kiv2_cn,
                hgvs=var.hgvs,
                ref_allele=var.ref,
                alt_allele=var.alt,
                positions=var.positions,
            )
            variant_calls.append(call)
            logger.info(
                "Variant %s: alt_cn=%d qual=%.2f (ref=%d alt=%d)",
                var.hgvs,
                call.alt_copy_number,
                call.qual,
                counts.ref,
                counts.alt,
            )

        result = LpaCall(
            kiv2_copy_number=kiv2_cn,
            ref_marker_allele_copy_number=ref_cn,
            alt_marker_allele_copy_number=alt_cn,
            call_type=call_type,
            variants=variant_calls,
        )

        out_prefix = Path(args.output_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = Path(str(out_prefix) + ".targeted.json")
        _write_json(json_path, args.sample_id, result, args.genome_build)
        logger.info("Wrote %s", json_path)

        if not args.no_vcf:
            contigs = list(zip(bam.references, bam.lengths, strict=False))
            vcf_path = Path(str(out_prefix) + ".targeted.vcf.gz")
            _write_vcf(
                vcf_path,
                args.sample_id,
                result,
                args.kiv2_chrom,
                args.kiv2_start,
                args.kiv2_end,
                args.kiv2_repeat_count,
                fasta,
                contigs,
            )
            logger.info("Wrote %s", vcf_path)
    finally:
        bam.close()
        fasta.close()


def main() -> None:
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
