"""
Genome ploidy estimation from CRAM (mosdepth) or VCF (SNP depth + BAF).

Two mutually exclusive modes:
  Mode 1 (VCF):  per-chr coverage from SNP DP + BAF analysis -> ploidy + karyotype
  Mode 2 (CRAM): mosdepth whole-genome summary -> ploidy + karyotype (no BAF)

Usage:
    estimate_ploidy --vcf <file> --sample-id <id> [--output-dir <dir>]
    estimate_ploidy --mosdepth-summary <file> --sample-id <id> [--output-dir <dir>]
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from array import array
from collections import defaultdict
from itertools import filterfalse
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from ugbio_core.vcfbed.bed_writer import parse_intervals_file
from ugbio_core.vcfbed.vcftools import is_pass_record

DEFAULT_SEX_CHROMOSOMES = ("chrX", "chrY", "X", "Y")
_STANDARD_BASES = {"A", "C", "G", "T"}

_MITO = re.compile(r"^(chrM|MT)$", re.IGNORECASE)
_SKIP = re.compile(r"_random$|_decoy$|^chrUn|^HLA|^EBV|_alt$", re.IGNORECASE)
_REGION_SUFFIX = re.compile(r"_region$", re.IGNORECASE)

# DRAGEN-compatible karyotype lookup: (x_min, x_max, y_min, y_max, label)
_KARYOTYPE_TABLE = [
    (0.75, 1.25, 0.00, 0.25, "XX"),
    (0.25, 0.75, 0.25, 0.75, "XY"),
    (0.75, 1.25, 0.25, 0.75, "XXY"),
    (0.25, 0.75, 0.75, 1.25, "XYY"),
    (0.25, 0.75, 0.00, 0.25, "X0"),
    (1.25, 1.75, 0.25, 0.75, "XXXY"),
    (1.25, 1.75, 0.00, 0.25, "XXX"),
]


def _normalize_chromosome_name(chromosome: str) -> str:
    return chromosome.lower().removeprefix("chr")


def _normalize_sex_chromosomes(sex_chromosomes: list[str] | tuple[str, ...]) -> set[str]:
    return {_normalize_chromosome_name(chromosome) for chromosome in sex_chromosomes}


def _is_sex_chromosome(chromosome: str, sex_chromosomes: set[str]) -> bool:
    return _normalize_chromosome_name(chromosome) in sex_chromosomes


def _is_x_chromosome(chromosome: str) -> bool:
    return _normalize_chromosome_name(chromosome) == "x"


def _is_y_chromosome(chromosome: str) -> bool:
    return _normalize_chromosome_name(chromosome) == "y"


def _is_skipped_chromosome(chromosome: str) -> bool:
    return chromosome == "total" or bool(_SKIP.search(chromosome)) or bool(_MITO.match(chromosome))


def _is_standard_biallelic_snp(ref: str, alts: tuple[str, ...] | None) -> bool:
    return ref in _STANDARD_BASES and alts is not None and len(alts) == 1 and alts[0] in _STANDARD_BASES


def _determine_karyotype(x_ratio: float, y_ratio: float) -> str:
    for x_min, x_max, y_min, y_max, label in _KARYOTYPE_TABLE:
        if x_min <= x_ratio <= x_max and y_min <= y_ratio <= y_max:
            return label
    return "UNDETERMINED"


def _sex_label_from_karyotype(karyotype: str) -> str:
    if karyotype in ("XX", "XXX"):
        return "female"
    if karyotype in ("XY", "XYY", "XXY", "XXXY"):
        return "male"
    if karyotype == "X0":
        return "female"
    return "unknown"


def parse_mosdepth_summary(summary_path: str | Path) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_path, sep="\t")
    summary_df.columns = [c.strip().lower() for c in summary_df.columns]
    return summary_df


def _is_position_excluded(exclude_regions: pd.DataFrame, chrom: str, pos_1based: int) -> bool:
    """Check whether a 1-based VCF position falls inside a 0-based half-open excluded interval."""
    if chrom not in exclude_regions.index:
        return False
    intervals = exclude_regions.loc[[chrom]]
    return bool(((intervals["start"] < pos_1based) & (intervals["end"] >= pos_1based)).any())


def _undetermined_ploidy_result(reason: str) -> dict:
    """Graceful fallback when there's not enough data to compute ploidy (e.g. a narrow input region).

    Mirrors the shape of a normal _compute_ploidy_from_chr_data result so callers
    (write_report, WDL karyotype extraction) don't need special-casing.
    """
    return {
        "per_chrom": [],
        "sex_label": "UNDETERMINED",
        "karyotype": "UNDETERMINED",
        "x_ratio": 0.0,
        "y_ratio": 0.0,
        "auto_mean": 0.0,
        "warnings": [reason],
    }


def _compute_ploidy_from_chr_data(
    chr_data: dict[str, dict],
    sex_chromosomes: list[str] | tuple[str, ...] = DEFAULT_SEX_CHROMOSOMES,
) -> dict:
    """Shared logic: given {chrom: {mean}} compute ploidy, karyotype."""
    sex_chromosome_names = _normalize_sex_chromosomes(sex_chromosomes)
    auto_chroms = {
        chrom: data
        for chrom, data in chr_data.items()
        if not _is_skipped_chromosome(chrom) and not _is_sex_chromosome(chrom, sex_chromosome_names)
    }
    if not auto_chroms:
        return _undetermined_ploidy_result("No autosomal contigs found; cannot estimate ploidy")

    auto_median = float(np.median([data["coverage"] for data in auto_chroms.values()]))
    warnings: list[str] = []

    if auto_median == 0:
        return _undetermined_ploidy_result("Autosomal median coverage is 0; cannot compute ploidy")

    x_mean = next(
        (
            data["coverage"]
            for chrom, data in chr_data.items()
            if _is_sex_chromosome(chrom, sex_chromosome_names) and _is_x_chromosome(chrom)
        ),
        0,
    )
    y_mean = next(
        (
            data["coverage"]
            for chrom, data in chr_data.items()
            if _is_sex_chromosome(chrom, sex_chromosome_names) and _is_y_chromosome(chrom)
        ),
        0,
    )
    x_ratio = x_mean / auto_median
    y_ratio = y_mean / auto_median

    karyotype = _determine_karyotype(x_ratio, y_ratio)
    sex_label = _sex_label_from_karyotype(karyotype)

    per_chrom = []
    for chrom in auto_chroms:
        data = chr_data[chrom]
        coverage = data["coverage"]
        ploidy = 2 * coverage / auto_median
        per_chrom.append({"chrom": chrom, "ploidy": round(ploidy, 3), "mean_cov": round(coverage, 2), "flag": ""})

    for sex_chrom, data in chr_data.items():
        if _is_sex_chromosome(sex_chrom, sex_chromosome_names):
            coverage = data["coverage"]
            ploidy = 2 * coverage / auto_median
            per_chrom.append(
                {"chrom": sex_chrom, "ploidy": round(ploidy, 3), "mean_cov": round(coverage, 2), "flag": "sex"}
            )

    return {
        "per_chrom": per_chrom,
        "sex_label": sex_label,
        "karyotype": karyotype,
        "x_ratio": round(x_ratio, 4),
        "y_ratio": round(y_ratio, 4),
        "auto_mean": round(auto_median, 2),
        "warnings": warnings,
    }


def estimate_ploidy_from_coverage(
    mosdepth_df: pd.DataFrame,
    sex_chromosomes: list[str] | tuple[str, ...] = DEFAULT_SEX_CHROMOSOMES,
) -> dict:
    """Mode 2: estimate ploidy from mosdepth interval summary rows.

    Mosdepth emits both whole-chromosome rows and rows for the BED intervals
    passed with ``--by``. The interval rows are the intended callable regions
    for this workflow, so whole-chromosome and aggregate rows are excluded.
    """
    region_mask = mosdepth_df["chrom"].apply(lambda chrom: bool(_REGION_SUFFIX.search(chrom)))
    df_filtered = mosdepth_df[region_mask].copy()
    if df_filtered.empty:
        raise ValueError("Mosdepth summary contains no *_region rows for ploidy estimation")

    chr_data = {}
    for _, row in df_filtered.iterrows():
        chrom = _REGION_SUFFIX.sub("", row["chrom"])
        if chrom not in chr_data:
            chr_data[chrom] = {"length": row["length"], "coverage": row["mean"]}

    result = _compute_ploidy_from_chr_data(chr_data, sex_chromosomes=sex_chromosomes)
    result["source"] = "mosdepth"
    return result


def _iter_vcf_sites(vcf_path: str | Path, sample_id: str, exclude_regions_bed: str | Path | None = None):
    """Yield (chrom, pos, dp, ref_ad, alt_ad, is_het) of PASS biallelic SNPs; AD is (0, 0) when missing."""
    exclude_regions = (
        parse_intervals_file(str(exclude_regions_bed)).set_index("chromosome") if exclude_regions_bed else None
    )

    with pysam.VariantFile(str(vcf_path)) as reader:
        if sample_id not in reader.header.samples:
            raise ValueError(
                f"Sample {sample_id!r} is not present in VCF; available samples: {', '.join(reader.header.samples)}"
            )

        def is_rejected(variant) -> bool:
            return (
                not _is_standard_biallelic_snp(variant.ref, variant.alts)
                or not is_pass_record(variant)
                or _is_skipped_chromosome(variant.chrom)
                or (exclude_regions is not None and _is_position_excluded(exclude_regions, variant.chrom, variant.pos))
            )

        for variant in filterfalse(is_rejected, reader):
            sample = variant.samples[sample_id]
            ad = sample.get("AD")
            has_ad = ad is not None and len(ad) >= 2 and None not in ad[:2]
            ref_ad, alt_ad = ad[:2] if has_ad else (0, 0)
            is_het = has_ad and sample.get("GT") in ((0, 1), (1, 0))
            yield variant.chrom, variant.pos, sample.get("DP") or 0, ref_ad, alt_ad, is_het


def estimate_ploidy_from_vcf(
    vcf_path: str | Path,
    sample_id: str,
    sex_chromosomes: list[str] | tuple[str, ...] = DEFAULT_SEX_CHROMOSOMES,
    exclude_regions_bed: str | Path | None = None,
    min_effect: float = 0.15,
    coverage_only_effect: float = 0.8,
    min_af_skew: float = 1.2,
) -> tuple[dict, dict]:
    """Mode 1: per-chr coverage from SNP DP + BAF. Returns (coverage_result, baf_result).

    exclude_regions_bed: optional BED (0-based, half-open) of low-confidence regions
    (e.g. PAR-adjacent chrY) whose SNPs are dropped before computing per-chr coverage.
    """
    sites = _iter_vcf_sites(vcf_path, sample_id, exclude_regions_bed)
    return _ploidy_from_sites(sites, sex_chromosomes, min_effect, coverage_only_effect, min_af_skew)


def _ploidy_from_sites(
    sites,
    sex_chromosomes: list[str] | tuple[str, ...] = DEFAULT_SEX_CHROMOSOMES,
    min_effect: float = 0.15,
    coverage_only_effect: float = 0.8,
    min_af_skew: float = 1.2,
) -> tuple[dict, dict]:
    """Estimate ploidy from (chrom, pos, dp, ref_ad, alt_ad, is_het) sites (see _iter_vcf_sites).

    Coverage: interpolated (grouped) median DP per chromosome after dropping DP > 3x the autosomal median;
    ploidy = 2 * cov / median of the autosomal coverages.
    Allelic skew: for each autosomal het with tot = ref_ad + alt_ad >= 8, u = (alt - tot/2)^2 / (tot/4)
    (squared binomial z-score, capped at 30). u is averaged in 1 Mb bins with >= 50 hets, and
    af_skew = median bin mean of the chromosome / median bin mean of all autosomes (~1 when balanced).
    het_af: d = (u - 1) / (4 (tot - 1)) is unbiased for (AF - 0.5)^2; het_af = 0.5 + sqrt(median bin mean of d
    of the chromosome - that of all autosomes), the typical major-allele fraction of the chromosome's hets.

    Parameters
    ----------
    sites : Iterable[tuple[str, int, int, int, int, bool]]
        (chrom, pos, dp, ref_ad, alt_ad, is_het) per site, as yielded by _iter_vcf_sites.
    sex_chromosomes : list[str] | tuple[str, ...], optional
        Chromosome names excluded from the autosomal baseline and from the allelic statistics.
    min_effect : float, optional
        Minimal |ploidy - 2| for an autosome to be considered at all (default 0.15, about a 15% mosaic
        single-copy change). Smaller deviations are treated as noise and never reported.
    coverage_only_effect : float, optional
        |ploidy - 2| at or above which an autosome is called on coverage alone (default 0.8, i.e. close to a
        full copy gained or lost). Needed because such events can leave af_skew uninformative: a monosomy has
        no hets and a balanced 2:2 tetrasomy keeps the alleles at 0.5.
    min_af_skew : float, optional
        Minimal af_skew that confirms a deviation between min_effect and coverage_only_effect (default 1.2;
        normal chromosomes stay below ~1.13, real events in the research data were >= 1.57). Without this
        confirmation the chromosome is reported as "Not confirmed by BAF" rather than called.

    Returns
    -------
    tuple[dict, dict]
        (coverage_result, baf_result): per-chromosome ploidy, af_skew, het_af and call plus karyotype, and the
        genome-wide BAF label from _call_whole_genome_ploidy_by_baf.
    """
    sex_chromosome_names = _normalize_sex_chromosomes(sex_chromosomes)
    chr_dps: dict[str, array] = {}
    bins: dict[tuple[str, int], list[float]] = {}  # (chrom, Mb) -> [sum u, sum d, n]
    baf_values = array("d")  # BAF of every autosomal het for the genome-wide label
    BIN_SIZE = 1_000_000

    # Go over heterozygous calls, aggregate their depths and VAF
    for chrom, pos, dp, ref_ad, alt_ad, is_het in sites:
        if dp > 0:
            chr_dps.setdefault(chrom, array("H")).append(min(dp, 65535))
        total = ref_ad + alt_ad
        if not is_het or total < 8 or _is_sex_chromosome(chrom, sex_chromosome_names):
            continue
        u = min((alt_ad - total / 2) ** 2 / (total / 4), 30.0)
        bin_sums = bins.setdefault((chrom, pos // BIN_SIZE), [0.0, 0.0, 0])
        bin_sums[0] += u
        bin_sums[1] += (u - 1) / (4 * (total - 1))
        bin_sums[2] += 1
        if total >= 10 and 0.1 <= alt_ad / total <= 0.9:
            baf_values.append(alt_ad / total)

    dps = {chrom: np.frombuffer(values, dtype=np.uint16) for chrom, values in chr_dps.items()}
    autosomal = [d for chrom, d in dps.items() if not _is_sex_chromosome(chrom, sex_chromosome_names)]
    cap = int(3 * np.median(np.concatenate(autosomal))) if autosomal else 0
    chr_data = {}
    for chrom, values in dps.items():
        capped = values[values <= cap]
        if len(capped) >= 20:
            chr_data[chrom] = {"coverage": statistics.median_grouped(capped.tolist())}

    coverage_result = _compute_ploidy_from_chr_data(chr_data, sex_chromosomes=sex_chromosomes)
    coverage_result["source"] = "VCF SNP median DP"

    chrom_bins = defaultdict(list)
    for (chrom, _), (sum_u, sum_d, n) in bins.items():
        if n >= 50:
            chrom_bins[chrom].append((sum_u / n, sum_d / n))
    all_bins = np.array([b for values in chrom_bins.values() for b in values]).reshape(-1, 2)
    u_genome, d_genome = np.median(all_bins, axis=0) if len(all_bins) else (np.nan, np.nan)
    for entry in coverage_result["per_chrom"]:
        if entry["flag"] == "sex":
            continue
        chrom_u, chrom_d = np.median(chrom_bins[entry["chrom"]], axis=0) if chrom_bins[entry["chrom"]] else (None, None)
        entry["af_skew"] = None if chrom_u is None else round(float(chrom_u / u_genome), 3)
        entry["het_af"] = None if chrom_d is None else round(0.5 + math.sqrt(max(0.0, chrom_d - d_genome)), 3)
        entry["call"] = _call_aneuploidy(
            entry["ploidy"], entry["af_skew"], min_effect, coverage_only_effect, min_af_skew
        )

    return coverage_result, _call_whole_genome_ploidy_by_baf(baf_values)


def _call_aneuploidy(
    ploidy: float, af_skew: float | None, min_effect: float, coverage_only_effect: float, min_af_skew: float
) -> str:
    """Large coverage deviations are called on coverage alone; smaller ones need allelic skew to confirm them."""
    deviation = abs(ploidy - 2)
    if deviation >= coverage_only_effect:
        return "coverage"
    if deviation < min_effect:
        return ""
    return "coverage+BAF" if af_skew is not None and af_skew >= min_af_skew else "unconfirmed"


def _call_whole_genome_ploidy_by_baf(baf_values: array[float]) -> dict:
    n_het = len(baf_values)
    if n_het < 50:
        return {"label": "INSUFFICIENT_DATA", "confidence": f"(<50 het SNPs, found {n_het})", "n_het": n_het}

    di_count = sum(1 for b in baf_values if 0.40 < b < 0.60)
    tri_count = sum(1 for b in baf_values if (0.25 <= b <= 0.4) or (0.60 <= b <= 0.75))
    di_frac = round(di_count / n_het * 100, 1)
    tri_frac = round(tri_count / n_het * 100, 1)

    if di_count > tri_count * 2.5:
        label, confidence = "DIPLOID", f"({di_frac}% hets in 0.4-0.6 BAF band, n={n_het})"
    elif tri_count > di_count * 2.5:  # symmetric to the DIPLOID dominance check above
        label, confidence = "TRIPLOID", f"(diplo={di_frac}%, tri={tri_frac}%, n={n_het})"
    elif di_count > tri_count * 1.5:
        label, confidence = "LIKELY_DIPLOID", f"({di_frac}% in 0.4-0.6 band, {tri_frac}% in triploid bands, n={n_het})"
    else:
        label, confidence = "INCONCLUSIVE", f"(diplo={di_frac}%, tri={tri_frac}%, n={n_het})"

    return {"label": label, "confidence": confidence, "n_het": n_het, "di_frac": di_frac, "tri_frac": tri_frac}


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def _aneuploidy_call_lines(per_chrom: list[dict]) -> list[str]:
    """Report lines for VCF-mode calls: one warning per flagged chromosome, then nominees BAF did not confirm."""
    flagged = [e for e in per_chrom if e.get("call") in ("coverage", "coverage+BAF")]
    lines = [
        f"  [WARNING] {e['chrom']} {'gain' if e['ploidy'] > 2 else 'loss'} ploidy={e['ploidy']:.3f} "
        f"af_skew={_fmt(e['af_skew'])} het_af={_fmt(e['het_af'])} ({e['call']})"
        for e in flagged
    ] or ["  No autosomal aneuploidy detected."]
    unconfirmed = [
        f"{e['chrom']} (ploidy={e['ploidy']:.3f}, het_af={_fmt(e['het_af'])})"
        for e in per_chrom
        if e.get("call") == "unconfirmed"
    ]
    if unconfirmed:
        lines.append(f"  Not confirmed by BAF: {', '.join(unconfirmed)}")
    return lines


def write_report(
    sample_id: str,
    coverage_result: dict,
    baf_result: dict | None,
    source_path: str,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{sample_id}.ploidy_report.txt"

    cr = coverage_result
    karyotype = cr.get("karyotype", "UNDETERMINED")
    wgs_ploidy = f"{baf_result['label']} {baf_result['confidence']}" if baf_result else "N/A (CRAM mode, no BAF)"
    coverage_source = cr.get("source", "mosdepth")

    lines = [
        "======================================================",
        f"  Genome Ploidy Report - {sample_id}",
        "======================================================",
        "",
        f"  Karyotype:          {karyotype}",
        f"  X/Y coverage ratios: X={cr['x_ratio']:.3f}, Y={cr['y_ratio']:.3f}",
        f"  Whole-genome ploidy: {wgs_ploidy}",
        f"  Autosomal median cov: {cr['auto_mean']:.1f}x",
        f"  Coverage source:    {coverage_source}",
        f"  Input:              {source_path}",
        "",
        "  Per-chromosome ploidy (relative to autosome median = 2.0):",
        "  --------------------------------------------------------",
    ]

    for warning in cr.get("warnings", []):
        lines.extend(["", f"  [WARNING] {warning}"])

    vcf_mode = any("call" in entry for entry in cr["per_chrom"])
    for entry in cr["per_chrom"]:
        icon = {"sex": "."}.get(entry["flag"], " ")
        skew = f"  af_skew={_fmt(entry['af_skew'])}" if "af_skew" in entry else ""
        lines.append(f"  {entry['chrom']:<6s} ploidy={entry['ploidy']:.3f}  cov={entry['mean_cov']:.2f}x{skew}  {icon}")

    lines.extend(["", "  (. = sex chromosome)"])
    if vcf_mode:
        lines.append(
            "  (af_skew = allele-fraction skew of het SNPs relative to the autosomal median;"
            " ~1.0 balanced, >=1.2 supports a copy-number change)"
        )
    lines.append("")
    pd.DataFrame(cr["per_chrom"]).to_csv(output_dir / f"{sample_id}.ploidy_per_chrom.tsv", sep="\t", index=False)

    if vcf_mode:
        lines.extend(_aneuploidy_call_lines(cr["per_chrom"]))
    else:
        any_warn = False
        for entry in cr["per_chrom"]:
            if entry["flag"] not in ("acro", "sex"):
                dev = abs(entry["ploidy"] - 2.0)
                if dev >= 0.35:
                    lines.append(
                        f"  [WARNING] {entry['chrom']} ploidy={entry['ploidy']:.3f} "
                        f"(deviation={dev:.2f}, cov={entry['mean_cov']:.2f}x)"
                    )
                    any_warn = True
        if not any_warn:
            lines.append("  No autosomal aneuploidy detected (all within +/-0.35 of 2.0).")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n")

    return report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate genome ploidy. Mode 1: --vcf (SNP DP + BAF). Mode 2: --mosdepth-summary (coverage only)."
    )
    parser.add_argument("--vcf", default=None, help="Mode 1: VCF input for coverage from SNP DP + BAF analysis")
    parser.add_argument("--mosdepth-summary", default=None, help="Mode 2: mosdepth summary for coverage-only ploidy")
    parser.add_argument("--sample-id", required=True, help="Sample identifier for output files")
    parser.add_argument(
        "--exclude-regions-bed",
        default=None,
        help="Mode 1: optional BED (0-based, half-open) of low-confidence regions to exclude from coverage",
    )
    parser.add_argument(
        "--sex-chromosomes",
        nargs="+",
        default=list(DEFAULT_SEX_CHROMOSOMES),
        help="Sex chromosome names to exclude from autosomal baseline; defaults to chrX chrY X Y",
    )
    parser.add_argument("--min-effect", type=float, default=0.15, help="Mode 1: min |ploidy-2| to nominate a chrom")
    parser.add_argument(
        "--coverage-only-effect", type=float, default=0.8, help="Mode 1: |ploidy-2| flagged on coverage alone"
    )
    parser.add_argument("--min-af-skew", type=float, default=1.2, help="Mode 1: min af_skew confirming a nominee")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args(argv)

    if args.vcf and args.mosdepth_summary:
        parser.error("Provide either --vcf or --mosdepth-summary, not both.")
    if not args.vcf and not args.mosdepth_summary:
        parser.error("Provide either --vcf or --mosdepth-summary.")

    if args.vcf:
        print(f"[estimate_ploidy] Mode 1 (VCF): {args.vcf}", file=sys.stderr)
        coverage_result, baf_result = estimate_ploidy_from_vcf(
            args.vcf,
            args.sample_id,
            args.sex_chromosomes,
            args.exclude_regions_bed,
            args.min_effect,
            args.coverage_only_effect,
            args.min_af_skew,
        )
        source_path = args.vcf
    else:
        print(f"[estimate_ploidy] Mode 2 (mosdepth): {args.mosdepth_summary}", file=sys.stderr)
        mosdepth_df = parse_mosdepth_summary(args.mosdepth_summary)
        coverage_result = estimate_ploidy_from_coverage(mosdepth_df, args.sex_chromosomes)
        baf_result = None
        source_path = args.mosdepth_summary

    print("[estimate_ploidy] Ploidy estimation completed.", file=sys.stderr)

    report_path = write_report(args.sample_id, coverage_result, baf_result, source_path, args.output_dir)
    print(report_path.read_text())
    print(f"[estimate_ploidy] Report: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
