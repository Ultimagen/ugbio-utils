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
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from ugbio_core.vcfbed.vcftools import is_pass_record

DEFAULT_SEX_CHROMOSOMES = ("chrX", "chrY", "X", "Y")
_STANDARD_BASES = {"A", "C", "G", "T"}
_COVERAGE_SAMPLE_COUNT = 5000

_AUTOSOME_CHR = re.compile(r"^chr(\d+)$")
_AUTOSOME_NOCHR = re.compile(r"^(\d+)$")
_MITO = re.compile(r"^(chrM|MT)$", re.IGNORECASE)
_SKIP = re.compile(r"_random$|_decoy$|^chrUn|^HLA|^EBV|_alt$", re.IGNORECASE)

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


def _detect_chr_prefix(contigs: list[str]) -> bool:
    return any(c.startswith("chr") for c in contigs if re.match(r"^chr\d+$", c))


def _autosome_number(contig: str, *, has_chr: bool) -> int | None:
    pattern = _AUTOSOME_CHR if has_chr else _AUTOSOME_NOCHR
    m = pattern.match(contig)
    return int(m.group(1)) if m else None


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


def _update_reservoir(
    reservoir: list[int | float], value: int | float, seen_count: int, sample_count: int, rng: random.Random
) -> int:
    """Add a value to a fixed-size uniform reservoir sample."""
    seen_count += 1
    if len(reservoir) < sample_count:
        reservoir.append(value)
    else:
        index = rng.randint(0, seen_count - 1)  # noqa: S311
        if index < sample_count:
            reservoir[index] = value
    return seen_count


def _determine_karyotype(x_ratio: float, y_ratio: float) -> str:
    for x_min, x_max, y_min, y_max, label in _KARYOTYPE_TABLE:
        if x_min <= x_ratio <= x_max and y_min <= y_ratio <= y_max:
            return label
    return "UNDETERMINED"


def _autosomal_baseline_coverage(auto_chroms: dict[str, dict]) -> tuple[float, list[str]]:
    """Calculate length-weighted autosomal coverage, falling back when lengths are missing."""
    warnings = []
    lengths = [data.get("length") for data in auto_chroms.values()]
    if lengths and all(length is not None and length > 0 for length in lengths):
        total_length = sum(lengths)
        return (
            sum(data["coverage"] * data["length"] for data in auto_chroms.values()) / total_length,
            warnings,
        )

    warnings.append("Missing contig lengths for autosomal coverage baseline; using unweighted median.")
    return float(np.median([data["coverage"] for data in auto_chroms.values()])), warnings


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


def _compute_ploidy_from_chr_data(
    chr_data: dict[str, dict],
    *,
    sex_chromosomes: list[str] | tuple[str, ...] = DEFAULT_SEX_CHROMOSOMES,
    has_chr: bool | None = None,  # noqa: ARG001
) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Shared logic: given {chrom: {mean}} compute ploidy, karyotype."""
    sex_chromosome_names = _normalize_sex_chromosomes(sex_chromosomes)
    auto_chroms = {
        chrom: data
        for chrom, data in chr_data.items()
        if not _is_skipped_chromosome(chrom) and not _is_sex_chromosome(chrom, sex_chromosome_names)
    }
    if not auto_chroms:
        raise ValueError("No autosomal contigs found")

    auto_median, warnings = _autosomal_baseline_coverage(auto_chroms)

    if auto_median == 0:
        raise ValueError("Autosomal median coverage is 0; cannot compute ploidy")

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
    x_ratio = x_mean / auto_median if auto_median > 0 else 0
    y_ratio = y_mean / auto_median if auto_median > 0 else 0

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
    """Mode 2: estimate ploidy from mosdepth summary."""
    df_filtered = mosdepth_df[~mosdepth_df["chrom"].apply(_is_skipped_chromosome)].copy()

    chr_data = {}
    for _, row in df_filtered.iterrows():
        chrom = row["chrom"]
        if chrom not in chr_data:
            chr_data[chrom] = {"length": row["length"], "coverage": row["mean"]}

    result = _compute_ploidy_from_chr_data(chr_data, sex_chromosomes=sex_chromosomes)
    result["source"] = "mosdepth"
    return result


def estimate_ploidy_from_vcf(  # noqa: C901, PLR0912, PLR0915
    vcf_path: str | Path,
    sample_id: str,
    het_sample_count: int = 5000,
    sex_chromosomes: list[str] | tuple[str, ...] = DEFAULT_SEX_CHROMOSOMES,
) -> tuple[dict, dict]:
    """Mode 1: per-chr coverage from SNP DP + BAF. Returns (coverage_result, baf_result)."""
    rng = random.Random(42)  # noqa: S311
    sex_chromosome_names = _normalize_sex_chromosomes(sex_chromosomes)

    reader = pysam.VariantFile(vcf_path)
    if sample_id not in reader.header.samples:
        raise ValueError(
            f"Sample {sample_id!r} is not present in VCF; available samples: {', '.join(reader.header.samples)}"
        )
    contig_lengths = {contig: reader.header.contigs[contig].length for contig in reader.header.contigs}
    chr_dps: dict[str, list[int]] = {}
    chr_dp_seen: dict[str, int] = {}
    # Reservoir sampling for BAF: uniform random sample over autosomal het SNPs
    baf_reservoir: list[float] = []
    baf_seen = 0

    for variant in reader:
        if not _is_standard_biallelic_snp(variant.ref, variant.alts):
            continue
        if not is_pass_record(variant):
            continue

        chrom = variant.chrom
        if _is_skipped_chromosome(chrom):
            continue

        sample = variant.samples[sample_id]
        dp_value = sample.get("DP")
        ad_value = sample.get("AD")

        if dp_value is not None and dp_value > 0:
            chr_dp_seen[chrom] = _update_reservoir(
                chr_dps.setdefault(chrom, []),
                dp_value,
                chr_dp_seen.get(chrom, 0),
                _COVERAGE_SAMPLE_COUNT,
                rng,
            )

        # BAF: reservoir sampling over autosomal het SNPs only
        is_autosome = not _is_sex_chromosome(chrom, sex_chromosome_names)
        if is_autosome and sample.get("GT") in ((0, 1), (1, 0)) and ad_value and len(ad_value) >= 2:  # noqa: PLR2004
            ref_count, alt_count = ad_value[:2]
            if ref_count is not None and alt_count is not None:
                total = ref_count + alt_count
                if total >= 10:  # noqa: PLR2004
                    baf = alt_count / total
                    if 0.1 <= baf <= 0.9:  # noqa: PLR2004
                        baf_seen = _update_reservoir(baf_reservoir, baf, baf_seen, het_sample_count, rng)

    reader.close()

    chr_data = {}
    for chrom, dps in chr_dps.items():
        if len(dps) >= 20:  # noqa: PLR2004
            median_dp = float(np.median(dps))
            chr_data[chrom] = {
                "length": contig_lengths.get(chrom),
                "coverage": median_dp,
            }

    coverage_result = _compute_ploidy_from_chr_data(chr_data, sex_chromosomes=sex_chromosomes)
    coverage_result["source"] = "VCF SNP median DP"

    baf_result = _classify_baf(baf_reservoir)
    return coverage_result, baf_result


def _classify_baf(baf_values: list[float]) -> dict:
    n_het = len(baf_values)
    if n_het < 50:  # noqa: PLR2004
        return {"label": "INSUFFICIENT_DATA", "confidence": f"(<50 het SNPs, found {n_het})", "n_het": n_het}

    di_count = sum(1 for b in baf_values if 0.40 < b < 0.60)  # noqa: PLR2004
    tri_count = sum(1 for b in baf_values if (0.25 <= b <= 0.4) or (0.60 <= b <= 0.75))  # noqa: PLR2004
    di_frac = round(di_count / n_het * 100, 1)
    tri_frac = round(tri_count / n_het * 100, 1)

    if di_count > tri_count * 2.5:  # noqa: PLR2004
        label, confidence = "DIPLOID", f"({di_frac}% hets in 0.4-0.6 BAF band, n={n_het})"
    elif tri_count > di_count * 0.4:  # noqa: PLR2004
        label, confidence = "TRIPLOID", f"(diplo={di_frac}%, tri={tri_frac}%, n={n_het})"
    elif di_count > tri_count * 1.5:  # noqa: PLR2004
        label, confidence = "LIKELY_DIPLOID", f"({di_frac}% in 0.4-0.6 band, {tri_frac}% in triploid bands, n={n_het})"
    else:
        label, confidence = "INCONCLUSIVE", f"(diplo={di_frac}%, tri={tri_frac}%, n={n_het})"

    return {"label": label, "confidence": confidence, "n_het": n_het, "di_frac": di_frac, "tri_frac": tri_frac}


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
        f"  Sex:                {cr['sex_label']} (X ratio={cr['x_ratio']:.3f}, Y ratio={cr['y_ratio']:.3f})",
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

    for entry in cr["per_chrom"]:
        icon = {"sex": "."}.get(entry["flag"], " ")
        lines.append(f"  {entry['chrom']:<6s} ploidy={entry['ploidy']:.3f}  cov={entry['mean_cov']:.2f}x  {icon}")

    lines.extend(
        [
            "",
            "  (. = sex chromosome)",
            "",
        ]
    )

    any_warn = False
    for entry in cr["per_chrom"]:
        if entry["flag"] not in ("acro", "sex"):
            dev = abs(entry["ploidy"] - 2.0)
            if dev >= 0.35:  # noqa: PLR2004
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
    parser.add_argument("--het-sample-count", type=int, default=5000, help="Max het SNPs to sample for BAF (mode 1)")
    parser.add_argument(
        "--sex-chromosomes",
        nargs="+",
        default=list(DEFAULT_SEX_CHROMOSOMES),
        help="Sex chromosome names to exclude from autosomal baseline; defaults to chrX chrY X Y",
    )
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args(argv)

    if args.vcf and args.mosdepth_summary:
        parser.error("Provide either --vcf or --mosdepth-summary, not both.")
    if not args.vcf and not args.mosdepth_summary:
        parser.error("Provide either --vcf or --mosdepth-summary.")

    if args.vcf:
        print(f"[estimate_ploidy] Mode 1 (VCF): {args.vcf}", file=sys.stderr)
        coverage_result, baf_result = estimate_ploidy_from_vcf(
            args.vcf, args.sample_id, args.het_sample_count, args.sex_chromosomes
        )
        source_path = args.vcf
    else:
        print(f"[estimate_ploidy] Mode 2 (mosdepth): {args.mosdepth_summary}", file=sys.stderr)
        mosdepth_df = parse_mosdepth_summary(args.mosdepth_summary)
        coverage_result = estimate_ploidy_from_coverage(mosdepth_df, args.sex_chromosomes)
        baf_result = None
        source_path = args.mosdepth_summary

    print(
        f"[estimate_ploidy] Karyotype={coverage_result['karyotype']}, "
        f"sex={coverage_result['sex_label']}, auto_mean={coverage_result['auto_mean']:.1f}x",
        file=sys.stderr,
    )

    report_path = write_report(args.sample_id, coverage_result, baf_result, source_path, args.output_dir)
    print(report_path.read_text())
    print(f"[estimate_ploidy] Report: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
