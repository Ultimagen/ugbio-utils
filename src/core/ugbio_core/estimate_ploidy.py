"""
Genome ploidy estimation from mosdepth summary and optional VCF BAF analysis.

Two orthogonal signals:
  1. Mosdepth coverage summary → per-chromosome ploidy + sex calling
  2. VCF B-allele frequency distribution → whole-genome ploidy (2N/3N/4N)

Usage:
    estimate_ploidy --mosdepth-summary <file> --sample-id <id> [--vcf <file>] [--output-dir <dir>]
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


# Known acrocentric chromosomes (lower coverage expected in short-read WGS)
_ACROCENTRICS_CHR = {"chr13", "chr14", "chr15", "chr21", "chr22"}
_ACROCENTRICS_NOCHR = {"13", "14", "15", "21", "22"}

# Regex patterns for autosome / sex / mito contigs
_AUTOSOME_CHR = re.compile(r"^chr(\d+)$")
_AUTOSOME_NOCHR = re.compile(r"^(\d+)$")
_SEX_CHR = re.compile(r"^(chr)?[XY]$", re.IGNORECASE)
_MITO = re.compile(r"^(chrM|MT)$", re.IGNORECASE)
_SKIP = re.compile(r"_random$|_decoy$|^chrUn|^HLA|^EBV|_alt$", re.IGNORECASE)


def _detect_chr_prefix(contigs: list[str]) -> bool:
    """Auto-detect whether contigs use 'chr' prefix."""
    return any(c.startswith("chr") for c in contigs if re.match(r"^chr\d+$", c))


def _autosome_number(contig: str, has_chr: bool) -> int | None:
    """Extract autosome number from contig name, or None."""
    pattern = _AUTOSOME_CHR if has_chr else _AUTOSOME_NOCHR
    m = pattern.match(contig)
    return int(m.group(1)) if m else None


def parse_mosdepth_summary(summary_path: str | Path) -> pd.DataFrame:
    """Parse mosdepth summary TSV into a DataFrame with columns: chrom, length, bases, mean, min, max."""
    df = pd.read_csv(summary_path, sep="\t")
    # Standardize column names (mosdepth uses 'chrom', 'length', 'bases', 'mean', 'min', 'max')
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def estimate_ploidy_from_coverage(
    df: pd.DataFrame,
) -> dict:
    """
    Estimate per-chromosome ploidy and sex from mosdepth summary.

    Returns a dict with:
        - per_chrom: list of dicts with chrom, ploidy, mean_cov, flag
        - sex_label: 'male' or 'female'
        - x_ploidy: 1 or 2
        - x_ratio: float
        - y_present: 'yes' or 'no'
        - auto_mean: weighted autosomal mean coverage
    """
    contigs = df["chrom"].tolist()
    has_chr = _detect_chr_prefix(contigs)
    acrocentrics = _ACROCENTRICS_CHR if has_chr else _ACROCENTRICS_NOCHR

    # Filter out alt/random/decoy/mito/total contigs
    df_filtered = df[~df["chrom"].str.contains(_SKIP, regex=True, na=False)].copy()
    df_filtered = df_filtered[~df_filtered["chrom"].str.match(_MITO, na=False)]
    df_filtered = df_filtered[df_filtered["chrom"] != "total"]

    # Separate autosomes and sex chromosomes
    chr_data = {}
    for _, row in df_filtered.iterrows():
        chrom = row["chrom"]
        auto_num = _autosome_number(chrom, has_chr)
        is_sex = bool(_SEX_CHR.match(chrom))
        if auto_num is not None or is_sex:
            chr_data[chrom] = {"length": row["length"], "mean": row["mean"]}

    # Weighted autosomal mean
    auto_sum_num = 0.0
    auto_sum_den = 0.0
    for chrom, data in chr_data.items():
        if _autosome_number(chrom, has_chr) is not None:
            auto_sum_num += data["length"] * data["mean"]
            auto_sum_den += data["length"]

    if auto_sum_den == 0:
        raise ValueError("Could not compute autosomal mean coverage — no autosomal contigs found")

    auto_mean = auto_sum_num / auto_sum_den

    # Sex calling
    x_name = "chrX" if has_chr else "X"
    y_name = "chrY" if has_chr else "Y"

    x_mean = chr_data.get(x_name, {}).get("mean", 0)
    x_ratio = x_mean / auto_mean if auto_mean > 0 else 0
    x_ploidy = 2 if x_ratio >= 0.65 else 1

    y_mean = chr_data.get(y_name, {}).get("mean", 0)
    y_present = "yes" if y_mean > 0.15 * auto_mean else "no"

    sex_label = "male" if x_ploidy == 1 else "female"

    # Per-chromosome ploidy
    per_chrom = []

    # Autosomes sorted by number
    auto_chroms = sorted(
        [c for c in chr_data if _autosome_number(c, has_chr) is not None],
        key=lambda c: _autosome_number(c, has_chr),
    )
    for chrom in auto_chroms:
        data = chr_data[chrom]
        ploidy = 2 * data["mean"] / auto_mean
        flag = "acro" if chrom in acrocentrics else ""
        per_chrom.append({"chrom": chrom, "ploidy": round(ploidy, 3), "mean_cov": round(data["mean"], 2), "flag": flag})

    # Sex chromosomes
    for sex_chrom in [x_name, y_name]:
        if sex_chrom in chr_data:
            data = chr_data[sex_chrom]
            ploidy = 2 * data["mean"] / auto_mean
            per_chrom.append(
                {"chrom": sex_chrom, "ploidy": round(ploidy, 3), "mean_cov": round(data["mean"], 2), "flag": "sex"}
            )

    return {
        "per_chrom": per_chrom,
        "sex_label": sex_label,
        "x_ploidy": x_ploidy,
        "x_ratio": round(x_ratio, 4),
        "y_present": y_present,
        "auto_mean": round(auto_mean, 2),
    }


def estimate_ploidy_from_baf(
    vcf_path: str | Path,
    het_sample_count: int = 5000,
) -> dict:
    """
    Estimate whole-genome ploidy from B-allele frequency distribution in a VCF.

    Samples het SNPs and checks BAF peak shape:
      - Peak at 0.5 → diploid
      - Shoulders at 0.33/0.67 → triploid

    Returns dict with: label, confidence, n_het, di_frac, tri_frac
    """
    # Use bcftools to extract het SNPs and compute BAF
    cmd = f"bcftools view -H {vcf_path}"
    try:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        return {"label": "BCFTOOLS_NOT_FOUND", "confidence": "(bcftools not available)", "n_het": 0}

    baf_values = []
    for line in proc.stdout:
        if len(baf_values) >= het_sample_count:
            break

        parts = line.strip().split("\t")
        if len(parts) < 10:
            continue

        # Check for het genotype
        gt_field = parts[9].split(":")[0] if parts[9] else ""
        if gt_field not in ("0/1", "0|1", "1|0"):
            continue

        # Extract AD field
        fmt_fields = parts[8].split(":")
        sample_fields = parts[9].split(":")
        ad_value = None
        for j, fmt in enumerate(fmt_fields):
            if fmt == "AD" and j < len(sample_fields):
                ad_value = sample_fields[j]
                break

        if not ad_value:
            continue

        ad_parts = ad_value.split(",")
        if len(ad_parts) < 2:
            continue

        try:
            ref_count = int(ad_parts[0])
            alt_count = int(ad_parts[1])
        except ValueError:
            continue

        total = ref_count + alt_count
        if total < 10:
            continue

        baf = alt_count / total
        if 0.1 <= baf <= 0.9:
            baf_values.append(baf)

    proc.terminate()
    proc.wait()

    n_het = len(baf_values)
    if n_het < 50:
        return {"label": "INSUFFICIENT_DATA", "confidence": f"(<50 het SNPs, found {n_het})", "n_het": n_het}

    # Classify based on BAF distribution
    di_count = sum(1 for b in baf_values if 0.40 <= b <= 0.60)
    tri_count = sum(1 for b in baf_values if (0.25 <= b <= 0.35) or (0.65 <= b <= 0.75))

    di_frac = round(di_count / n_het * 100, 1)
    tri_frac = round(tri_count / n_het * 100, 1)

    if di_count > tri_count * 2.5:
        label = "DIPLOID"
        confidence = f"({di_frac}% hets in 0.4-0.6 BAF band, n={n_het})"
    elif tri_count > di_count * 0.4:
        label = "TRIPLOID"
        confidence = f"(diplo={di_frac}%, tri={tri_frac}%, n={n_het})"
    elif di_count > tri_count * 1.5:
        label = "LIKELY_DIPLOID"
        confidence = f"({di_frac}% in 0.4-0.6 band, {tri_frac}% in triploid bands, n={n_het})"
    else:
        label = "INCONCLUSIVE"
        confidence = f"(diplo={di_frac}%, tri={tri_frac}%, n={n_het})"

    return {"label": label, "confidence": confidence, "n_het": n_het, "di_frac": di_frac, "tri_frac": tri_frac}


def write_report(
    sample_id: str,
    coverage_result: dict,
    baf_result: dict | None,
    mosdepth_summary_path: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write ploidy report and per-chromosome TSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{sample_id}.ploidy_report.txt"
    tsv_path = output_dir / f"{sample_id}.per_chromosome_ploidy.tsv"

    # BAF result
    if baf_result:
        wgs_ploidy = f"{baf_result['label']} {baf_result['confidence']}"
    else:
        wgs_ploidy = "N/A (no VCF provided)"

    cr = coverage_result

    # Write report
    lines = [
        "======================================================",
        f"  Genome Ploidy Report - {sample_id}",
        "======================================================",
        "",
        f"  Whole-genome ploidy: {wgs_ploidy}",
        f"  Sex:                {cr['sex_label']} (chrX ploidy={cr['x_ploidy']}, "
        f"chrX/auto ratio={cr['x_ratio']:.3f}, Y present={cr['y_present']})",
        f"  Autosomal mean cov: {cr['auto_mean']:.1f}x",
        f"  Mosdepth source:    {mosdepth_summary_path}",
        "",
        "  Per-chromosome ploidy (relative to autosome mean = 2.0):",
        "  --------------------------------------------------------",
    ]

    for entry in cr["per_chrom"]:
        icon = {"acro": "*", "sex": "."}.get(entry["flag"], " ")
        lines.append(f"  {entry['chrom']:<6s} ploidy={entry['ploidy']:.3f}  cov={entry['mean_cov']:.2f}x  {icon}")

    lines.extend([
        "",
        "  (* = acrocentric, lower coverage typical in short-read WGS)",
        "  (. = sex chromosome)",
        "",
    ])

    # Check for aneuploidy
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

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)

    # Write TSV
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chrom", "ploidy", "mean_cov", "flag"], delimiter="\t")
        writer.writeheader()
        writer.writerows(cr["per_chrom"])

    return report_path, tsv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate genome ploidy from mosdepth summary and optional VCF BAF analysis."
    )
    parser.add_argument("--mosdepth-summary", required=True, help="Path to mosdepth summary TSV file")
    parser.add_argument("--sample-id", required=True, help="Sample identifier for output files")
    parser.add_argument("--vcf", default=None, help="Optional filtered VCF for BAF-based ploidy estimation")
    parser.add_argument("--het-sample-count", type=int, default=5000, help="Number of het SNPs to sample for BAF")
    parser.add_argument("--output-dir", default=".", help="Output directory for report and TSV")

    args = parser.parse_args(argv)

    # Signal 1: Coverage-based ploidy
    print(f"[estimate_ploidy] Parsing mosdepth summary: {args.mosdepth_summary}", file=sys.stderr)
    df = parse_mosdepth_summary(args.mosdepth_summary)
    coverage_result = estimate_ploidy_from_coverage(df)
    print(
        f"[estimate_ploidy] Sex={coverage_result['sex_label']}, "
        f"auto_mean={coverage_result['auto_mean']:.1f}x",
        file=sys.stderr,
    )

    # Signal 2: BAF-based ploidy (optional)
    baf_result = None
    if args.vcf:
        print(f"[estimate_ploidy] Running BAF analysis on: {args.vcf}", file=sys.stderr)
        baf_result = estimate_ploidy_from_baf(args.vcf, args.het_sample_count)
        print(f"[estimate_ploidy] BAF result: {baf_result['label']}", file=sys.stderr)

    # Write outputs
    report_path, tsv_path = write_report(
        args.sample_id, coverage_result, baf_result, args.mosdepth_summary, args.output_dir
    )

    # Print report to stdout
    print(report_path.read_text())
    print(f"[estimate_ploidy] Report: {report_path}", file=sys.stderr)
    print(f"[estimate_ploidy] TSV: {tsv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
