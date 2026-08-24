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
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ACROCENTRICS_CHR = {"chr13", "chr14", "chr15", "chr21", "chr22"}
_ACROCENTRICS_NOCHR = {"13", "14", "15", "21", "22"}

_AUTOSOME_CHR = re.compile(r"^chr(\d+)$")
_AUTOSOME_NOCHR = re.compile(r"^(\d+)$")
_SEX_CHR = re.compile(r"^(chr)?[XY]$", re.IGNORECASE)
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


def _compute_ploidy_from_chr_data(chr_data: dict[str, dict], *, has_chr: bool) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Shared logic: given {chrom: {mean, length(optional)}} compute ploidy, karyotype."""
    acrocentrics = _ACROCENTRICS_CHR if has_chr else _ACROCENTRICS_NOCHR
    x_name = "chrX" if has_chr else "X"
    y_name = "chrY" if has_chr else "Y"

    auto_chroms = {c: d for c, d in chr_data.items() if _autosome_number(c, has_chr=has_chr) is not None}
    if not auto_chroms:
        raise ValueError("No autosomal contigs found")

    has_length = all("length" in d for d in auto_chroms.values())
    if has_length:
        auto_sum_num = sum(d["length"] * d["mean"] for d in auto_chroms.values())
        auto_sum_den = sum(d["length"] for d in auto_chroms.values())
        auto_mean = auto_sum_num / auto_sum_den
    else:
        auto_mean = float(np.median([d["mean"] for d in auto_chroms.values()]))

    if auto_mean == 0:
        raise ValueError("Autosomal mean coverage is 0; cannot compute ploidy")

    x_mean = chr_data.get(x_name, {}).get("mean", 0)
    y_mean = chr_data.get(y_name, {}).get("mean", 0)
    x_ratio = x_mean / auto_mean if auto_mean > 0 else 0
    y_ratio = y_mean / auto_mean if auto_mean > 0 else 0

    karyotype = _determine_karyotype(x_ratio, y_ratio)
    sex_label = _sex_label_from_karyotype(karyotype)

    per_chrom = []
    sorted_autos = sorted(auto_chroms.keys(), key=lambda c: _autosome_number(c, has_chr=has_chr))
    for chrom in sorted_autos:
        data = chr_data[chrom]
        ploidy = 2 * data["mean"] / auto_mean
        flag = "acro" if chrom in acrocentrics else ""
        per_chrom.append({"chrom": chrom, "ploidy": round(ploidy, 3), "mean_cov": round(data["mean"], 2), "flag": flag})

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
        "karyotype": karyotype,
        "x_ratio": round(x_ratio, 4),
        "y_ratio": round(y_ratio, 4),
        "auto_mean": round(auto_mean, 2),
    }


def estimate_ploidy_from_coverage(mosdepth_df: pd.DataFrame) -> dict:
    """Mode 2: estimate ploidy from mosdepth summary."""
    contigs = mosdepth_df["chrom"].tolist()
    has_chr = _detect_chr_prefix(contigs)

    df_filtered = mosdepth_df[~mosdepth_df["chrom"].str.contains(_SKIP, regex=True, na=False)].copy()
    df_filtered = df_filtered[~df_filtered["chrom"].str.match(_MITO, na=False)]
    df_filtered = df_filtered[df_filtered["chrom"] != "total"]

    chr_data = {}
    for _, row in df_filtered.iterrows():
        chrom = row["chrom"]
        if _autosome_number(chrom, has_chr=has_chr) is not None or bool(_SEX_CHR.match(chrom)):
            chr_data[chrom] = {"length": row["length"], "mean": row["mean"]}

    result = _compute_ploidy_from_chr_data(chr_data, has_chr=has_chr)
    result["source"] = "mosdepth"
    return result


def estimate_ploidy_from_vcf(  # noqa: C901, PLR0912, PLR0915
    vcf_path: str | Path, het_sample_count: int = 5000
) -> tuple[dict, dict]:
    """Mode 1: per-chr coverage from SNP DP + BAF. Returns (coverage_result, baf_result)."""
    import random  # noqa: PLC0415

    random.seed(42)  # noqa: S311

    cmd = f"bcftools view -H -v snps {vcf_path}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # noqa: S602

    chr_dps: dict[str, list[int]] = {}
    # Reservoir sampling for BAF: uniform random sample over autosomal het SNPs
    baf_reservoir: list[float] = []
    baf_seen = 0

    for line in proc.stdout:
        parts = line.strip().split("\t")
        if len(parts) < 10:  # noqa: PLR2004
            continue

        chrom = parts[0]
        if _SKIP.search(chrom) or _MITO.match(chrom):
            continue

        fmt_fields = parts[8].split(":")
        sample_fields = parts[9].split(":")
        dp_value = None
        ad_value = None
        for j, fmt in enumerate(fmt_fields):
            if fmt == "DP" and j < len(sample_fields):
                try:
                    dp_value = int(sample_fields[j])
                except ValueError:
                    pass  # skip unparseable DP fields
            if fmt == "AD" and j < len(sample_fields):
                ad_value = sample_fields[j]

        if dp_value is not None and dp_value > 0:
            chr_dps.setdefault(chrom, []).append(dp_value)

        # BAF: reservoir sampling over autosomal het SNPs only (chr1-22)
        has_chr_local = chrom.startswith("chr")
        is_autosome = _autosome_number(chrom, has_chr_local) is not None
        gt_field = sample_fields[0] if sample_fields else ""
        if is_autosome and gt_field in ("0/1", "0|1", "1|0") and ad_value:
            ad_parts = ad_value.split(",")
            if len(ad_parts) >= 2:  # noqa: PLR2004
                try:
                    ref_count, alt_count = int(ad_parts[0]), int(ad_parts[1])
                    total = ref_count + alt_count
                    if total >= 10:  # noqa: PLR2004
                        baf = alt_count / total
                        if 0.1 <= baf <= 0.9:  # noqa: PLR2004
                            baf_seen += 1
                            if len(baf_reservoir) < het_sample_count:
                                baf_reservoir.append(baf)
                            else:
                                idx = random.randint(0, baf_seen - 1)  # noqa: S311
                                if idx < het_sample_count:
                                    baf_reservoir[idx] = baf
                except ValueError:
                    pass  # skip unparseable AD fields

    proc.wait()
    if proc.returncode and proc.returncode != 0:
        stderr_text = proc.stderr.read() if proc.stderr else ""
        print(
            f"[estimate_ploidy] WARNING: bcftools exited with code {proc.returncode}: {stderr_text[:200]}",
            file=sys.stderr,
        )

    contigs = list(chr_dps.keys())
    has_chr = _detect_chr_prefix(contigs)

    chr_data = {}
    for chrom, dps in chr_dps.items():
        if (_autosome_number(chrom, has_chr=has_chr) is not None or bool(_SEX_CHR.match(chrom))) and len(dps) >= 20:  # noqa: PLR2004
            chr_data[chrom] = {"mean": float(np.median(dps))}

    coverage_result = _compute_ploidy_from_chr_data(chr_data, has_chr=has_chr)
    coverage_result["source"] = "VCF SNP median DP"

    baf_result = _classify_baf(baf_reservoir)
    return coverage_result, baf_result


def _classify_baf(baf_values: list[float]) -> dict:
    n_het = len(baf_values)
    if n_het < 50:  # noqa: PLR2004
        return {"label": "INSUFFICIENT_DATA", "confidence": f"(<50 het SNPs, found {n_het})", "n_het": n_het}

    di_count = sum(1 for b in baf_values if 0.40 <= b <= 0.60)  # noqa: PLR2004
    tri_count = sum(1 for b in baf_values if (0.25 <= b <= 0.35) or (0.65 <= b <= 0.75))  # noqa: PLR2004
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
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{sample_id}.ploidy_report.txt"
    tsv_path = output_dir / f"{sample_id}.per_chromosome_ploidy.tsv"

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
        f"  Autosomal mean cov: {cr['auto_mean']:.1f}x",
        f"  Coverage source:    {coverage_source}",
        f"  Input:              {source_path}",
        "",
        "  Per-chromosome ploidy (relative to autosome mean = 2.0):",
        "  --------------------------------------------------------",
    ]

    for entry in cr["per_chrom"]:
        icon = {"acro": "*", "sex": "."}.get(entry["flag"], " ")
        lines.append(f"  {entry['chrom']:<6s} ploidy={entry['ploidy']:.3f}  cov={entry['mean_cov']:.2f}x  {icon}")

    lines.extend(
        [
            "",
            "  (* = acrocentric, lower coverage typical in short-read WGS)",
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

    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chrom", "ploidy", "mean_cov", "flag"], delimiter="\t")
        writer.writeheader()
        writer.writerows(cr["per_chrom"])

    return report_path, tsv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate genome ploidy. Mode 1: --vcf (SNP DP + BAF). Mode 2: --mosdepth-summary (coverage only)."
    )
    parser.add_argument("--vcf", default=None, help="Mode 1: VCF input for coverage from SNP DP + BAF analysis")
    parser.add_argument("--mosdepth-summary", default=None, help="Mode 2: mosdepth summary for coverage-only ploidy")
    parser.add_argument("--sample-id", required=True, help="Sample identifier for output files")
    parser.add_argument("--het-sample-count", type=int, default=5000, help="Max het SNPs to sample for BAF (mode 1)")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args(argv)

    if args.vcf and args.mosdepth_summary:
        parser.error("Provide either --vcf or --mosdepth-summary, not both.")
    if not args.vcf and not args.mosdepth_summary:
        parser.error("Provide either --vcf or --mosdepth-summary.")

    if args.vcf:
        print(f"[estimate_ploidy] Mode 1 (VCF): {args.vcf}", file=sys.stderr)
        coverage_result, baf_result = estimate_ploidy_from_vcf(args.vcf, args.het_sample_count)
        source_path = args.vcf
    else:
        print(f"[estimate_ploidy] Mode 2 (mosdepth): {args.mosdepth_summary}", file=sys.stderr)
        mosdepth_df = parse_mosdepth_summary(args.mosdepth_summary)
        coverage_result = estimate_ploidy_from_coverage(mosdepth_df)
        baf_result = None
        source_path = args.mosdepth_summary

    print(
        f"[estimate_ploidy] Karyotype={coverage_result['karyotype']}, "
        f"sex={coverage_result['sex_label']}, auto_mean={coverage_result['auto_mean']:.1f}x",
        file=sys.stderr,
    )

    report_path, tsv_path = write_report(args.sample_id, coverage_result, baf_result, source_path, args.output_dir)
    print(report_path.read_text())
    print(f"[estimate_ploidy] Report: {report_path}", file=sys.stderr)
    print(f"[estimate_ploidy] TSV: {tsv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
