"""
Generate local sample MRD reports from the unit-test resources.

Usage (from repo root, with venv activated or via uv run):

    # All standard variants (output to ./sample_reports/)
    uv run python src/mrd/generate_sample_reports.py

    # Custom output directory
    uv run python src/mrd/generate_sample_reports.py --output-dir /tmp/mrd_reports

    # Single variant
    uv run python src/mrd/generate_sample_reports.py --only test_report

    # With optional filters enabled (adds QC sensitivity sections to QC report)
    uv run python src/mrd/generate_sample_reports.py --thresh-noise-lq-reads 3 --thresh-multi-read-pvalue 0.01

Report variants
---------------
test_report      Standard read filter (snvq>60 + mapq≥60), 5 synthetic controls
nomapq_report    No mapq filter (snvq>60 only),            5 synthetic controls
snvq40_report    Lenient SNV quality (snvq>40 + mapq≥60),  5 synthetic controls
example_report   Standard read filter,                     20 synthetic controls
"""

import argparse
import shutil
import sys
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent / "tests" / "resources" / "report"
SAMPLE = "Pa_46_333_LuNgs_08"
MATCHED_VCF = str(RESOURCES_DIR / "Pa_46_FreshFrozen.ann.chr20.filtered.vcf.gz")
CONTROL_VCF = str(RESOURCES_DIR / "Pa_67_FFPE.ann.chr20.filtered.vcf.gz")
COVERAGE_BED = str(RESOURCES_DIR / f"{SAMPLE}.regions.bed.gz")
FEATUREMAP = str(RESOURCES_DIR / f"{SAMPLE}.featuremap_df.10k.parquet")
SRSNV_META = str(RESOURCES_DIR / f"{SAMPLE}.srsnv_metadata.json")
SIG_FILTER = "(norm_coverage <= 2.5) and (norm_coverage >= 0.6)"

# Intersection parquets per signature type
MATCHED_PARQUET = str(RESOURCES_DIR / f"{SAMPLE}.Pa_46_FreshFrozen.matched.intersection.parquet")
CONTROL_PARQUET = str(RESOURCES_DIR / f"{SAMPLE}.Pa_67_FFPE.control.intersection.parquet")


def _syn_parquets(n: int) -> list[str]:
    return [str(RESOURCES_DIR / f"{SAMPLE}.syn{i}_Pa_46_FreshFrozen.db_control.intersection.parquet") for i in range(n)]


def _syn_vcfs(n: int) -> list[str]:
    return [
        str(RESOURCES_DIR / f"syn{i}_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz")
        for i in range(n)
    ]


# Report variant definitions: (basename, read_filter_query, n_syn_controls)
VARIANTS = {
    "test_report": ("filt>0 and snvq>60 and mapq>=60", 5),
    "nomapq_report": ("filt>0 and snvq>60", 5),
    "snvq40_report": ("filt>0 and snvq>40 and mapq>=60", 5),
    "example_report": ("filt>0 and snvq>60 and mapq>=60", 20),
}


def generate(
    basename: str,
    read_filter_query: str,
    n_syn: int,
    output_dir: Path,
    thresh_noise_lq_reads: int | None,
    thresh_noise_hq_exemption: int,
    thresh_multi_read_pvalue: float | None,
) -> None:
    from ugbio_mrd.generate_mrd_report import MrdReportInputs, generate_mrd_report

    out = output_dir / basename
    out.mkdir(parents=True, exist_ok=True)

    inputs = MrdReportInputs(
        intersected_featuremaps_parquet=[MATCHED_PARQUET, CONTROL_PARQUET] + _syn_parquets(n_syn),
        matched_signatures_vcf_files=[MATCHED_VCF],
        control_signatures_vcf_files=[CONTROL_VCF],
        db_control_signatures_vcf_files=_syn_vcfs(n_syn),
        coverage_bed=COVERAGE_BED,
        output_dir=str(out),
        output_basename=basename,
        featuremap_file=FEATUREMAP,
        srsnv_metadata_json=SRSNV_META,
        signature_filter_query=SIG_FILTER,
        read_filter_query=read_filter_query,
        thresh_noise_lq_reads=thresh_noise_lq_reads,
        thresh_noise_hq_exemption=thresh_noise_hq_exemption,
        thresh_multi_read_pvalue=thresh_multi_read_pvalue,
    )

    results_html, qc_html = generate_mrd_report(inputs)
    print(f"  analysis : {results_html}")
    print(f"  qc       : {qc_html}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--output-dir",
        default="sample_reports",
        help="Directory to write reports into (default: ./sample_reports)",
    )
    ap.add_argument(
        "--only",
        choices=list(VARIANTS),
        default=None,
        help="Generate only one variant instead of all four",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before generating",
    )
    ap.add_argument(
        "--thresh-noise-lq-reads",
        type=int,
        default=None,
        metavar="N",
        help="Enable noise locus filter: remove loci with >= N low-quality reads (default: disabled)",
    )
    ap.add_argument(
        "--thresh-noise-hq-exemption",
        type=int,
        default=3,
        metavar="N",
        help="HQ-read exemption for noise filter (default: 3)",
    )
    ap.add_argument(
        "--thresh-multi-read-pvalue",
        type=float,
        default=None,
        metavar="P",
        help="Enable multi-read locus filter: Bonferroni Poisson p-value threshold (default: disabled)",
    )
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    variants_to_run = {args.only: VARIANTS[args.only]} if args.only else VARIANTS

    for basename, (read_filter, n_syn) in variants_to_run.items():
        print(f"\n[{basename}]  read_filter={read_filter!r}  n_syn={n_syn}")
        generate(
            basename=basename,
            read_filter_query=read_filter,
            n_syn=n_syn,
            output_dir=output_dir,
            thresh_noise_lq_reads=args.thresh_noise_lq_reads,
            thresh_noise_hq_exemption=args.thresh_noise_hq_exemption,
            thresh_multi_read_pvalue=args.thresh_multi_read_pvalue,
        )

    print(f"\nDone. Reports written to: {output_dir.resolve()}/")


if __name__ == "__main__":
    main(sys.argv[1:])
