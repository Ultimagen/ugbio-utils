"""Prepare annotation VCFs for snvfind: add Flag fields, convert to BCF, create filter JSONs.

Usage::

    prepare_annotation_vcfs \
        --exclude-vcf exclude.vcf.gz --exclude-field EXCLUDE_TRAINING \
        --include-vcf include.vcf.gz --include-field INCLUDE_INFERENCE \
        --pcawg-vcf pcawg.vcf.gz --pcawg-field PCAWG \
        --read-filters read_filters.json \
        --output-dir .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger


def _add_flag_and_convert_to_bcf(vcf_path: str, field_name: str, output_bcf: str) -> None:
    """Add a Flag INFO field to every record in a VCF and convert to indexed BCF."""
    logger.info(f"Converting {vcf_path} to BCF with Flag field '{field_name}'")

    cmd = (
        f"(bcftools view -h {vcf_path} | grep -v '^#CHROM';"
        f" echo '##INFO=<ID={field_name},Number=0,Type=Flag,Description=\"{field_name} annotation flag\">';"
        f" bcftools view -h {vcf_path} | grep '^#CHROM';"
        f" bcftools view -H {vcf_path} | awk -F'\\t' -v OFS='\\t'"
        f' \'{{if($8==".") $8="{field_name}"; else $8=$8";{field_name}"; print}}\''
        f") | bcftools view -Ob -o {output_bcf}"
    )
    print_and_execute(cmd)
    print_and_execute(f"bcftools index {output_bcf}")
    logger.info(f"Created {output_bcf}")


def _inject_exclusion_filter(filters_json: dict, field_name: str) -> dict:
    """Add an is_null exclusion filter for the given field to both filter sets."""
    entry = {"name": f"not_in_{field_name}", "type": "region", "field": field_name, "op": "is_null"}
    if "filters_full_output" in filters_json:
        filters_json["filters_full_output"].append(entry)
    if "filters_random_sample" in filters_json:
        filters_json["filters_random_sample"].append(entry)
    logger.info(f"Injected is_null filter for '{field_name}' into read_filters JSON")
    return filters_json


def _create_inference_filters(fields: list[str], output_path: str) -> None:
    """Create inference_filters.json with any_not_null for the given fields."""
    filters = {
        "filters_inference": [{"name": "in_inference_set", "type": "region", "op": "any_not_null", "fields": fields}]
    }
    Path(output_path).write_text(json.dumps(filters, indent=2) + "\n")
    logger.info(f"Created {output_path} with any_not_null on {fields}")


def _update_coverage_threshold(filters_json: dict, coverage_threshold: int) -> dict:
    """Update coverage_le_max filter value in both filter sets."""
    for key in ("filters_full_output", "filters_random_sample"):
        if key in filters_json:
            for entry in filters_json[key]:
                if entry.get("name") == "coverage_le_max":
                    entry["value"] = coverage_threshold
    logger.info(f"Updated coverage_le_max threshold to {coverage_threshold}")
    return filters_json


def run(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    read_filters = None
    if args.read_filters:
        with open(args.read_filters) as f:
            read_filters = json.load(f)
        if args.coverage_threshold is not None:
            read_filters = _update_coverage_threshold(read_filters, args.coverage_threshold)

    inference_fields: list[str] = []

    if args.exclude_vcf:
        bcf_path = str(output_dir / "exclude_annot.bcf")
        _add_flag_and_convert_to_bcf(args.exclude_vcf, args.exclude_field, bcf_path)
        if read_filters:
            read_filters = _inject_exclusion_filter(read_filters, args.exclude_field)

    if args.include_vcf:
        bcf_path = str(output_dir / "include_annot.bcf")
        _add_flag_and_convert_to_bcf(args.include_vcf, args.include_field, bcf_path)
        inference_fields.append(args.include_field)

    if args.pcawg_vcf:
        bcf_path = str(output_dir / "pcawg_annot.bcf")
        _add_flag_and_convert_to_bcf(args.pcawg_vcf, args.pcawg_field, bcf_path)
        if read_filters:
            read_filters = _inject_exclusion_filter(read_filters, args.pcawg_field)
        inference_fields.append(args.pcawg_field)

    if read_filters:
        out_json = str(output_dir / "read_filters_with_max_coverage.json")
        Path(out_json).write_text(json.dumps(read_filters, indent=2) + "\n")
        logger.info(f"Written augmented read_filters to {out_json}")

    if inference_fields:
        _create_inference_filters(inference_fields, str(output_dir / "inference_filters.json"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare annotation VCFs for snvfind")
    parser.add_argument("--exclude-vcf", help="Exclude-from-training annotation VCF")
    parser.add_argument("--exclude-field", default="EXCLUDE_TRAINING", help="Field name for exclusion flag")
    parser.add_argument("--include-vcf", help="Include-in-inference annotation VCF")
    parser.add_argument("--include-field", default="INCLUDE_INFERENCE", help="Field name for inclusion flag")
    parser.add_argument("--pcawg-vcf", help="PCAWG annotation VCF")
    parser.add_argument("--pcawg-field", default="PCAWG", help="Field name for PCAWG flag")
    parser.add_argument("--read-filters", help="Input read_filters JSON to augment with exclusion filters")
    parser.add_argument(
        "--coverage-threshold", type=int, default=None, help="Coverage threshold to update in read_filters"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for BCF files and JSON")
    return parser.parse_args(argv)


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
