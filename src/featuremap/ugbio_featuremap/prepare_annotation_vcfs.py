"""Prepare read_filters JSON and inference_filters JSON for annotation-based filtering.

Handles:
- Coverage threshold update in read_filters JSON
- Annotation exclusion filter injection (is_null for EXCLUDE_TRAINING, PCAWG)
- Inference inclusion filter creation (any_not_null for INCLUDE_INFERENCE, PCAWG)

Usage::

    prepare_annotation_vcfs \
        --exclude-field EXCLUDE_TRAINING \
        --include-field INCLUDE_INFERENCE \
        --pcawg-field PCAWG \
        --read-filters read_filters.json \
        --coverage-threshold 42 \
        --output-dir .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ugbio_core.logger import logger

from ugbio_featuremap.filter_dataframe import (
    KEY_FIELD,
    KEY_FIELDS,
    KEY_NAME,
    KEY_OP,
    KEY_TYPE,
    OP_ANY_NOT_NULL,
    OP_IS_NULL,
    TYPE_REGION,
)


def _update_coverage_threshold(filters_json: dict, coverage_threshold: int) -> dict:
    """Update coverage_le_max filter value in both filter sets."""
    for key in ("filters_full_output", "filters_random_sample"):
        if key in filters_json:
            for entry in filters_json[key]:
                if entry.get("name") == "coverage_le_max":
                    entry["value"] = coverage_threshold
    logger.info(f"Updated coverage_le_max threshold to {coverage_threshold}")
    return filters_json


def _inject_exclusion_filter(filters_json: dict, field_name: str) -> dict:
    """Add an is_null exclusion filter for the given field to both filter sets."""
    entry = {KEY_NAME: f"not_in_{field_name}", KEY_TYPE: TYPE_REGION, KEY_FIELD: field_name, KEY_OP: OP_IS_NULL}
    if "filters_full_output" in filters_json:
        filters_json["filters_full_output"].append(entry)
    if "filters_random_sample" in filters_json:
        filters_json["filters_random_sample"].append(entry.copy())
    logger.info(f"Injected is_null filter for '{field_name}' into read_filters JSON")
    return filters_json


def _create_inference_filters(fields: list[str], output_path: str) -> None:
    """Create inference_filters.json with any_not_null for the given fields."""
    filters = {
        "filters_inference": [
            {KEY_NAME: "in_inference_set", KEY_TYPE: TYPE_REGION, KEY_OP: OP_ANY_NOT_NULL, KEY_FIELDS: fields}
        ]
    }
    Path(output_path).write_text(json.dumps(filters, indent=2) + "\n")
    logger.info(f"Created {output_path} with any_not_null on {fields}")


def _build_inference_fields(include_field: str | None, pcawg_field: str | None) -> list[str]:
    """Build the list of fields for inference inclusion filtering."""
    fields: list[str] = []
    if include_field:
        fields.append(include_field)
        # PCAWG contributes to inference inclusion only when include VCFs are also provided
        if pcawg_field:
            fields.append(pcawg_field)
    return fields


def run(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    read_filters = None
    if args.read_filters:
        with open(args.read_filters, encoding="utf-8") as f:
            read_filters = json.load(f)
        if args.coverage_threshold is not None:
            read_filters = _update_coverage_threshold(read_filters, args.coverage_threshold)

    if args.exclude_field and read_filters:
        read_filters = _inject_exclusion_filter(read_filters, args.exclude_field)

    if args.pcawg_field and read_filters:
        read_filters = _inject_exclusion_filter(read_filters, args.pcawg_field)

    if read_filters:
        out_json = str(output_dir / "read_filters_with_max_coverage.json")
        Path(out_json).write_text(json.dumps(read_filters, indent=2) + "\n")
        logger.info(f"Written augmented read_filters to {out_json}")

    inference_fields = _build_inference_fields(args.include_field, args.pcawg_field)
    if inference_fields:
        _create_inference_filters(inference_fields, str(output_dir / "inference_filters.json"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare filter JSONs for annotation-based filtering")
    parser.add_argument("--exclude-field", default=None, help="Field name for training exclusion (inject is_null)")
    parser.add_argument("--include-field", default=None, help="Field name for inference inclusion")
    parser.add_argument("--pcawg-field", default=None, help="Field name for PCAWG (exclude + include)")
    parser.add_argument("--read-filters", help="Input read_filters JSON to augment")
    parser.add_argument(
        "--coverage-threshold", type=int, default=None, help="Coverage threshold to update in read_filters"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for JSON files")
    return parser.parse_args(argv)


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
