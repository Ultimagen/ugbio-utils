"""Merge per-shard snvfind stats JSON files into a single unified stats file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ugbio_core.logger import logger


def _sum_filters(sections: list[dict[str, Any]], filter_names: list[str]) -> dict[str, dict[str, Any]]:
    """Sum funnel/pass values across shards for each filter."""
    merged: dict[str, dict[str, Any]] = {}
    for name in filter_names:
        entry = dict(sections[0]["filters"][name])
        entry["funnel"] = 0
        if "pass" in entry:
            entry["pass"] = 0
        merged[name] = entry

    for section in sections:
        section_filters = section["filters"]
        if list(section_filters.keys()) != filter_names:
            raise ValueError(
                f"Filter mismatch across shards: expected {filter_names}, got {list(section_filters.keys())}"
            )
        for name in filter_names:
            merged[name]["funnel"] += section_filters[name]["funnel"]
            if "pass" in section_filters[name]:
                merged[name]["pass"] += section_filters[name]["pass"]

    return merged


def _sum_combinations(sections: list[dict[str, Any]]) -> tuple[dict[str, int], int]:
    """Sum combination counts across shards."""
    merged: dict[str, int] = {}
    total = 0
    for section in sections:
        if "combinations" in section:
            for pattern, count in section["combinations"].items():
                merged[pattern] = merged.get(pattern, 0) + count
        if "combinations_total" in section:
            total += section["combinations_total"]
    return merged, total


def _merge_stats_section(sections: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge multiple stats sections by summing numeric fields.

    Each section has:
        filters: dict of filter_name -> {funnel: N, pass: N, name, type, field, op, value, ...}
        combinations: dict of binary_pattern -> count
        combinations_total: int
    """
    if not sections:
        raise ValueError("No sections to merge")

    filter_names = list(sections[0]["filters"].keys())
    merged_filters = _sum_filters(sections, filter_names)
    merged_combinations, merged_total = _sum_combinations(sections)

    result: dict[str, Any] = {"filters": merged_filters}
    if merged_combinations:
        result["combinations"] = merged_combinations
        result["combinations_total"] = merged_total

    return result


def merge_snvfind_stats(input_files: list[str | Path], output_file: str | Path) -> None:
    """
    Merge multiple snvfind stats JSON files into one.

    Handles the unified format (with filters_full_output/filters_random_sample sections)
    and the simple format (section directly at top level).

    Parameters
    ----------
    input_files
        Paths to per-shard stats JSON files
    output_file
        Path to write merged JSON
    """
    all_data = []
    for f in input_files:
        with open(f, encoding="utf-8") as fh:
            all_data.append(json.load(fh))

    if not all_data:
        raise ValueError("No input files provided")

    first = all_data[0]
    is_unified = "filters_full_output" in first

    if is_unified:
        full_output_sections = [d["filters_full_output"] for d in all_data]
        random_sample_sections = [d["filters_random_sample"] for d in all_data]
        merged = {
            "filters_full_output": _merge_stats_section(full_output_sections),
            "filters_random_sample": _merge_stats_section(random_sample_sections),
        }
    else:
        merged = _merge_stats_section(all_data)

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)

    logger.info("Merged %d stats files into %s", len(all_data), output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-shard snvfind stats JSONs")
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        dest="inputs",
        help="Input stats JSON file (repeat for each shard)",
    )
    parser.add_argument("--output", required=True, help="Output merged JSON file path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    merge_snvfind_stats(args.inputs, args.output)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
