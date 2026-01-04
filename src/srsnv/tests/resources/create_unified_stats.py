#!/usr/bin/env python3
"""
Helper script to create unified stats file for testing from separate stats files.
This converts the old format (3 separate files) to the new unified format.
"""

import json
import sys


def convert_legacy_to_unified_stats(pos_stats_path, neg_stats_path, raw_stats_path, output_path):
    """
    Convert legacy separate stats files to unified format.
    Args:
        pos_stats_path: Path to positive stats JSON (will become f2_filters)
        neg_stats_path: Path to negative stats JSON (not used in new format)
        raw_stats_path: Path to raw featuremap stats JSON (will become filters)
        output_path: Path for output unified JSON
    """

    # Read the separate files
    with open(pos_stats_path) as f:
        pos_stats = json.load(f)

    with open(raw_stats_path) as f:
        raw_stats = json.load(f)

    # Note: neg_stats_path is not used in the new format

    # Convert to new unified format
    def convert_filters_to_dict(filters_list):
        """Convert from list format to dict format"""
        result = {}
        for filter_entry in filters_list:
            name = filter_entry["name"]
            filter_dict = {"funnel": filter_entry["rows"]}

            # Copy other fields
            for key in ["type", "field", "op", "value"]:
                if key in filter_entry:
                    filter_dict[key] = filter_entry[key]

            result[name] = filter_dict
        return result

    # Create unified format - single section with f2_filters (positive) and filters (negative/FP)
    unified_stats = {
        "filtering_stats_random_sample": {
            "f2_filters": convert_filters_to_dict(pos_stats["filters"]),
            "filters": convert_filters_to_dict(raw_stats["filters"]),
        }
    }

    # Write unified stats file
    with open(output_path, "w") as f:
        json.dump(unified_stats, f, indent=2)

    print(f"Created unified stats file: {output_path}")
    print(f"- f2_filters (TP): {pos_stats_path}")
    print(f"- filters (FP): {raw_stats_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python create_unified_stats.py <pos_stats> <neg_stats> <raw_stats> <output>")
        sys.exit(1)

    convert_legacy_to_unified_stats(*sys.argv[1:])
