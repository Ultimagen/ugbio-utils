from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import polars as pl

try:
    from ugbio_core.logger import logger
except ImportError:
    # Fallback to standard logging if ugbio_core is not available
    logger = logging.getLogger(__name__)

# ───────────────────────────── utilities ──────────────────────────────────
_OPS = {
    "eq": lambda c, v: c == v,
    "ne": lambda c, v: c != v,
    "lt": lambda c, v: c < v,
    "le": lambda c, v: c <= v,
    "gt": lambda c, v: c > v,
    "ge": lambda c, v: c >= v,
    "in": lambda c, v: c.is_in(v),
    "not_in": lambda c, v: ~c.is_in(v),
    "between": lambda c, v: (c >= v[0]) & (c <= v[1]),
    "regex": lambda c, v: c.str.contains(v),
}


def _validate_filter(rule: dict[str, Any], index: int) -> None:
    """Validate a single filter rule."""
    if not isinstance(rule, dict):
        raise ValueError(f"Filter {index} must be a dictionary")

    # Check required fields
    if "field" not in rule:
        raise ValueError(f"Filter {index} missing required 'field' key")
    if "op" not in rule:
        raise ValueError(f"Filter {index} missing required 'op' key")
    if "type" not in rule:
        raise ValueError(f"Filter {index} missing required 'type' key")

    # Check operator
    if rule["op"] not in _OPS:
        raise ValueError(f"Filter {index} has unsupported operator: {rule['op']}")

    # Check type
    valid_types = {"quality", "region", "label"}
    if rule["type"] not in valid_types:
        raise ValueError(f"Filter {index} has invalid type '{rule['type']}'. Must be one of: {valid_types}")

    # Check value/value_field
    if "value" not in rule and "values" not in rule and "value_field" not in rule:
        raise ValueError(f"Filter {index} must have either 'value', 'values', or 'value_field'")


def _validate_downsample(ds: dict[str, Any]) -> None:
    """Validate downsample configuration."""
    if not isinstance(ds, dict):
        raise ValueError("'downsample' must be a dictionary")
    if "size" not in ds:
        raise ValueError("'downsample' must have 'size' key")
    if not isinstance(ds["size"], int) or ds["size"] <= 0:
        raise ValueError("'downsample.size' must be a positive integer")

    method = ds.get("method", "random")
    if method not in {"head", "random"}:
        raise ValueError(f"Invalid downsample method: {method}. Must be 'head' or 'random'")


def validate_filter_config(cfg: dict[str, Any]) -> None:
    """
    Validate the filter configuration JSON structure.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary loaded from JSON

    Raises
    ------
    ValueError
        If the configuration is invalid
    """
    if "filters" not in cfg:
        raise ValueError("Configuration must contain 'filters' key")

    if not isinstance(cfg["filters"], list):
        raise ValueError("'filters' must be a list")

    for i, rule in enumerate(cfg["filters"]):
        _validate_filter(rule, i)

    # Validate downsample if present
    if "downsample" in cfg:
        _validate_downsample(cfg["downsample"])


def _mask_for_rule(featuremap_dataframe: pl.LazyFrame, rule: dict[str, Any]) -> pl.Expr:
    """Return a boolean expression for a single rule."""
    field = rule["field"]
    op = rule["op"]
    if op not in _OPS:
        raise ValueError(f"Unsupported op: {op}")
    if "value_field" in rule:
        rhs = pl.col(rule["value_field"])
    else:
        rhs = rule.get("value") or rule.get("values")
    return _OPS[op](pl.col(field), rhs)


def _create_filter_columns(
    featuremap_dataframe: pl.LazyFrame,
    filters: list[dict[str, Any]],
) -> tuple[pl.LazyFrame, list[str]]:
    """Create binary columns for each filter."""
    filter_cols = []

    logger.info(f"Creating binary columns for {len(filters)} filters")

    for rule in filters:
        name = rule.get("name") or f"{rule['field']}_{rule['op']}"
        col_name = f"__filter_{name}"
        filter_cols.append(col_name)

        mask_expr = _mask_for_rule(featuremap_dataframe, rule)
        featuremap_dataframe = featuremap_dataframe.with_columns(mask_expr.alias(col_name))

        logger.debug(f"Created filter column: {col_name} (type: {rule.get('type', 'unknown')})")

    return featuremap_dataframe, filter_cols


def _create_downsample_column(
    featuremap_dataframe: pl.LazyFrame,
    filter_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[pl.LazyFrame, str | None]:
    """Create downsample column for rows passing all filters."""
    if "downsample" not in cfg:
        return featuremap_dataframe, None

    logger.info("Creating downsample column")

    # Create combined filter mask
    all_filters_mask = pl.all_horizontal(filter_cols)

    # Add row number for rows passing all filters
    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.when(all_filters_mask)
        .then(pl.int_range(pl.len()).over(all_filters_mask))
        .otherwise(None)
        .alias("__row_num_filtered")
    )

    # Create downsample mask
    size = cfg["downsample"]["size"]
    method = cfg["downsample"].get("method", "random")

    if method == "head":
        downsample_expr = pl.col("__row_num_filtered") < size
    else:  # method == "random"
        # For random sampling, we need to sample from the filtered rows
        # We'll mark rows for sampling based on their position after filtering
        downsample_expr = pl.col("__row_num_filtered").is_not_null()

    downsample_col = "__filter_downsample"
    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.when(all_filters_mask).then(downsample_expr).otherwise(None).alias(downsample_col)
    ).drop("__row_num_filtered")

    logger.debug(f"Created downsample column: {downsample_col}")

    return featuremap_dataframe, downsample_col


def _create_final_filter_column(
    featuremap_dataframe: pl.LazyFrame,
    filter_cols: list[str],
    downsample_col: str | None,
) -> pl.LazyFrame:
    """Create final AND column combining all filters."""
    logger.info("Creating final filter column combining all filters")

    # For the final column, treat None in downsample as True if downsample doesn't exist
    if downsample_col:
        final_expr = pl.all_horizontal(filter_cols) & pl.col(downsample_col).fill_null(value=False)
    else:
        final_expr = pl.all_horizontal(filter_cols)

    return featuremap_dataframe.with_columns(final_expr.alias("__filter_final"))


def _calculate_statistics(
    featuremap_dataframe: pl.LazyFrame,
    filter_cols: list[str],
    downsample_col: str | None,
    filters: list[dict[str, Any]],
    total_rows: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Calculate filter statistics using binary columns."""
    logger.info("Calculating filter statistics")

    # Funnel statistics
    funnel = [("raw", total_rows)]

    # Calculate cumulative filter effects
    cumulative_mask = pl.lit(value=True)
    for _i, (col, rule) in enumerate(zip(filter_cols, filters, strict=False)):
        name = rule.get("name") or f"{rule['field']}_{rule['op']}"
        cumulative_mask = cumulative_mask & pl.col(col)
        count = featuremap_dataframe.select(cumulative_mask.sum()).collect().item()
        funnel.append((name, count))

    # Add downsample to funnel if present
    if "downsample" in cfg:
        # Get the count after all filters
        filtered_count = funnel[-1][1]
        downsample_size = cfg["downsample"]["size"]
        final_count = min(filtered_count, downsample_size)
        funnel.append(("downsample", final_count))

    # Single effect statistics
    single_effect = {}
    for col, rule in zip(filter_cols, filters, strict=False):
        name = rule.get("name") or f"{rule['field']}_{rule['op']}"
        count = featuremap_dataframe.select(pl.col(col).sum()).collect().item()
        single_effect[name] = count

    # Combination statistics
    logger.info("Calculating combination statistics")

    if len(filter_cols) == 0:
        # No filters, so no combinations
        combos = []
    else:
        # Create binary string representation for combinations
        # Cast boolean to int (0/1) then to string
        combo_expr = pl.concat_str(
            [pl.when(pl.col(col)).then(pl.lit("1")).otherwise(pl.lit("0")) for col in filter_cols]
        )

        combos_df = featuremap_dataframe.group_by(combo_expr.alias("pattern")).agg(pl.len().alias("count")).collect()

        # The combinations will naturally include all patterns that exist in the data
        # Including "00...00" if there are rows that fail all filters

        combos = combos_df.to_dicts()

    # Add filter types to the statistics
    filters_with_types = []
    for name, rows in funnel:
        filter_type = None
        if name not in {"raw", "downsample"}:
            # Find the corresponding filter to get its type
            for rule in filters:
                rule_name = rule.get("name") or f"{rule['field']}_{rule['op']}"
                if rule_name == name:
                    filter_type = rule.get("type")
                    break
        filters_with_types.append({"name": name, "rows": rows, "type": filter_type})

    return {
        "filters": filters_with_types,
        "single_effect": single_effect,
        "combinations": combos,
    }


def filter_parquet(
    in_path: str,
    out_path: str | None,
    out_path_full: str | None,
    cfg_path: str,
    stats_path: str,
) -> None:
    """
    Filter a parquet file based on configuration.

    Parameters
    ----------
    in_path : str
        Input parquet file path
    out_path : str | None
        Output path for filtered data (optional)
    out_path_full : str | None
        Output path for full data with filter columns (optional)
    cfg_path : str
        Configuration JSON file path
    stats_path : str
        Output statistics JSON file path
    """
    logger.info(f"Starting filter_parquet: input={in_path}")

    # Load and validate configuration
    with open(cfg_path) as f:
        cfg = json.load(f)

    validate_filter_config(cfg)
    logger.info(f"Loaded configuration {cfg_path} with {len(cfg['filters'])} filters")

    # Create lazy frame for efficient processing
    featuremap_dataframe = pl.scan_parquet(in_path)

    # Get total row count
    total_rows = pl.read_parquet(in_path).height
    logger.info(f"Total rows in input: {total_rows:,}")

    # Create filter columns
    featuremap_dataframe, filter_cols = _create_filter_columns(featuremap_dataframe, cfg["filters"])

    # Create downsample column if needed
    featuremap_dataframe, downsample_col = _create_downsample_column(featuremap_dataframe, filter_cols, cfg)

    # Create final filter column
    featuremap_dataframe = _create_final_filter_column(featuremap_dataframe, filter_cols, downsample_col)

    # Calculate statistics
    stats = _calculate_statistics(featuremap_dataframe, filter_cols, downsample_col, cfg["filters"], total_rows, cfg)

    # Write outputs
    if out_path:
        logger.info(f"Writing filtered output to {out_path}")

        # First get all rows passing filters
        filtered_df = (
            featuremap_dataframe.filter(pl.all_horizontal(filter_cols)).select(pl.exclude("^__filter_.*$")).collect()
        )

        # Apply downsampling if configured
        if "downsample" in cfg:
            size = cfg["downsample"]["size"]
            if filtered_df.height > size:
                method = cfg["downsample"].get("method", "random")
                if method == "head":
                    filtered_df = filtered_df.head(size)
                else:  # random
                    seed = cfg["downsample"].get("seed", 0)
                    filtered_df = filtered_df.sample(n=size, shuffle=True, seed=seed)
                logger.info(f"Downsampled from {filtered_df.height} to {size} rows")

        filtered_df.write_parquet(out_path)
        logger.info(f"Wrote filtered data: {filtered_df.height:,} rows")

    if out_path_full:
        logger.info(f"Writing full output with filter columns to {out_path_full}")
        full_df = featuremap_dataframe.collect()
        full_df.write_parquet(out_path_full)
        logger.info(f"Wrote full data with filters: {full_df.height:,} rows")

    # Write statistics
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote statistics to {stats_path}")


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Filter / down-sample featuremap Parquet")
    ap.add_argument("--in", dest="inp", required=True, help="input parquet")
    ap.add_argument("--out", help="output parquet with filtered rows (optional)")
    ap.add_argument("--out-full", help="output parquet with all rows and filter columns (optional)")
    ap.add_argument("--config", required=True, help="JSON with filters + downsample")
    ap.add_argument("--stats", required=True, help="output JSON with statistics")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main() -> None:
    args = _build_cli().parse_args()

    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Also set the handler level
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    # Validate that at least one output is specified
    if not args.out and not args.out_full:
        logger.error("At least one of --out or --out-full must be specified")
        raise ValueError("No output file specified")

    filter_parquet(args.inp, args.out, args.out_full, args.config, args.stats)


if __name__ == "__main__":
    import logging

    main()
