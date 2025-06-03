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

# ───────────────────────────── constants ──────────────────────────────────
# Configuration keys
KEY_FIELD = "field"
KEY_OP = "op"
KEY_TYPE = "type"
KEY_VALUE = "value"
KEY_VALUES = "values"
KEY_VALUE_FIELD = "value_field"
KEY_NAME = "name"
KEY_FILTERS = "filters"
KEY_DOWNSAMPLE = "downsample"
KEY_SIZE = "size"
KEY_METHOD = "method"
KEY_SEED = "seed"

# Filter types
TYPE_QUALITY = "quality"
TYPE_REGION = "region"
TYPE_LABEL = "label"

# Downsample methods
METHOD_HEAD = "head"
METHOD_RANDOM = "random"

# Column prefixes
COL_PREFIX_FILTER = "__filter_"
COL_FILTER_FINAL = "__filter_final"
COL_ROW_NUM_FILTERED = "__row_num_filtered"
COL_FILTER_DOWNSAMPLE = "__filter_downsample"

# Statistics keys
STAT_RAW = "raw"
STAT_DOWNSAMPLE = "downsample"
STAT_PATTERN = "pattern"
STAT_COUNT = "count"
STAT_ROWS = "rows"

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
    if KEY_FIELD not in rule:
        raise ValueError(f"Filter {index} missing required '{KEY_FIELD}' key")
    if KEY_OP not in rule:
        raise ValueError(f"Filter {index} missing required '{KEY_OP}' key")
    if KEY_TYPE not in rule:
        raise ValueError(f"Filter {index} missing required '{KEY_TYPE}' key")

    # Check operator
    if rule[KEY_OP] not in _OPS:
        raise ValueError(f"Filter {index} has unsupported operator: {rule[KEY_OP]}")

    # Check type
    valid_types = {TYPE_QUALITY, TYPE_REGION, TYPE_LABEL}
    if rule[KEY_TYPE] not in valid_types:
        raise ValueError(f"Filter {index} has invalid type '{rule[KEY_TYPE]}'. Must be one of: {valid_types}")

    # Check value/value_field
    if KEY_VALUE not in rule and KEY_VALUES not in rule and KEY_VALUE_FIELD not in rule:
        raise ValueError(f"Filter {index} must have either '{KEY_VALUE}', '{KEY_VALUES}', or '{KEY_VALUE_FIELD}'")


def _validate_downsample(ds: dict[str, Any]) -> None:
    """Validate downsample configuration."""
    if not isinstance(ds, dict):
        raise ValueError(f"'{KEY_DOWNSAMPLE}' must be a dictionary")
    if KEY_SIZE not in ds:
        raise ValueError(f"'{KEY_DOWNSAMPLE}' must have '{KEY_SIZE}' key")
    if not isinstance(ds[KEY_SIZE], int) or ds[KEY_SIZE] <= 0:
        raise ValueError(f"'{KEY_DOWNSAMPLE}.{KEY_SIZE}' must be a positive integer")

    method = ds.get(KEY_METHOD, METHOD_RANDOM)
    if method not in {METHOD_HEAD, METHOD_RANDOM}:
        raise ValueError(f"Invalid downsample method: {method}. Must be '{METHOD_HEAD}' or '{METHOD_RANDOM}'")


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
    if KEY_FILTERS not in cfg:
        raise ValueError(f"Configuration must contain '{KEY_FILTERS}' key")

    if not isinstance(cfg[KEY_FILTERS], list):
        raise ValueError(f"'{KEY_FILTERS}' must be a list")

    for i, rule in enumerate(cfg[KEY_FILTERS]):
        _validate_filter(rule, i)

    # Validate downsample if present
    if KEY_DOWNSAMPLE in cfg:
        _validate_downsample(cfg[KEY_DOWNSAMPLE])


def _mask_for_rule(featuremap_dataframe: pl.LazyFrame, rule: dict[str, Any]) -> pl.Expr:
    """Return a boolean expression for a single rule."""
    field = rule[KEY_FIELD]
    op = rule[KEY_OP]
    if op not in _OPS:
        raise ValueError(f"Unsupported op: {op}")
    if KEY_VALUE_FIELD in rule:
        rhs = pl.col(rule[KEY_VALUE_FIELD])
    else:
        rhs = rule.get(KEY_VALUE) or rule.get(KEY_VALUES)
    return _OPS[op](pl.col(field), rhs)


def _create_filter_columns(
    featuremap_dataframe: pl.LazyFrame,
    filters: list[dict[str, Any]],
) -> tuple[pl.LazyFrame, list[str]]:
    """Create binary columns for each filter."""
    filter_cols = []

    logger.info(f"Creating binary columns for {len(filters)} filters")

    for rule in filters:
        name = rule.get(KEY_NAME) or f"{rule[KEY_FIELD]}_{rule[KEY_OP]}"
        col_name = f"{COL_PREFIX_FILTER}{name}"
        filter_cols.append(col_name)

        mask_expr = _mask_for_rule(featuremap_dataframe, rule)
        featuremap_dataframe = featuremap_dataframe.with_columns(mask_expr.alias(col_name))

        logger.debug(f"Created filter column: {col_name} (type: {rule.get(KEY_TYPE, 'unknown')})")

    return featuremap_dataframe, filter_cols


def _create_downsample_column(
    featuremap_dataframe: pl.LazyFrame,
    filter_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[pl.LazyFrame, str | None]:
    """Create downsample column for rows passing all filters."""
    if KEY_DOWNSAMPLE not in cfg:
        return featuremap_dataframe, None

    logger.info("Creating downsample column")

    # Create combined filter mask
    all_filters_mask = pl.all_horizontal(filter_cols)

    # Add row number for rows passing all filters
    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.when(all_filters_mask)
        .then(pl.int_range(pl.len()).over(all_filters_mask))
        .otherwise(None)
        .alias(COL_ROW_NUM_FILTERED)
    )

    # Create downsample mask
    size = cfg[KEY_DOWNSAMPLE][KEY_SIZE]
    method = cfg[KEY_DOWNSAMPLE].get(KEY_METHOD, METHOD_RANDOM)

    if method == METHOD_HEAD:
        downsample_expr = pl.col(COL_ROW_NUM_FILTERED) < size
    else:  # method == METHOD_RANDOM
        # For random sampling, we need to sample from the filtered rows
        # We'll mark rows for sampling based on their position after filtering
        downsample_expr = pl.col(COL_ROW_NUM_FILTERED).is_not_null()

    downsample_col = COL_FILTER_DOWNSAMPLE
    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.when(all_filters_mask).then(downsample_expr).otherwise(None).alias(downsample_col)
    ).drop(COL_ROW_NUM_FILTERED)

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

    return featuremap_dataframe.with_columns(final_expr.alias(COL_FILTER_FINAL))


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
    funnel = [(STAT_RAW, total_rows)]

    # Calculate cumulative filter effects
    cumulative_mask = pl.lit(value=True)
    for _i, (col, rule) in enumerate(zip(filter_cols, filters, strict=False)):
        name = rule.get(KEY_NAME) or f"{rule[KEY_FIELD]}_{rule[KEY_OP]}"
        cumulative_mask = cumulative_mask & pl.col(col)
        count = featuremap_dataframe.select(cumulative_mask.sum()).collect().item()
        funnel.append((name, count))

    # Add downsample to funnel if present
    if KEY_DOWNSAMPLE in cfg:
        # Get the count after all filters
        filtered_count = funnel[-1][1]
        downsample_size = cfg[KEY_DOWNSAMPLE][KEY_SIZE]
        final_count = min(filtered_count, downsample_size)
        funnel.append((STAT_DOWNSAMPLE, final_count))

    # Single effect statistics
    single_effect = {}
    for col, rule in zip(filter_cols, filters, strict=False):
        name = rule.get(KEY_NAME) or f"{rule[KEY_FIELD]}_{rule[KEY_OP]}"
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

        combos_df = (
            featuremap_dataframe.group_by(combo_expr.alias(STAT_PATTERN)).agg(pl.len().alias(STAT_COUNT)).collect()
        )

        # The combinations will naturally include all patterns that exist in the data
        # Including "00...00" if there are rows that fail all filters

        combos = combos_df.to_dicts()

    # Add filter types to the statistics
    filters_with_types = []
    for name, rows in funnel:
        filter_type = None
        if name not in {STAT_RAW, STAT_DOWNSAMPLE}:
            # Find the corresponding filter to get its type
            for rule in filters:
                rule_name = rule.get(KEY_NAME) or f"{rule[KEY_FIELD]}_{rule[KEY_OP]}"
                if rule_name == name:
                    filter_type = rule.get(KEY_TYPE)
                    break
        filters_with_types.append({KEY_NAME: name, STAT_ROWS: rows, KEY_TYPE: filter_type})

    return {
        KEY_FILTERS: filters_with_types,
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
    logger.info(f"Loaded configuration {cfg_path} with {len(cfg[KEY_FILTERS])} filters")

    # Create lazy frame for efficient processing
    featuremap_dataframe = pl.scan_parquet(in_path)

    # Get total row count
    total_rows = pl.read_parquet(in_path).height
    logger.info(f"Total rows in input: {total_rows:,}")

    # Create filter columns
    featuremap_dataframe, filter_cols = _create_filter_columns(featuremap_dataframe, cfg[KEY_FILTERS])

    # Create downsample column if needed
    featuremap_dataframe, downsample_col = _create_downsample_column(featuremap_dataframe, filter_cols, cfg)

    # Create final filter column
    featuremap_dataframe = _create_final_filter_column(featuremap_dataframe, filter_cols, downsample_col)

    # Calculate statistics
    stats = _calculate_statistics(featuremap_dataframe, filter_cols, downsample_col, cfg[KEY_FILTERS], total_rows, cfg)

    # Write outputs
    if out_path:
        logger.info(f"Writing filtered output to {out_path}")

        # First get all rows passing filters
        filtered_df = (
            featuremap_dataframe.filter(pl.all_horizontal(filter_cols))
            .select(pl.exclude(f"^{COL_PREFIX_FILTER}.*$"))
            .collect()
        )

        # Apply downsampling if configured
        if KEY_DOWNSAMPLE in cfg:
            size = cfg[KEY_DOWNSAMPLE][KEY_SIZE]
            if filtered_df.height > size:
                method = cfg[KEY_DOWNSAMPLE].get(KEY_METHOD, METHOD_RANDOM)
                if method == METHOD_HEAD:
                    filtered_df = filtered_df.head(size)
                else:  # random
                    seed = cfg[KEY_DOWNSAMPLE].get(KEY_SEED, 0)
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
