from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

try:
    from ugbio_core.logger import logger
except ImportError:
    # Fallback to standard logging if ugbio_core is not available
    logger = logging.getLogger(__name__)
pl.enable_string_cache()
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
TYPE_MAPPING = "mapping"
TYPE_DOWNSAMPLE = "downsample"
TYPE_RAW = "raw"

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

# Skip combination stats when there are "too many" filters
MAX_COMBINATION_FILTERS = 20  # default threshold

# CLI parsing constants
CLI_FILTER_MIN_PARTS = 4  # minimum required parts for CLI filter
CLI_DOWNSAMPLE_MIN_PARTS = 2  # method:size
CLI_DOWNSAMPLE_MAX_PARTS = 3  # method:size:seed
BETWEEN_VALUE_PARTS = 2  # min,max

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
    valid_types = {TYPE_QUALITY, TYPE_REGION, TYPE_MAPPING, TYPE_LABEL}
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


def _try_to_convert_to_number_or_boolean(value: str) -> Any:
    """Try to convert a string to a boolean, int, or float, otherwise return the original string."""
    # Handle boolean literals
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_value_based_on_operation(value_str: str, op: str) -> Any:
    """Parse value string based on the operation type."""
    # Handle in/not_in
    if op in ["in", "not_in"]:
        # For list operations, split on comma
        return [_try_to_convert_to_number_or_boolean(value) for value in value_str.split(",")]
    # Handle single values
    else:
        # Try to convert to number, otherwise keep as string
        return _try_to_convert_to_number_or_boolean(value_str)


def _parse_cli_filter(filter_spec: str) -> dict[str, Any]:
    """Parse a CLI filter specification into a filter dictionary.

    Format: name=value:field=value:op=value:value=value:type=value
    or:     name=value:field=value:op=value:value_field=value:type=value
    """
    parts = filter_spec.split(":")
    if len(parts) < CLI_FILTER_MIN_PARTS:
        raise ValueError(
            f"Filter specification must have at least {CLI_FILTER_MIN_PARTS} parts separated by ':'. "
            f"Got: {filter_spec}"
        )

    # Parse key=value pairs
    parsed = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Each part must be in key=value format. Invalid part: '{part}'")
        key, value = part.split("=", 1)
        parsed[key] = value

    # Validate required keys
    required_keys = {KEY_NAME, KEY_FIELD, KEY_OP, KEY_TYPE}
    missing_keys = required_keys - parsed.keys()
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    # Validate that we have either 'value' or 'value_field'
    if KEY_VALUE not in parsed and KEY_VALUE_FIELD not in parsed:
        raise ValueError(f"Must specify either '{KEY_VALUE}' or '{KEY_VALUE_FIELD}'")

    # Build the filter dictionary
    filter_dict = {
        KEY_NAME: parsed[KEY_NAME],
        KEY_FIELD: parsed[KEY_FIELD],
        KEY_OP: parsed[KEY_OP],
        KEY_TYPE: parsed[KEY_TYPE],
    }

    # Handle value or value_field
    if KEY_VALUE_FIELD in parsed:
        filter_dict[KEY_VALUE_FIELD] = parsed[KEY_VALUE_FIELD]
    else:
        filter_dict[KEY_VALUE] = _parse_value_based_on_operation(parsed[KEY_VALUE], parsed[KEY_OP])

    return filter_dict


def _parse_cli_downsample(downsample_spec: str) -> dict[str, Any]:
    """Parse a CLI downsample specification into a downsample dictionary.

    Format: method:size:seed (seed is optional for random method)
    """
    parts = downsample_spec.split(":")
    if len(parts) < CLI_DOWNSAMPLE_MIN_PARTS or len(parts) > CLI_DOWNSAMPLE_MAX_PARTS:
        raise ValueError(
            f"Downsample specification must have {CLI_DOWNSAMPLE_MIN_PARTS}-{CLI_DOWNSAMPLE_MAX_PARTS} "
            f"parts separated by ':'. Got: {downsample_spec}"
        )

    method, size_str = parts[0], parts[1]

    try:
        size = int(size_str)
    except ValueError as err:
        raise ValueError(f"Downsample size must be an integer. Got: {size_str}") from err

    downsample_config = {
        KEY_METHOD: method,
        KEY_SIZE: size,
    }

    # Add seed if provided
    if len(parts) == CLI_DOWNSAMPLE_MAX_PARTS:
        try:
            seed = int(parts[2])
            downsample_config[KEY_SEED] = seed
        except ValueError as err:
            raise ValueError(f"Downsample seed must be an integer. Got: {parts[2]}") from err

    return downsample_config


def _merge_config_and_cli(
    config_path: str | None, cli_filters: list[str] | None, cli_downsample: str | None
) -> dict[str, Any]:
    """Merge JSON config with CLI arguments to create final configuration."""
    # Start with config from file if provided
    cfg = {}
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)

    # Initialize filters list if not present
    if KEY_FILTERS not in cfg:
        cfg[KEY_FILTERS] = []

    # Add CLI filters
    if cli_filters:
        for filter_spec in cli_filters:
            filter_dict = _parse_cli_filter(filter_spec)
            cfg[KEY_FILTERS].append(filter_dict)

    # Override downsample if provided via CLI
    if cli_downsample:
        cfg[KEY_DOWNSAMPLE] = _parse_cli_downsample(cli_downsample)

    return cfg


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


def _mask_for_rule(rule: dict[str, Any]) -> pl.Expr:
    """Return a boolean expression for a single rule."""
    field = rule[KEY_FIELD]
    op = rule[KEY_OP]
    if op not in _OPS:
        raise ValueError(f"Unsupported op: {op}")

    # Extract RHS without relying on truthiness (0, "", [] must be accepted)
    if KEY_VALUE_FIELD in rule:
        # When we compare two *columns* (value_field) and either side may be
        # Categorical/Enum, cast both to Utf8 first to avoid StringCacheMismatchError.
        lhs = pl.col(field).cast(pl.Utf8)
        rhs = pl.col(rule[KEY_VALUE_FIELD]).cast(pl.Utf8)
    elif KEY_VALUE in rule:
        lhs = pl.col(field)
        rhs = rule[KEY_VALUE]
    elif KEY_VALUES in rule:
        lhs = pl.col(field)
        rhs = rule[KEY_VALUES]
    else:  # Should never happen – config is already validated
        raise ValueError(
            f"Filter rule for field '{field}' is missing a comparison target "
            f"(expected one of '{KEY_VALUE}', '{KEY_VALUES}', or '{KEY_VALUE_FIELD}')."
        )

    return _OPS[op](lhs, rhs)


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

        mask_expr = _mask_for_rule(rule)
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

    # Build cumulative index for rows that pass all filters
    row_idx_expr = (pl.when(all_filters_mask).then(1).otherwise(0)).cum_sum() - 1

    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.when(all_filters_mask).then(row_idx_expr).otherwise(None).alias(COL_ROW_NUM_FILTERED)
    )

    # Create downsample mask
    size = cfg[KEY_DOWNSAMPLE][KEY_SIZE]
    method = cfg[KEY_DOWNSAMPLE].get(KEY_METHOD, METHOD_RANDOM)

    if method == METHOD_HEAD:
        downsample_expr = pl.col(COL_ROW_NUM_FILTERED) < size
        tmp_cols = [COL_ROW_NUM_FILTERED]
    else:  # METHOD_RANDOM
        seed = cfg[KEY_DOWNSAMPLE].get(KEY_SEED, 0)
        col_rand = "__rand_val"
        col_rank = "__rand_rank"

        # Assign a deterministic pseudo-random value to each row that passed all filters
        featuremap_dataframe = featuremap_dataframe.with_columns(
            pl.when(pl.col(COL_ROW_NUM_FILTERED).is_not_null())
            .then(pl.col(COL_ROW_NUM_FILTERED).hash(seed=seed))
            .otherwise(None)
            .alias(col_rand)
        ).with_columns(pl.col(col_rand).rank(method="dense").alias(col_rank))

        # Keep the first `size` rows in that random order
        downsample_expr = (pl.col(col_rank) - 1) < size
        tmp_cols = [COL_ROW_NUM_FILTERED, col_rand, col_rank]

    downsample_col = COL_FILTER_DOWNSAMPLE
    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.when(all_filters_mask).then(downsample_expr).otherwise(None).alias(downsample_col)
    ).drop(tmp_cols)

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
        combos: dict[str, int] = {}
    elif len(filter_cols) > MAX_COMBINATION_FILTERS:
        logger.info(
            "Skipping combination statistics: %d filters > %d",
            len(filter_cols),
            MAX_COMBINATION_FILTERS,
        )
        combos = {}
    else:
        combo_expr = pl.concat_str(
            [pl.when(pl.col(col)).then(pl.lit("1")).otherwise(pl.lit("0")) for col in filter_cols]
        )

        combos_df = (
            featuremap_dataframe.group_by(combo_expr.alias(STAT_PATTERN)).agg(pl.len().alias(STAT_COUNT)).collect()
        )

        # Map existing patterns to counts
        combos = {row[STAT_PATTERN]: row[STAT_COUNT] for row in combos_df.to_dicts()}

        # Ensure every possible pattern appears (even if count == 0)
        num_patterns = 1 << len(filter_cols)  # 2 ** n
        width = len(filter_cols)
        for i in range(num_patterns):
            pattern = format(i, f"0{width}b")
            combos.setdefault(pattern, 0)

    # Add filter types to the statistics
    filters_with_types = []
    for name, rows in funnel:
        if name == STAT_RAW:
            filters_with_types.append({KEY_NAME: name, STAT_ROWS: rows, KEY_TYPE: TYPE_RAW})
        elif name == STAT_DOWNSAMPLE:
            # include method / seed from the downsample section
            ds_cfg = cfg.get(KEY_DOWNSAMPLE, {})
            entry = {
                KEY_NAME: name,
                STAT_ROWS: rows,
                KEY_TYPE: TYPE_DOWNSAMPLE,
                KEY_METHOD: ds_cfg.get(KEY_METHOD),
                KEY_SEED: ds_cfg.get(KEY_SEED),
            }
            entry = {k: v for k, v in entry.items() if v is not None}
            filters_with_types.append(entry)
        else:
            rule = next(
                (r for r in filters if (r.get(KEY_NAME) or f"{r[KEY_FIELD]}_{r[KEY_OP]}") == name),
                {},
            )
            entry = {
                KEY_NAME: name,
                STAT_ROWS: rows,
                KEY_TYPE: rule.get(KEY_TYPE),
                KEY_FIELD: rule.get(KEY_FIELD),
                KEY_OP: rule.get(KEY_OP),
                KEY_VALUE: rule.get(KEY_VALUE, rule.get(KEY_VALUES)),
                KEY_VALUE_FIELD: rule.get(KEY_VALUE_FIELD),
            }
            # drop keys whose value is None
            entry = {k: v for k, v in entry.items() if v is not None}
            filters_with_types.append(entry)

    return {
        KEY_FILTERS: filters_with_types,
        "single_effect": single_effect,
        "combinations": combos,
    }


def filter_parquet(
    in_path: str,
    out_path: str | None,
    out_path_full: str | None,
    cfg_path: str | None,
    stats_path: str,
    cli_filters: list[str] | None = None,
    cli_downsample: str | None = None,
) -> None:
    """
    Filter a parquet file based on configuration and generate statistics.

    This function applies a series of filters to a parquet file according to a JSON
    configuration, optionally downsamples the results, and generates comprehensive
    statistics about the filtering process.

    Parameters
    ----------
    in_path : str
        Path to the input parquet file containing the data to be filtered.

    out_path : str | None
        Path for the output parquet file containing only the rows that pass all filters.
        If downsampling is configured, the output will be downsampled to the specified size.
        If None, no filtered output file is created.

    out_path_full : str | None
        Path for the output parquet file containing all original rows plus additional
        binary columns indicating which filters each row passes/fails.
        If None, no full output file is created.

    cfg_path : str | None
        Path to the JSON configuration file that defines the filters and optional
        downsampling. If None, only CLI filters will be used. The configuration should have the following structure:
        {
            "filters": [
                {
                    "field": "column_name",
                    "op": "operator",  # eq, ne, lt, le, gt, ge, in, not_in
                    "type": "filter_type",  # quality, region, or label
                    "value": "value_to_compare",  # or "values" for list, or "value_field" for column
                    "name": "optional_filter_name"  # optional custom name
                }
            ],
            "downsample": {  # optional
                "size": 1000,  # number of rows to keep
                "method": "head" or "random",  # default: "random"
                "seed": 42  # optional, for reproducible random sampling
            }
        }

    stats_path : str
        Path for the output JSON file containing detailed statistics about the filtering
        process, including:
        - Funnel statistics showing cumulative effect of filters
        - Single effect statistics for each filter independently
        - Combination statistics showing all unique filter pass/fail patterns

    cli_filters : list[str] | None, optional
        List of CLI filter specifications in the format "name=value:field=value:op=value:value=value:type=value".
        These filters are appended to any filters specified in the config file.

    cli_downsample : str | None, optional
        CLI downsample specification in the format "method:size:seed" (seed optional).
        This overrides any downsampling specified in the config file.

    Raises
    ------
    ValueError
        If the configuration is invalid or if neither out_path nor out_path_full is specified.

    Notes
    -----
    - At least one of out_path or out_path_full must be specified.
    - All filters are applied with AND logic - a row must pass ALL filters to be included
      in the output. The order of filters in the configuration only affects the funnel
      statistics reporting, not the filtering results.
    - The statistics file provides insights into how each filter affects the data:
      * Funnel statistics show the cumulative effect of filters in the order specified
      * Single effect statistics show each filter's impact independently
      * Combination statistics show all unique pass/fail patterns across filters
    - Binary filter columns in out_path_full are prefixed with "__filter_".
    """
    logger.info(f"Starting filter_parquet: input={in_path}")

    # Merge config from file and CLI arguments
    cfg = _merge_config_and_cli(cfg_path, cli_filters, cli_downsample)

    validate_filter_config(cfg)
    logger.info(f"Configuration has {len(cfg[KEY_FILTERS])} filters")

    # Create lazy frame for efficient processing
    featuremap_dataframe = pl.scan_parquet(in_path)

    # Get total row count from lazy frame
    total_rows = featuremap_dataframe.select(pl.len()).collect().item()
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
        pl.Config.set_streaming_chunk_size(100_000)
        logger.info(f"Writing filtered output to {out_path}")
        (
            featuremap_dataframe.filter(pl.col(COL_FILTER_FINAL))
            .select(pl.exclude(f"^{COL_PREFIX_FILTER}.*$"))
            .sink_parquet(out_path, row_group_size=100_000)
        )

        # Get row count for logging
        written_rows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
        logger.info(f"Wrote filtered data: {written_rows:,} rows ({out_path})")

    if out_path_full:
        logger.info(f"Writing full output with filter columns to {out_path_full}")
        featuremap_dataframe.sink_parquet(out_path_full)

        # Get row count for logging
        full_rows = pl.scan_parquet(out_path_full).select(pl.len()).collect().item()
        logger.info(f"Wrote full data with filters: {full_rows:,} rows ({out_path_full})")
        logger.info(pl.read_parquet(out_path_full).select(f"^{COL_PREFIX_FILTER}.*$").sum())  # TODO remove this line

    # Write statistics
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote statistics to {stats_path}")


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Filter / down-sample featuremap Parquet")
    ap.add_argument("--in", dest="inp", required=True, help="input parquet")
    ap.add_argument("--out", help="output parquet with filtered rows (optional)")
    ap.add_argument("--out-full", help="output parquet with all rows and filter columns (optional)")
    ap.add_argument("--config", help="JSON with filters + downsample (optional)")
    ap.add_argument("--stats", required=True, help="output JSON with statistics")
    ap.add_argument(
        "--filter",
        action="append",
        dest="filters",
        help=(
            "Filter specification: name=value:field=value:op=value:value=value:type=value "
            "(can be used multiple times)"
        ),
    )
    ap.add_argument("--downsample", help="Downsample specification: method:size:seed (optional seed for random method)")
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

    # Validate that either config file or CLI filters are provided
    if not args.config and not args.filters:
        logger.error("Either --config or --filter must be specified")
        raise ValueError("No filters specified")

    filter_parquet(args.inp, args.out, args.out_full, args.config, args.stats, args.filters, args.downsample)


if __name__ == "__main__":
    # Minimal logging configuration so that messages appear when executed directly
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()


# -------------- internal helper -------------------------------------------
def _validate_stats_dict(data: dict, where: str) -> None:  # noqa C901
    """Raise ValueError if *data* is not a well-formed filtering-stats dict."""
    # ---------- basic keys -------------------------------------------------
    required_keys = {"filters", "single_effect", "combinations"}
    if missing := required_keys - data.keys():
        raise ValueError(f"stats JSON{where}: missing key(s): {sorted(missing)}")

    # ---------- filters ----------------------------------------------------
    filters = data["filters"]
    if not isinstance(filters, list) or not filters:
        raise ValueError(f"stats JSON{where}: 'filters' must be a non-empty list")

    for idx, f in enumerate(filters):
        if not isinstance(f, dict):
            raise ValueError(f"stats JSON{where}: filters[{idx}] is not an object")
        if {"name", "rows"} - f.keys():
            raise ValueError(f"stats JSON{where}: filters[{idx}] missing 'name' or 'rows'")
        if not isinstance(f["rows"], int) or f["rows"] < 0:
            raise ValueError(f"stats JSON{where}: filters[{idx}]['rows'] must be a non-negative integer")

    if filters[0]["name"] != "raw":
        raise ValueError(f"stats JSON{where}: first filter must have name 'raw'")

    # ---------- single_effect ---------------------------------------------
    se = data["single_effect"]
    if not isinstance(se, dict):
        raise ValueError(f"stats JSON{where}: 'single_effect' must be an object")
    if not all(isinstance(v, int) and v >= 0 for v in se.values()):
        raise ValueError(f"stats JSON{where}: all 'single_effect' values must be non-negative integers")

    # ---------- combinations ----------------------------------------------
    if (
        "combinations" in data
        and isinstance(combos := data["combinations"], dict)
        and not all(isinstance(v, int) and v >= 0 for v in combos.values())
    ):
        raise ValueError(f"stats JSON{where}: all 'combinations' values must be non-negative integers")


# -------------- public loader ---------------------------------------------
def read_filtering_stats_json(path: str | Path) -> dict:
    """
    Read a statistics JSON written by `filter_parquet --stats`.

    The file is validated; any structural problem raises ValueError.
    On success the parsed dictionary is returned.
    """
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)

    _validate_stats_dict(data, where=f" ({p})")
    return data
