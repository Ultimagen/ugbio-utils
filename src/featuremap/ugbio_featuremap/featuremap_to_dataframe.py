#!/usr/bin/env python3
# Copyright 2023 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Add additional feature annotations to featuremap, to be used from single-read SNV qual recalibration
# featuremap_to_dataframe.py
# --------------------------

# Convert a *feature-map* VCF (one sample, FORMAT lists = supporting reads)
# to a tidy **per-read** Parquet dataframe.

# Pipeline:

# 1.  **Header parse** via `bcftools view -h`
#     • Collect INFO / FORMAT IDs, Number, Type, enum list `{A,C,G,T}`
# 2.  **Query** necessary columns with `bcftools query`
#     • Streaming to a temporary TSV to keep memory low
# 3.  **Polars ingest**
#     • POS → Int64
#     • INFO scalar casting   (`cast_scalar`)
#     • FORMAT list handling  (`cast_list`)
#       – split, null-replace
#       – numeric cast when appropriate
#       – list columns stay Utf8
# 4.  **Enum padding**
#     • Append stub rows so every categorical column registers all levels
# 5.  **Explode** list columns → one row per read
# 6.  **Write** Parquet

# The whole conversion runs inside a `pl.StringCache()` context so any
# categorical columns created later share the same global dictionary.

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as _mp  # NEW
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import polars as pl

from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_featuremap.filter_dataframe import (
    FIELD_GNOMAD_AF,
    FIELD_ID,
    FIELD_UG_HCR,
    KEY_FIELD,
    KEY_FILTERS,
    KEY_NAME,
    KEY_OP,
    KEY_TYPE,
    KEY_VALUE,
    KEY_VALUE_FIELD,
    TYPE_DOWNSAMPLE,
    TYPE_RAW,
    TYPE_REGION,
    _create_filter_columns,
    _create_final_filter_column,
    validate_filter_config,
)

log = logging.getLogger(__name__)

# Configuration constants
DEFAULT_JOBS = 0  # 0 means auto-detect CPU cores
CHUNK_BP_DEFAULT = 10_000_000  # 10 Mbp per processing chunk


@dataclass
class ColumnConfig:
    """Configuration for column processing."""

    info_ids: list[str]
    scalar_fmt_ids: list[str]
    list_fmt_ids: list[str]
    info_meta: dict
    fmt_meta: dict


@dataclass
class VCFJobConfig:
    """Static metadata shared by all workers."""

    bcftools_path: str
    awk_script: str
    columns: list[str]
    list_indices: list[int]
    schema: dict[str, pl.PolarsDataType]
    info_meta: dict
    fmt_meta: dict
    info_ids: list[str]
    scalar_fmt_ids: list[str]
    list_fmt_ids: list[str]
    read_filters: dict | None = None  # Filter configuration from JSON file


def _load_read_filters(read_filters_json: str | None, read_filter_json_key: str | None = None) -> dict | None:
    """
    Load read filters from JSON file.

    Parameters
    ----------
    read_filters_json : str | None
        Path to JSON file containing read filters
    read_filter_json_key : str | None
        Specific key in the JSON file to extract filters from.
        If None, uses the entire JSON content.

    Returns
    -------
    dict | None
        Filter configuration dictionary or None if no filters provided
    """
    if not read_filters_json:
        log.debug("No read_filters_json provided")
        return None

    log.info(f"Loading read filters from: {read_filters_json}")
    try:
        with open(read_filters_json) as f:
            filter_data = json.load(f)

        if read_filter_json_key:
            log.info(f"Extracting filters from key: '{read_filter_json_key}'")
            if read_filter_json_key not in filter_data:
                raise ValueError(f"Key '{read_filter_json_key}' not found in filter JSON file {read_filters_json}")
            result = filter_data[read_filter_json_key]
        else:
            log.info("Using entire JSON file content as filters")
            result = filter_data

        log.info(f"Loaded filter configuration: {json.dumps(result, indent=2)[:500]}...")  # Log first 500 chars
        return result

    except FileNotFoundError as e:
        raise RuntimeError(f"File not found: {read_filters_json}\n{e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to load read filters from {read_filters_json}\n{e}") from e


def _apply_id_null_handling(
    lazy_frame: pl.LazyFrame, col_name: str, filter_name: str, op: str, value: str
) -> pl.LazyFrame:
    """
    Apply special null handling for ID filters.

    In VCF files, variants not in dbSNP have ID = ".". However, Polars converts
    "." to null during TSV parsing (via null_values=["."]).

    For ID filters checking for "." (not in dbSNP), we need to treat null values
    as passing the filter, since null means the original value was ".".

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        The lazy frame containing the filter column
    col_name : str
        The name of the filter column to modify
    filter_name : str
        The name of the filter for logging
    op : str
        The operation (e.g., "eq")
    value : str
        The value being compared (e.g., ".")

    Returns
    -------
    pl.LazyFrame
        Modified lazy frame with updated filter column
    """
    # Only apply null handling when checking for "." with equality
    if op == "eq" and value == ".":
        log.debug(f"Adding null handling for ID filter: '{filter_name}' (treating null as '.')")
        return lazy_frame.with_columns((pl.col(col_name) | pl.col(FIELD_ID).is_null()).alias(col_name))
    # For other comparisons (ne, in, etc.), null values should fail the filter
    return lazy_frame


def _apply_gnomad_null_handling(lazy_frame: pl.LazyFrame, col_name: str, filter_name: str) -> pl.LazyFrame:
    """
    Apply special null handling for gnomAD_AF filters.

    In VCF files, variants not present in gnomAD have null/missing gnomAD_AF values.
    For filtering purposes, missing gnomAD_AF should be treated as rare (passing the filter).
    This function modifies the filter column to include null values as passing.

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        The lazy frame containing the filter column
    col_name : str
        The name of the filter column to modify
    filter_name : str
        The name of the filter for logging

    Returns
    -------
    pl.LazyFrame
        Modified lazy frame with updated filter column
    """
    log.debug(f"Adding null handling for gnomAD_AF filter: '{filter_name}'")
    return lazy_frame.with_columns((pl.col(col_name) | pl.col(FIELD_GNOMAD_AF).is_null()).alias(col_name))


def _apply_ug_hcr_null_handling(lazy_frame: pl.LazyFrame, col_name: str, filter_name: str) -> pl.LazyFrame:
    """
    Apply special null handling for UG_HCR filters.

    For UG_HCR (Ultimagen High Confidence Region), null values indicate the variant
    is NOT in the HCR and should be filtered out. This function ensures only
    non-null TRUE values pass the filter.

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        The lazy frame containing the filter column
    col_name : str
        The name of the filter column to modify
    filter_name : str
        The name of the filter for logging

    Returns
    -------
    pl.LazyFrame
        Modified lazy frame with updated filter column requiring non-null values
    """
    log.debug(f"Adding NOT-null requirement for UG_HCR filter: '{filter_name}'")
    return lazy_frame.with_columns((pl.col(col_name) & pl.col(FIELD_UG_HCR).is_not_null()).alias(col_name))


def _prepare_filter_config(read_filters: dict | list | None) -> list[dict]:
    """
    Extract and validate filter configuration from input.

    Parameters
    ----------
    read_filters : dict | list | None
        Filter configuration in various formats

    Returns
    -------
    list[dict]
        List of validated filter dictionaries, excluding raw/downsample filters

    Raises
    ------
    RuntimeError
        If filter validation fails
    """
    # Handle both dict and list formats
    if isinstance(read_filters, list):
        all_filters = read_filters
    else:
        all_filters = read_filters.get(KEY_FILTERS, [])

    # Create a copy of filters excluding 'raw' and 'downsample' entries for validation
    filters_to_apply = [f for f in all_filters if f.get(KEY_TYPE) not in {TYPE_RAW, TYPE_DOWNSAMPLE}]

    if not filters_to_apply:
        log.debug("No applicable filters found (only raw/downsample filters present)")
        return []

    # Log the filters being applied
    log.debug(f"Filters to apply ({len(filters_to_apply)}):")
    for i, f in enumerate(filters_to_apply):
        log.debug(
            f"  {i+1}. {f.get(KEY_NAME, 'unnamed')}: {f.get(KEY_FIELD)} {f.get(KEY_OP)} {f.get(KEY_VALUE, f.get(KEY_VALUE_FIELD, 'N/A'))}"  # noqa E501
        )

    # Validate filter configuration
    filter_config = {KEY_FILTERS: filters_to_apply}
    try:
        validate_filter_config(filter_config)
    except Exception as e:
        filter_names = [f.get(KEY_NAME, f.get(KEY_TYPE, "unnamed")) for f in filters_to_apply]
        log.error(f"FATAL: Filter validation failed. Filters: {filter_names}")
        log.error(f"Validation error: {e}")
        raise RuntimeError(f"Filter validation failed for filters {filter_names}: {e}") from e

    return filters_to_apply


def _apply_null_handling_to_filters(
    lazy_frame: pl.LazyFrame, filters_to_apply: list[dict], filter_cols: list[str]
) -> pl.LazyFrame:
    """
    Apply special null-handling logic for specific fields.

    This ensures records with missing values are handled correctly per field semantics:
    - gnomAD_AF: null = not in gnomAD = rare (should pass filter)
    - ID: null = "." in VCF = not in dbSNP (special handling)
    - UG_HCR: null = not in HCR (should fail filter)

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        The lazy frame with filter columns
    filters_to_apply : list[dict]
        List of filter configurations
    filter_cols : list[str]
        List of filter column names

    Returns
    -------
    pl.LazyFrame
        Lazy frame with updated filter columns
    """
    for f in filters_to_apply:
        field_name = f.get(KEY_FIELD, "")
        filter_name = f.get(KEY_NAME, f"{field_name}_filter")
        col_name = f"__filter_{filter_name}"

        if col_name not in filter_cols:
            continue

        # gnomAD_AF: include null values (missing = not in gnomAD = rare)
        if field_name == FIELD_GNOMAD_AF and f.get(KEY_TYPE) == TYPE_REGION:
            lazy_frame = _apply_gnomad_null_handling(lazy_frame, col_name, filter_name)

        # ID: VCF uses "." for variants not in dbSNP, but Polars converts "." to null during parsing
        # Need to treat null as "." for ID filters
        elif field_name == FIELD_ID and f.get(KEY_TYPE) == TYPE_REGION:
            lazy_frame = _apply_id_null_handling(lazy_frame, col_name, filter_name, f.get(KEY_OP), f.get(KEY_VALUE))

        # UG_HCR: require non-null values (null = not in HCR, should be filtered out)
        elif field_name == FIELD_UG_HCR and f.get(KEY_TYPE) == TYPE_REGION and f.get(KEY_OP) == "ne":
            lazy_frame = _apply_ug_hcr_null_handling(lazy_frame, col_name, filter_name)

    return lazy_frame


def _log_filter_statistics(df_with_filters: pl.DataFrame, filter_cols: list[str]) -> None:
    """
    Log per-filter statistics showing pass/fail counts.

    Parameters
    ----------
    df_with_filters : pl.DataFrame
        DataFrame with filter columns
    filter_cols : list[str]
        List of filter column names to report on
    """
    log.debug(f"Filter statistics for region (total rows: {df_with_filters.height:,}):")
    for col in filter_cols:
        if col in df_with_filters.columns:
            pass_count = df_with_filters[col].sum()
            fail_count = df_with_filters.height - pass_count
            pct = 100.0 * pass_count / df_with_filters.height if df_with_filters.height > 0 else 0
            log.debug(f"  {col}: {pass_count:,} pass ({pct:.1f}%) / {fail_count:,} fail")


def _execute_filters_with_diagnostics(
    df_final: pl.DataFrame, filters_to_apply: list[dict], frame: pl.DataFrame
) -> pl.DataFrame:
    """
    Execute final filtering with diagnostic error handling.

    If filtering fails, test each filter individually to identify the problematic one.

    Parameters
    ----------
    df_final : pl.DataFrame
        DataFrame with __filter_final column
    filters_to_apply : list[dict]
        List of filter configurations for diagnostic purposes
    frame : pl.DataFrame
        Original frame for diagnostic testing

    Returns
    -------
    pl.DataFrame
        Filtered dataframe with filter columns removed

    Raises
    ------
    RuntimeError
        If filter execution fails
    """
    try:
        return df_final.filter(pl.col("__filter_final")).select(pl.exclude("^__filter_.*$"))
    except Exception as e:
        # Error during filter execution - test each filter individually to identify the problem
        log.error(f"Error during filter execution: {e}")
        log.error("Testing each filter individually to identify the problematic filter...")

        problematic_filter = None
        for i, f in enumerate(filters_to_apply):
            filter_name = f.get(KEY_NAME, "unnamed")
            filter_type = f.get(KEY_TYPE, "unknown")
            try:
                # Test this filter alone - full execution with collect
                test_lazy = frame.lazy()
                test_lazy, test_cols = _create_filter_columns(test_lazy, [f])
                test_lazy = _create_final_filter_column(test_lazy, test_cols, None)
                _ = test_lazy.filter(pl.col("__filter_final")).select(pl.exclude("^__filter_.*$")).collect()
                log.debug(
                    f"  ✓ Filter {i + 1}/{len(filters_to_apply)}: {filter_name} (type={filter_type}) - EXECUTED OK"
                )
            except Exception as filter_error:
                problematic_filter = f
                log.error(
                    f"  ✗ Filter {i + 1}/{len(filters_to_apply)}: "
                    f"{filter_name} (type={filter_type}) - EXECUTION FAILED"
                )
                log.error(f"    Error: {filter_error}")
                log.error(f"    Filter configuration: {json.dumps(f, indent=2)}")

                # Additional diagnostic info
                if "column" in f:
                    col_name = f["column"]
                    if col_name in frame.columns:
                        col_dtype = frame[col_name].dtype
                        log.error(f"    Column '{col_name}' has dtype: {col_dtype}")
                        if "value" in f:
                            value_type = type(f["value"]).__name__
                            log.error(f"    Filter is trying to compare with value: {f['value']} (type: {value_type})")
                            log.error(f"    → This appears to be a type mismatch: {col_dtype} vs {value_type}")
                break

        if problematic_filter:
            log.error(f"FATAL: Identified problematic filter: {problematic_filter.get(KEY_NAME, 'unnamed')}")
            log.error(f"Full filter configuration: {json.dumps(problematic_filter, indent=2)}")
            log.error("Fix: Ensure the filter value type matches the column type")
            log.error("     - If column is string/Enum, use string values in quotes")
            log.error("     - If column is numeric, use numeric values without quotes")
            filter_name = problematic_filter.get(KEY_NAME, "unnamed")
            raise RuntimeError(
                f"Filter '{filter_name}' execution failed. "
                f"Configuration: {json.dumps(problematic_filter)}. "
                f"Original error: {e}"
            ) from e
        else:
            log.error("FATAL: Could not identify specific problematic filter through individual testing.")
            raise RuntimeError(f"Filter execution failed but could not identify specific filter. Error: {e}") from e


def _apply_read_filters(frame: pl.DataFrame, read_filters: dict | list | None) -> pl.DataFrame:
    """
    Apply read filters to a dataframe using the existing filter framework.

    Parameters
    ----------
    frame : pl.DataFrame
        Input dataframe to filter
    read_filters : dict | list | None
        Filter configuration in the same format as used by filter_dataframe.py
        Can be either:
        - A dict with "filters" key containing list of filters
        - A list of filter dictionaries directly
        - None for no filtering

    Returns
    -------
    pl.DataFrame
        Filtered dataframe containing only reads that pass all filters
    """
    if not read_filters or frame.is_empty():
        if not read_filters:
            log.info("No read filters provided - skipping filter step")
        if frame.is_empty():
            log.debug("Empty dataframe - skipping filter step")
        return frame

    log.info(f"Starting read filter application on {frame.height:,} rows")

    try:
        # Extract and validate filter configuration
        filters_to_apply = _prepare_filter_config(read_filters)
        if not filters_to_apply:
            log.info("No read filters provided - skipping filter step")
            return frame

        log.debug(f"Applying {len(filters_to_apply)} filters to dataframe with {frame.height:,} rows")

        # Convert to lazy frame for efficient processing
        lazy_frame = frame.lazy()

        # Create filter columns and apply filters
        lazy_frame, filter_cols = _create_filter_columns(lazy_frame, filters_to_apply)
        log.debug(f"Created filter columns: {filter_cols}")

        # Apply special null-handling for specific fields BEFORE collecting statistics
        lazy_frame = _apply_null_handling_to_filters(lazy_frame, filters_to_apply, filter_cols)

        # Collect with filter columns to check individual filter statistics (after null handling)
        df_with_filters = lazy_frame.collect()

        # Log per-filter statistics (DEBUG level)
        _log_filter_statistics(df_with_filters, filter_cols)

        # Back to lazy for final filter column creation
        lazy_frame = df_with_filters.lazy()
        lazy_frame = _create_final_filter_column(lazy_frame, filter_cols, None)

        # Log final filter statistics
        df_final = lazy_frame.collect()
        final_pass = df_final["__filter_final"].sum()
        final_fail = df_final.height - final_pass
        pct = 100.0 * final_pass / df_final.height if df_final.height > 0 else 0
        log.debug(f"  __filter_final (ALL COMBINED): {final_pass:,} pass ({pct:.1f}%) / {final_fail:,} fail")

        # Apply filters with diagnostic error handling
        filtered_frame = _execute_filters_with_diagnostics(df_final, filters_to_apply, frame)

        log.debug(f"Read filtering: {frame.height:,} → {filtered_frame.height:,} reads")
        return filtered_frame

    except Exception as e:
        # Catch any unexpected errors
        filter_summary = []
        try:
            all_filters = read_filters if isinstance(read_filters, list) else read_filters.get("filters", [])
            for f in all_filters:
                filter_summary.append(f.get(KEY_NAME, f.get(KEY_TYPE, "unnamed")))
        except Exception:
            filter_summary = ["<error parsing filters>"]

        log.error(f"FATAL: Unexpected error applying read filters. Filters: {filter_summary}")
        log.error(f"Error: {e}")
        raise RuntimeError(f"Unexpected error applying read filters {filter_summary}: {e}") from e


# ───────────────── header helpers ────────────────────────────────────────────
# Allow escaped quotes (\") within the Description text
_QUOTED_VALUE = r'"((?:[^"\\]|\\.)*)'

INFO_RE = re.compile(rf"##INFO=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description={_QUOTED_VALUE}")
FORMAT_RE = re.compile(rf"##FORMAT=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description={_QUOTED_VALUE}")

_POLARS_DTYPE = {"Integer": pl.Int64, "Float": pl.Float64, "Flag": pl.Boolean}
# VCF column names – imported from FeatureMapFields for consistency
CHROM = FeatureMapFields.CHROM.value
POS = FeatureMapFields.POS.value
REF = FeatureMapFields.REF.value
ALT = FeatureMapFields.ALT.value
QUAL = FeatureMapFields.QUAL.value
FILTER = FeatureMapFields.FILTER.value
ID = FeatureMapFields.ID.value
SAMPLE = FeatureMapFields.SAMPLE.value
X_ALT = FeatureMapFields.X_ALT.value
MQUAL = FeatureMapFields.MQUAL.value
SNVQ = FeatureMapFields.SNVQ.value
# Reserved/fixed VCF columns (cannot be overridden)
RESERVED = {CHROM, POS, ID, REF, ALT, QUAL, FILTER}

# Category dictionaries
ALT_ALLELE_CATS = ["A", "C", "G", "T"]
REF_ALLELE_CATS = ALT_ALLELE_CATS + [
    "R",
    "Y",
    "K",
    "M",
    "S",
    "W",
    "B",
    "D",
    "H",
    "V",
    "N",
]


def _enum(desc: str) -> list[str] | None:
    """
    Extract a comma-separated enumeration from an INFO/FORMAT description.

    Example
    -------
    >>> _enum("Reference base {A,C,G,T}")
    ['A', 'C', 'G', 'T']

    Returns
    -------
    list[str] | None
        Ordered list of category strings if a {...} pattern is found,
        otherwise ``None``.
    """
    m = re.search(r"\{([^}]*)}", desc)
    if not m:
        return None
    # Return deterministically-ordered categories
    return sorted(m.group(1).split(","))


def header_meta(vcf: str, bcftools_path: str, threads: int = 0) -> tuple[dict, dict]:
    """
    Parse the VCF header and build dictionaries with tag metadata.

    Parameters
    ----------
    vcf
        Path to input VCF/BCF (bgzipped ok).
    bcftools_path
        Absolute path to the ``bcftools`` executable.
    threads
        Number of threads for bcftools (0 = auto-detect).

    Returns
    -------
    (info_meta, format_meta)
        Two dicts keyed by tag name.  Each value is::

            {
                "num":   str,          # Number= in header
                "type":  str,          # Type= in header
                "cat":   list[str] | None   # enumeration from {_,..._}
            }
    """
    cmd = [bcftools_path, "view", "-h"]
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    cmd.append(vcf)

    txt = subprocess.check_output(cmd, text=True)
    info, fmt = {}, {}
    for m in INFO_RE.finditer(txt):
        k, n, t, d = m.groups()
        info[k] = {"num": n, "type": t, "cat": _enum(d)}
    for m in FORMAT_RE.finditer(txt):
        k, n, t, d = m.groups()
        fmt[k] = {"num": n, "type": t, "cat": _enum(d)}
    return info, fmt


def _cast_expr(col: str, meta: dict) -> pl.Expr:
    """
    Build a Polars expression that
    1. normalises "" / "." → null
    2. fills remaining nulls
    3. casts to the final dtype
    """
    base = pl.when(pl.col(col).cast(pl.Utf8, strict=False).is_in(["", "."])).then(None).otherwise(pl.col(col))

    # ---- categorical handling -------------------------------------------
    if meta["cat"]:
        cats = meta["cat"] + ([] if "" in meta["cat"] else [""])
        return base.fill_null(value="").cast(pl.Enum(cats), strict=True).alias(col)
    elif meta["type"] in ("Integer", "Float"):
        return base.fill_null(value=0).cast(_POLARS_DTYPE[meta["type"]], strict=True).alias(col)
    elif meta["type"] == "Flag":
        return base.fill_null(value=False).cast(pl.Boolean, strict=True).alias(col)

    # ---- default (Utf8) --------------------------------------------------
    return base.alias(col)


def _resolve_bcftools_command() -> str:
    """
    Locate ``bcftools`` on $PATH and return its absolute path.

    Raises
    ------
    RuntimeError
        If the executable is not found.
    """
    path = shutil.which("bcftools")
    if path is None:
        raise RuntimeError("bcftools not found in $PATH")
    return path


def _resolve_bedtools_command() -> str:
    """Locate ``bedtools`` on $PATH."""
    path = shutil.which("bedtools")
    if path is None:
        raise RuntimeError("bedtools not found in $PATH")
    return path


def _generate_genomic_regions(
    vcf_path: str,
    jobs: int,
    bcftools_path: str,
    bedtools_path: str,
    window_size: int,
) -> list[str]:
    """
    Return a *large* list of fixed-size windows (chr:start-end).

    If *window_size* is None it is chosen so that the number of windows is
    ~10× ``jobs``.
    """
    # Extract contig lengths from header
    contig_re = re.compile(r"##contig=<ID=([^,>]+),length=(\d+)")
    header = subprocess.check_output([bcftools_path, "view", "-h", vcf_path], text=True)
    # Keep only IDs that are strictly alphanumeric (letters / digits only)
    contigs: list[tuple[str, int]] = [
        (cid, int(length))
        for cid, length in (m.groups() for m in contig_re.finditer(header))
        if re.fullmatch(r"[A-Za-z0-9]+", cid)
    ]

    if not contigs:
        raise RuntimeError("No suitable ##contig lines found in VCF header")
    log.debug(f"Found {len(contigs)} contigs: {contigs[:5]}{'...' if len(contigs) > 5 else ''}")  # noqa PLR2004

    # Write genome.sizes tmp file
    with tempfile.NamedTemporaryFile("w", delete=False) as fh:
        for chrom, length in contigs:
            fh.write(f"{chrom}\t{length}\n")
        genome_sizes_path = fh.name

    try:
        # bedtools makewindows
        proc = subprocess.run(
            [
                bedtools_path,
                "makewindows",
                "-g",
                genome_sizes_path,
                "-w",
                str(window_size),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    finally:
        Path(genome_sizes_path).unlink(missing_ok=True)

    regions: list[str] = []
    for line in proc.stdout.strip().split("\n"):
        chrom, start0, end = line.split("\t")
        regions.append(f"{chrom}:{int(start0) + 1}-{end}")  # convert to 1-based inclusive
    return regions


def _sanity_check_format_numbers(format_ids: list[str], fmt_meta: dict) -> None:
    """Raise if any FORMAT tag has a Number we do not support."""
    for tag in format_ids:
        num = fmt_meta[tag]["num"]
        if num not in {"1", "."}:
            raise ValueError(
                f"Unsupported FORMAT field {tag}: Number={num}. "
                "Only '1' (scalar) or '.' (variable list) are supported. "
                "Use --drop-format to exclude this tag."
            )


def _split_format_ids(format_ids: list[str], fmt_meta: dict) -> tuple[list[str], list[str]]:
    """Return (scalar_ids, list_ids) according to Number=."""
    scalar = [k for k in format_ids if fmt_meta[k]["num"] == "1"]
    return scalar, [k for k in format_ids if k not in scalar]


def _make_query_string(format_ids: list[str], query_info: list[str]) -> str:
    """Build the bcftools query format string (now includes %QUAL and %ID)."""
    bracket = "[" + "\t".join(f"%{t}" for t in format_ids) + "]"
    return (
        "\t".join(
            [
                "%CHROM",
                "%POS",
                "%ID",
                "%QUAL",
                "%REF",
                "%ALT",
                *[f"%INFO/{t}" for t in query_info],
                bracket,
            ]
        )
        + "\n"
    )


def _get_awk_script_path() -> str:
    """Get path to the AWK script for list explosion."""
    # First try the standard approach (should work in development and installed package)
    script_dir = Path(__file__).parent
    awk_script = script_dir / "explode_lists.awk"
    if awk_script.exists():
        return str(awk_script)

    # Fallback using importlib.resources for robust package resource access
    try:
        import importlib.resources as pkg_resources

        try:
            # Python 3.9+
            ref = pkg_resources.files("ugbio_featuremap") / "explode_lists.awk"
            with pkg_resources.as_file(ref) as awk_path:
                return str(awk_path)
        except AttributeError:
            # Python 3.8 fallback
            with pkg_resources.path("ugbio_featuremap", "explode_lists.awk") as awk_path:
                return str(awk_path)
    except ImportError:
        pass

    raise FileNotFoundError(f"AWK script not found: {awk_script}")


def _merge_parquet_files_lazy(  # noqa: PLR0912, E501
    parquet_files: list[str],
    output_path: str,
    downsample_reads: int | None = None,
    downsample_seed: int | None = None,
) -> None:
    """
    Merge multiple Parquet files using Polars lazy evaluation for memory efficiency.

    Parameters
    ----------
    parquet_files : list[str]
        List of parquet file paths to merge
    output_path : str
        Output path for merged parquet file
    downsample_reads : int | None
        If specified, downsample to this number of reads. If the total number of reads
        is less than this value, all reads are returned.
    downsample_seed : int | None
        Random seed for downsampling (optional, for reproducibility)
    """
    if not parquet_files:
        log.warning("No Parquet files to merge - creating empty output file")
        # Create an empty Parquet file with minimal structure
        empty_df = pl.DataFrame({"CHROM": [], "POS": [], "REF": [], "ALT": []})
        empty_df.write_parquet(output_path)
        return

    log.debug(f"Received {len(parquet_files)} parquet files: {','.join(parquet_files)}")
    if len(parquet_files) == 1:
        log.debug("Only one Parquet file present - skipping merge")
        if downsample_reads is None:
            shutil.move(parquet_files[0], output_path)
        else:
            # Need to collect to check height
            dataframe_size = pl.scan_parquet(parquet_files[0]).select(pl.len()).collect().item()
            if dataframe_size <= downsample_reads:
                log.debug(
                    f"Dataset has {dataframe_size} rows, which is <= requested {downsample_reads} - keeping all rows"  # noqa E501
                )
                shutil.move(parquet_files[0], output_path)
            else:
                log.info(f"Sampling {downsample_reads} rows from {dataframe_size} total rows")
                pl.read_parquet(parquet_files[0]).sample(n=downsample_reads, seed=downsample_seed).write_parquet(
                    output_path
                )
            Path(parquet_files[0]).unlink(missing_ok=True)
        return

    log.debug(f"Merging {len(parquet_files)} Parquet files lazily")

    # Use lazy scanning and streaming write for memory efficiency
    lazy_frames = [pl.scan_parquet(f) for f in parquet_files]

    lazy_frames_size = [frame.select(pl.len()).collect().item() for frame in lazy_frames]
    log.debug(f"Individual Parquet file sizes (rows): {', '.join(map(str, lazy_frames_size))}")

    # Concatenate all lazy frames
    merged_lazy = pl.concat(lazy_frames, how="vertical")

    # Apply downsampling if requested
    if downsample_reads is None:
        merged_lazy.sink_parquet(output_path)
    else:
        # Need to collect to check height and perform sampling
        merged_df = merged_lazy.collect()
        if merged_df.height <= downsample_reads:
            log.debug(
                f"Dataset has {merged_df.height} rows, which is <= requested {downsample_reads} - keeping all rows"
            )
            merged_df.write_parquet(output_path)
        else:
            log.info(f"Sampling {downsample_reads} rows from {merged_df.height} total rows")
            merged_df.sample(n=downsample_reads, seed=downsample_seed).write_parquet(output_path)

    # Clean up temporary files
    for f in parquet_files:
        Path(f).unlink(missing_ok=True)

    log.debug(f"Merged Parquet files written to: {output_path}")


def _build_explicit_schema(cols: list[str], info_meta: dict, fmt_meta: dict) -> dict[str, pl.PolarsDataType]:
    """
    Build an explicit Polars schema for the bcftools TSV based on VCF metadata.
    Only scalar columns get strict numeric types; everything else is Utf8.
    """
    schema: dict[str, pl.PolarsDataType] = {}
    meta_lookup = {**info_meta, **fmt_meta}
    for col in cols:
        if col in (CHROM, REF, ALT, FILTER, ID):
            schema[col] = pl.Utf8
        elif col == POS:
            schema[col] = pl.Int64
        elif col == QUAL:
            schema[col] = pl.Float64
        elif col in meta_lookup and meta_lookup[col]["type"] in _POLARS_DTYPE:
            schema[col] = _POLARS_DTYPE[meta_lookup[col]["type"]]
        else:
            schema[col] = pl.Utf8
    return schema


def _assert_vcf_index_exists(vcf: str) -> None:
    """
    Ensure the VCF file has an index (.tbi or .csi) present.

    Raises
    ------
    RuntimeError
        If no index is found.
    """
    index_path = f"{vcf}.tbi"
    if not Path(index_path).is_file():
        index_path = f"{vcf}.csi"
        if not Path(index_path).is_file():
            raise RuntimeError(f"VCF index not found: {index_path}")


def _run_region_jobs(
    *,
    regions: list[str],
    vcf_path: str,
    fmt_str: str,
    job_cfg: VCFJobConfig,
    jobs: int,
    tmpdir: str,
    spawn_ctx: _mp.context.BaseContext,
) -> list[str]:
    """Run region conversions in parallel and return created part-files.

    Raises
    ------
    RuntimeError
        If any region fails to convert. The original worker exception is chained
        as the cause.
    """
    part_files: list[str] = []
    successful_parts = 0

    with ProcessPoolExecutor(max_workers=jobs, mp_context=spawn_ctx) as executor:
        futures: dict[Future[str], tuple[str, str]] = {}
        for i, region in enumerate(regions):
            part_path = Path(tmpdir) / f"part_{i:06d}.parquet"
            part_files.append(str(part_path))
            fut = executor.submit(
                _process_region_to_parquet,
                region,
                vcf_path,
                fmt_str,
                job_cfg,
                str(part_path),
            )
            futures[fut] = (region, str(part_path))

        failures: list[tuple[str, str, BaseException]] = []
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                region, part_path = futures[fut]
                failures.append((region, part_path, exc))
                log.error(
                    "Error processing region %s (part=%s)\n%s",
                    region,
                    part_path,
                    "".join(traceback.format_exception(exc)),
                )
            else:
                # Check result - empty string means no data (either before or after filtering)
                result = fut.result()
                if result:
                    successful_parts += 1
                # We can't distinguish between empty before/after filtering without more info

        if failures:
            first_region, first_part, first_exc = failures[0]
            raise RuntimeError(
                f"{len(failures)} region(s) failed during VCF→Parquet conversion; "
                f"first failure: {first_region} (part={first_part})"
            ) from first_exc

        log.info(f"Region processing complete: {successful_parts}/{len(regions)} regions produced data")

    result_files = [p for p in part_files if Path(p).is_file()]
    log.info(f"Created {len(result_files)} parquet part files")
    return result_files


def _setup_parallel_jobs(jobs: int) -> tuple[int, str, str]:
    """
    Configure parallel job settings and resolve tool paths.

    Parameters
    ----------
    jobs : int
        Number of parallel jobs (0 = auto-detect CPU cores)

    Returns
    -------
    tuple[int, str, str]
        (job_count, bcftools_path, bedtools_path)
    """
    # Auto-detect optimal job count if not specified
    if jobs == 0:
        jobs = os.cpu_count() or 4

    log.info(f"Using {jobs} parallel jobs for region processing")

    # Resolve tool paths
    bcftools = _resolve_bcftools_command()
    bedtools = _resolve_bedtools_command()

    return jobs, bcftools, bedtools


def _prepare_vcf_metadata(
    vcf: str,
    bcftools: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
) -> tuple[dict, dict, list[str], list[str], list[str]]:
    """
    Parse VCF header and prepare metadata dictionaries.

    Parameters
    ----------
    vcf : str
        Path to VCF file
    bcftools : str
        Path to bcftools executable
    drop_info : set[str] | None
        INFO fields to exclude
    drop_format : set[str] | None
        FORMAT fields to exclude

    Returns
    -------
    tuple[dict, dict, list[str], list[str], list[str]]
        (info_meta, fmt_meta, info_ids, scalar_fmt_ids, list_fmt_ids)
    """
    # Parse VCF header
    info_meta, fmt_meta = header_meta(vcf, bcftools, threads=1)

    # Filter dropped fields
    if drop_info:
        info_meta = {k: v for k, v in info_meta.items() if k not in drop_info}
    if drop_format:
        fmt_meta = {k: v for k, v in fmt_meta.items() if k not in drop_format}

    # Validate FORMAT fields
    format_ids = list(fmt_meta.keys())
    _sanity_check_format_numbers(format_ids, fmt_meta)

    # Split FORMAT fields
    scalar_fmt_ids, list_fmt_ids = _split_format_ids(format_ids, fmt_meta)
    info_ids = list(info_meta.keys())

    return info_meta, fmt_meta, info_ids, scalar_fmt_ids, list_fmt_ids


def _setup_column_configuration(
    info_ids: list[str],
    format_ids: list[str],
    list_fmt_ids: list[str],
) -> tuple[str, list[str], list[int]]:
    """
    Build query string, column names, and list column indices.

    Parameters
    ----------
    info_ids : list[str]
        INFO field names
    format_ids : list[str]
        FORMAT field names
    list_fmt_ids : list[str]
        FORMAT fields with list values

    Returns
    -------
    tuple[str, list[str], list[int]]
        (fmt_str, cols, list_fmt_indices)
    """
    # Build query string and column names
    fmt_str = _make_query_string(format_ids, info_ids)
    cols = [CHROM, POS, ID, QUAL, REF, ALT] + info_ids + format_ids

    # Find indices of list columns for AWK script
    list_fmt_indices = []
    base_cols = [CHROM, POS, ID, QUAL, REF, ALT] + info_ids
    for list_col in list_fmt_ids:
        idx = len(base_cols) + format_ids.index(list_col)
        list_fmt_indices.append(idx)

    return fmt_str, cols, list_fmt_indices


def _execute_parallel_conversion(
    vcf: str,
    out: str,
    regions: list[str],
    fmt_str: str,
    job_cfg: VCFJobConfig,
    jobs: int,
    read_filters: dict | list | None,
    downsample_reads: int | None,
    downsample_seed: int | None,
) -> None:
    """
    Execute parallel region conversion and merge results.

    Parameters
    ----------
    vcf : str
        Path to input VCF file
    out : str
        Path to output Parquet file
    regions : list[str]
        Genomic regions to process
    fmt_str : str
        bcftools query format string
    job_cfg : VCFJobConfig
        Job configuration
    jobs : int
        Number of parallel jobs
    read_filters : dict | list | None
        Read filter configuration
    downsample_reads : int | None
        Number of reads to downsample to
    downsample_seed : int | None
        Random seed for downsampling
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        spawn_ctx = _mp.get_context("spawn")
        part_files = _run_region_jobs(
            regions=regions,
            vcf_path=vcf,
            fmt_str=fmt_str,
            job_cfg=job_cfg,
            jobs=jobs,
            tmpdir=tmpdir,
            spawn_ctx=spawn_ctx,
        )
        if not part_files:
            log.error(
                "No Parquet part-files were produced\n"
                "This could mean:\n"
                "  1. All regions were empty (no variants in the VCF)\n"
                "  2. All data was filtered out by read filters\n"
                "  3. All regions failed to process (check errors above)"
            )
            if read_filters:
                log.error("")
                log.error("Read filters were applied. Check if filters are too strict:")
                if "filters" in read_filters:
                    for f in read_filters["filters"]:
                        value_display = f.get("value", f.get("value_field", "N/A"))
                        log.error(f"  - {f.get('name', 'unnamed')}: {f.get('field')} {f.get('op')} {value_display}")
            raise RuntimeError(
                "No Parquet part-files were produced – all regions empty or failed. "
                "Check log for details. If using read filters, they may be filtering out all data."
            )

        _merge_parquet_files_lazy(part_files, out, downsample_reads, downsample_seed)


def vcf_to_parquet(
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
    chunk_bp: int = CHUNK_BP_DEFAULT,
    jobs: int = DEFAULT_JOBS,
    read_filters_json: str | None = None,
    read_filter_json_key: str | None = None,
    downsample_reads: int | None = None,
    downsample_seed: int | None = None,
) -> None:
    """
    Convert VCF to Parquet using region-based parallel processing.

    This implementation uses:
    1. bedtools to split genome into equal-sized regions
    2. bcftools + AWK pipeline per region in parallel
    3. ProcessPoolExecutor for parallel region processing
    4. Lazy merging of Parquet files using Polars streaming

    Parameters
    ----------
    vcf : str
        Path to input VCF file
    out : str
        Path to output Parquet file
    drop_info : set[str] | None
        INFO fields to exclude
    drop_format : set[str] | None
        FORMAT fields to exclude
    chunk_bp : int
        Maximum number of base-pairs per chunk (default 300 Mbp).
    jobs : int
        Number of parallel jobs (0 = auto-detect CPU cores)
    read_filters_json : str | None
        Path to JSON file containing read filters
    read_filter_json_key : str | None
        Key in JSON file for read filters (uses entire JSON if None)
    downsample_reads : int | None
        If specified, downsample to this number of reads. If the total number of reads
        is less than this value, all reads are returned.
    downsample_seed : int | None
        Random seed for downsampling (optional, for reproducibility)
    """
    log.info(f"Converting {vcf} to {out} using region-based parallel processing")
    _assert_vcf_index_exists(vcf)

    # Setup parallel jobs and resolve tool paths
    jobs, bcftools, bedtools = _setup_parallel_jobs(jobs)

    # Prepare VCF metadata
    info_meta, fmt_meta, info_ids, scalar_fmt_ids, list_fmt_ids = _prepare_vcf_metadata(
        vcf, bcftools, drop_info, drop_format
    )

    # Setup column configuration
    format_ids = list(fmt_meta.keys())
    fmt_str, cols, list_fmt_indices = _setup_column_configuration(info_ids, format_ids, list_fmt_ids)

    with pl.StringCache():
        # Load read filters if provided
        read_filters = None
        if read_filters_json:
            read_filters = _load_read_filters(read_filters_json, read_filter_json_key)
            log.info(f"Loaded read filters from {read_filters_json}")
            if KEY_FILTERS in read_filters:
                log.info(f"Filter configuration contains {len(read_filters[KEY_FILTERS])} filter rules")

        # Generate genomic regions (fixed windows via bedtools)
        regions = _generate_genomic_regions(
            vcf,
            jobs,
            bcftools,
            bedtools,
            window_size=chunk_bp,
        )
        log.info(f"Created {len(regions)} regions: {regions[:5]}{'...' if len(regions) > 5 else ''}")  # noqa PLR2004

        # Build immutable job configuration (shared by every worker)
        job_cfg = VCFJobConfig(
            bcftools_path=bcftools,
            awk_script=_get_awk_script_path(),
            columns=cols,
            list_indices=list_fmt_indices,
            schema=_build_explicit_schema(cols, info_meta, fmt_meta),
            info_meta=info_meta,
            fmt_meta=fmt_meta,
            info_ids=info_ids,
            scalar_fmt_ids=scalar_fmt_ids,
            list_fmt_ids=list_fmt_ids,
            read_filters=read_filters,
        )

        # Execute parallel conversion
        _execute_parallel_conversion(
            vcf, out, regions, fmt_str, job_cfg, jobs, read_filters, downsample_reads, downsample_seed
        )

        log.info(f"Conversion completed: {out}")


def _process_region_to_parquet(
    region: str,
    vcf_path: str,
    fmt_str: str,
    job_cfg: VCFJobConfig,
    output_file: str,
) -> str:
    """
    Process a single genomic region and write a Parquet part-file.

    This function is designed to be pickle-safe for multiprocessing.
    All arguments are simple types that can be safely pickled.

    Parameters
    ----------
    region : str
        Genomic region in format "chr:start-end" or "chr"
    vcf_path : str
        Path to VCF file
    fmt_str : str
        bcftools query format string
    job_cfg : VCFJobConfig
        Job configuration object with processing metadata
    output_file : str
        Path for output parquet file

    Returns
    -------
    str
        Path to created parquet file, empty string if no data
    """
    # Configure logging for worker process (spawn context doesn't inherit parent's config)
    # Use INFO level to see all important messages from workers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,  # Reconfigure even if already set
    )

    try:
        with pl.StringCache():
            frame = _stream_region_to_polars(
                region=region,
                vcf_path=vcf_path,
                fmt_str=fmt_str,
                job_cfg=job_cfg,
            )
            if frame.is_empty():
                log.debug(f"Region {region}: No data (empty VCF region)")
                return ""

            rows_after_awk = frame.height
            frame = _cast_column_data_types(frame, job_cfg)
            log.debug(f"Region {region}: {rows_after_awk:,} rows after AWK explosion & type casting")

            # Apply read filters if configured
            if job_cfg.read_filters:
                rows_before_filter = frame.height
                frame = _apply_read_filters(frame, job_cfg.read_filters)
                rows_after_filter = frame.height
                pct_remaining = (rows_after_filter / rows_before_filter * 100) if rows_before_filter > 0 else 0
                log.info(
                    f"Region {region}: {rows_before_filter:,} → {rows_after_filter:,} rows after filtering "
                    f"({pct_remaining:.1f}% retained)"
                )
            else:
                log.debug(f"Region {region}: No filters applied, {frame.height:,} rows")

            # Skip writing if no reads remain after filtering
            if frame.is_empty():
                log.info(f"Region {region}: All reads filtered out")
                return ""

            frame.write_parquet(output_file)
            log.debug(f"Region {region}: Wrote {frame.height:,} rows to {output_file}")
            return output_file

    except Exception:
        # Emit full traceback inside the worker
        log.exception("Error processing region %s", region)
        raise


def _bcftools_awk_stdout(
    *,
    region: str,
    vcf_path: str,
    fmt_str: str,
    bcftools: str,
    awk_script: str,
    list_indices: list[int],
) -> str:
    """Return TSV (string) produced by `bcftools | awk` for a region."""
    bcftools_cmd = [bcftools, "query", "-f", fmt_str, vcf_path]
    if region:
        bcftools_cmd.insert(2, "-r")
        bcftools_cmd.insert(3, region)

    awk_cmd = [
        "awk",
        "-v",
        f"list_indices={','.join(map(str, list_indices))}",
        "-f",
        awk_script,
    ]

    bcftool = subprocess.Popen(bcftools_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    awk = subprocess.Popen(
        awk_cmd,
        stdin=bcftool.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if bcftool.stdout:  # close in parent
        bcftool.stdout.close()

    out, awk_err = awk.communicate()
    bcftool.wait()

    # Capture bcftools stderr
    bcftool_err = bcftool.stderr.read() if bcftool.stderr else ""

    # Check for errors or warnings
    if bcftool.returncode:  # pragma: no cover
        log.error(f"bcftools failed for region {region}: return code {bcftool.returncode}")
        log.error(f"bcftools stderr: {bcftool_err}")
        raise subprocess.CalledProcessError(bcftool.returncode, bcftools_cmd, bcftool_err)

    if awk.returncode:  # pragma: no cover
        log.error(f"AWK failed for region {region}: return code {awk.returncode}")
        log.error(f"AWK stderr: {awk_err}")
        raise subprocess.CalledProcessError(awk.returncode, awk_cmd, awk_err)

    # Log warnings from bcftools even if exit code is 0
    if bcftool_err and bcftool_err.strip():
        log.warning(f"bcftools warning for region {region}: {bcftool_err.strip()}")

    # Log warnings from awk even if exit code is 0
    if awk_err and awk_err.strip():
        log.warning(f"AWK warning for region {region}: {awk_err.strip()}")

    return out.strip()


def _frame_from_tsv(tsv: str, *, cols: list[str], schema: dict[str, pl.PolarsDataType]) -> pl.DataFrame:
    """String TSV → Polars DataFrame (empty frame if no rows)."""
    if not tsv:
        return pl.DataFrame({c: [] for c in cols})
    return pl.read_csv(
        StringIO(tsv),
        separator="\t",
        has_header=False,
        new_columns=cols,
        null_values=["."],
        schema=schema,
    )


def _stream_region_to_polars(
    region: str,
    vcf_path: str,
    fmt_str: str,
    job_cfg: VCFJobConfig,
) -> pl.DataFrame:
    """Run bcftools→awk and return a typed Polars DataFrame for *region*."""
    tsv = _bcftools_awk_stdout(
        region=region,
        vcf_path=vcf_path,
        fmt_str=fmt_str,
        bcftools=job_cfg.bcftools_path,
        awk_script=job_cfg.awk_script,
        list_indices=job_cfg.list_indices,
    )

    frame = _frame_from_tsv(tsv, cols=job_cfg.columns, schema=job_cfg.schema)
    if frame.is_empty():
        # Count raw TSV lines to detect truncation (only when there's a problem)
        # Note: After strip(), a non-empty string with N lines has N-1 newlines
        tsv_lines = (tsv.count("\n") + 1) if tsv else 0
        if tsv_lines > 0:
            log.warning(
                f"Region {region}: AWK produced {tsv_lines} TSV lines but DataFrame is empty - possible parsing error"
            )
        return frame

    log.debug(f"Region {region}: {frame.height:,} DataFrame rows")
    return _cast_column_data_types(frame, job_cfg)


def _cast_ref_alt_columns() -> list[pl.Expr]:
    """
    Return casting expressions for the fixed REF / ALT columns.
    X_ALT is *not* included here because it is already processed via the
    INFO/FORMAT loops; adding it again would create duplicate aliases.
    """
    return [
        _cast_expr(REF, {"type": "String", "cat": REF_ALLELE_CATS}),
        _cast_expr(ALT, {"type": "String", "cat": ALT_ALLELE_CATS}),
    ]


def _cast_column_data_types(featuremap_dataframe: pl.DataFrame, job_cfg: VCFJobConfig) -> pl.DataFrame:
    """Apply column casting and categorical processing to a region DataFrame."""
    exprs: list[pl.Expr] = [pl.col(POS).cast(pl.Int64)]

    # build expressions for INFO / FORMAT
    exprs.extend(_cast_expr(tag, job_cfg.info_meta[tag]) for tag in job_cfg.info_ids)
    exprs.extend(_cast_expr(tag, job_cfg.fmt_meta[tag]) for tag in job_cfg.scalar_fmt_ids)
    exprs.extend(_cast_expr(tag, job_cfg.fmt_meta[tag]) for tag in job_cfg.list_fmt_ids)

    # QUAL ─ force Float64 even if all values are missing
    if QUAL in featuremap_dataframe.columns:
        exprs.append(_cast_expr(QUAL, {"type": "Float", "cat": None}))

    # REF / ALT
    exprs.extend(_cast_ref_alt_columns())

    # Single materialisation
    return featuremap_dataframe.with_columns(exprs)


# ────────────────────────────── CLI entry point ─────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """
    Minimal command-line interface, e.g.:

    $ python -m ugbio_featuremap.featuremap_to_dataframe  \
         --in sample.vcf.gz --out sample.parquet --jobs 4 --drop-format GT AD
    """
    parser = argparse.ArgumentParser(description="Convert feature-map VCF → Parquet", allow_abbrev=True)
    parser.add_argument("--input", required=True, help="Input VCF/BCF (bgzipped ok)")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="Parallel jobs (0 = auto)")
    parser.add_argument("--drop-info", nargs="*", default=[], help="INFO tags to drop")
    parser.add_argument("--drop-format", nargs="*", default=["GT", "AD", "X_TCM"], help="FORMAT tags to drop")
    parser.add_argument(
        "--chunk-bp",
        type=int,
        default=CHUNK_BP_DEFAULT,
        help=f"Base-pairs per processing chunk (default {CHUNK_BP_DEFAULT} bp)",
    )
    parser.add_argument(
        "--read_filters_json",
        "--read_filter_json",  # Allow both singular and plural forms
        required=False,
        default=None,
        help="JSON file with read filters",
    )
    parser.add_argument(
        "--read_filter_json_key", required=False, default=None, help="Key in JSON file for read filters"
    )
    parser.add_argument(
        "--downsample_reads",
        type=int,
        default=None,
        help="Downsample to this number of reads (optional). If total reads < this value, all reads are kept.",
    )
    parser.add_argument(
        "--downsample_seed",
        type=int,
        default=None,
        help="Random seed for downsampling (optional, for reproducibility)",
    )
    # ───────────── new verbose flag ─────────────
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    # -------------------------------------------
    args = parser.parse_args(argv)

    # ───────────── logging setup ───────────────
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        log.setLevel(logging.DEBUG)
    # -------------------------------------------

    vcf_to_parquet(
        vcf=args.input,
        out=args.output,
        drop_info=set(args.drop_info),
        drop_format=set(args.drop_format),
        chunk_bp=args.chunk_bp,
        jobs=args.jobs,
        read_filters_json=args.read_filters_json,
        read_filter_json_key=args.read_filter_json_key,
        downsample_reads=args.downsample_reads,
        downsample_seed=args.downsample_seed,
    )


if __name__ == "__main__":
    main()
