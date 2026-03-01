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
import multiprocessing as _mp
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
import pysam
from ugbio_core.logger import logger as log

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
    TYPE_UNKNOWN,
    TYPE_UNNAMED,
    _create_filter_columns,
    _create_final_filter_column,
    validate_filter_config,
)

# Configuration constants
DEFAULT_JOBS = 0  # 0 means auto-detect CPU cores
CHUNK_BP_DEFAULT = 10_000_000  # 10 Mbp per processing chunk
MAX_DEBUG_FILES_TO_SHOW = 3  # Number of file paths to show in debug logs


def _configure_logging(log_level: int, *, check_worker_cache: bool = False) -> None:
    """
    Configure logging for the current process.

    Parameters
    ----------
    log_level : int
        Logging level to configure (e.g., logging.INFO, logging.DEBUG)
    check_worker_cache : bool
        If True, check if logging is already configured at this level (for worker processes).
        If False, always reconfigure (for main process).
    """
    # Use function attribute to track configured log level in worker processes
    # This avoids using global variables
    if not hasattr(_configure_logging, "_worker_log_level"):
        _configure_logging._worker_log_level = None  # type: ignore[attr-defined]

    # For worker processes, only configure if not already configured or if level changed
    if check_worker_cache and _configure_logging._worker_log_level == log_level:  # type: ignore[attr-defined]
        return

    # Reconfigure the root logger for this process.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    root_logger.setLevel(log_level)
    log.setLevel(log_level)
    if log.handlers:
        # Avoid double logging when the shared logger already has handlers.
        log.propagate = False
        for handler in log.handlers:
            handler.setLevel(log_level)

    # Track configured level for worker processes
    if check_worker_cache:
        _configure_logging._worker_log_level = log_level  # type: ignore[attr-defined]


@dataclass
class ColumnConfig:
    """Configuration for column processing."""

    # Metadata dictionaries
    info_meta: dict  # INFO field metadata (e.g. {X_HMER_REF: {"num": "1", "type": "String", "cat": None}})
    fmt_meta: dict  # FORMAT fields metadata (e.g. {DP: {"num": "1", "type": "Integer", "cat": None}})

    # Field names
    info_ids: list[str]  # INFO field names (e.g. X_NEXT1)
    fmt_ids: list[str]  # FORMAT field names (e.g. DP, MQUAL)
    scalar_fmt_ids: list[str]  # FORMAT field names with scalar values (e.g. DP)
    list_fmt_ids: list[str]  # FORMAT field names with list values (e.g. MQUAL)

    # Expand columns
    expand_columns: dict[str, int] | None = None  # columns to expand into multiple columns (e.g. {"AD": 2})
    expand_sizes: list[int] | None = None  # Size of each expand column

    # Indices
    list_indices: list[int] | None = None  # 0-based column indices for list columns
    expand_indices: list[int] | None = None  # 0-based column indices for expand columns


@dataclass
class VCFJobConfig:
    """Static metadata shared by all workers."""

    bcftools_path: str
    awk_script: str
    columns: list[str]
    schema: dict[str, pl.PolarsDataType]
    column_config: ColumnConfig
    sample_list: list[str]
    log_level: int
    read_filters: dict | None = None


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
RESERVED = {CHROM, POS, REF, ALT, QUAL, FILTER}

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

# Aggregation types and their Polars dtypes for aggregate mode
AGGREGATION_TYPES = [
    ("mean", pl.Float64),
    ("min", pl.Float64),
    ("max", pl.Float64),
    ("count", pl.Int64),
    ("count_zero", pl.Int64),
]


# ───────────────── read filter functions ─────────────────────────────────────


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

        log.info(f"Loaded filter configuration: {json.dumps(result, indent=2)[:500]}...")
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
    """
    if op == "eq" and value == ".":
        log.debug(f"Adding null handling for ID filter: '{filter_name}' (treating null as '.')")
        return lazy_frame.with_columns((pl.col(col_name) | pl.col(FIELD_ID).is_null()).alias(col_name))
    return lazy_frame


def _apply_gnomad_null_handling(lazy_frame: pl.LazyFrame, col_name: str, filter_name: str) -> pl.LazyFrame:
    """
    Apply special null handling for gnomAD_AF filters.

    Variants not present in gnomAD have null/missing gnomAD_AF values.
    For filtering, missing gnomAD_AF should be treated as rare (passing the filter).
    """
    log.debug(f"Adding null handling for gnomAD_AF filter: '{filter_name}'")
    return lazy_frame.with_columns((pl.col(col_name) | pl.col(FIELD_GNOMAD_AF).is_null()).alias(col_name))


def _apply_ug_hcr_null_handling(lazy_frame: pl.LazyFrame, col_name: str, filter_name: str) -> pl.LazyFrame:
    """
    Apply special null handling for UG_HCR filters.

    For UG_HCR (Ultimagen High Confidence Region), null values indicate the variant
    is NOT in the HCR and should be filtered out. Only non-null TRUE values pass.
    """
    log.debug(f"Adding NOT-null requirement for UG_HCR filter: '{filter_name}'")
    return lazy_frame.with_columns((pl.col(col_name) & pl.col(FIELD_UG_HCR).is_not_null()).alias(col_name))


def _prepare_filter_config(read_filters: dict | list | None) -> list[dict]:
    """
    Extract and validate filter configuration from input.

    Returns list of validated filter dictionaries, excluding raw/downsample filters.
    """
    if isinstance(read_filters, list):
        all_filters = read_filters
    else:
        all_filters = read_filters.get(KEY_FILTERS, [])

    filters_to_apply = [f for f in all_filters if f.get(KEY_TYPE) not in {TYPE_RAW, TYPE_DOWNSAMPLE}]

    if not filters_to_apply:
        log.debug("No applicable filters found (only raw/downsample filters present)")
        return []

    log.debug(f"Filters to apply ({len(filters_to_apply)}):")
    for i, f in enumerate(filters_to_apply):
        log.debug(
            f"  {i+1}. {f.get(KEY_NAME, TYPE_UNNAMED)}: {f.get(KEY_FIELD)} {f.get(KEY_OP)} "
            f"{f.get(KEY_VALUE, f.get(KEY_VALUE_FIELD, 'N/A'))}"
        )

    filter_config = {KEY_FILTERS: filters_to_apply}
    try:
        validate_filter_config(filter_config)
    except Exception as e:
        filter_names = [f.get(KEY_NAME, f.get(KEY_TYPE, TYPE_UNNAMED)) for f in filters_to_apply]
        log.error(f"FATAL: Filter validation failed. Filters: {filter_names}")
        log.error(f"Validation error: {e}")
        raise RuntimeError(f"Filter validation failed for filters {filter_names}: {e}") from e

    return filters_to_apply


def _apply_null_handling_to_filters(
    lazy_frame: pl.LazyFrame, filters_to_apply: list[dict], filter_cols: list[str]
) -> pl.LazyFrame:
    """
    Apply special null-handling logic for specific fields.

    - gnomAD_AF: null = not in gnomAD = rare (should pass filter)
    - ID: null = "." in VCF = not in dbSNP (special handling)
    - UG_HCR: null = not in HCR (should fail filter)
    """
    for f in filters_to_apply:
        field_name = f.get(KEY_FIELD, "")
        filter_name = f.get(KEY_NAME, f"{field_name}_filter")
        col_name = f"__filter_{filter_name}"

        if col_name not in filter_cols:
            continue

        if field_name == FIELD_GNOMAD_AF and f.get(KEY_TYPE) == TYPE_REGION:
            lazy_frame = _apply_gnomad_null_handling(lazy_frame, col_name, filter_name)
        elif field_name == FIELD_ID and f.get(KEY_TYPE) == TYPE_REGION:
            lazy_frame = _apply_id_null_handling(lazy_frame, col_name, filter_name, f.get(KEY_OP), f.get(KEY_VALUE))
        elif field_name == FIELD_UG_HCR and f.get(KEY_TYPE) == TYPE_REGION and f.get(KEY_OP) == "ne":
            lazy_frame = _apply_ug_hcr_null_handling(lazy_frame, col_name, filter_name)

    return lazy_frame


def _log_filter_statistics(df_with_filters: pl.DataFrame, filter_cols: list[str]) -> None:
    """Log per-filter statistics showing pass/fail counts."""
    log.debug(f"Filter statistics for region (total rows: {df_with_filters.height:,}):")
    for col in filter_cols:
        if col in df_with_filters.columns:
            pass_count = df_with_filters[col].sum()
            fail_count = df_with_filters.height - pass_count
            pct = 100.0 * pass_count / df_with_filters.height if df_with_filters.height > 0 else 0
            log.debug(f"  {col}: {pass_count:,} pass ({pct:.1f}%) / {fail_count:,} fail")


def _apply_read_filters(frame: pl.DataFrame, read_filters: dict | list | None) -> pl.DataFrame:
    """
    Apply read filters to a dataframe using the existing filter framework.

    Parameters
    ----------
    frame : pl.DataFrame
        Input dataframe to filter
    read_filters : dict | list | None
        Filter configuration in the same format as used by filter_dataframe.py

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
        filters_to_apply = _prepare_filter_config(read_filters)
        if not filters_to_apply:
            log.info("No read filters provided - skipping filter step")
            return frame

        log.debug(f"Applying {len(filters_to_apply)} filters to dataframe with {frame.height:,} rows")

        lazy_frame = frame.lazy()
        lazy_frame, filter_cols = _create_filter_columns(lazy_frame, filters_to_apply)
        log.debug(f"Created filter columns: {filter_cols}")

        lazy_frame = _apply_null_handling_to_filters(lazy_frame, filters_to_apply, filter_cols)

        df_with_filters = lazy_frame.collect()
        _log_filter_statistics(df_with_filters, filter_cols)

        lazy_frame = df_with_filters.lazy()
        lazy_frame = _create_final_filter_column(lazy_frame, filter_cols, None)

        df_final = lazy_frame.collect()
        final_pass = df_final["__filter_final"].sum()
        final_fail = df_final.height - final_pass
        pct = 100.0 * final_pass / df_final.height if df_final.height > 0 else 0
        log.debug(f"  __filter_final (ALL COMBINED): {final_pass:,} pass ({pct:.1f}%) / {final_fail:,} fail")

        filtered_frame = df_final.filter(pl.col("__filter_final")).select(pl.exclude("^__filter_.*$"))

        log.debug(f"Read filtering: {frame.height:,} → {filtered_frame.height:,} reads")
        return filtered_frame

    except Exception as e:
        filter_summary = []
        try:
            all_filters = read_filters if isinstance(read_filters, list) else read_filters.get("filters", [])
            for f in all_filters:
                filter_summary.append(f.get(KEY_NAME, f.get(KEY_TYPE, TYPE_UNNAMED)))
        except Exception:
            filter_summary = ["<error parsing filters>"]

        log.error(f"FATAL: Unexpected error applying read filters. Filters: {filter_summary}")
        log.error(f"Error: {e}")
        raise RuntimeError(f"Unexpected error applying read filters {filter_summary}: {e}") from e


# ───────────────── header helpers ────────────────────────────────────────────


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
    log.debug(f"Parsing VCF header from {vcf}")
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
    log.info(f"Parsed header: {len(info)} INFO fields, {len(fmt)} FORMAT fields")
    log.debug(f"INFO fields: {list(info.keys())}")
    log.debug(f"FORMAT fields: {list(fmt.keys())}")
    return info, fmt


def _cast_expr(col: str, meta: dict) -> pl.Expr:
    """
    Build a Polars expression that
    1. normalises "" / "." → null
    2. fills remaining nulls
    3. casts to the final dtype
    """
    base = pl.when(pl.col(col).cast(pl.Utf8).is_in(["", "."])).then(None).otherwise(pl.col(col))

    # ---- categorical handling -------------------------------------------
    if meta["cat"]:
        cats = meta["cat"] + ([] if "" in meta["cat"] else [""])
        return base.fill_null(value="").cast(pl.Enum(cats), strict=True).alias(col)
    elif meta["type"] == "Flag":
        return base.fill_null(value=False).cast(pl.Boolean, strict=True).alias(col)
    elif meta["type"] in _POLARS_DTYPE:
        return base.fill_null(value=0).cast(_POLARS_DTYPE[meta["type"]], strict=True).alias(col)

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
    log.debug("Extracting contig information from VCF header")
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
        log.info(f"Generating genomic regions with window size {window_size:,} bp")
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
        regions.append(f"{chrom}:{int(start0)+1}-{end}")  # convert to 1-based inclusive
    return regions


def _sanity_check_format_numbers(
    format_ids: list[str], fmt_meta: dict, expand_columns: dict[str, int] | None = None
) -> None:
    """
    Raise an error if any FORMAT tag has a Number we do not support. (not "1" or ".")
    When in aggregate mode, we allow all Number values for expand columns.
    """
    expand_columns = expand_columns or {}
    for tag in format_ids:
        num = fmt_meta[tag]["num"]
        if num not in {"1", "."}:
            if tag not in expand_columns:
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
    """Build the bcftools query format string (includes %QUAL and %ID)."""
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


def _get_awk_script_path(mode: str = "explode") -> str:
    """Get path to the AWK script for list handling.

    Parameters
    ----------
    mode : str
        Either "explode" or "aggregate" to select the appropriate script.

    Returns
    -------
    str
        Path to the AWK script.
    """
    if mode not in ("explode", "aggregate"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'explode' or 'aggregate'")

    script_name = "explode_lists.awk" if mode == "explode" else "aggregate_lists.awk"

    # First try the standard approach (should work in development and installed package)
    script_dir = Path(__file__).parent
    awk_script = script_dir / script_name
    if awk_script.exists():
        return str(awk_script)

    # Fallback using importlib.resources for robust package resource access
    try:
        import importlib.resources as pkg_resources

        ref = pkg_resources.files("ugbio_featuremap") / script_name
        with pkg_resources.as_file(ref) as awk_path:
            return str(awk_path)
    except ImportError:
        pass

    raise FileNotFoundError(f"AWK script not found: {awk_script}")


def _merge_parquet_files_lazy(  # noqa: PLR0912
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
        empty_df = pl.DataFrame({"CHROM": [], "POS": [], "REF": [], "ALT": []})
        empty_df.write_parquet(output_path)
        return

    if len(parquet_files) == 1:
        log.debug(f"Single part file, moving to output: {output_path}")
        if downsample_reads is None:
            shutil.move(parquet_files[0], output_path)
        else:
            dataframe_size = pl.scan_parquet(parquet_files[0]).select(pl.len()).collect().item()
            if dataframe_size <= downsample_reads:
                log.debug(
                    f"Dataset has {dataframe_size} rows, which is <= requested {downsample_reads} - keeping all rows"
                )
                shutil.move(parquet_files[0], output_path)
            else:
                log.info(f"Sampling {downsample_reads} rows from {dataframe_size} total rows")
                pl.read_parquet(parquet_files[0]).sample(n=downsample_reads, seed=downsample_seed).write_parquet(
                    output_path
                )
            Path(parquet_files[0]).unlink(missing_ok=True)
        return

    log.info(f"Merging {len(parquet_files)} Parquet part-file(s) into final output")
    log.debug(
        f"Part-files: {parquet_files[:MAX_DEBUG_FILES_TO_SHOW]}"
        f"{'...' if len(parquet_files) > MAX_DEBUG_FILES_TO_SHOW else ''}"
    )

    lazy_frames = [pl.scan_parquet(f) for f in parquet_files]

    lazy_frames_size = [frame.select(pl.len()).collect().item() for frame in lazy_frames]
    log.debug(f"Individual Parquet file sizes (rows): {', '.join(map(str, lazy_frames_size))}")

    merged_lazy = pl.concat(lazy_frames, how="vertical")

    if downsample_reads is None:
        log.debug(f"Writing merged Parquet to: {output_path}")
        merged_lazy.sink_parquet(output_path)
    else:
        merged_df = merged_lazy.collect()
        if merged_df.height <= downsample_reads:
            log.debug(
                f"Dataset has {merged_df.height} rows, which is <= requested {downsample_reads} - keeping all rows"
            )
            merged_df.write_parquet(output_path)
        else:
            log.info(f"Sampling {downsample_reads} rows from {merged_df.height} total rows")
            merged_df.sample(n=downsample_reads, seed=downsample_seed).write_parquet(output_path)

    for f in parquet_files:
        Path(f).unlink(missing_ok=True)

    log.debug(f"Successfully merged {len(parquet_files)} part-file(s)")


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


def _transform_cols_and_schema_for_aggregate(  # noqa: C901
    cols: list[str],
    list_fmt_ids: list[str],
    info_meta: dict,
    fmt_meta: dict,
    expand_columns: dict[str, int] | None = None,
) -> tuple[list[str], dict[str, pl.PolarsDataType]]:
    """
    Transform columns and schema for aggregate mode.

    In aggregate mode:
    - Each list column is replaced with aggregation columns:
      {col}_mean, {col}_min, {col}_max, {col}_count, {col}_count_zero
      (defined by AGGREGATION_TYPES)
    - Each expand column is replaced with indexed columns:
      {col}_0, {col}_1, ... (up to size)

    Parameters
    ----------
    cols : list[str]
        Original column names
    list_fmt_ids : list[str]
        List of FORMAT field IDs that are lists
    info_meta : dict
        INFO field metadata
    fmt_meta : dict
        FORMAT field metadata
    expand_columns : dict[str, int] | None
        Mapping of column names to their expand sizes (e.g., {"AD": 2})

    Returns
    -------
    tuple[list[str], dict[str, pl.PolarsDataType]]
        Transformed columns and schema
    """
    transformed_cols: list[str] = []
    expand_columns = expand_columns or {}

    for col in cols:
        if col in expand_columns:
            # Replace expand column with indexed columns
            size = expand_columns[col]
            transformed_cols.extend([f"{col}_{i}" for i in range(size)])
        elif col in list_fmt_ids:
            # Replace list column with aggregation columns
            transformed_cols.extend([f"{col}_{suffix}" for suffix, _ in AGGREGATION_TYPES])
        else:
            # Keep non-list columns as-is
            transformed_cols.append(col)

    # Build schema for transformed columns using existing function
    schema = _build_explicit_schema(transformed_cols, info_meta, fmt_meta)

    # Override schema for aggregation columns (they're not in metadata)
    for col in list_fmt_ids:
        if col not in expand_columns:  # Don't add aggregation schema for expand columns
            for suffix, dtype in AGGREGATION_TYPES:
                schema[f"{col}_{suffix}"] = dtype

    # Override schema for expand columns (determine type from metadata)
    for col, size in expand_columns.items():
        if col in fmt_meta and fmt_meta[col]["type"] in _POLARS_DTYPE:
            col_type = _POLARS_DTYPE[fmt_meta[col]["type"]]
        else:
            col_type = pl.Utf8

        for i in range(size):
            schema[f"{col}_{i}"] = col_type

    return transformed_cols, schema


def _validate_expand_columns(
    expand_columns: dict[str, int] | None,
    list_mode: str,
    fmt_meta: dict[str, dict] | None = None,
) -> None:
    """
    Validate expand_columns parameter.

    Parameters
    ----------
    expand_columns : dict[str, int] | None
        Mapping of column names to their expand sizes
    list_mode : str
        How to handle list format fields: "explode" or "aggregate"
    fmt_meta : dict[str, dict] | None
        FORMAT metadata mapping from header_meta. When provided, validates that expand_columns
        only refers to list-type FORMAT fields present in the header.

    Raises
    ------
    ValueError
        If expand_columns is used in explode mode, if any size is non-positive, or if
        expand_columns contains missing or scalar FORMAT fields.
    """
    if not expand_columns:
        return

    if list_mode == "explode":
        raise ValueError(
            f"expand_columns is not supported in explode mode (got {expand_columns}). "
            "Use list_mode='aggregate' to use expand_columns."
        )

    for col, size in expand_columns.items():
        if size <= 0:
            raise ValueError(
                f"expand_columns size must be positive (got {col}:{size}). "
                "Each expand column must have a size greater than 0."
            )

    if fmt_meta is None:
        return

    missing = [col for col in expand_columns if col not in fmt_meta]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "expand_columns must refer to FORMAT fields present in the VCF header. " f"Missing: {missing_str}."
        )

    scalar_fields = [col for col in expand_columns if fmt_meta[col]["num"] == "1"]
    if scalar_fields:
        scalar_str = ", ".join(sorted(scalar_fields))
        raise ValueError(
            "expand_columns can only be used with list-type FORMAT fields (Number != 1). "
            f"Scalar fields: {scalar_str}."
        )


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
    completed = 0
    total = len(regions)

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
            if exc is None:
                completed += 1
                region, part_path = futures[fut]
                log.debug(f"Completed region {region} ({completed}/{total})")
                if completed % max(1, total // 10) == 0 or completed == total:
                    log.info(f"Progress: {completed}/{total} regions processed ({100*completed//total}%)")
            else:
                region, part_path = futures[fut]
                failures.append((region, part_path, exc))
                log.error(
                    "Error processing region %s (part=%s)\n%s",
                    region,
                    part_path,
                    "".join(traceback.format_exception(exc)),
                )

        if failures:
            first_region, first_part, first_exc = failures[0]
            raise RuntimeError(
                f"{len(failures)} region(s) failed during VCF→Parquet conversion; "
                f"first failure: {first_region} (part={first_part})"
            ) from first_exc

    valid_files = [p for p in part_files if Path(p).is_file()]
    log.info(f"Successfully processed {len(valid_files)} region(s) with data")
    return valid_files


def _drop_fields(
    info_meta: dict,
    fmt_meta: dict,
    drop_info: set[str] | None,
    drop_format: set[str] | None,
    expand_columns: dict[str, int] | None = None,
) -> tuple[dict, dict]:
    """
    Drop fields from INFO and FORMAT metadata.
    If expand_columns are provided, any FORMAT fields that are used for expand columns will not be dropped.
    """
    if drop_info:
        dropped_info = [k for k in info_meta.keys() if k in drop_info]
        info_meta = {k: v for k, v in info_meta.items() if k not in drop_info}
        if dropped_info:
            log.info(f"Dropping {len(dropped_info)} INFO field(s): {dropped_info}")
    if drop_format:
        if expand_columns:
            new_drop_format = drop_format - set(expand_columns.keys())
            if new_drop_format != drop_format:
                drop_format = new_drop_format
                log.info(
                    f"One or more FORMAT fields were asked to be dropped, but are used for expand columns. "
                    f"Note they will not be dropped. {drop_format=}, {expand_columns=}"
                )
        dropped_fmt = [k for k in fmt_meta.keys() if k in drop_format]
        fmt_meta = {k: v for k, v in fmt_meta.items() if k not in drop_format}
        if dropped_fmt:
            log.info(f"Dropping {len(dropped_fmt)} FORMAT field(s): {dropped_fmt}")
    return info_meta, fmt_meta


def vcf_to_parquet(  # noqa: PLR0915, C901, PLR0912, PLR0913
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
    chunk_bp: int = CHUNK_BP_DEFAULT,
    jobs: int = DEFAULT_JOBS,
    list_mode: str = "explode",
    log_level: int = logging.INFO,
    expand_columns: dict[str, int] | None = None,
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
        Maximum number of base-pairs per chunk (default 10 Mbp).
    jobs : int
        Number of parallel jobs (0 = auto-detect CPU cores)
    list_mode : str
        How to handle list format fields: "explode" (one row per list element) or
        "aggregate" (mean, min, max, count, count_zero metrics). Default is "explode".
    log_level : int
        Logging level (e.g., logging.DEBUG, logging.INFO)
    expand_columns : dict[str, int] | None
        Mapping of list-type FORMAT field names to their expand sizes. In aggregate mode,
        these columns are expanded into indexed columns (e.g., {"AD": 2} produces
        AD_0, AD_1) instead of being aggregated. Only used when list_mode="aggregate".
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
    log.info(f"Input: {vcf}")
    log.info(f"Output: {out}")
    log.info(f"List mode: {list_mode}")

    _assert_vcf_index_exists(vcf)
    log.debug("VCF index found")

    # Auto-detect optimal job count if not specified
    if jobs == 0:
        jobs = os.cpu_count() or 4

    log.info(f"Using {jobs} parallel job(s) for region processing")

    # Resolve tool paths
    bcftools = _resolve_bcftools_command()
    bedtools = _resolve_bedtools_command()

    # Parse VCF header
    info_meta, fmt_meta = header_meta(vcf, bcftools, threads=1)

    _validate_expand_columns(expand_columns, list_mode, fmt_meta)

    info_meta, fmt_meta = _drop_fields(info_meta, fmt_meta, drop_info, drop_format, expand_columns)

    # Validate FORMAT fields
    format_ids = list(fmt_meta.keys())
    _sanity_check_format_numbers(format_ids, fmt_meta, expand_columns)

    # Split FORMAT fields
    scalar_fmt_ids, list_fmt_ids = _split_format_ids(format_ids, fmt_meta)
    info_ids = list(info_meta.keys())

    # Handle expand columns: remove them from list_fmt_ids for aggregation
    expand_columns = expand_columns or {}
    effective_list_fmt_ids = [col for col in list_fmt_ids if col not in expand_columns]
    expand_ids = [col for col in list_fmt_ids if col in expand_columns]

    if expand_columns:
        log.info(f"Expand columns (will be expanded): {expand_ids}")
    log.info(
        f"Processing {len(scalar_fmt_ids)} scalar FORMAT field(s), {len(effective_list_fmt_ids)} list FORMAT field(s)"
    )
    log.debug(f"Scalar FORMAT fields: {scalar_fmt_ids}")
    log.debug(f"List FORMAT fields: {effective_list_fmt_ids}")

    # Build query string and column names
    fmt_str = _make_query_string(format_ids, info_ids)
    cols = [CHROM, POS, ID, QUAL, REF, ALT] + info_ids + format_ids

    # Find indices of list columns for AWK script (excluding fixed-tuple columns)
    list_fmt_indices = []
    base_cols = [CHROM, POS, ID, QUAL, REF, ALT] + info_ids
    for list_col in effective_list_fmt_ids:
        idx = len(base_cols) + format_ids.index(list_col)
        list_fmt_indices.append(idx)

    # Find indices of expand columns for AWK script
    expand_indices = []
    expand_sizes = []
    for expand_col in expand_ids:
        idx = len(base_cols) + format_ids.index(expand_col)
        expand_indices.append(idx)
        expand_sizes.append(expand_columns[expand_col])

    # Fetch sample list ONCE in main process
    sample_list = _get_sample_list(vcf)
    log.info(f"Found {len(sample_list)} sample(s): {sample_list}")

    if not sample_list:
        raise ValueError(
            f"VCF file '{vcf}' contains no samples. "
            "VCF files without samples cannot be processed as variant data requires sample-specific FORMAT fields."
        )

    # Cache fmt_ids
    fmt_ids = list(fmt_meta.keys())

    # Transform columns and schema for aggregate mode
    if list_mode == "aggregate":
        log.info("Using aggregate mode: list fields will be aggregated (mean, min, max, count, count_zero)")
        cols, schema = _transform_cols_and_schema_for_aggregate(
            cols, effective_list_fmt_ids, info_meta, fmt_meta, expand_columns
        )
    else:
        log.info("Using explode mode: list fields will be expanded to one row per element")
        schema = _build_explicit_schema(cols, info_meta, fmt_meta)

    with pl.StringCache():
        # Load read filters if provided
        read_filters = None
        if read_filters_json:
            read_filters = _load_read_filters(read_filters_json, read_filter_json_key)
            log.info(f"Loaded read filters from {read_filters_json}")
            if isinstance(read_filters, dict) and KEY_FILTERS in read_filters:
                log.info(f"Filter configuration contains {len(read_filters[KEY_FILTERS])} filter rules")

        # Generate genomic regions (fixed windows via bedtools)
        regions = _generate_genomic_regions(
            vcf,
            jobs,
            bcftools,
            bedtools,
            window_size=chunk_bp,
        )
        log.info(f"Created {len(regions)} regions for parallel processing")
        log.debug(f"First 5 regions: {regions[:5]}{'...' if len(regions) > 5 else ''}")  # noqa PLR2004

        # Build column configuration
        col_cfg = ColumnConfig(
            info_meta=info_meta,
            fmt_meta=fmt_meta,
            info_ids=info_ids,
            fmt_ids=fmt_ids,
            scalar_fmt_ids=scalar_fmt_ids,
            list_fmt_ids=effective_list_fmt_ids,
            expand_columns=expand_columns if list_mode == "aggregate" else None,
            expand_sizes=expand_sizes if list_mode == "aggregate" else None,
            expand_indices=expand_indices if list_mode == "aggregate" else None,
            list_indices=list_fmt_indices,
        )

        job_cfg = VCFJobConfig(
            bcftools_path=bcftools,
            awk_script=_get_awk_script_path(mode=list_mode),
            columns=cols,
            schema=schema,
            column_config=col_cfg,
            sample_list=sample_list,
            log_level=log_level,
            read_filters=read_filters,
        )

        # Temporary directory for ordered part-files
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
                    if isinstance(read_filters, dict) and KEY_FILTERS in read_filters:
                        for f in read_filters[KEY_FILTERS]:
                            value_display = f.get(KEY_VALUE, f.get(KEY_VALUE_FIELD, "N/A"))
                            log.error(
                                f"  - {f.get(KEY_NAME, TYPE_UNNAMED)}: {f.get(KEY_FIELD)} "
                                f"{f.get(KEY_OP, TYPE_UNKNOWN)} {value_display}"
                            )
                raise RuntimeError(
                    "No Parquet part-files were produced – all regions empty or failed. "
                    "Check log for details. If using read filters, they may be filtering out all data."
                )

            _merge_parquet_files_lazy(part_files, out, downsample_reads, downsample_seed)

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
    # Configure logging once per worker process
    _configure_logging(job_cfg.log_level, check_worker_cache=True)
    try:
        log.debug(f"Processing region {region}")
        with pl.StringCache():
            frame = _stream_region_to_polars(
                region=region,
                vcf_path=vcf_path,
                fmt_str=fmt_str,
                job_cfg=job_cfg,
            )
            if frame.is_empty():
                log.debug(f"Region {region} contains no data, skipping")
                return ""

            rows_after_awk = frame.height
            log.debug(f"Region {region}: {rows_after_awk:,} rows after AWK explosion & type casting")

            # Apply read filters for single-sample VCFs (multi-sample filters
            # are applied per-sample inside _convert_sample_frames_to_polars)
            if job_cfg.read_filters and len(job_cfg.sample_list) == 1:
                rows_before_filter = frame.height
                frame = _apply_read_filters(frame, job_cfg.read_filters)
                rows_after_filter = frame.height
                pct_remaining = (rows_after_filter / rows_before_filter * 100) if rows_before_filter > 0 else 0
                log.info(
                    f"Region {region}: {rows_before_filter:,} -> {rows_after_filter:,} rows after filtering "
                    f"({pct_remaining:.1f}% retained)"
                )
                if frame.is_empty():
                    log.debug(f"Region {region}: all rows filtered out")
                    return ""

            log.debug(f"Region {region}: writing {frame.height} rows to {output_file}")
            frame.write_parquet(output_file)
            return output_file

    except Exception:
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
    sample_name: str,
    expand_indices: list[int] | None = None,
    expand_sizes: list[int] | None = None,
) -> str:
    """Return TSV (string) produced by `bcftools | awk` for a region.

    Parameters
    ----------
    region : str
        Genomic region in format "chr:start-end"
    vcf_path : str
        Path to the VCF file
    fmt_str : str
        bcftools query format string
    bcftools : str
        Path to bcftools executable
    awk_script : str
        Path to the AWK script
    list_indices : list[int]
        0-based column indices for list columns to aggregate
    sample_name : str
        Name of the sample to extract
    expand_indices : list[int] | None
        0-based column indices for expand columns to split
    expand_sizes : list[int] | None
        Size of each expand column (parallel array with expand_indices)

    Returns
    -------
    str
        TSV string output from bcftools | awk pipeline
    """
    bcftools_cmd = [bcftools, "query", "-s", sample_name, "-f", fmt_str, vcf_path]
    if region:
        bcftools_cmd.insert(2, "-r")
        bcftools_cmd.insert(3, region)

    awk_cmd = [
        "awk",
        "-v",
        f"list_indices={','.join(map(str, list_indices))}",
    ]

    # Add expand column parameters if provided
    if expand_indices and expand_sizes:
        awk_cmd.extend(
            [
                "-v",
                f"expand_indices={','.join(map(str, expand_indices))}",
                "-v",
                f"expand_sizes={','.join(map(str, expand_sizes))}",
            ]
        )

    awk_cmd.extend(["-f", awk_script])

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

    if bcftool.returncode:  # pragma: no cover
        err_msg = bcftool.stderr.read() if bcftool.stderr else ""
        raise subprocess.CalledProcessError(bcftool.returncode, bcftools_cmd, err_msg)
    if awk.returncode:  # pragma: no cover
        raise subprocess.CalledProcessError(awk.returncode, awk_cmd, awk_err)
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


def _get_sample_list(vcf: str) -> list[str]:
    """
    Get the list of samples in the VCF file.

    Parameters
    ----------
    vcf : str
        Path to input VCF/BCF file

    Returns
    -------
    list[str]
        List of samples in the VCF
    """
    try:
        with pysam.VariantFile(vcf) as h_vcf:
            return list(h_vcf.header.samples)
    except Exception as e:
        log.error(f"Could not determine sample list from VCF {vcf}: {e}.")
        raise RuntimeError(f"Could not determine sample list from VCF {vcf}: {e}.") from e


def _convert_sample_frames_to_polars(
    region: str,
    vcf_path: str,
    fmt_str: str,
    job_cfg: VCFJobConfig,
) -> dict[str, pl.DataFrame]:
    """Collect DataFrames for each sample in the region.

    Read filters are applied per-sample before the multi-sample join,
    since FORMAT columns still have their original names (e.g. DP) at this stage.
    After the join, they get sample suffixes (e.g. DP_SAMPLE1).
    """
    frames: dict[str, pl.DataFrame] = {}

    for sample in job_cfg.sample_list:
        log.debug(f"Processing sample {sample} for region {region}")
        single_sample_tsv = _bcftools_awk_stdout(
            region=region,
            vcf_path=vcf_path,
            fmt_str=fmt_str,
            bcftools=job_cfg.bcftools_path,
            awk_script=job_cfg.awk_script,
            list_indices=job_cfg.column_config.list_indices,
            sample_name=sample,
            expand_indices=job_cfg.column_config.expand_indices,
            expand_sizes=job_cfg.column_config.expand_sizes,
        )
        if not single_sample_tsv.strip():
            log.debug(f"Sample {sample} has no data in region {region}")
            continue
        frame = _frame_from_tsv(single_sample_tsv, cols=job_cfg.columns, schema=job_cfg.schema)
        if not frame.is_empty():
            frame = _cast_column_data_types(frame, job_cfg)
            if job_cfg.read_filters and len(job_cfg.sample_list) > 1:
                rows_before = frame.height
                frame = _apply_read_filters(frame, job_cfg.read_filters)
                log.debug(f"Sample {sample}, region {region}: {rows_before:,} -> {frame.height:,} rows after filtering")
            if not frame.is_empty():
                frames[sample] = frame

    return frames


def _join_multi_sample_frames(frames: dict[str, pl.DataFrame], job_cfg: VCFJobConfig) -> pl.DataFrame:
    """Join multiple sample DataFrames, renaming FORMAT columns and dropping duplicates."""
    aggregation_suffixes = {suffix for suffix, _ in AGGREGATION_TYPES}

    # Build set of all columns that need sample suffix (FORMAT + aggregate columns + expand columns)
    cols_to_rename = set(job_cfg.column_config.fmt_ids)
    for list_id in job_cfg.column_config.list_fmt_ids:
        for suffix in aggregation_suffixes:
            cols_to_rename.add(f"{list_id}_{suffix}")

    # Add expand column split columns (e.g., AD_0, AD_1)
    if job_cfg.column_config.expand_columns:
        for col, size in job_cfg.column_config.expand_columns.items():
            for i in range(size):
                cols_to_rename.add(f"{col}_{i}")

    # Rename FORMAT columns and aggregate columns with sample suffix
    for sample, frame in frames.items():
        rename_map = {col: f"{col}_{sample}" for col in frame.columns if col in cols_to_rename}
        frames[sample] = frame.rename(rename_map)

    frame_list = list(frames.values())
    final_frame = frame_list[0]

    # Join keys: CHROM, POS, REF, ALT (genomic coordinates and alleles)
    join_keys = [CHROM, POS, REF, ALT]

    # Columns to drop from subsequent frames (QUAL, INFO) - they're identical across samples
    non_sample_specific_cols = {QUAL} | set(job_cfg.column_config.info_ids)
    cols_to_drop = non_sample_specific_cols - set(join_keys) - set(job_cfg.column_config.fmt_ids)

    # Join remaining frames, dropping duplicate columns
    for frame in frame_list[1:]:
        frame_dropped = frame.drop([col for col in cols_to_drop if col in frame.columns])
        final_frame = final_frame.join(frame_dropped, on=join_keys, how="outer", coalesce=True)

    return final_frame


def _stream_region_to_polars(
    region: str,
    vcf_path: str,
    fmt_str: str,
    job_cfg: VCFJobConfig,
) -> pl.DataFrame:
    """
    Run bcftools→awk and return a typed Polars DataFrame for *region*.
    If VCF has multiple samples, create a separate dataframe for each sample and join them on CHROM, POS, REF, ALT.
    The final dataframe will have the same columns as the input VCF, but with the FORMAT column name suffixed with
    the sample name to avoid column name conflicts in the join.

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

    Returns
    -------
    pl.DataFrame
        Typed Polars DataFrame for the region
    """
    frames = _convert_sample_frames_to_polars(region, vcf_path, fmt_str, job_cfg)

    if not frames:
        return pl.DataFrame()

    if len(job_cfg.sample_list) == 1:
        return list(frames.values())[0]

    return _join_multi_sample_frames(frames, job_cfg)


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
    col_cfg = job_cfg.column_config
    exprs.extend(_cast_expr(tag, col_cfg.info_meta[tag]) for tag in col_cfg.info_ids)
    exprs.extend(_cast_expr(tag, col_cfg.fmt_meta[tag]) for tag in col_cfg.scalar_fmt_ids)
    # Only cast list columns if they exist (they're replaced with aggregate columns in aggregate mode)
    exprs.extend(
        _cast_expr(tag, col_cfg.fmt_meta[tag]) for tag in col_cfg.list_fmt_ids if tag in featuremap_dataframe.columns
    )

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
        "--list-mode",
        choices=["explode", "aggregate"],
        default="explode",
        help=(
            "How to handle list format fields: 'explode' (one row per list element) or 'aggregate' "
            "(mean, min, max, count, count_zero metrics). Default is 'explode'."
        ),
    )
    parser.add_argument(
        "--expand-columns",
        nargs="*",
        default=[],
        metavar="COL:SIZE",
        help=(
            "Columns to expand into multiple columns instead of aggregate (only in aggregate mode). "
            "Format: COL:SIZE, e.g., 'AD:2' expands AD into AD_0, AD_1."
        ),
    )
    parser.add_argument(
        "--read-filters-json",
        default=None,
        help="Path to JSON file containing read filters to apply during conversion",
    )
    parser.add_argument(
        "--read-filter-json-key",
        default=None,
        help="Key in read filters JSON file to extract filters from (uses entire JSON if not specified)",
    )
    parser.add_argument(
        "--downsample-reads",
        type=int,
        default=None,
        help="Downsample to this number of reads after conversion (keeps all if fewer)",
    )
    parser.add_argument(
        "--downsample-seed",
        type=int,
        default=None,
        help="Random seed for reproducible downsampling",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    # Parse expand columns
    expand_columns: dict[str, int] | None = None
    if args.expand_columns:
        expand_columns = {}
        for spec in args.expand_columns:
            if ":" not in spec:
                parser.error(f"Invalid expand-columns format: {spec}. Expected COL:SIZE, e.g., 'AD:2'")
            col, size_str = spec.split(":", 1)
            try:
                size = int(size_str)
            except ValueError:
                parser.error(f"Invalid size in expand-columns: {spec}. Size must be an integer.")
            expand_columns[col] = size

    log_level = logging.DEBUG if args.verbose else logging.INFO
    _configure_logging(log_level, check_worker_cache=False)

    vcf_to_parquet(
        vcf=args.input,
        out=args.output,
        drop_info=set(args.drop_info),
        drop_format=set(args.drop_format),
        chunk_bp=args.chunk_bp,
        jobs=args.jobs,
        list_mode=args.list_mode,
        log_level=log_level,
        expand_columns=expand_columns,
        read_filters_json=args.read_filters_json,
        read_filter_json_key=args.read_filter_json_key,
        downsample_reads=args.downsample_reads,
        downsample_seed=args.downsample_seed,
    )


if __name__ == "__main__":
    main()
