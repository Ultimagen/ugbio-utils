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

# CHANGELOG in reverse chronological order

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from polars.exceptions import ShapeError

log = logging.getLogger(__name__)

# Configuration constants
CHUNK_SIZE = 100_000  # Default chunk size for parallel processing
DEFAULT_MAX_WORKERS = min(8, (os.cpu_count() or 4))  # Use up to 8 cores or available CPUs


def _log_memory_usage(stage: str) -> None:
    """Log current stage for debugging."""
    log.debug(f"[{stage}] Processing stage completed")


@dataclass
class ColumnConfig:
    """Configuration for column processing."""

    info_ids: list[str]
    scalar_fmt_ids: list[str]
    list_fmt_ids: list[str]
    info_meta: dict
    fmt_meta: dict


@dataclass
class ChunkProcessingArgs:
    """Arguments for parallel chunk processing."""

    chunk_file: str
    cols: list[str]
    info_ids: list[str]
    scalar_fmt_ids: list[str]
    list_fmt_ids: list[str]
    info_meta: dict
    fmt_meta: dict
    output_file: str
    is_first_chunk: bool
    overrides: dict


# ───────────────── header helpers ────────────────────────────────────────────
INFO_RE = re.compile(r'##INFO=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]*)')
FORMAT_RE = re.compile(r'##FORMAT=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]*)')

_POLARS_DTYPE = {"Integer": pl.Int64, "Float": pl.Float64, "Flag": pl.Boolean}
CHROM, POS, REF, ALT, QUAL, FILTER, ID, SAMPLE = "CHROM", "POS", "REF", "ALT", "QUAL", "FILTER", "ID", "SAMPLE"
# Reserved/fixed VCF columns (cannot be overridden)
RESERVED = {CHROM, POS, REF, ALT, QUAL, FILTER}
ALLELE_CATS = ["A", "C", "G", "T"]  # for REF / ALT


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
    return m.group(1).split(",") if m else None


def header_meta(vcf: str, bcftools_path: str) -> tuple[dict, dict]:
    """
    Parse the VCF header and build dictionaries with tag metadata.

    Parameters
    ----------
    vcf
        Path to input VCF/BCF (bgzipped ok).
    bcftools_path
        Absolute path to the ``bcftools`` executable.

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
    txt = subprocess.check_output([bcftools_path, "view", "-h", vcf], text=True)
    info, fmt = {}, {}
    for m in INFO_RE.finditer(txt):
        k, n, t, d = m.groups()
        info[k] = {"num": n, "type": t, "cat": _enum(d)}
    for m in FORMAT_RE.finditer(txt):
        k, n, t, d = m.groups()
        fmt[k] = {"num": n, "type": t, "cat": _enum(d)}
    return info, fmt


# ───────────────── enum helpers ──────────────────────────────
def _ensure_scalar_categories(featuremap_dataframe: pl.DataFrame, col: str, cats: list[str]) -> pl.DataFrame:
    """
    Ensure a categorical *scalar* column registers every category value.

    Polars ≥ 1.27 no longer provides ``set_categories``/``set_order``.
    Instead we append a one-row *stub block* that contains each category
    exactly once, concatenate it to the original frame so Polars merges
    the dictionaries, then trim the extra row(s) away.

    Parameters
    ----------
    featuremap_dataframe :
        The DataFrame containing the column.
    col :
        Name of the categorical scalar column that was already
        ``cast(pl.Categorical)``.
    cats :
        Ordered list of category strings extracted from the VCF header
        (e.g. ``["A", "C", "G", "T"]``).

    Returns
    -------
    pl.DataFrame
        A DataFrame of identical shape to *featuremap_dataframe* (same rows/columns) but
        whose *col* now recognises **all** values in *cats*.
    """
    # build a DataFrame that matches the full schema
    # ensure the empty-string category is present
    if "" not in cats:
        cats = cats + [""]
    rows = len(cats)
    stub_dict: dict[str, list] = {c: [None] * rows for c in featuremap_dataframe.columns}
    stub_dict[col] = cats  # each category once

    # Create DataFrame with same schema as original (skip casting for now)
    stub = pl.DataFrame(stub_dict).with_columns(pl.col(col).cast(pl.Categorical))

    return pl.concat([featuremap_dataframe, stub], how="vertical").head(featuremap_dataframe.height)


# ───────────────── casting helpers ──────────────────────────────────────────
def _cast_scalar(
    featuremap_dataframe: pl.DataFrame,
    col: str,
    meta: dict,
) -> pl.DataFrame:
    """
    Cast a scalar INFO column.

    Parameters
    ----------
    featuremap_dataframe
        Current DataFrame.
    col
        Column name.
    meta
        Dict with keys {"type", "cat"} parsed from the VCF header.
    enable_debug_logging
        Whether to enable debug logging for this call.

    Returns
    -------
    DataFrame with `col` recast and (if categorical) stub-padded so
    all categories from `meta["cat"]` are present.
    """
    utf_null = pl.when(pl.col(col).cast(pl.Utf8).is_in(["", "."])).then(None).otherwise(pl.col(col).cast(pl.Utf8))

    if meta["cat"]:
        featuremap_dataframe = featuremap_dataframe.with_columns(utf_null.cast(pl.Categorical).alias(col))
        return _ensure_scalar_categories(featuremap_dataframe, col, meta["cat"])
    if meta["type"] in _POLARS_DTYPE:
        return featuremap_dataframe.with_columns(utf_null.cast(_POLARS_DTYPE[meta["type"]], strict=False).alias(col))

    return featuremap_dataframe.with_columns(utf_null.alias(col))


def _cast_list(featuremap_dataframe: pl.DataFrame, col: str, meta: dict) -> pl.DataFrame:
    """
    Convert a comma-separated FORMAT list column.

    Steps
    -----
    1. `str.split(",")`  → list<str>
    2. Replace ``''`` and ``'.'`` with null.
    3. If the header declares ``Type=Integer/Float/Flag`` cast each
       element accordingly (string remains Utf8).
    4. *No* categorical cast inside the list because Polars ≥ 1.27
       forbids it; after `explode()` scalars are handled by
       :func:`cast_scalar`.

    Returns the DataFrame with *col* recast.
    """
    # Handle null values first - replace null with "." so we can split it consistently
    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.when(pl.col(col).is_null()).then(pl.lit(".")).otherwise(pl.col(col)).alias(col)
    )

    # split to list<str>
    featuremap_dataframe = featuremap_dataframe.with_columns(pl.col(col).str.split(",").alias(col))

    # null-replace on each element, and handle the case where split results in [null]
    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.col(col)
        .list.eval(
            pl.when(pl.element().cast(pl.Utf8).is_in(["", "."])).then(None).otherwise(pl.element().cast(pl.Utf8))
        )
        .alias(col)
    )

    # numeric / boolean element cast if requested
    if meta["type"] in _POLARS_DTYPE and meta["type"] != "String":
        elem_dt = _POLARS_DTYPE[meta["type"]]
        featuremap_dataframe = featuremap_dataframe.with_columns(
            pl.col(col).list.eval(pl.element().cast(elem_dt, strict=False)).alias(col)
        )

    return featuremap_dataframe


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


def _get_override_categories(
    categories_json: str | None,
):
    """
    Read a JSON file with user-specified categorical features.
    The JSON file should contain a dictionary with the key
    "categorical_features" and a dictionary of feature names
    as keys and their corresponding categories as values.
    The function returns a dictionary with the feature names
    as keys and their corresponding categories as values.
    If the JSON file is not provided, an empty dictionary is returned.

    Parameters
    ----------
    categories_json
        Path to the JSON file containing user-specified categorical features.
    Returns
    -------
    dict
        A dictionary with the feature names as keys and their corresponding
        categories as values. If the JSON file is not provided, an empty
        dictionary is returned.
    Notes
    -----
    * The JSON file should contain a dictionary with the key
      "categorical_features" and a dictionary of feature names
      as keys and their corresponding categories as values.
    * The function will log a warning if any of the feature names
      in the JSON file are reserved columns (CHROM, POS, REF, ALT,
      QUAL, FILTER).
    * The function will log a warning if the JSON file is not
      provided or if the "categorical_features" key is not found
      in the JSON file.
    """
    overrides = {}
    if categories_json:
        with open(categories_json) as jh:
            user_map = json.load(jh).get("categorical_features", {})
            if not user_map:
                log.warning("No categorical_features found in JSON file")
            for k, v in user_map.items():
                if k in RESERVED:
                    log.warning("Ignoring JSON category override for reserved column %s", k)
                else:
                    overrides[k] = v
    return overrides


def _find_bad_list_columns(featuremap_dataframe: pl.DataFrame, cols: list[str]) -> list[str]:
    """
    Return list columns whose element count differs from the row-wise maximum
    in the first row that shows a mismatch.
    """
    for row in featuremap_dataframe.select(cols).iter_rows(named=True):
        lens = {k: (len(v) if v is not None else 0) for k, v in row.items()}
        if len(set(lens.values())) > 1:
            max_len = max(lens.values())
            return [k for k, v in lens.items() if v != max_len]
    return []  # fallback – should not happen


# ───────────────── misc helpers ─────────────────────────────────────────────
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
    """Build the bcftools query format string."""
    bracket = "[" + "\t".join(f"%{t}" for t in format_ids) + "]"
    return "\t".join(["%CHROM", "%POS", "%REF", "%ALT", *[f"%INFO/{t}" for t in query_info], bracket]) + "\n"


def _load_vcf_as_lazy_frame(
    bcftools: str,
    vcf: str,
    fmt_str: str,
    cols: list[str],
    info_meta: dict,
    fmt_meta: dict,
    infer_schema_length: int = 10000,  # Keep as fallback but prefer explicit schema
) -> pl.LazyFrame:
    """Run bcftools query and load the TSV into a LazyFrame for memory-efficient processing."""
    with tempfile.NamedTemporaryFile("w+b", delete=False) as tmp:
        subprocess.run([bcftools, "query", "-f", fmt_str, vcf], stdout=tmp, check=True)
        path = tmp.name

    # Build explicit schema from VCF metadata to avoid parsing issues
    schema = _build_explicit_schema(cols, info_meta, fmt_meta)

    # Use scan_csv with explicit schema - don't delete the file yet as lazy frames need it
    # The caller is responsible for cleanup if needed
    return pl.scan_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=cols,
        low_memory=True,
        null_values=["."],
        schema=schema,  # Use explicit schema instead of inference
    )


def _load_vcf_as_dataframe(
    bcftools: str,
    vcf: str,
    fmt_str: str,
    cols: list[str],
    info_meta: dict,
    fmt_meta: dict,
    infer_schema_length: int = 10000,  # Keep as fallback but prefer explicit schema
) -> pl.DataFrame:
    """Run bcftools query and load the TSV into a DataFrame."""
    with tempfile.NamedTemporaryFile("w+b", delete=False) as tmp:
        subprocess.run([bcftools, "query", "-f", fmt_str, vcf], stdout=tmp, check=True)
        path = tmp.name
    try:
        # Build explicit schema from VCF metadata to avoid parsing issues
        schema = _build_explicit_schema(cols, info_meta, fmt_meta)

        return pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=cols,
            low_memory=True,
            decimal_comma=True,
            null_values=["."],
            schema=schema,  # Use explicit schema instead of inference
        )
    finally:
        Path(path).unlink(missing_ok=True)


def _cast_scalar_lazy(
    lazy_df: pl.LazyFrame,
    col: str,
    meta: dict,
) -> pl.LazyFrame:
    """
    Cast a scalar INFO column on a LazyFrame.

    Parameters
    ----------
    lazy_df
        Current LazyFrame.
    col
        Column name.
    meta
        Dict with keys {"type", "cat"} parsed from the VCF header.
    enable_debug_logging
        Whether to enable debug logging for this call.

    Returns
    -------
    LazyFrame with `col` recast.
    """
    utf_null = pl.when(pl.col(col).cast(pl.Utf8).is_in(["", "."])).then(None).otherwise(pl.col(col).cast(pl.Utf8))

    if meta["cat"]:
        lazy_df = lazy_df.with_columns(utf_null.cast(pl.Categorical).alias(col))
        # Note: Lazy frames don't support ensure_scalar_categories operation
        return lazy_df
    if meta["type"] in _POLARS_DTYPE:
        return lazy_df.with_columns(utf_null.cast(_POLARS_DTYPE[meta["type"]], strict=False).alias(col))

    return lazy_df.with_columns(utf_null.alias(col))


def _cast_list_lazy(lazy_df: pl.LazyFrame, col: str, meta: dict) -> pl.LazyFrame:
    """
    Convert a comma-separated FORMAT list column on a LazyFrame.

    Steps
    -----
    1. `str.split(",")`  → list<str>
    2. Replace ``''`` and ``'.'`` with null.
    3. If the header declares ``Type=Integer/Float/Flag`` cast each
       element accordingly (string remains Utf8).

    Returns the LazyFrame with *col* recast.
    """
    # Handle null values first - replace null with "." so we can split it consistently
    lazy_df = lazy_df.with_columns(pl.when(pl.col(col).is_null()).then(pl.lit(".")).otherwise(pl.col(col)).alias(col))

    # split to list<str>
    lazy_df = lazy_df.with_columns(pl.col(col).str.split(",").alias(col))

    # null-replace on each element
    lazy_df = lazy_df.with_columns(
        pl.col(col)
        .list.eval(
            pl.when(pl.element().cast(pl.Utf8).is_in(["", "."])).then(None).otherwise(pl.element().cast(pl.Utf8))
        )
        .alias(col)
    )

    # numeric / boolean element cast if requested
    if meta["type"] in _POLARS_DTYPE and meta["type"] != "String":
        elem_dt = _POLARS_DTYPE[meta["type"]]
        lazy_df = lazy_df.with_columns(pl.col(col).list.eval(pl.element().cast(elem_dt, strict=False)).alias(col))

    return lazy_df


def _cast_all_columns_lazy(
    lazy_df: pl.LazyFrame,
    info_ids: list[str],
    scalar_fmt_ids: list[str],
    list_fmt_ids: list[str],
    info_meta: dict,
    fmt_meta: dict,
) -> pl.LazyFrame:
    """Apply casting helpers to every INFO / FORMAT column on a LazyFrame."""
    for tag in info_ids:
        lazy_df = _cast_scalar_lazy(lazy_df, tag, info_meta[tag])
    for tag in scalar_fmt_ids:
        lazy_df = _cast_scalar_lazy(lazy_df, tag, fmt_meta[tag])
    for tag in list_fmt_ids:
        lazy_df = _cast_list_lazy(lazy_df, tag, fmt_meta[tag])
    # REF / ALT
    for allele in (REF, ALT):
        lazy_df = _cast_scalar_lazy(
            lazy_df,
            allele,
            {"type": "String", "cat": ALLELE_CATS},
        )

    return lazy_df


def _cast_all_columns(
    featuremap_dataframe: pl.DataFrame,
    info_ids: list[str],
    scalar_fmt_ids: list[str],
    list_fmt_ids: list[str],
    info_meta: dict,
    fmt_meta: dict,
) -> pl.DataFrame:
    """Apply casting helpers to every INFO / FORMAT column."""
    for tag in info_ids:
        featuremap_dataframe = _cast_scalar(featuremap_dataframe, tag, info_meta[tag])
    for tag in scalar_fmt_ids:
        featuremap_dataframe = _cast_scalar(featuremap_dataframe, tag, fmt_meta[tag])
    for tag in list_fmt_ids:
        featuremap_dataframe = _cast_list(featuremap_dataframe, tag, fmt_meta[tag])
    # REF / ALT
    for allele in (REF, ALT):
        featuremap_dataframe = _cast_scalar(featuremap_dataframe, allele, {"type": "String", "cat": ALLELE_CATS})

    return featuremap_dataframe


def _explode_with_retry(
    featuremap_dataframe: pl.DataFrame,
    list_fmt_ids: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """
    Try to explode *list_fmt_ids*; if mismatched lengths are detected, drop
    the offending columns and retry until success.
    """
    while list_fmt_ids:
        try:
            return featuremap_dataframe.explode(list_fmt_ids), list_fmt_ids  # success
        except ShapeError:
            bad_cols = _find_bad_list_columns(featuremap_dataframe, list_fmt_ids)
            if not bad_cols:
                raise

            log.warning("Dropping list columns with inconsistent length: %s", ", ".join(bad_cols))
            featuremap_dataframe = featuremap_dataframe.drop(bad_cols)
            list_fmt_ids = [c for c in list_fmt_ids if c not in bad_cols]
    return featuremap_dataframe, []  # nothing exploded


def _process_line_for_explosion(fields: list[str], list_fmt_indices: list[int]) -> list[list[str]]:
    """Process a single TSV line and return exploded rows."""
    # Check if any list columns exist and get their lengths
    list_lengths = {}
    max_length = 1

    for idx in list_fmt_indices:
        if idx < len(fields) and fields[idx] and fields[idx] != ".":
            # Split the list and handle null values
            list_values = [v if v not in ("", ".") else "." for v in fields[idx].split(",")]
            list_lengths[idx] = list_values
            max_length = max(max_length, len(list_values))
        else:
            list_lengths[idx] = ["."]
    # If no lists or all lists are empty, just return the row as-is
    if max_length == 1:
        return [fields]

    # Create exploded rows
    exploded = []
    for i in range(max_length):
        new_row = fields.copy()
        for idx in list_fmt_indices:
            if idx in list_lengths and i < len(list_lengths[idx]):
                new_row[idx] = list_lengths[idx][i]
            else:
                new_row[idx] = "."
        exploded.append(new_row)
    return exploded


def _explode_tsv_rows(
    bcftools: str,
    vcf: str,
    fmt_str: str,
    cols: list[str],
    list_fmt_indices: list[int],
) -> pl.DataFrame:
    """
    Run bcftools query and explode list columns during TSV parsing to avoid memory issues.

    This processes the TSV line by line, exploding list columns immediately,
    which prevents the massive memory usage that occurs when loading all data
    and then exploding.

    Parameters
    ----------
    bcftools : str
        Path to bcftools executable
    vcf : str
        Path to VCF file
    fmt_str : str
        bcftools query format string
    cols : list[str]
        Column names
    list_fmt_indices : list[int]
        Indices of columns that contain comma-separated lists to explode

    Returns
    -------
    pl.DataFrame
        Exploded dataframe with one row per read
    """
    # Run bcftools and capture output
    result = subprocess.run([bcftools, "query", "-f", fmt_str, vcf], capture_output=True, text=True, check=True)

    exploded_rows = []

    # Process each line and explode immediately
    for line_num, line in enumerate(result.stdout.strip().split("\n")):
        if not line.strip():
            continue

        fields = line.split("\t")
        if len(fields) != len(cols):
            log.warning(f"Line {line_num}: expected {len(cols)} fields, got {len(fields)}")
            continue

        # Process line and get exploded rows
        line_exploded = _process_line_for_explosion(fields, list_fmt_indices)
        exploded_rows.extend(line_exploded)

    # Return empty DataFrame if no data
    if not exploded_rows:
        return pl.DataFrame({col: [] for col in cols})

    # Create DataFrame from exploded rows
    df_data = {col: [] for col in cols}
    for row in exploded_rows:
        for i, col in enumerate(cols):
            df_data[col].append(row[i] if i < len(row) else ".")

    return pl.DataFrame(df_data)


def _explode_tsv_rows_generator(
    bcftools: str,
    vcf: str,
    fmt_str: str,
    cols: list[str],
    list_fmt_indices: list[int],
):
    """
    Generator version of _explode_tsv_rows that yields individual exploded rows.

    This processes the TSV line by line, exploding list columns immediately,
    and yields each exploded row one at a time for memory efficiency.
    """
    log.debug(f"[EXPLODE] Starting bcftools query for {vcf}")

    # Run bcftools query with streaming output
    process = subprocess.Popen(
        [bcftools, "query", "-f", fmt_str, vcf], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if process.stdout is None:
        raise RuntimeError("Failed to get stdout from bcftools process")

    line_num = 0
    try:
        for raw_line in process.stdout:
            line_content = raw_line.strip()
            if not line_content:
                continue

            line_num += 1
            fields = line_content.split("\t")
            if len(fields) != len(cols):
                log.warning(f"Line {line_num}: expected {len(cols)} fields, got {len(fields)}")
                continue

            # Process line and get exploded rows
            line_exploded = _process_line_for_explosion(fields, list_fmt_indices)
            yield from line_exploded

    finally:
        # Ensure process cleanup
        if process.stdout:
            process.stdout.close()
        exit_code = process.wait()
        if exit_code != 0:
            stderr_output = process.stderr.read() if process.stderr else "No stderr"
            raise subprocess.CalledProcessError(exit_code, [bcftools, "query"], stderr_output)


# ───────────────── core converter ────────────────────────────────────────────
CHUNK_SIZE = 10000


def vcf_to_parquet_streaming(
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
    categories_json: str | None = None,
    chunk_size: int = 100_000,
) -> None:
    """
    Convert a single-sample feature-map VCF into a per-read Parquet file using streaming/chunked processing.

    Parameters
    ----------
    vcf
        Input VCF/BCF path (may be ``.gz`` + ``.tbi``).
    out
        Output Parquet path.
    drop_info
        INFO tags to exclude.  Default: keep all.
    drop_format
        FORMAT tags to exclude.  Default: ``{"GT"}``.
    categories_json
        Path to JSON file with categorical overrides.
    chunk_size
        Number of variants to process at once.    Notes
    -----
    * Uses lazy evaluation and streaming to reduce memory usage.
    * Maintains categorical column functionality.
    """
    log.debug("🚀 Starting vcf_to_parquet_streaming")
    _log_memory_usage("streaming_start")

    overrides = _get_override_categories(categories_json)

    # resolve info and format tags from header
    drop_info = drop_info or set()
    drop_format = drop_format or {"GT"}
    if "GT" not in drop_format:
        log.warning(
            "GT not dropped; this may break downstream logic. Include 'GT' in --drop-format to "
            "drop if an error occurs."
        )

    bcftools = _resolve_bcftools_command()
    info_meta, fmt_meta = header_meta(vcf, bcftools)

    # Merge JSON overrides
    for k, v in overrides.items():
        target = fmt_meta if k in fmt_meta else info_meta
        target.setdefault(k, {"num": "1", "type": "String", "cat": None})["cat"] = v

    info_ids = [k for k in info_meta if k not in drop_info]
    format_ids = [k for k in fmt_meta if k not in drop_format]

    _sanity_check_format_numbers(format_ids, fmt_meta)
    scalar_fmt_ids, list_fmt_ids = _split_format_ids(format_ids, fmt_meta)

    query_info = [k for k in info_meta if k not in RESERVED and k not in drop_info]
    fmt_str = _make_query_string(format_ids, query_info)
    cols = list(OrderedDict.fromkeys([CHROM, POS, REF, ALT, *info_ids, *format_ids]))

    log.debug(f"📋 Columns configured: {len(cols)} total, {len(list_fmt_ids)} list columns")

    # Process using streaming approach for better memory efficiency
    with pl.StringCache():
        log.debug("⚠️ Loading entire VCF into memory - THIS IS A MEMORY-INTENSIVE STEP")
        _log_memory_usage("before_loading_vcf")

        # Load data eagerly but process in chunks
        featuremap_dataframe = _load_vcf_as_dataframe(bcftools, vcf, fmt_str, cols, info_meta, fmt_meta, chunk_size)

        _log_memory_usage("after_loading_vcf")
        log.debug(f"📊 Loaded dataframe: {featuremap_dataframe.shape[0]} rows × {featuremap_dataframe.shape[1]} cols")

        featuremap_dataframe = featuremap_dataframe.with_columns(pl.col(POS).cast(pl.Int64))

        # Apply all column casting
        log.debug("🎭 Applying column casting")
        _log_memory_usage("before_casting")
        featuremap_dataframe = _cast_all_columns(
            featuremap_dataframe, info_ids, scalar_fmt_ids, list_fmt_ids, info_meta, fmt_meta
        )
        _log_memory_usage("after_casting")

        # Use the retry mechanism for exploding - THIS IS ANOTHER MEMORY BOTTLENECK!
        log.debug("💥 Starting explode operation - MEMORY INTENSIVE")
        _log_memory_usage("before_explode")
        featuremap_dataframe, list_fmt_ids = _explode_with_retry(featuremap_dataframe, list_fmt_ids)
        _log_memory_usage("after_explode")
        log.debug(f"📊 After explode: {featuremap_dataframe.shape[0]} rows × {featuremap_dataframe.shape[1]} cols")

        # After explode, cast the exploded columns as scalars
        for tag in list_fmt_ids:
            featuremap_dataframe = _cast_scalar(featuremap_dataframe, tag, fmt_meta[tag])

        # Apply categorical category registration for any categorical columns
        for tag in info_ids + scalar_fmt_ids + list_fmt_ids + [REF, ALT]:
            if tag in featuremap_dataframe.columns:
                col_meta = (
                    info_meta.get(tag) or fmt_meta.get(tag) or {"cat": ALLELE_CATS if tag in [REF, ALT] else None}
                )
                cats = col_meta.get("cat")
                if cats and featuremap_dataframe[tag].dtype == pl.Categorical:
                    featuremap_dataframe = _ensure_scalar_categories(featuremap_dataframe, tag, cats)

    log.debug("💾 Writing final parquet file")
    _log_memory_usage("before_writing")
    featuremap_dataframe.write_parquet(out)
    _log_memory_usage("after_writing")
    log.info("✅  %s: %d rows × %d cols", out, *featuremap_dataframe.shape)


def vcf_to_parquet(
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
    categories_json: str | None = None,
) -> None:
    """
    Convert a single-sample feature-map VCF into a per-read Parquet file.

    Parameters
    ----------
    vcf
        Input VCF/BCF path (may be ``.gz`` + ``.tbi``).
    out
        Output Parquet path.
    drop_info
        INFO tags to exclude.  Default: keep all.
    drop_format
        FORMAT tags to exclude.  Default: ``{"GT"}``.

    Notes
    -----
    * Runs inside a ``pl.StringCache()`` context so all categorical
      columns share one global dictionary.
    * List‐valued FORMAT fields remain ``list<Utf8>`` due to Polars 1.27
      limitations; after ``explode`` they become scalars and are cast
      like INFO fields.
    * Now uses memory-efficient approach that explodes TSV rows directly.
    """
    # Use the memory-efficient version for better performance
    return vcf_to_parquet_memory_efficient(
        vcf=vcf,
        out=out,
        drop_info=drop_info,
        drop_format=drop_format,
        categories_json=categories_json,
    )


def vcf_to_parquet_memory_efficient(
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
    categories_json: str | None = None,
) -> None:
    """
    Convert a single-sample feature-map VCF into a per-read Parquet file using memory-efficient TSV explosion.
    """
    _vcf_to_parquet_memory_efficient_impl(vcf, out, drop_info, drop_format, categories_json)


def _vcf_to_parquet_memory_efficient_impl(
    vcf: str,
    out: str,
    drop_info: set[str] | None,
    drop_format: set[str] | None,
    categories_json: str | None,
) -> None:
    log.debug("🚀 Starting vcf_to_parquet_memory_efficient")
    _log_memory_usage("memory_efficient_start")
    overrides = _get_override_categories(categories_json)
    drop_info = drop_info or set()
    drop_format = drop_format or {"GT"}
    if "GT" not in drop_format:
        log.warning(
            "GT not dropped; this may break downstream logic. Include 'GT' in --drop-format to "
            "drop if an error occurs."
        )
    log.debug("📋 Parsing VCF header metadata")
    bcftools = _resolve_bcftools_command()
    info_meta, fmt_meta = header_meta(vcf, bcftools)
    for k, v in overrides.items():
        target = fmt_meta if k in fmt_meta else info_meta
        target.setdefault(k, {"num": "1", "type": "String", "cat": None})["cat"] = v
    info_ids = [k for k in info_meta if k not in drop_info]
    format_ids = [k for k in fmt_meta if k not in drop_format]
    _sanity_check_format_numbers(format_ids, fmt_meta)
    scalar_fmt_ids, list_fmt_ids = _split_format_ids(format_ids, fmt_meta)
    query_info = [k for k in info_meta if k not in RESERVED and k not in drop_info]
    fmt_str = _make_query_string(format_ids, query_info)
    cols = list(OrderedDict.fromkeys([CHROM, POS, REF, ALT, *info_ids, *format_ids]))
    list_fmt_indices = [cols.index(col) for col in list_fmt_ids if col in cols]
    log.debug(f"📋 Columns configured: {len(cols)} total, {len(list_fmt_ids)} list columns to explode")
    log.debug(f"🎯 List column indices: {list_fmt_indices}")

    with pl.StringCache():
        log.debug("🔄 Starting streaming batch processing with direct write")
        _log_memory_usage("before_streaming")

        column_config = ColumnConfig(
            info_ids=info_ids,
            scalar_fmt_ids=scalar_fmt_ids,
            list_fmt_ids=list_fmt_ids,
            info_meta=info_meta,
            fmt_meta=fmt_meta,
        )

        # Process and write batches directly to output file
        total_rows = _stream_and_write_batches_directly(
            bcftools,
            vcf,
            fmt_str,
            cols,
            list_fmt_indices,
            column_config,
            out,
        )

        _log_memory_usage("after_streaming")
        log.info("✅  %s: %d rows × %d cols", out, total_rows, len(cols))


def _stream_and_process_chunks(
    bcftools: str,
    vcf: str,
    fmt_str: str,
    cols: list[str],
    list_fmt_indices: list[int],
    column_config: ColumnConfig,
    out: str,
) -> list[str]:
    temp_parquet_files = []
    chunk_rows = []
    chunk_idx = 0
    for row in _explode_tsv_rows_generator(bcftools, vcf, fmt_str, cols, list_fmt_indices):
        chunk_rows.append(row)
        if len(chunk_rows) >= CHUNK_SIZE:
            _log_memory_usage(f"before_chunk_{chunk_idx}")
            temp_file = _process_and_write_chunk(
                chunk_rows,
                cols,
                column_config.info_ids,
                column_config.scalar_fmt_ids,
                column_config.list_fmt_ids,
                column_config.info_meta,
                column_config.fmt_meta,
                out,
                chunk_idx,
            )
            temp_parquet_files.append(temp_file)
            _log_memory_usage(f"after_chunk_{chunk_idx}")
            chunk_rows = []
            chunk_idx += 1
    if chunk_rows:
        _log_memory_usage(f"before_chunk_{chunk_idx}")
        temp_file = _process_and_write_chunk(
            chunk_rows,
            cols,
            column_config.info_ids,
            column_config.scalar_fmt_ids,
            column_config.list_fmt_ids,
            column_config.info_meta,
            column_config.fmt_meta,
            out,
            chunk_idx,
        )
        temp_parquet_files.append(temp_file)
        _log_memory_usage(f"after_chunk_{chunk_idx}")
    log.debug(f"[CHUNK] All chunks processed, {len(temp_parquet_files)} temp files created")
    return temp_parquet_files


def _create_chunk_dataframe(chunk_rows: list[list[str]], cols: list[str]) -> pl.DataFrame | None:
    """Create DataFrame from chunk rows."""
    if not chunk_rows:
        return None

    df_data = {col: [] for col in cols}
    for row in chunk_rows:
        for i, col in enumerate(cols):
            df_data[col].append(row[i] if i < len(row) else ".")

    chunk_df = pl.DataFrame(df_data)
    return chunk_df if chunk_df.height > 0 else None


def _apply_chunk_casting(
    chunk_df: pl.DataFrame,
    info_ids: list[str],
    scalar_fmt_ids: list[str],
    list_fmt_ids: list[str],
    info_meta: dict,
    fmt_meta: dict,
) -> pl.DataFrame:
    """Apply column casting to chunk DataFrame."""
    # Cast POS to Int64
    chunk_df = chunk_df.with_columns(pl.col(POS).cast(pl.Int64))

    # Apply column casting - list columns are already exploded as scalars
    for tag in info_ids:
        chunk_df = _cast_scalar(chunk_df, tag, info_meta[tag])
    for tag in scalar_fmt_ids:
        chunk_df = _cast_scalar(chunk_df, tag, fmt_meta[tag])
    for tag in list_fmt_ids:
        # These were exploded so now treat as scalars
        chunk_df = _cast_scalar(chunk_df, tag, fmt_meta[tag])

    # REF / ALT
    for allele in (REF, ALT):
        chunk_df = _cast_scalar(chunk_df, allele, {"type": "String", "cat": ALLELE_CATS})

    return chunk_df


def _apply_chunk_categories(
    chunk_df: pl.DataFrame,
    info_ids: list[str],
    scalar_fmt_ids: list[str],
    list_fmt_ids: list[str],
    info_meta: dict,
    fmt_meta: dict,
) -> pl.DataFrame:
    """Apply categorical category registration for chunk DataFrame."""
    for tag in info_ids + scalar_fmt_ids + list_fmt_ids + [REF, ALT]:
        if tag in chunk_df.columns:
            col_meta = info_meta.get(tag) or fmt_meta.get(tag) or {"cat": ALLELE_CATS if tag in [REF, ALT] else None}
            cats = col_meta.get("cat")
            if cats and chunk_df[tag].dtype == pl.Categorical:
                chunk_df = _ensure_scalar_categories(chunk_df, tag, cats)
    return chunk_df


def _process_and_write_chunk(
    chunk_rows: list[list[str]],
    cols: list[str],
    info_ids: list[str],
    scalar_fmt_ids: list[str],
    list_fmt_ids: list[str],
    info_meta: dict,
    fmt_meta: dict,
    out: str,
    chunk_num: int,
) -> str | None:
    """Process a chunk and write it to a temporary parquet file."""
    # Create DataFrame from exploded rows
    chunk_df = _create_chunk_dataframe(chunk_rows, cols)
    if chunk_df is None:
        return None

    # Apply column casting and categories
    chunk_df = _apply_chunk_casting(chunk_df, info_ids, scalar_fmt_ids, list_fmt_ids, info_meta, fmt_meta)
    chunk_df = _apply_chunk_categories(chunk_df, info_ids, scalar_fmt_ids, list_fmt_ids, info_meta, fmt_meta)

    # Write to temporary parquet file
    temp_file = f"{out}.temp_{chunk_num}.parquet"
    chunk_df.write_parquet(temp_file)
    return temp_file


def _write_first_batch_to_file(batch_df: pl.DataFrame, out: str) -> None:
    """Create the output parquet file with the first batch of data."""
    batch_df.write_parquet(out)
    log.debug(f"[BATCH_WRITE] Created parquet file {out} with {batch_df.height} rows")


def _append_batch_to_file(batch_df: pl.DataFrame, out: str) -> None:
    """Append a batch DataFrame to an existing parquet file."""
    # Read existing file, concatenate with new batch, and write back
    # This is still memory-efficient because we only hold 2 batches in memory at once
    existing_df = pl.read_parquet(out)
    combined_df = pl.concat([existing_df, batch_df])
    combined_df.write_parquet(out)
    log.debug(f"[BATCH_WRITE] Appended {batch_df.height} rows to {out}, total: {combined_df.height} rows")


def _stream_and_write_batches_directly(
    bcftools: str,
    vcf: str,
    fmt_str: str,
    cols: list[str],
    list_fmt_indices: list[int],
    column_config: ColumnConfig,
    out: str,
) -> int:
    """
    Stream and process batches, writing them directly to the output parquet file incrementally.

    This processes each batch and immediately appends it to the output file,
    avoiding the need to accumulate all data in memory.
    Returns the total number of rows written.
    """
    log.debug(f"[BATCH] Starting _stream_and_write_batches_directly for {vcf}")

    chunk_rows = []
    chunk_idx = 0
    total_rows = 0
    first_batch = True

    for row in _explode_tsv_rows_generator(bcftools, vcf, fmt_str, cols, list_fmt_indices):
        chunk_rows.append(row)

        if len(chunk_rows) >= CHUNK_SIZE:
            _log_memory_usage(f"before_batch_{chunk_idx}")

            # Process the batch
            batch_df = _process_batch_dataframe(
                chunk_rows,
                cols,
                column_config.info_ids,
                column_config.scalar_fmt_ids,
                column_config.list_fmt_ids,
                column_config.info_meta,
                column_config.fmt_meta,
            )

            if batch_df is not None:
                rows_in_batch = batch_df.height
                total_rows += rows_in_batch

                # Write batch to file
                if first_batch:
                    _write_first_batch_to_file(batch_df, out)
                    first_batch = False
                else:
                    _append_batch_to_file(batch_df, out)

            _log_memory_usage(f"after_batch_{chunk_idx}")
            chunk_rows = []
            chunk_idx += 1

    # Process final batch if any rows remain
    if chunk_rows:
        _log_memory_usage(f"before_batch_{chunk_idx}")

        batch_df = _process_batch_dataframe(
            chunk_rows,
            cols,
            column_config.info_ids,
            column_config.scalar_fmt_ids,
            column_config.list_fmt_ids,
            column_config.info_meta,
            column_config.fmt_meta,
        )

        if batch_df is not None:
            rows_in_batch = batch_df.height
            total_rows += rows_in_batch

            # Write final batch to file
            if first_batch:
                _write_first_batch_to_file(batch_df, out)
                first_batch = False
            else:
                _append_batch_to_file(batch_df, out)

        _log_memory_usage(f"after_batch_{chunk_idx}")

    # Handle case where no data was processed
    if first_batch:
        empty_df = pl.DataFrame({col: [] for col in cols})
        empty_df.write_parquet(out)
        total_rows = 0

    log.debug(f"[BATCH] All batches processed, total rows written: {total_rows}")
    return total_rows


def _process_batch_dataframe(
    chunk_rows: list[list[str]],
    cols: list[str],
    info_ids: list[str],
    scalar_fmt_ids: list[str],
    list_fmt_ids: list[str],
    info_meta: dict,
    fmt_meta: dict,
) -> pl.DataFrame | None:
    """Process a batch and return the processed DataFrame."""
    # Create DataFrame from exploded rows
    chunk_df = _create_chunk_dataframe(chunk_rows, cols)
    if chunk_df is None:
        return None

    # Apply column casting and categories
    chunk_df = _apply_chunk_casting(chunk_df, info_ids, scalar_fmt_ids, list_fmt_ids, info_meta, fmt_meta)
    chunk_df = _apply_chunk_categories(chunk_df, info_ids, scalar_fmt_ids, list_fmt_ids, info_meta, fmt_meta)

    return chunk_df


# ─────────────────────── NEW PARALLEL ARCHITECTURE ─────────────────────────
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


def _explode_and_split_with_bcftools_awk_streaming(
    vcf: str,
    fmt_str: str,
    list_fmt_indices: list[int],
    bcftools_path: str,
    chunk_size: int,
    chunk_directory: Path,
    num_threads: int = 1,
) -> subprocess.Popen:
    """
    Start bcftools + AWK pipeline that creates chunks as they stream.

    Returns the process handle so the caller can monitor for new chunk files
    as they're created concurrently.
    """
    awk_script = _get_awk_script_path()

    # Build the command pipeline: bcftools query | awk | split
    indices_str = ",".join(map(str, list_fmt_indices))

    # Use split command to create chunks directly from the pipeline
    cmd = [
        "bash",
        "-c",
        f"{bcftools_path} query -f '{fmt_str}' '{vcf}' | "
        f"awk -v list_indices='{indices_str}' -f '{awk_script}' | "
        f"split -l {chunk_size} - '{chunk_directory}/chunk_' --suffix-length=6 --numeric-suffixes",
    ]

    log.debug(f"Starting streaming bcftools + AWK + split pipeline: {' '.join(cmd)}")

    # Start the pipeline process (non-blocking)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=None,  # Allow proper signal handling
    )

    return process


# Constants for monitoring logic
_MAX_NO_ACTIVITY_CYCLES = 100
_PIPELINE_DONE_WAIT_CYCLES = 5


def _check_completed_futures(
    pending_futures: dict, processed_files: list[str], max_log_groups: int, last_logged_group: int
):
    """Check for completed futures and handle their results."""
    from concurrent.futures import as_completed

    completed_futures = []
    try:
        for future in as_completed(pending_futures, timeout=0.1):
            try:
                result = future.result()
                chunk_name = pending_futures[future]
                if result:
                    processed_files.append(result)

                    # Group-based logging - only log at intervals
                    current_count = len(processed_files)
                    if max_log_groups > 0:
                        group_size = max(1, current_count // max_log_groups)
                        current_group = current_count // group_size
                        if current_group > last_logged_group and current_count % group_size == 0:
                            log.info(f"Processed {current_count} chunks so far...")
                            last_logged_group = current_group

                completed_futures.append(future)
            except Exception as e:
                chunk_name = pending_futures[future]
                log.error(f"Error processing chunk {chunk_name}: {e}")
                completed_futures.append(future)
                # Re-raise the exception to ensure failure propagates
                raise RuntimeError(f"Chunk processing failed for {chunk_name}") from e
    except TimeoutError:
        # TimeoutError is expected when no futures complete within timeout
        pass

    return completed_futures, last_logged_group


def _should_exit_monitoring(
    *,
    pipeline_done: bool,
    chunk_counter: int,
    found_new_chunk: bool,
    no_activity_cycles: int,
    process: subprocess.Popen,
    pending_futures_count: int,
) -> tuple[bool, int]:
    """Determine if monitoring should exit and return updated activity cycles."""
    # Reset cycles when we find new chunks
    if found_new_chunk:
        return False, 0

    # Never exit while we have pending work
    if pending_futures_count > 0:
        return False, no_activity_cycles

    # Handle pipeline completion cases
    if pipeline_done:
        if chunk_counter == 0:
            log.debug("Pipeline done and no chunks were created")
            return True, no_activity_cycles

        new_cycles = no_activity_cycles + 1
        should_exit = new_cycles >= _PIPELINE_DONE_WAIT_CYCLES
        if should_exit:
            log.debug("Pipeline done and no more chunks expected")
        return should_exit, new_cycles

    # Handle active pipeline cases - no new chunks found
    new_cycles = no_activity_cycles + 1
    if new_cycles >= _MAX_NO_ACTIVITY_CYCLES:
        log.warning(f"No new chunks found for {_MAX_NO_ACTIVITY_CYCLES} cycles, checking pipeline status")
        if process.poll() is not None:
            log.debug("Pipeline has terminated, stopping monitoring")
            return True, new_cycles
        # Reset and continue if pipeline still running
        new_cycles = 0

    return False, new_cycles


def _monitor_and_process_chunks_concurrent(
    process: subprocess.Popen,
    chunk_directory: Path,
    chunk_args_template: dict,
    max_workers: int,
    output_files: list[str],
) -> list[str]:
    """
    Monitor for new chunk files as the pipeline creates them and process
    them immediately using a ProcessPoolExecutor.

    This implements true producer-consumer concurrency where TSV chunks
    are processed as soon as they're created.
    """
    import time
    from concurrent.futures import Future, as_completed

    processed_files = []
    chunk_counter = 0
    pending_futures: dict[Future, str] = {}
    no_activity_cycles = 0

    # Progress tracking - limit to 30 log messages max
    max_log_groups = 30
    last_logged_group = -1

    log.info(f"Starting concurrent chunk monitoring with {max_workers} workers")
    log.info(f"System has {os.cpu_count()} CPU cores available")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while True:
            # Check if pipeline process is still running
            pipeline_done = process.poll() is not None

            # Look for new chunk files
            expected_chunk = f"chunk_{chunk_counter:06d}"
            chunk_path = chunk_directory / expected_chunk

            found_new_chunk = False
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                found_new_chunk = True
                no_activity_cycles = 0  # Reset counter

                # Prepare arguments for this chunk
                output_file = (
                    output_files[chunk_counter]
                    if chunk_counter < len(output_files)
                    else str(chunk_directory / f"output_{chunk_counter:06d}.parquet")
                )

                chunk_args = ChunkProcessingArgs(
                    chunk_file=str(chunk_path),
                    cols=chunk_args_template["cols"],
                    info_ids=chunk_args_template["info_ids"],
                    scalar_fmt_ids=chunk_args_template["scalar_fmt_ids"],
                    list_fmt_ids=chunk_args_template["list_fmt_ids"],
                    info_meta=chunk_args_template["info_meta"],
                    fmt_meta=chunk_args_template["fmt_meta"],
                    output_file=output_file,
                    is_first_chunk=(chunk_counter == 0),
                    overrides=chunk_args_template["overrides"],
                )

                # Submit chunk for processing
                future = executor.submit(_process_chunk_to_parquet, chunk_args)
                pending_futures[future] = expected_chunk
                chunk_counter += 1

            # Check for completed chunks
            completed_futures, last_logged_group = _check_completed_futures(
                pending_futures, processed_files, max_log_groups, last_logged_group
            )

            # Remove completed futures
            for future in completed_futures:
                del pending_futures[future]

            # Exit conditions with better logic
            exit_monitor, no_activity_cycles = _should_exit_monitoring(
                pipeline_done=pipeline_done,
                chunk_counter=chunk_counter,
                found_new_chunk=found_new_chunk,
                no_activity_cycles=no_activity_cycles,
                process=process,
                pending_futures_count=len(pending_futures),
            )
            if exit_monitor:
                break

            # Always sleep to avoid busy waiting
            time.sleep(0.1)

        # Wait for all remaining chunks to complete
        log.debug(f"Waiting for {len(pending_futures)} remaining chunks to complete")
        if pending_futures:  # Only wait if there are pending futures
            for future in as_completed(pending_futures):
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result)
                except Exception as e:
                    log.error(f"Error in final chunk processing: {e}")

    # Check pipeline exit status
    exit_code = process.wait()
    if exit_code != 0:
        stderr_output = process.stderr.read() if process.stderr else "No stderr"
        raise RuntimeError(f"bcftools + AWK + split pipeline failed with exit code {exit_code}: {stderr_output}")

    log.info(f"Completed processing {len(processed_files)} chunks concurrently")
    return processed_files


def _process_chunk_to_parquet(args: ChunkProcessingArgs) -> str:
    """
    Process a single chunk and convert it to Parquet.

    This is the worker function for parallel processing.
    Note: Parquet writing is often I/O bound, which may limit CPU utilization.
    """
    try:
        # Read the chunk TSV into DataFrame
        chunk_df = pl.read_csv(
            args.chunk_file, separator="\t", has_header=False, new_columns=args.cols, null_values=["."]
        )

        if chunk_df.height == 0:
            return ""  # Empty chunk

        # Apply casting with debug logging only for first chunk
        chunk_df = _apply_chunk_casting(
            chunk_df,
            args.info_ids,
            args.scalar_fmt_ids,
            args.list_fmt_ids,
            args.info_meta,
            args.fmt_meta,
        )

        # Apply categories
        chunk_df = _apply_chunk_categories(
            chunk_df, args.info_ids, args.scalar_fmt_ids, args.list_fmt_ids, args.info_meta, args.fmt_meta
        )

        # Write to Parquet
        chunk_df.write_parquet(args.output_file)

        # Clean up input chunk file
        Path(args.chunk_file).unlink(missing_ok=True)

        return args.output_file

    except Exception as e:
        log.error(f"Error processing chunk {args.chunk_file}: {e}")
        # Clean up on error
        Path(args.chunk_file).unlink(missing_ok=True)
        if Path(args.output_file).exists():
            Path(args.output_file).unlink(missing_ok=True)
        raise


def _merge_parquet_files_lazy(parquet_files: list[str], output_path: str) -> None:
    """
    Merge multiple Parquet files using Polars lazy evaluation for memory efficiency.
    """
    if not parquet_files:
        raise ValueError("No Parquet files to merge")

    if len(parquet_files) == 1:
        # Single file - just move it
        shutil.move(parquet_files[0], output_path)
        return

    log.debug(f"Merging {len(parquet_files)} Parquet files lazily")

    # Use lazy scanning and streaming write for memory efficiency
    lazy_frames = [pl.scan_parquet(f) for f in parquet_files]

    # Concatenate all lazy frames
    merged_lazy = pl.concat(lazy_frames, how="vertical")

    # Stream write to final output
    merged_lazy.sink_parquet(output_path)

    # Clean up temporary files
    for f in parquet_files:
        Path(f).unlink(missing_ok=True)

    log.debug(f"Merged Parquet files written to: {output_path}")


def vcf_to_parquet_parallel(
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
    categories_json: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    bcftools_threads: int = 1,
) -> None:
    """
    Convert VCF to Parquet using new parallel architecture.

    This implementation uses:
    1. bcftools + AWK for list explosion (external pipeline)
    2. Python ProcessPoolExecutor for parallel chunk processing
    3. Immediate TSV-to-Parquet conversion with equal-sized chunks
    4. Lazy merging of Parquet files using Polars streaming
    5. Debug logging only from first chunk

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
    categories_json : str | None
        Path to JSON file with categorical overrides
    chunk_size : int
        Number of rows per chunk for parallel processing
    max_workers : int
        Maximum number of parallel workers
    bcftools_threads : int
        Number of threads for bcftools
    """
    log.info(f"Converting {vcf} to {out} using parallel architecture")

    # Resolve bcftools path
    bcftools = _resolve_bcftools_command()

    # Parse VCF header
    info_meta, fmt_meta = header_meta(vcf, bcftools)

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

    # Build query string and column names
    fmt_str = _make_query_string(format_ids, info_ids)
    cols = [CHROM, POS, REF, ALT] + info_ids + format_ids

    # Find indices of list columns for AWK script
    list_fmt_indices = []
    base_cols = [CHROM, POS, REF, ALT] + info_ids
    for list_col in list_fmt_ids:
        idx = len(base_cols) + format_ids.index(list_col)
        list_fmt_indices.append(idx)

    with pl.StringCache():
        # Create temporary directory for chunk files
        chunk_directory = Path(tempfile.mkdtemp(prefix="vcf_chunks_"))

        # Get category overrides
        overrides = _get_override_categories(categories_json)

        # Prepare output files for chunks (we'll estimate initial count)
        estimated_chunks = 50  # Start with reasonable estimate
        output_files = []
        for i in range(estimated_chunks):
            output_file = tempfile.NamedTemporaryFile(suffix=f"_chunk_{i}.parquet", delete=False).name
            output_files.append(output_file)

        try:
            # Step 1: Start the bcftools + AWK + split pipeline (non-blocking)
            log.info("Step 1: Starting streaming bcftools + AWK + split pipeline")
            pipeline_process = _explode_and_split_with_bcftools_awk_streaming(
                vcf, fmt_str, list_fmt_indices, bcftools, chunk_size, chunk_directory, bcftools_threads
            )

            # Prepare template arguments for chunk processing
            chunk_args_template = {
                "cols": cols,
                "info_ids": info_ids,
                "scalar_fmt_ids": scalar_fmt_ids,
                "list_fmt_ids": list_fmt_ids,  # These are now scalar after explosion
                "info_meta": info_meta,
                "fmt_meta": fmt_meta,
                "overrides": overrides,
            }

            # Step 2: Monitor and process chunks concurrently as they're created
            log.info(f"Step 2: Starting concurrent chunk processing with {max_workers} workers")
            parquet_files = _monitor_and_process_chunks_concurrent(
                pipeline_process, chunk_directory, chunk_args_template, max_workers, output_files
            )

            # Step 3: Merge Parquet files lazily
            log.info(f"Step 3: Merging {len(parquet_files)} Parquet files")
            _merge_parquet_files_lazy(parquet_files, out)

        finally:
            # Clean up chunk files and directory
            if chunk_directory.exists():
                for chunk_file in chunk_directory.glob("chunk_*"):
                    chunk_file.unlink(missing_ok=True)
                chunk_directory.rmdir()

            # Clean up output files
            for output_file in output_files:
                Path(output_file).unlink(missing_ok=True)

    log.info(f"Conversion completed: {out}")


def _build_explicit_schema(cols: list[str], info_meta: dict, fmt_meta: dict) -> dict[str, pl.DataType]:
    """
    Build explicit Polars schema from VCF metadata to avoid schema inference issues.

    This function maps VCF type information to appropriate Polars data types,
    ensuring decimal values are correctly parsed as floats instead of integers.

    Parameters
    ----------
    cols : list[str]
        Column names from the featuremap
    info_meta : dict
        INFO field metadata from VCF header
    fmt_meta : dict
        FORMAT field metadata from VCF header

    Returns
    -------
    dict[str, pl.DataType]
        Schema mapping column names to Polars data types
    """
    schema = {}

    # Standard VCF columns - these are always strings or ints
    standard_vcf_cols = {
        CHROM: pl.Utf8,
        POS: pl.Int64,
        ID: pl.Utf8,
        REF: pl.Utf8,
        ALT: pl.Utf8,
        QUAL: pl.Float64,  # QUAL can be float
        FILTER: pl.Utf8,
        SAMPLE: pl.Utf8,
    }

    for col in cols:
        if col in standard_vcf_cols:
            schema[col] = standard_vcf_cols[col]
        else:
            # Look up type information from VCF metadata
            meta = info_meta.get(col) or fmt_meta.get(col)
            if meta:
                vcf_type = meta.get("type", "String")
                # Map VCF types to Polars types
                if vcf_type == "Integer":
                    schema[col] = pl.Int64
                elif vcf_type == "Float":
                    schema[col] = pl.Float64
                elif vcf_type == "Flag":
                    schema[col] = pl.Boolean
                elif vcf_type == "String":
                    schema[col] = pl.Utf8
                else:
                    # Default to string for unknown types
                    schema[col] = pl.Utf8
            else:
                # Default to string if no metadata found
                schema[col] = pl.Utf8

    return schema


# ─────────────────────────── CLI wrapper ─────────────────────────────────────
def main() -> None:
    """
    CLI entry-point.

    Examples
    --------
    >>> python featuremap_to_dataframe.py \\
            --vcf sample.featuremap.vcf.gz \\
            --out-parquet sample.featuremap.parquet
    """
    ap = argparse.ArgumentParser(description="Featuremap VCF to DataFrame conversion saved in a Parquet format")
    ap.add_argument("--vcf", required=True, help="input VCF/BCF (bgz ok)")
    ap.add_argument("--out-parquet", required=True, help="output .parquet")
    ap.add_argument("--drop-info", default="", help="comma-separated INFO tags to drop")
    ap.add_argument("--drop-format", default="", help="comma-separated FORMAT tags to drop")
    ap.add_argument(
        "--categories-json",
        help="JSON file whose `categorical_features` map overrides/extends " "automatically-inferred enumerations.",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="use fast streaming version instead of memory-efficient version (may use more memory)",
    )
    ap.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="use new parallel architecture with bcftools+AWK (default, recommended)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"chunk size for parallel processing (default: {CHUNK_SIZE})",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"maximum number of parallel workers (default: {DEFAULT_MAX_WORKERS})",
    )
    ap.add_argument(
        "--bcftools-threads",
        type=int,
        default=1,
        help="number of threads for bcftools (default: 1)",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="emit DEBUG‐level logs")
    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    if args.parallel:
        # Use new parallel architecture (default, recommended)
        vcf_to_parquet_parallel(
            vcf=args.vcf,
            out=args.out_parquet,
            drop_info=set(filter(None, args.drop_info.split(","))),
            drop_format=set(filter(None, args.drop_format.split(","))),
            categories_json=args.categories_json,
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
            bcftools_threads=args.bcftools_threads,
        )
    elif args.fast:
        # Use streaming version for faster processing (may use more memory)
        vcf_to_parquet_streaming(
            vcf=args.vcf,
            out=args.out_parquet,
            drop_info=set(filter(None, args.drop_info.split(","))),
            drop_format=set(filter(None, args.drop_format.split(","))),
            categories_json=args.categories_json,
        )
    else:
        # Use memory-efficient version
        vcf_to_parquet_memory_efficient(
            vcf=args.vcf,
            out=args.out_parquet,
            drop_info=set(filter(None, args.drop_info.split(","))),
            drop_format=set(filter(None, args.dropFormat.split(","))),
            categories_json=args.categories_json,
        )


if __name__ == "__main__":
    main()
