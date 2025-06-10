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
import re
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from polars.exceptions import ShapeError

log = logging.getLogger(__name__)


def _log_memory_usage(stage: str) -> None:
    """Log current stage for debugging."""
    log.debug(f"[{stage}] Processing stage completed")


def _log_dataframe_info(df: pl.DataFrame, stage: str) -> None:
    """Log DataFrame info for debugging."""
    log.debug(f"[{stage}] DataFrame shape: {df.shape}")
    log.debug(f"[{stage}] DataFrame columns: {df.columns}")
    log.debug(f"[{stage}] DataFrame dtypes: {dict(zip(df.columns, df.dtypes, strict=True))}")


@dataclass
class ColumnConfig:
    """Configuration for column processing."""

    info_ids: list[str]
    scalar_fmt_ids: list[str]
    list_fmt_ids: list[str]
    info_meta: dict
    fmt_meta: dict


# ───────────────── header helpers ────────────────────────────────────────────
INFO_RE = re.compile(r'##INFO=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]*)')
FORMAT_RE = re.compile(r'##FORMAT=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]*)')

_POLARS_DTYPE = {"Integer": pl.Int64, "Float": pl.Float64, "Flag": pl.Boolean}
CHROM, POS, REF, ALT, QUAL, FILTER = "CHROM", "POS", "REF", "ALT", "QUAL", "FILTER"
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
    stub_dict = {c: [None] * rows for c in featuremap_dataframe.columns}
    stub_dict[col] = cats  # each category once

    stub = (
        pl.DataFrame(stub_dict)
        .cast(featuremap_dataframe.schema, strict=False)  # keep dtypes identical
        .with_columns(pl.col(col).cast(pl.Categorical))
    )

    return pl.concat([featuremap_dataframe, stub], how="vertical").head(featuremap_dataframe.height)


# ───────────────── casting helpers ──────────────────────────────────────────
def _cast_scalar(featuremap_dataframe: pl.DataFrame, col: str, meta: dict) -> pl.DataFrame:
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

    Returns
    -------
    DataFrame with `col` recast and (if categorical) stub-padded so
    all categories from `meta["cat"]` are present.
    """

    utf_null = pl.when(pl.col(col).cast(pl.Utf8).is_in(["", "."])).then(None).otherwise(pl.col(col).cast(pl.Utf8))

    if meta["cat"]:
        featuremap_dataframe = featuremap_dataframe.with_columns(utf_null.cast(pl.Categorical).alias(col))
        log.debug(f"Scalar column {col} cast to Categorical with {len(meta['cat'])} categories")
        return _ensure_scalar_categories(featuremap_dataframe, col, meta["cat"])
    if meta["type"] in _POLARS_DTYPE:
        return featuremap_dataframe.with_columns(utf_null.cast(_POLARS_DTYPE[meta["type"]], strict=False).alias(col))

    log.debug("List column %s processed (type=%s)", col, meta["type"])
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
) -> pl.LazyFrame:
    """Run bcftools query and load the TSV into a LazyFrame for memory-efficient processing."""
    with tempfile.NamedTemporaryFile("w+b", delete=False) as tmp:
        subprocess.run([bcftools, "query", "-f", fmt_str, vcf], stdout=tmp, check=True)
        path = tmp.name

    # Use scan_csv for lazy loading - don't delete the file yet as lazy frames need it
    # The caller is responsible for cleanup if needed
    return pl.scan_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=cols,
        low_memory=True,
        null_values=["."],
        infer_schema_length=0,
    )


def _load_vcf_as_dataframe(
    bcftools: str,
    vcf: str,
    fmt_str: str,
    cols: list[str],
) -> pl.DataFrame:
    """Run bcftools query and load the TSV into a DataFrame."""
    with tempfile.NamedTemporaryFile("w+b", delete=False) as tmp:
        subprocess.run([bcftools, "query", "-f", fmt_str, vcf], stdout=tmp, check=True)
        path = tmp.name
    try:
        return pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=cols,
            low_memory=True,
            decimal_comma=True,
            null_values=["."],
            infer_schema_length=0,
        )
    finally:
        Path(path).unlink(missing_ok=True)


def _cast_scalar_lazy(lazy_df: pl.LazyFrame, col: str, meta: dict) -> pl.LazyFrame:
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

    Returns
    -------
    LazyFrame with `col` recast.
    """
    utf_null = pl.when(pl.col(col).cast(pl.Utf8).is_in(["", "."])).then(None).otherwise(pl.col(col).cast(pl.Utf8))

    if meta["cat"]:
        lazy_df = lazy_df.with_columns(utf_null.cast(pl.Categorical).alias(col))
        log.debug(f"Scalar column {col} cast to Categorical with {len(meta['cat'])} categories")
        return lazy_df
    if meta["type"] in _POLARS_DTYPE:
        return lazy_df.with_columns(utf_null.cast(_POLARS_DTYPE[meta["type"]], strict=False).alias(col))

    log.debug("Scalar column %s processed (type=%s)", col, meta["type"])
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
        lazy_df = _cast_scalar_lazy(lazy_df, allele, {"type": "String", "cat": ALLELE_CATS})
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
    log.debug(f"max_length: {max_length}, list_lengths: {list_lengths}")
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
        Number of variants to process at once.

    Notes
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
        log.debug("⚠️  Loading entire VCF into memory - THIS IS THE MEMORY-INTENSIVE STEP")
        _log_memory_usage("before_loading_vcf")

        # Load data eagerly but process in chunks - THIS IS THE MEMORY BOTTLENECK!
        featuremap_dataframe = _load_vcf_as_dataframe(bcftools, vcf, fmt_str, cols)

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
        log.debug("🔄 Starting streaming chunk processing")
        _log_memory_usage("before_streaming")
        temp_parquet_files = []
        try:
            column_config = ColumnConfig(
                info_ids=info_ids,
                scalar_fmt_ids=scalar_fmt_ids,
                list_fmt_ids=list_fmt_ids,
                info_meta=info_meta,
                fmt_meta=fmt_meta,
            )
            temp_parquet_files = _stream_and_process_chunks(
                bcftools,
                vcf,
                fmt_str,
                cols,
                list_fmt_indices,
                column_config,
                out,
            )
            _log_memory_usage("after_streaming")
            if temp_parquet_files:
                log.debug(f"🔗 Concatenating {len(temp_parquet_files)} temporary parquet files")
                _log_memory_usage("before_concatenation")
                lazy_frames = [pl.scan_parquet(f) for f in temp_parquet_files]
                final_df = pl.concat(lazy_frames, how="vertical").collect()
                _log_memory_usage("after_concatenation")
                log.debug(f"📝 Writing final parquet: {out} with shape {final_df.shape}")
                final_df.write_parquet(out)
                log.info("✅  %s: %d rows × %d cols", out, *final_df.shape)
            else:
                empty_df = pl.DataFrame({col: [] for col in cols})
                empty_df.write_parquet(out)
                log.info("✅  %s: 0 rows × %d cols (empty)", out, len(cols))
        finally:
            log.debug(f"🧹 Cleaning up {len(temp_parquet_files)} temporary files")
            for temp_file in temp_parquet_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception as e:
                    log.warning(f"Failed to delete temporary file {temp_file}: {e}")


def _stream_and_process_chunks(
    bcftools: str,
    vcf: str,
    fmt_str: str,
    cols: list[str],
    list_fmt_indices: list[int],
    column_config: ColumnConfig,
    out: str,
) -> list[str]:
    log.debug(f"[CHUNK] Starting _stream_and_process_chunks for {vcf}")
    temp_parquet_files = []
    chunk_rows = []
    chunk_idx = 0
    for row in _explode_tsv_rows_generator(bcftools, vcf, fmt_str, cols, list_fmt_indices):
        chunk_rows.append(row)
        if len(chunk_rows) >= CHUNK_SIZE:
            log.debug(f"[CHUNK] Processing chunk {chunk_idx} with {len(chunk_rows)} rows")
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
            log.debug(f"[CHUNK] Finished chunk {chunk_idx}, temp file: {temp_file}")
            _log_memory_usage(f"after_chunk_{chunk_idx}")
            chunk_rows = []
            chunk_idx += 1
    if chunk_rows:
        log.debug(f"[CHUNK] Processing final chunk {chunk_idx} with {len(chunk_rows)} rows")
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
        log.debug(f"[CHUNK] Finished final chunk {chunk_idx}, temp file: {temp_file}")
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

    if args.fast:
        # Use streaming version for faster processing (may use more memory)
        vcf_to_parquet_streaming(
            vcf=args.vcf,
            out=args.out_parquet,
            drop_info=set(filter(None, args.drop_info.split(","))),
            drop_format=set(filter(None, args.drop_format.split(","))),
            categories_json=args.categories_json,
        )
    else:
        # Use memory-efficient version by default
        vcf_to_parquet_memory_efficient(
            vcf=args.vcf,
            out=args.out_parquet,
            drop_info=set(filter(None, args.drop_info.split(","))),
            drop_format=set(filter(None, args.drop_format.split(","))),
            categories_json=args.categories_json,
        )


if __name__ == "__main__":
    main()
