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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from polars.exceptions import ShapeError

log = logging.getLogger(__name__)

# Configuration constants
DEFAULT_JOBS = 0  # 0 means auto-detect CPU cores


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


# ChunkProcessingArgs dataclass removed - no longer needed with region-based architecture


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


# ─────────────────────── REGION-BASED PARALLELISM ─────────────────────────
def _extract_chromosome_sizes_from_vcf(vcf_path: str, bcftools_path: str) -> dict[str, int]:
    """
    Extract chromosome names and sizes from VCF header.

    Parameters
    ----------
    vcf_path : str
        Path to VCF file
    bcftools_path : str
        Path to bcftools executable

    Returns
    -------
    dict[str, int]
        Dictionary mapping chromosome names to their lengths
    """
    # Get header with contig information
    result = subprocess.run([bcftools_path, "view", "-h", vcf_path], capture_output=True, text=True, check=True)

    chrom_sizes = {}
    contig_pattern = re.compile(r"##contig=<ID=([^,>]+)(?:,length=(\d+))?")

    for line in result.stdout.split("\n"):
        match = contig_pattern.match(line)
        if match:
            chrom_id = match.group(1)
            length = int(match.group(2)) if match.group(2) else 1000000  # Default 1MB if no length
            chrom_sizes[chrom_id] = length

    # If no contigs found in header, extract from data
    if not chrom_sizes:
        result = subprocess.run(
            [bcftools_path, "query", "-f", "%CHROM\n", vcf_path], capture_output=True, text=True, check=True
        )
        unique_chroms = set(result.stdout.strip().split("\n"))
        chrom_sizes = {chrom: 1000000 for chrom in unique_chroms if chrom}  # Default 1MB

    return chrom_sizes


def _generate_genomic_regions(vcf_path: str, num_regions: int, bcftools_path: str) -> list[str]:
    """
    Generate exactly num_regions regions that collectively cover all chromosomes.

    Distributes chromosomes across regions based on cumulative genomic size to
    ensure roughly equal workload per region.

    Parameters
    ----------
    vcf_path : str
        Path to VCF file
    num_regions : int
        Number of regions to create
    bcftools_path : str
        Path to bcftools executable

    Returns
    -------
    list[str]
        List of region strings - either "" for entire file, or comma-separated chromosome lists
    """
    # Special case: single region processes entire file
    if num_regions == 1:
        return [""]  # Empty string means entire file

    # Extract chromosome sizes from VCF header
    chrom_sizes = _extract_chromosome_sizes_from_vcf(vcf_path, bcftools_path)

    if not chrom_sizes:
        raise RuntimeError("Could not extract chromosome information from VCF")

    # Filter to only chromosomes that have data in the VCF
    result = subprocess.run(
        [bcftools_path, "query", "-f", "%CHROM\n", vcf_path], capture_output=True, text=True, check=True
    )

    data_chroms = []
    seen = set()
    for line in result.stdout.strip().split("\n"):
        chrom = line.strip()
        if chrom and chrom not in seen:
            data_chroms.append(chrom)
            seen.add(chrom)

    # Keep only chromosomes with data, preserving order
    filtered_chrom_sizes = [(chrom, chrom_sizes[chrom]) for chrom in data_chroms if chrom in chrom_sizes]

    if not filtered_chrom_sizes:
        raise RuntimeError("No chromosomes with data found in VCF")

    log.info(f"Found {len(filtered_chrom_sizes)} chromosomes with data")

    # Calculate total genome size and target size per region
    total_genome_size = sum(size for _, size in filtered_chrom_sizes)
    target_size_per_region = total_genome_size / num_regions

    log.info(f"Total genome size: {total_genome_size:,} bp")
    log.info(f"Target size per region: {target_size_per_region:,.0f} bp")

    # Distribute chromosomes across regions by cumulative size
    regions = []
    current_region_chroms = []
    current_region_size = 0

    for chrom, size in filtered_chrom_sizes:
        # Add chromosome to current region
        current_region_chroms.append(chrom)
        current_region_size += size

        # Check if we should close current region
        should_close_region = False

        if len(regions) == num_regions - 1:
            # Last region - include all remaining chromosomes
            should_close_region = chrom == filtered_chrom_sizes[-1][0]  # Last chromosome
        # Close region if it's large enough
        elif current_region_size >= target_size_per_region * 0.8:
            should_close_region = True

        if should_close_region:
            # Close current region
            if len(current_region_chroms) == 1:
                regions.append(current_region_chroms[0])
            else:
                regions.append(",".join(current_region_chroms))

            # Reset for next region
            current_region_chroms = []
            current_region_size = 0

    # Handle any remaining chromosomes (shouldn't happen with correct logic)
    if current_region_chroms:
        if len(current_region_chroms) == 1:
            regions.append(current_region_chroms[0])
        else:
            regions.append(",".join(current_region_chroms))

    # Ensure we have exactly num_regions
    while len(regions) < num_regions:
        regions.append("")  # Empty regions for unused workers

    if len(regions) > num_regions:
        # Too many regions - merge last ones
        merged_chroms = []
        for region in regions[num_regions - 1 :]:
            if region:
                merged_chroms.extend(region.split(","))
        regions = regions[: num_regions - 1]
        if merged_chroms:
            regions.append(",".join(merged_chroms))

    log.info(f"Generated {len(regions)} regions:")
    for i, region in enumerate(regions):
        if region:
            chroms = region.split(",")
            log.info(f"  Region {i+1}: {len(chroms)} chromosome(s) - {region}")
        else:
            log.info(f"  Region {i+1}: empty")

    return regions


def _process_region_to_parquet(
    region: str,
    vcf_path: str,
    fmt_str: str,
    list_fmt_indices: list[int],
    cols: list[str],
    info_ids: list[str],
    scalar_fmt_ids: list[str],
    list_fmt_ids: list[str],
    info_meta: dict,
    fmt_meta: dict,
    bcftools_path: str,
    awk_script: str,
    output_file: str,
) -> str:
    """
    Process a single genomic region and return parquet file path.

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
    list_fmt_indices : list[int]
        Indices of list columns for AWK processing
    cols : list[str]
        Column names
    info_ids : list[str]
        INFO field IDs
    scalar_fmt_ids : list[str]
        Scalar FORMAT field IDs
    list_fmt_ids : list[str]
        List FORMAT field IDs
    info_meta : dict
        INFO metadata
    fmt_meta : dict
        FORMAT metadata
    bcftools_path : str
        Path to bcftools executable
    awk_script : str
        Path to AWK script
    output_file : str
        Path for output parquet file

    Returns
    -------
    str
        Path to created parquet file, empty string if no data
    """
    import polars as pl

    try:
        # Recreate schema and processing args inside the worker process
        schema = _build_explicit_schema(cols, info_meta, fmt_meta)

        processing_args = {
            "bcftools_path": bcftools_path,
            "awk_script": awk_script,
            "cols": cols,
            "schema": schema,
            "info_ids": info_ids,
            "scalar_fmt_ids": scalar_fmt_ids,
            "list_fmt_ids": list_fmt_ids,
            "info_meta": info_meta,
            "fmt_meta": fmt_meta,
        }

        # Enable StringCache in worker process
        with pl.StringCache():
            # Stream region data directly to Polars DataFrame
            df = _stream_region_to_polars(
                region=region,
                vcf_path=vcf_path,
                fmt_str=fmt_str,
                list_fmt_indices=list_fmt_indices,
                processing_args=processing_args,
            )

            if df.height == 0:
                return ""  # No data in this region

            # Apply processing
            df = _apply_region_processing(df, processing_args)

            # Write to parquet
            df.write_parquet(output_file)
            return output_file

    except Exception as e:
        log.error(f"Error processing region {region}: {e}")
        # Clean up on error
        if Path(output_file).exists():
            Path(output_file).unlink(missing_ok=True)
        raise


def _stream_region_to_polars(
    region: str,
    vcf_path: str,
    fmt_str: str,
    list_fmt_indices: list[int],
    processing_args: dict,
) -> pl.DataFrame:
    """
    Stream bcftools output directly to Polars DataFrame for a specific region.

    Parameters
    ----------
    region : str
        Genomic region to process
    vcf_path : str
        Path to VCF file
    fmt_str : str
        bcftools query format string
    list_fmt_indices : list[int]
        Indices of list columns for explosion
    processing_args : dict
        Processing arguments containing metadata and config

    Returns
    -------
    pl.DataFrame
        Processed DataFrame for the region
    """
    bcftools_path = processing_args["bcftools_path"]
    awk_script = processing_args["awk_script"]
    cols = processing_args["cols"]
    schema = processing_args["schema"]

    # Build bcftools command with or without region
    if region:
        bcftools_cmd = [bcftools_path, "query", "-r", region, "-f", fmt_str, vcf_path]
    else:
        # Empty region means process entire file
        bcftools_cmd = [bcftools_path, "query", "-f", fmt_str, vcf_path]

    # Build AWK command
    indices_str = ",".join(map(str, list_fmt_indices))
    awk_cmd = ["awk", "-v", f"list_indices={indices_str}", "-f", awk_script]

    # Create pipeline: bcftools | awk
    log.debug(f"Processing region {region}")

    bcftools_process = subprocess.Popen(bcftools_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    awk_process = subprocess.Popen(
        awk_cmd, stdin=bcftools_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Close bcftools stdout in parent to allow proper pipeline behavior
    if bcftools_process.stdout:
        bcftools_process.stdout.close()

    try:
        # Read AWK output and create DataFrame
        awk_output, awk_stderr = awk_process.communicate()

        # Check for errors
        bcftools_exit = bcftools_process.wait()
        awk_exit = awk_process.wait()

        if bcftools_exit != 0:
            bcftools_stderr = bcftools_process.stderr.read() if bcftools_process.stderr else ""
            raise subprocess.CalledProcessError(bcftools_exit, bcftools_cmd, bcftools_stderr)

        if awk_exit != 0:
            raise subprocess.CalledProcessError(awk_exit, awk_cmd, awk_stderr)

        # Process AWK output
        if not awk_output.strip():
            return pl.DataFrame({col: [] for col in cols})

        # Convert to DataFrame using StringIO-like approach
        lines = awk_output.strip().split("\n")
        if not lines:
            return pl.DataFrame({col: [] for col in cols})

        # Parse lines into data
        data = {col: [] for col in cols}
        for line in lines:
            fields = line.split("\t")
            for i, col in enumerate(cols):
                if i < len(fields):
                    value = fields[i] if fields[i] != "." else None
                else:
                    value = None
                data[col].append(value)

        # Create DataFrame with explicit schema, allowing mixed types initially
        try:
            df = pl.DataFrame(data, schema=schema, strict=False)
        except Exception as e:
            log.warning(f"Failed to create DataFrame with strict schema, falling back to loose typing: {e}")
            # Create without schema first, then cast
            df = pl.DataFrame(data)
            # Apply type conversions
            for col, dtype in schema.items():
                if col in df.columns:
                    try:
                        if dtype == pl.Int64:
                            # Handle integer conversion with null values
                            df = df.with_columns(pl.col(col).str.replace("^\\.$", "").cast(dtype, strict=False))
                        elif dtype == pl.Float64:
                            # Handle float conversion with null values
                            df = df.with_columns(pl.col(col).str.replace("^\\.$", "").cast(dtype, strict=False))
                        else:
                            df = df.with_columns(pl.col(col).cast(dtype, strict=False))
                    except Exception as cast_error:
                        log.warning(f"Failed to cast column {col} to {dtype}: {cast_error}")
                        # Keep as string if casting fails

        if df.height == 0:
            return df

        # Apply processing
        df = _apply_region_processing(df, processing_args)

        return df

    finally:
        # Cleanup processes
        if bcftools_process.stdout and not bcftools_process.stdout.closed:
            bcftools_process.stdout.close()
        if bcftools_process.stderr:
            bcftools_process.stderr.close()
        if awk_process.stderr:
            awk_process.stderr.close()


def _apply_region_processing(df: pl.DataFrame, processing_args: dict) -> pl.DataFrame:
    """Apply column casting and categorical processing to a region DataFrame."""
    info_ids = processing_args["info_ids"]
    scalar_fmt_ids = processing_args["scalar_fmt_ids"]
    list_fmt_ids = processing_args["list_fmt_ids"]
    info_meta = processing_args["info_meta"]
    fmt_meta = processing_args["fmt_meta"]

    # Cast POS to Int64
    df = df.with_columns(pl.col(POS).cast(pl.Int64))

    # Apply column casting - list columns are already exploded as scalars
    for tag in info_ids:
        df = _cast_scalar(df, tag, info_meta[tag])
    for tag in scalar_fmt_ids:
        df = _cast_scalar(df, tag, fmt_meta[tag])
    for tag in list_fmt_ids:
        # These were exploded by AWK so now treat as scalars
        df = _cast_scalar(df, tag, fmt_meta[tag])

    # REF / ALT
    for allele in (REF, ALT):
        df = _cast_scalar(df, allele, {"type": "String", "cat": ALLELE_CATS})

    # Apply categories
    for tag in info_ids + scalar_fmt_ids + list_fmt_ids + [REF, ALT]:
        if tag in df.columns:
            col_meta = info_meta.get(tag) or fmt_meta.get(tag) or {"cat": ALLELE_CATS if tag in [REF, ALT] else None}
            cats = col_meta.get("cat")
            if cats and df[tag].dtype == pl.Categorical:
                df = _ensure_scalar_categories(df, tag, cats)

    return df


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
# Only parallel processing is supported for optimal performance and CPU utilization


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


# Old batch processing functions removed - replaced by region-based parallel processing


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


# Old streaming chunk functions removed - replaced by region-based parallel processing


# Old chunk processing monitoring functions removed - replaced by region-based parallel processing


# Old chunk monitoring function removed - replaced by region-based parallel processing


# Old chunk processing function removed - replaced by region-based parallel processing


def _merge_parquet_files_lazy(parquet_files: list[str], output_path: str) -> None:
    """
    Merge multiple Parquet files using Polars lazy evaluation for memory efficiency.
    """
    if not parquet_files:
        log.warning("No Parquet files to merge - creating empty output file")
        # Create an empty Parquet file with minimal structure
        empty_df = pl.DataFrame({"CHROM": [], "POS": [], "REF": [], "ALT": []})
        empty_df.write_parquet(output_path)
        return

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


def vcf_to_parquet(
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
    categories_json: str | None = None,
    jobs: int = DEFAULT_JOBS,
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
    categories_json : str | None
        Path to JSON file with categorical overrides
    jobs : int
        Number of parallel jobs (0 = auto-detect CPU cores)
    """
    log.info(f"Converting {vcf} to {out} using region-based parallel processing")

    # Auto-detect optimal job count if not specified
    if jobs == 0:
        jobs = os.cpu_count() or 4

    log.info(f"Using {jobs} parallel jobs for region processing")

    # Resolve bcftools path
    bcftools = _resolve_bcftools_command()

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
        # Special case: if jobs=1, process entire file without region splitting
        if jobs == 1:
            log.info("Processing entire VCF file without region splitting (jobs=1)")

            # Get AWK script path
            awk_script = _get_awk_script_path()

            # Build explicit schema for consistent processing
            schema = _build_explicit_schema(cols, info_meta, fmt_meta)

            # Prepare processing arguments
            processing_args = {
                "bcftools_path": bcftools,
                "awk_script": awk_script,
                "cols": cols,
                "schema": schema,
                "info_ids": info_ids,
                "scalar_fmt_ids": scalar_fmt_ids,
                "list_fmt_ids": list_fmt_ids,
                "info_meta": info_meta,
                "fmt_meta": fmt_meta,
            }

            # Process entire file as single region (no region parameter)
            df = _stream_region_to_polars(
                region="",  # Empty region means process entire file
                vcf_path=vcf,
                fmt_str=fmt_str,
                list_fmt_indices=list_fmt_indices,
                processing_args=processing_args,
            )

            if df.height == 0:
                log.warning("No data processed from VCF file")
                return

            # Apply processing and write directly to output
            df = _apply_region_processing(df, processing_args)
            df.write_parquet(out)

            log.info(f"Conversion completed: {out}")
            return

        # Generate genomic regions for parallel processing
        log.info(f"Generating {jobs} genomic regions for parallel processing")
        regions = _generate_genomic_regions(vcf, jobs, bcftools)
        log.info(f"Created {len(regions)} regions: {regions[:5]}{'...' if len(regions) > 5 else ''}")

        # Get AWK script path
        awk_script = _get_awk_script_path()

        # Build explicit schema for consistent processing
        schema = _build_explicit_schema(cols, info_meta, fmt_meta)

        # Prepare processing arguments
        processing_args = {
            "bcftools_path": bcftools,
            "awk_script": awk_script,
            "cols": cols,
            "schema": schema,
            "info_ids": info_ids,
            "scalar_fmt_ids": scalar_fmt_ids,
            "list_fmt_ids": list_fmt_ids,
            "info_meta": info_meta,
            "fmt_meta": fmt_meta,
        }

        # Process regions in parallel
        parquet_files = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            # Submit all region processing tasks
            future_to_region = {}
            for i, region in enumerate(regions):
                output_file = tempfile.NamedTemporaryFile(suffix=f"_region_{i}.parquet", delete=False).name
                future = executor.submit(
                    _process_region_to_parquet,
                    region,
                    vcf,
                    fmt_str,
                    list_fmt_indices,
                    cols,
                    info_ids,
                    scalar_fmt_ids,
                    list_fmt_ids,
                    info_meta,
                    fmt_meta,
                    bcftools,
                    awk_script,
                    output_file,
                )
                future_to_region[future] = (region, output_file)

            # Collect results in order
            log.info("Processing regions in parallel...")
            for i, future in enumerate(as_completed(future_to_region)):
                region, output_file = future_to_region[future]
                try:
                    result = future.result()
                    if result:  # Non-empty result
                        parquet_files.append(result)
                    if (i + 1) % max(1, len(regions) // 10) == 0:
                        log.info(f"Completed {i + 1}/{len(regions)} regions")
                except Exception as e:
                    log.error(f"Error processing region {region}: {e}")
                    # Clean up failed output file
                    Path(output_file).unlink(missing_ok=True)
                    raise

        # Sort parquet files to ensure consistent order
        # Extract region info for sorting
        region_order = {}
        for i, region in enumerate(regions):
            region_order[f"_region_{i}.parquet"] = i

        parquet_files.sort(key=lambda f: next((order for suffix, order in region_order.items() if suffix in f), 999))

        # Merge Parquet files in order
        log.info(f"Merging {len(parquet_files)} Parquet files in order")
        _merge_parquet_files_lazy(parquet_files, out)

    log.info(f"Conversion completed: {out}")


# Legacy alias for backward compatibility
vcf_to_parquet_parallel = vcf_to_parquet


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
        help="JSON file whose `categorical_features` map overrides/extends automatically-inferred enumerations.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"number of parallel jobs (0 = auto-detect CPU cores, default: {DEFAULT_JOBS})",
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

    # Run the region-based parallel conversion
    vcf_to_parquet(
        vcf=args.vcf,
        out=args.out_parquet,
        drop_info=set(filter(None, args.drop_info.split(","))),
        drop_format=set(filter(None, args.drop_format.split(","))),
        categories_json=args.categories_json,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
