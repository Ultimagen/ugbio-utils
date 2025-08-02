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
import logging
import multiprocessing as _mp  # NEW
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import polars as pl

from ugbio_featuremap.featuremap_utils import FeatureMapFields

log = logging.getLogger(__name__)

# Configuration constants
DEFAULT_JOBS = 0  # 0 means auto-detect CPU cores
CHUNK_BP_DEFAULT = 300_000_000  # 300 Mbp per processing chunk


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
REF_ALLELE_CATS = ALT_ALLELE_CATS + ["R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", "N"]


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


# ───────────────── category helper ────────────────────────────────────────
def _ensure_scalar_categories(
    featuremap_dataframe: pl.DataFrame,
    col: str,
    cats: list[str],
) -> pl.DataFrame:
    """
    Cast *col* to a Polars Enum whose category dictionary is **exactly**
    `cats` plus an empty-string entry (""), ensuring round-trip fidelity.

    Any literal “.” is treated as missing (null); the empty string stays a
    valid category so downstream code can faithfully round-trip values.
    """
    if "" not in cats:
        cats = cats + [""]
    enum_dtype = pl.Enum(cats)

    cleaned = (
        pl.when(pl.col(col).cast(pl.Utf8) == ".")
        .then(None)  # “.”  →  null
        .otherwise(pl.col(col).cast(pl.Utf8))
    )
    return featuremap_dataframe.with_columns(cleaned.cast(enum_dtype, strict=False).alias(col))


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
        # Use Enum with fixed dictionary
        featuremap_dataframe = featuremap_dataframe.with_columns(utf_null.alias(col))
        return _ensure_scalar_categories(featuremap_dataframe, col, meta["cat"])

    if meta["type"] in _POLARS_DTYPE:
        return featuremap_dataframe.with_columns(utf_null.cast(_POLARS_DTYPE[meta["type"]], strict=False).alias(col))

    return featuremap_dataframe.with_columns(utf_null.alias(col))


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
            [bedtools_path, "makewindows", "-g", genome_sizes_path, "-w", str(window_size)],
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
    """Build the bcftools query format string (now includes %QUAL)."""
    bracket = "[" + "\t".join(f"%{t}" for t in format_ids) + "]"
    return "\t".join(["%CHROM", "%POS", "%QUAL", "%REF", "%ALT", *[f"%INFO/{t}" for t in query_info], bracket]) + "\n"


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


# ───────────────── schema helper ────────────────────────────────────────────
def _build_explicit_schema(cols: list[str], info_meta: dict, fmt_meta: dict) -> dict[str, pl.PolarsDataType]:
    """
    Build an explicit Polars schema for the bcftools TSV based on VCF metadata.
    Only scalar columns get strict numeric types; everything else is Utf8.
    """
    schema: dict[str, pl.PolarsDataType] = {}
    meta_lookup = {**info_meta, **fmt_meta}
    for col in cols:
        if col in (CHROM, REF, ALT, QUAL, FILTER, ID):
            schema[col] = pl.Utf8
        elif col == POS:
            schema[col] = pl.Int64
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


def vcf_to_parquet(
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
    chunk_bp: int = CHUNK_BP_DEFAULT,
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
    chunk_bp : int
        Maximum number of base-pairs per chunk (default 300 Mbp).
    jobs : int
        Number of parallel jobs (0 = auto-detect CPU cores)
    """
    log.info(f"Converting {vcf} to {out} using region-based parallel processing")
    _assert_vcf_index_exists(vcf)

    # Auto-detect optimal job count if not specified
    if jobs == 0:
        jobs = os.cpu_count() or 4

    log.info(f"Using {jobs} parallel jobs for region processing")

    # Resolve tool paths
    bcftools = _resolve_bcftools_command()
    bedtools = _resolve_bedtools_command()

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
    cols = [CHROM, POS, QUAL, REF, ALT] + info_ids + format_ids

    # Find indices of list columns for AWK script
    list_fmt_indices = []
    base_cols = [CHROM, POS, QUAL, REF, ALT] + info_ids
    for list_col in list_fmt_ids:
        idx = len(base_cols) + format_ids.index(list_col)
        list_fmt_indices.append(idx)

    with pl.StringCache():
        # Generate genomic regions (fixed windows via bedtools)
        regions = _generate_genomic_regions(
            vcf,
            jobs,
            bcftools,
            bedtools,
            window_size=chunk_bp,
        )
        log.info(f"Created {len(regions)} regions: {regions[:5]}{'...' if len(regions) > 5 else ''}")  # noqa PLR2004

        # Build immutable job configuration (shared by every worker)  ▼ NEW
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
        )

        # Temporary directory for ordered part-files
        with tempfile.TemporaryDirectory() as tmpdir:
            spawn_ctx = _mp.get_context("spawn")
            part_files: list[str] = []

            with ProcessPoolExecutor(max_workers=jobs, mp_context=spawn_ctx) as executor:
                futures = {}
                for i, region in enumerate(regions):
                    part_path = Path(tmpdir) / f"part_{i:06d}.parquet"
                    part_files.append(str(part_path))
                    futures[
                        executor.submit(
                            _process_region_to_parquet,
                            region,
                            vcf,
                            fmt_str,
                            job_cfg,
                            str(part_path),
                        )
                    ] = str(part_path)

                for fut in as_completed(futures):
                    if fut.exception():
                        log.error("Region %s failed: %s", futures[fut], fut.exception())

            # keep only those parts that were actually created
            part_files = [p for p in part_files if Path(p).is_file()]
            if not part_files:
                raise RuntimeError("No Parquet part-files were produced – all regions empty or failed")

            _merge_parquet_files_lazy(part_files, out)

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
    try:
        with pl.StringCache():
            frame = _stream_region_to_polars(
                region=region,
                vcf_path=vcf_path,
                fmt_str=fmt_str,
                job_cfg=job_cfg,
            )
            if frame.is_empty():
                return ""
            frame = _cast_column_data_types(frame, job_cfg)
            frame.write_parquet(output_file)
            return output_file

    except Exception as e:
        log.error(f"Error processing region {region}: {e}")
        # Clean up on error
        if Path(output_file).exists():
            Path(output_file).unlink(missing_ok=True)
        raise


# ──────────────────────── new tiny helpers ────────────────────────────────
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

    awk_cmd = ["awk", "-v", f"list_indices={','.join(map(str, list_indices))}", "-f", awk_script]

    bcftool = subprocess.Popen(bcftools_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    awk = subprocess.Popen(awk_cmd, stdin=bcftool.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        return frame
    return _cast_column_data_types(frame, job_cfg)


# ────────────────────────── new small helpers ───────────────────────────
def _cast_ref_alt_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast REF/ALT/X_ALT using their dedicated category dictionaries."""
    frame = _cast_scalar(frame, REF, {"type": "String", "cat": REF_ALLELE_CATS})
    frame = _cast_scalar(frame, ALT, {"type": "String", "cat": ALT_ALLELE_CATS})
    if X_ALT in frame.columns:
        frame = _cast_scalar(frame, X_ALT, {"type": "String", "cat": REF_ALLELE_CATS})
    return frame


def _apply_scalar_categories(
    frame: pl.DataFrame,
    touched_tags: list[str],
    info_meta: dict,
    fmt_meta: dict,
) -> pl.DataFrame:
    """Ensure all categories from header metadata are present."""
    for tag in touched_tags:
        if tag not in frame.columns:
            continue
        col_meta = (
            info_meta.get(tag)
            or fmt_meta.get(tag)
            or {"cat": REF_ALLELE_CATS if tag in (REF, X_ALT) else ALT_ALLELE_CATS if tag == ALT else None}
        )
        cats = col_meta.get("cat")
        if cats:
            frame = _ensure_scalar_categories(frame, tag, cats)
    return frame


def _handle_nulls_values(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill categorical nulls with '' and numeric nulls with 0, logging actions."""
    for col, dtype in frame.schema.items():
        # ----- categorical (Enum) -----
        if isinstance(dtype, pl.Enum):
            n_null = frame[col].null_count()
            if n_null:
                log.debug(f'Filling {n_null} missing values in categorical column "{col}" with ""')
                frame = frame.with_columns(pl.col(col).fill_null("").alias(col))

        # ----- numeric -----
        if dtype in pl.NUMERIC_DTYPES:
            n_null = frame[col].null_count()
            if n_null:
                log.debug(f"Filling {n_null} missing values in numeric column '{col}' with 0")
                frame = frame.with_columns(pl.col(col).fill_null(0).alias(col))
    return frame


def _cast_column_data_types(featuremap_dataframe: pl.DataFrame, job_cfg: VCFJobConfig) -> pl.DataFrame:
    """Apply column casting and categorical processing to a region DataFrame."""
    info_ids = job_cfg.info_ids
    scalar_fmt_ids = job_cfg.scalar_fmt_ids
    list_fmt_ids = job_cfg.list_fmt_ids
    info_meta = job_cfg.info_meta
    fmt_meta = job_cfg.fmt_meta

    # Cast POS to Int64
    featuremap_dataframe = featuremap_dataframe.with_columns(pl.col(POS).cast(pl.Int64))

    # Apply column casting - list columns are already exploded as scalars
    for tag in info_ids:
        featuremap_dataframe = _cast_scalar(featuremap_dataframe, tag, info_meta[tag])
    for tag in scalar_fmt_ids:
        featuremap_dataframe = _cast_scalar(featuremap_dataframe, tag, fmt_meta[tag])
    for tag in list_fmt_ids:
        featuremap_dataframe = _cast_scalar(featuremap_dataframe, tag, fmt_meta[tag])

    # QUAL ─ force Float64 even if all values are missing
    if QUAL in featuremap_dataframe.columns:
        featuremap_dataframe = _cast_scalar(featuremap_dataframe, QUAL, {"type": "Float", "cat": None})

    # ----- REF / ALT / X_ALT -------------------------------------------
    featuremap_dataframe = _cast_ref_alt_columns(featuremap_dataframe)

    # Apply categories for every scalar we touched
    touched = info_ids + scalar_fmt_ids + list_fmt_ids + [REF, ALT, X_ALT]
    featuremap_dataframe = _apply_scalar_categories(featuremap_dataframe, touched, info_meta, fmt_meta)

    # Handle nulls in categorical & numeric columns
    featuremap_dataframe = _handle_nulls_values(featuremap_dataframe)

    return featuremap_dataframe


# ────────────────────────────── CLI entry point ─────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """
    Minimal command-line interface, e.g.:

    $ python -m ugbio_featuremap.featuremap_to_dataframe  \
         --in sample.vcf.gz --out sample.parquet --jobs 4
    """
    parser = argparse.ArgumentParser(description="Convert feature-map VCF → Parquet", allow_abbrev=True)
    parser.add_argument("--input", required=True, help="Input VCF/BCF (bgzipped ok)")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="Parallel jobs (0 = auto)")
    parser.add_argument("--drop-info", nargs="*", default=[], help="INFO tags to drop")
    parser.add_argument("--drop-format", nargs="*", default=["GT"], help="FORMAT tags to drop")
    parser.add_argument(
        "--chunk-bp",
        type=int,
        default=CHUNK_BP_DEFAULT,
        help="Base-pairs per processing chunk (default 300 Mbp)",
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
    )


if __name__ == "__main__":  # pragma: no cover
    main()
