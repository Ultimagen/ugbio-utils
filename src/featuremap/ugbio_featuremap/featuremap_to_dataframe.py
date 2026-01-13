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
import logging
import multiprocessing as _mp  # NEW
import os
from polars.dataframe.frame import DataFrame
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
    sample_list: list[str]
    fmt_ids: list[str]


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
    return (
        "\t".join(
            [
                "%CHROM",
                "%POS",
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
                continue
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

    return [p for p in part_files if Path(p).is_file()]


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

    # Fetch sample list ONCE in main process
    sample_list = _get_sample_list(vcf, bcftools)
    log.info(f"Found {len(sample_list)} sample(s): {sample_list}")

    # Cache fmt_ids
    fmt_ids = list(fmt_meta.keys())

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
            sample_list=sample_list,
            fmt_ids=fmt_ids,
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
            frame.write_parquet(output_file)
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
    sample_name: str,
) -> str:
    """Return TSV (string) produced by `bcftools | awk` for a region."""
    bcftools_cmd = [bcftools, "query", "-s", sample_name, "-f", fmt_str, vcf_path]
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

def _get_sample_list(vcf: str, bcftools: str) -> list[str]:
    """
    Get the list of samples in the VCF file.
    
    Parameters
    ----------
    vcf : str
        Path to input VCF file
    bcftools : str
        Path to bcftools executable
    
    Returns
    -------
    list[str]
        List of samples in the VCF
    """
    cmd = [bcftools, "query", "-l", vcf]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        samples = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        return samples
    except subprocess.CalledProcessError as e:
        log.error(f"Could not determine sample list from {cmd}: {e}.")
        raise RuntimeError(f"Could not determine sample list from {cmd}: {e}.")
    
def _stream_region_to_polars(
    region: str,
    vcf_path: str,
    fmt_str: str,
    job_cfg: VCFJobConfig,
) -> pl.DataFrame:
    """
    Run bcftools→awk and return a typed Polars DataFrame for *region*.
    If VCF has multiple samples, create a separate dataframe for each sample and join them on CHROM and POS.
    The final dataframe will have the same columns as the input VCF, but with the sample name prefixed to the FORMAT columns to avoid column name conflicts in the join.

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
    frames: dict[str, pl.DataFrame] = {}

    for sample in job_cfg.sample_list:
        single_sample_tsv = _bcftools_awk_stdout(
            region=region,
            vcf_path=vcf_path,
            fmt_str=fmt_str,
            bcftools=job_cfg.bcftools_path,
            awk_script=job_cfg.awk_script,
            list_indices=job_cfg.list_indices,
            sample_name=sample
        )
        if not single_sample_tsv.strip():
            continue
        frame = _frame_from_tsv(single_sample_tsv, cols=job_cfg.columns, schema=job_cfg.schema)
        if not frame.is_empty():
            frame = _cast_column_data_types(frame, job_cfg)
            frames[sample] = frame
    if not frames:
        final_frame = pl.DataFrame()
    elif len(frames) > 1:
        # Multi-sample case
        # Add sample name prefix to the FORMAT fields to avoid column name conflicts in the join
        fmt_ids = job_cfg.fmt_ids

        for sample, frame in frames.items():
            frames[sample] = frame.rename({col: f"{sample}_{col}" for col in frame.columns if col in fmt_ids})
            
        frame_list = list(frames.values())
        final_frame = frame_list[0]

        # Use CHROM and POS as join keys (genomic coordinates)
        join_keys = [CHROM, POS]
        for frame in frame_list[1:]:
            final_frame = final_frame.join(frame, on=join_keys, how="outer", coalesce=True)
        
    else:
        # Single sample case
        final_frame = list(frames.values())[0]
    return final_frame


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


if __name__ == "__main__":
    main()
