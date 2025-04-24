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
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

# ───────────────── header helpers ────────────────────────────────────────────
INFO_RE = re.compile(r'##INFO=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]*)')
FORMAT_RE = re.compile(r'##FORMAT=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]*)')

_POLARS_DTYPE = {"Integer": pl.Int64, "Float": pl.Float64, "Flag": pl.Boolean}
CHROM, POS, REF, ALT = "CHROM", "POS", "REF", "ALT"


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
def _ensure_scalar_categories(df: pl.DataFrame, col: str, cats: list[str]) -> pl.DataFrame:
    """
    Ensure a categorical *scalar* column registers every category value.

    Polars ≥ 1.27 no longer provides ``set_categories``/``set_order``.
    Instead we append a one-row *stub block* that contains each category
    exactly once, concatenate it to the original frame so Polars merges
    the dictionaries, then trim the extra row(s) away.

    Parameters
    ----------
    df :
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
        A DataFrame of identical shape to *df* (same rows/columns) but
        whose *col* now recognises **all** values in *cats*.
    """
    # build a DataFrame that matches the full schema
    rows = len(cats)
    stub_dict = {c: [None] * rows for c in df.columns}
    stub_dict[col] = cats  # each category once

    stub = (
        pl.DataFrame(stub_dict)
        .cast(df.schema, strict=False)  # keep dtypes identical
        .with_columns(pl.col(col).cast(pl.Categorical))
    )

    return pl.concat([df, stub], how="vertical").head(df.height)


# ───────────────── casting helpers ──────────────────────────────────────────
def _cast_scalar(featuremap_dataframe: pl.DataFrame, col: str, meta: dict) -> pl.DataFrame:
    """
    Cast a scalar INFO column.

    Parameters
    ----------
    df
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
    # split to list<str>
    featuremap_dataframe = featuremap_dataframe.with_columns(pl.col(col).str.split(",").alias(col))

    # null-replace on each element
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


# ───────────────── core converter ────────────────────────────────────────────
def vcf_to_parquet(
    vcf: str,
    out: str,
    drop_info: set[str] | None = None,
    drop_format: set[str] | None = None,
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
    """
    drop_info = drop_info or set()
    drop_format = drop_format or {"GT"}

    bcftools = _resolve_bcftools_command()
    info_meta, fmt_meta = header_meta(vcf, bcftools)

    info_ids = [k for k in info_meta if k not in drop_info]
    format_ids = [k for k in fmt_meta if k not in drop_format]

    bracket = "[" + "\t".join(f"%{t}" for t in format_ids) + "]"
    fmt_str = "\t".join(["%CHROM", "%POS", "%REF", "%ALT", *[f"%INFO/{t}" for t in info_ids], bracket]) + "\n"

    csv_path: str
    try:
        with tempfile.NamedTemporaryFile("w+b", delete=False) as tmp:
            subprocess.run([bcftools, "query", "-f", fmt_str, vcf], stdout=tmp, check=True)
            csv_path = tmp.name

        cols = [CHROM, POS, REF, ALT, *info_ids, *format_ids]
        featuremap_dataframe = pl.read_csv(
            csv_path, separator="\t", has_header=False, new_columns=cols, low_memory=True
        )
    finally:
        Path(csv_path).unlink(missing_ok=True)

    featuremap_dataframe = featuremap_dataframe.with_columns(pl.col(POS).cast(pl.Int64))

    with pl.StringCache():
        # Without this cache, Polars keeps a private dictionary per categorical column per DataFrame → a wide pipeline
        # can have dozens of duplicated lookup tables.
        # With the cache all those columns share a single mapping table, so large workflows typically use less RAM.
        for tag in info_ids:
            featuremap_dataframe = _cast_scalar(featuremap_dataframe, tag, info_meta[tag])
        for tag in format_ids:
            featuremap_dataframe = _cast_list(featuremap_dataframe, tag, fmt_meta[tag])

    featuremap_dataframe = featuremap_dataframe.explode(format_ids)
    featuremap_dataframe.write_parquet(out)
    print(f"✅  {out}: {featuremap_dataframe.shape[0]:,} rows × {featuremap_dataframe.shape[1]} cols")


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
    args = ap.parse_args()

    vcf_to_parquet(
        args.vcf,
        args.out_parquet,
        set(filter(None, args.drop_info.split(","))),
        set(filter(None, args.drop_format.split(","))),
    )


if __name__ == "__main__":
    main()
