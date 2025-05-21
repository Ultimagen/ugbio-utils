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
from pathlib import Path

import polars as pl
from polars.exceptions import ShapeError

log = logging.getLogger(__name__)

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
            null_values=["."],  # ← treat “.” as NA so numeric columns parse
        )
    finally:
        Path(path).unlink(missing_ok=True)


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


# ───────────────── core converter ────────────────────────────────────────────
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
    """
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

    # Merge JSON overrides ---------------------------------------------------
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
    featuremap_dataframe = _load_vcf_as_dataframe(bcftools, vcf, fmt_str, cols).with_columns(pl.col(POS).cast(pl.Int64))

    with pl.StringCache():
        featuremap_dataframe = _cast_all_columns(
            featuremap_dataframe, info_ids, scalar_fmt_ids, list_fmt_ids, info_meta, fmt_meta
        )
        featuremap_dataframe, list_fmt_ids = _explode_with_retry(featuremap_dataframe, list_fmt_ids)
        # after explode, list columns became scalars
        for tag in list_fmt_ids:
            featuremap_dataframe = _cast_scalar(featuremap_dataframe, tag, fmt_meta[tag])

    featuremap_dataframe.write_parquet(out)
    log.info("✅  %s: %,d rows × %d cols", out, *featuremap_dataframe.shape)


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

    vcf_to_parquet(
        vcf=args.vcf,
        out=args.out_parquet,
        drop_info=set(filter(None, args.drop_info.split(","))),
        drop_format=set(filter(None, args.drop_format.split(","))),
        categories_json=args.categories_json,
    )


if __name__ == "__main__":
    main()
