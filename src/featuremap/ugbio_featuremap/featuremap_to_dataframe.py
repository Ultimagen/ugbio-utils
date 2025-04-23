#!/usr/bin/env python3
# … header unchanged …

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
    m = re.search(r"\{([^}]*)}", desc)
    return m.group(1).split(",") if m else None


def header_meta(vcf: str, bcftools_path: str) -> tuple[dict, dict]:
    txt = subprocess.check_output([bcftools_path, "view", "-h", vcf], text=True)
    info, fmt = {}, {}
    for m in INFO_RE.finditer(txt):
        k, n, t, d = m.groups()
        info[k] = {"num": n, "type": t, "cat": _enum(d)}
    for m in FORMAT_RE.finditer(txt):
        k, n, t, d = m.groups()
        fmt[k] = {"num": n, "type": t, "cat": _enum(d)}
    return info, fmt


# ───────────────── enum helpers (Polars ≥1.27) ──────────────────────────────
def _stub_row(df: pl.DataFrame, col: str, value) -> pl.DataFrame:
    """Return a 1-row DataFrame matching `df`'s schema, with `value` in `col`."""
    data = {c: [None] for c in df.columns}
    data[col] = [value]
    return pl.DataFrame(data).cast(df.schema, strict=False)


def _ensure_scalar_categories(df: pl.DataFrame, col: str, cats: list[str]) -> pl.DataFrame:
    """
    Append a stub block that contains *every* category value so Polars
    registers them, then trim back to the original row count.
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


def _ensure_list_categories(df: pl.DataFrame, col: str, cats: list[str]) -> pl.DataFrame:
    stub = _stub_row(df, col, cats)  # value is the full list
    return pl.concat([df, stub], how="vertical").head(df.height)


# ───────────────── casting helpers ──────────────────────────────────────────
def cast_scalar(featuremap_dataframe: pl.DataFrame, col: str, meta: dict) -> pl.DataFrame:
    utf_null = pl.when(pl.col(col).cast(pl.Utf8).is_in(["", "."])).then(None).otherwise(pl.col(col).cast(pl.Utf8))

    if meta["cat"]:
        featuremap_dataframe = featuremap_dataframe.with_columns(utf_null.cast(pl.Categorical).alias(col))
        return _ensure_scalar_categories(featuremap_dataframe, col, meta["cat"])
    if meta["type"] in _POLARS_DTYPE:
        return featuremap_dataframe.with_columns(utf_null.cast(_POLARS_DTYPE[meta["type"]], strict=False).alias(col))

    return featuremap_dataframe.with_columns(utf_null.alias(col))


def cast_list(featuremap_dataframe: pl.DataFrame, col: str, meta: dict) -> pl.DataFrame:
    """Split comma-lists, replace '' / '.' with nulls, cast numeric lists."""
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


def resolve_bcftools_command() -> str:
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
    drop_info = drop_info or set()
    drop_format = drop_format or {"GT"}

    bcftools = resolve_bcftools_command()
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

    for tag in info_ids:
        featuremap_dataframe = cast_scalar(featuremap_dataframe, tag, info_meta[tag])
    for tag in format_ids:
        featuremap_dataframe = cast_list(featuremap_dataframe, tag, fmt_meta[tag])

    featuremap_dataframe = featuremap_dataframe.explode(format_ids)
    featuremap_dataframe.write_parquet(out)
    print(f"✅  {out}: {featuremap_dataframe.shape[0]:,} rows × {featuremap_dataframe.shape[1]} cols")


# ─────────────────────────── CLI wrapper ─────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Featuremap VCF to DataFrame conversion saved in a Parquet format")
    ap.add_argument("--vcf", required=True, help="input VCF/BCF (bgz ok)")
    ap.add_argument("--out-parquet", required=True, help="output .parquet")
    ap.add_argument("--drop-info", default="", help="comma-separated INFO tags to drop")
    ap.add_argument("--drop-format", default="", help="comma-separated FORMAT tags to drop")
    args = ap.parse_args()

    with pl.StringCache():
        vcf_to_parquet(
            args.vcf,
            args.out_parquet,
            set(filter(None, args.drop_info.split(","))),
            set(filter(None, args.drop_format.split(","))),
        )


if __name__ == "__main__":
    main()
