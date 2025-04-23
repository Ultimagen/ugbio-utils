#!/usr/bin/env python3
# Copyright 2025 Ultima Genomics Inc.
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
#    Converts featuremap VCF file to dataframe
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
CHROM = "CHROM"
POS = "POS"
REF = "REF"
ALT = "ALT"


def _enum(desc: str) -> list[str] | None:
    m = re.search(r"\{([^}]*)}", desc)
    return m.group(1).split(",") if m else None


def header_meta(vcf: str, bcftools_path: str) -> tuple[dict[str, dict], dict[str, dict]]:
    txt = subprocess.check_output([bcftools_path, "view", "-h", vcf], text=True)
    info, fmt = {}, {}
    for m in INFO_RE.finditer(txt):
        k, n, t, d = m.groups()
        info[k] = {"num": n, "type": t, "cat": _enum(d)}
    for m in FORMAT_RE.finditer(txt):
        k, n, t, d = m.groups()
        fmt[k] = {"num": n, "type": t, "cat": _enum(d)}
    return info, fmt


# ───────────────── casting helpers ──────────────────────────────────────────
def cast_scalar(featuremap_dataframe: pl.DataFrame, col: str, meta: dict) -> pl.DataFrame:
    # 1️⃣  force Utf8; Polars materialises this as a new column expression
    utf = pl.col(col).cast(pl.Utf8)

    # 2️⃣  replace "" or "." with null on the Utf8 view
    utf_null = pl.when(utf.is_in(["", "."])).then(None).otherwise(utf)

    # 3️⃣  final cast to declared dtype / categorical
    if meta["cat"]:
        final = (
            utf_null.cast(pl.Categorical).cat.set_categories(meta["cat"], ordered=True)  # ← replace
        )
    elif meta["type"] in _POLARS_DTYPE:
        final = utf_null.cast(_POLARS_DTYPE[meta["type"]], strict=False)
    else:
        final = utf_null

    return featuremap_dataframe.with_columns(final.alias(col))


def cast_list(featuremap_dataframe: pl.DataFrame, col: str, meta: dict):
    elem_dt = _POLARS_DTYPE.get(meta["type"], pl.Utf8)

    featuremap_dataframe = featuremap_dataframe.with_columns(
        pl.col(col)
        .str.split(",")  # list<str>
        .list.eval(
            # cast each element to Utf8, then null-replace
            pl.when(pl.element().cast(pl.Utf8).is_in(["", "."]))
            .then(None)
            .otherwise(pl.element().cast(pl.Utf8))
            # final element cast
            .cast(pl.Categorical)
            .cat.set_categories(meta["cat"], ordered=True)  # ← replace
            if meta["cat"]
            else pl.when(pl.element().cast(pl.Utf8).is_in(["", "."]))
            .then(None)
            .otherwise(pl.element().cast(pl.Utf8))
            .cast(elem_dt, strict=False)
        )
        .alias(col)
    )
    return featuremap_dataframe


def resolve_bcftools_command() -> str:
    # resolve absolute bcftools path once
    bcftools_path = shutil.which("bcftools")
    if bcftools_path is None:
        raise RuntimeError("bcftools not found in $PATH")
    return bcftools_path  # prepend to every call


# ───────────────── core converter ───────────────────────────────────────────
def vcf_to_parquet(vcf: str, out: str, drop_info: set[str] | None = None, drop_format: set[str] | None = None) -> None:
    drop_info = drop_info or set()
    drop_format = drop_format or {"GT"}

    bcftools_path = resolve_bcftools_command()
    info_meta, fmt_meta = header_meta(vcf, bcftools_path)
    info_ids = [k for k in info_meta if k not in drop_info]
    format_ids = [k for k in fmt_meta if k not in drop_format]

    bracket = "[" + "\t".join(f"%{t}" for t in format_ids) + "]"
    fmt_str = "\t".join(["%CHROM", "%POS", "%REF", "%ALT", *[f"%INFO/{t}" for t in info_ids], bracket]) + "\n"

    try:
        with tempfile.NamedTemporaryFile("w+b", delete=False) as tmp:
            subprocess.run([bcftools_path, "query", "-f", fmt_str, vcf], stdout=tmp, check=True)
            csv_path = tmp.name

        cols = [CHROM, POS, REF, ALT, *info_ids, *format_ids]
        featuremap_dataframe = pl.read_csv(
            csv_path, separator="\t", has_header=False, new_columns=cols, low_memory=True
        )
    finally:
        Path.unlink(csv_path, missing_ok=True)

    featuremap_dataframe = featuremap_dataframe.with_columns(pl.col("POS").cast(pl.Int64))

    for tag in info_ids:
        featuremap_dataframe = cast_scalar(featuremap_dataframe, tag, info_meta[tag])
    for tag in format_ids:
        featuremap_dataframe = cast_list(featuremap_dataframe, tag, fmt_meta[tag])

    featuremap_dataframe = featuremap_dataframe.explode(format_ids)
    featuremap_dataframe.write_parquet(out)
    print(f"✅  {out}: {featuremap_dataframe.shape[0]:,} rows × {featuremap_dataframe.shape[1]} cols")


# ─────────────────────────── CLI wrapper ─────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="VCF ➜ per-read Parquet")
    ap.add_argument("--vcf", required=True, help="input VCF/BCF (bgz ok)")
    ap.add_argument("--out-parquet", required=True, help="output .parquet")
    ap.add_argument("--drop-info", default="", help="comma-separated list of INFO tags to drop from the output")
    ap.add_argument("--drop-format", default="", help="comma-separated list of FORMAT tags to drop from the output")
    args = ap.parse_args()

    vcf_to_parquet(
        args.vcf,
        args.out_parquet,
        set(filter(None, args.drop_info.split(","))),
        set(filter(None, args.drop_format.split(","))),
    )


if __name__ == "__main__":
    main()
