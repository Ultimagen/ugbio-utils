from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap import featuremap_to_dataframe

bcftools_missing = shutil.which("bcftools") is None


@pytest.fixture
def input_featuremap():
    return Path(__file__).parent.parent / "resources" / "416119-L7402-Z0296-CATCTATCAGGCGAT.few_examples.vcf.gz"


@pytest.fixture
def input_categorical_features():
    return Path(__file__).parent.parent / "resources" / "416119-L7402-Z0296-CATCTATCAGGCGAT.categorical_features.json"


def test_vcf_to_parquet_end_to_end(tmp_path: Path, input_featuremap: Path) -> None:
    """Full pipeline should yield 5 per-read rows and include key columns."""
    out_path = str(tmp_path / "416119-L7402-Z0296-CATCTATCAGGCGAT.few_examples.parquet")

    # run conversion (drop GT by default)
    featuremap_to_dataframe.vcf_to_parquet(
        vcf=str(input_featuremap),
        out=out_path,
        drop_info=set(),
        drop_format={"GT"},
    )

    featuremap_dataframe = pl.read_parquet(out_path)
    # vcf has 5 variants, one has 2 reads -> 6 exploded rows
    assert featuremap_dataframe.shape == (6, len(featuremap_dataframe.columns))
    # sanity-check a few expected columns and types
    assert {"RN", "RL", "X_PREV1"}.issubset(featuremap_dataframe.columns)
    assert featuremap_dataframe["RN"].dtype == pl.Utf8
    assert featuremap_dataframe["POS"].dtype == pl.Int64


def test_enum_column_is_categorical(tmp_path: Path, input_featuremap: Path) -> None:
    """
    Columns whose description lists {A,C,G,T} should be stored as categorical
    with exactly those four categories.
    """
    out_path = str(tmp_path / "416119-L7402-Z0296-CATCTATCAGGCGAT.few_examples.parquet")
    featuremap_to_dataframe.vcf_to_parquet(
        vcf=str(input_featuremap),
        out=out_path,
        drop_info=set(),
        drop_format={"GT"},
    )

    featuremap_dataframe = pl.read_parquet(out_path)
    col = featuremap_dataframe["X_PREV1"]
    assert col.dtype == pl.Categorical

    cats = set(col.cat.get_categories())
    assert cats == {"", "A", "C", "G", "T"}


def test_roundtrip(tmp_path: Path, input_featuremap: Path):
    """Parquet row count == total RN elements in source VCF."""
    out = tmp_path / "out.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(input_featuremap), str(out))

    featuremap_dataframe = pl.read_parquet(out)

    # count RN elements straight from bcftools (no header confusion)
    rn_bytes = subprocess.check_output(
        ["bcftools", "query", "-f", "[%RN\n]", str(input_featuremap)],
        text=False,
    )
    rn_len = sum(len(line.strip().split(b",")) for line in rn_bytes.splitlines())

    assert featuremap_dataframe.height == rn_len


# ------------- categorical-override test ----------------------------------
def test_json_override(tmp_path: Path, input_featuremap: Path, input_categorical_features: Path):
    out = tmp_path / "override.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(input_featuremap), str(out), categories_json=str(input_categorical_features)
    )
    featuremap_dataframe = pl.read_parquet(out)
    for tag, cats in json.load(open(input_categorical_features))["categorical_features"].items():
        if tag == "REF" or tag == "ALT":
            # REF/ALT are reserved, so we don't override them
            assert set(featuremap_dataframe[tag].cat.get_categories()) == {"", "A", "C", "G", "T"}
        else:
            assert set(featuremap_dataframe[tag].cat.get_categories()) == set([""] + cats)


# ------------- REF/ALT default categories ---------------------------------
def test_ref_alt_defaults(tmp_path: Path, input_featuremap: Path):
    out = tmp_path / "def.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(input_featuremap), str(out))
    featuremap_dataframe = pl.read_parquet(out)
    for tag in ("REF", "ALT"):
        assert set(featuremap_dataframe[tag].cat.get_categories()) == {"", "A", "C", "G", "T"}


# ------------- missing-value line encoded correctly -----------------------
def test_missing_values(tmp_path: Path, input_featuremap: Path):
    vcf = tmp_path / "with_missing.vcf"
    # copy header + records + extra missing line
    header = subprocess.check_output(["bcftools", "view", "-h", str(input_featuremap)], text=True)
    with vcf.open("w") as fh:
        fh.write(header)
        subprocess.run(["bcftools", "view", "-H", str(input_featuremap)], stdout=fh, text=True, check=True)
        fh.write("chr1\t2000000\t.\tA\tG\t0\t.\tX_PREV1=.;X_VAF=.;X_READ_COUNT=.;X_NEXT1=.;\tBCSQ:RN\t.:.\n")

    out = tmp_path / "miss.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(vcf), str(out))
    featuremap_dataframe = pl.read_parquet(out)
    # last INFO field null?
    assert featuremap_dataframe.filter(pl.col("POS") == 2000000)["X_PREV1"].null_count() == 1
    # FORMAT field XN exists and is null for that read
    assert featuremap_dataframe.filter(pl.col("POS") == 2000000)["BCSQ"].null_count() == 1


# ------------- tiny unit tests per helper ---------------------------------
def test_enum():
    assert featuremap_to_dataframe._enum("foo {A,B}") == ["A", "B"]
    assert featuremap_to_dataframe._enum("no enum") is None


def test_header_meta(input_featuremap):
    bcftools = featuremap_to_dataframe._resolve_bcftools_command()
    info, fmt = featuremap_to_dataframe.header_meta(str(input_featuremap), bcftools)
    assert "X_VAF" in info and "RN" in fmt


def test_ensure_scalar_categories():
    featuremap_dataframe = pl.DataFrame({"x": pl.Series(["A"], dtype=pl.Categorical)})
    featuremap_dataframe_2 = featuremap_to_dataframe._ensure_scalar_categories(featuremap_dataframe, "x", ["A", "B"])
    assert set(featuremap_dataframe_2["x"].cat.get_categories()) == {"", "A", "B"}


def test_json_override_and_reserved_warning(tmp_path, input_featuremap: Path, input_categorical_features: Path, caplog):
    out = tmp_path / "out.parquet"

    caplog.set_level(logging.WARNING)
    featuremap_to_dataframe.vcf_to_parquet(
        str(input_featuremap), str(out), categories_json=str(input_categorical_features)
    )

    # Reserved override ignored?
    assert "Ignoring JSON category override for reserved column REF" in caplog.text

    featuremap_dataframe = pl.read_parquet(out)
    cats = json.load(open(input_categorical_features))["categorical_features"]

    # st / et overridden
    assert set(featuremap_dataframe["tm"].cat.get_categories()) == set([""] + cats["tm"])
    # REF remains default
    assert set(featuremap_dataframe["REF"].cat.get_categories()) == {"", "A", "C", "G", "T"}


def test_selected_dtypes(tmp_path: Path, input_featuremap: Path):
    out = tmp_path / "full.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(input_featuremap), str(out))

    featuremap_dataframe = pl.read_parquet(out)

    expected = {
        "CHROM": pl.Utf8,  # string
        "POS": pl.Int64,  # integer
        "REF": pl.Categorical,  # categorical
        "ALT": pl.Categorical,  # categorical
        "X_VAF": pl.Float64,  # float
        "RN": pl.Utf8,  # exploded list -> string
    }
    for col, dt in expected.items():
        assert featuremap_dataframe[col].dtype == dt, f"{col} dtype {featuremap_dataframe[col].dtype} ≠ {dt}"
