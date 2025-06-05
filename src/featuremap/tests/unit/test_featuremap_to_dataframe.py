from __future__ import annotations

import json
import logging
import shutil
import subprocess
import warnings  # NEW
from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap import featuremap_to_dataframe

bcftools_missing = shutil.which("bcftools") is None


# --- fixtures --------------------------------------------------------------
@pytest.fixture(
    params=[
        # "416119_L7402.raw.featuremap.vcf.gz",
        # "416119_L7402.random_sample.featuremap.vcf.gz",
        "416119_L7402.random_sample.featuremap.manually_cleaned.vcf"
    ]
)
def input_featuremap(request):
    """Return each sample VCF in turn."""
    return Path(__file__).parent.parent / "resources" / request.param


@pytest.fixture
def input_categorical_features():
    return Path(__file__).parent.parent / "resources" / "416119-L7402-Z0296-CATCTATCAGGCGAT.categorical_features.json"


def test_vcf_to_parquet_end_to_end(tmp_path: Path, input_featuremap: Path) -> None:
    """Full pipeline should yield the correct per-read row count and include key columns."""
    out_path = str(tmp_path / input_featuremap.name.replace(".vcf.gz", ".parquet"))
    out_path_2 = str(tmp_path / input_featuremap.name.replace(".2.vcf.gz", ".parquet"))

    # Capture warnings to ensure no "Dropping list columns with inconsistent length" warning is raised
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # run conversion (drop GT by default)
        featuremap_to_dataframe.vcf_to_parquet(
            vcf=str(input_featuremap),
            out=out_path,
            drop_info=set(),
            drop_format={"GT"},
        )
    # Assert the specific warning was NOT raised
    assert not any(
        "Dropping list columns with inconsistent length" in str(w.message) for w in caught
    ), "Unexpected warning: 'Dropping list columns with inconsistent length'"

    featuremap_dataframe = pl.read_parquet(out_path)
    featuremap_dataframe.write_parquet(out_path_2)

    # hard-coded expected row counts per sample
    expected_rows = {
        "416119_L7402.raw.featuremap.vcf.gz": 2664,
        "416119_L7402.random_sample.featuremap.vcf.gz": 619,
        "416119_L7402.random_sample.featuremap.manually_cleaned.vcf": 6577,
    }[input_featuremap.name]
    assert featuremap_dataframe.shape[0] == expected_rows

    # sanity-check a few expected columns and types
    assert {"RN", "RL", "X_PREV1"}.issubset(featuremap_dataframe.columns)
    assert featuremap_dataframe["RN"].dtype == pl.Utf8
    assert featuremap_dataframe["POS"].dtype == pl.Int64


def test_enum_column_is_categorical(tmp_path: Path, input_featuremap: Path) -> None:
    """
    Columns whose description lists {A,C,G,T} should be stored as categorical
    with exactly those four categories.
    """
    out_path = str(tmp_path / input_featuremap.name.replace(".vcf.gz", ".parquet"))
    featuremap_to_dataframe.vcf_to_parquet(
        vcf=str(input_featuremap),
        out=out_path,
        drop_info=set(),
        drop_format={"GT"},
    )

    featuremap_dataframe = pl.read_parquet(out_path)
    print(featuremap_dataframe.schema)
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


# ------------- tiny unit tests per helper ---------------------------------
def test_enum():
    assert featuremap_to_dataframe._enum("foo {A,B}") == ["A", "B"]
    assert featuremap_to_dataframe._enum("no enum") is None


def test_header_meta(input_featuremap):
    bcftools = featuremap_to_dataframe._resolve_bcftools_command()
    info, fmt = featuremap_to_dataframe.header_meta(str(input_featuremap), bcftools)
    assert "X_PREV1" in info
    assert "RN" in fmt


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
        "VAF": pl.Float64,  # float
        "RN": pl.Utf8,  # exploded list -> string
    }
    for col, dt in expected.items():
        assert featuremap_dataframe[col].dtype == dt, f"{col} dtype {featuremap_dataframe[col].dtype} ≠ {dt}"
