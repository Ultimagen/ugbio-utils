from __future__ import annotations

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


@pytest.mark.skipif(bcftools_missing, reason="bcftools not available")
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
    # example4.vcf has 4 variants, last one has 2 reads -> 5 exploded rows
    assert featuremap_dataframe.shape == (5, len(featuremap_dataframe.columns))
    # sanity-check a few expected columns and types
    assert {"RN", "RL", "X_PREV1"}.issubset(featuremap_dataframe.columns)
    assert featuremap_dataframe["RN"].dtype == pl.Utf8
    assert featuremap_dataframe["POS"].dtype == pl.Int64


@pytest.mark.skipif(bcftools_missing, reason="bcftools not available")
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
    assert cats == {"A", "C", "G", "T"}


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
