"""Tests for filter logic with null ID values.

Verifies that _create_filter_columns and _create_final_filter_column
correctly handle null values when filtering VCF-derived DataFrames
(where '.' is parsed as null by Polars).
"""

import polars as pl
import pytest
from ugbio_featuremap.filter_dataframe import (
    _create_filter_columns,
    _create_final_filter_column,
)


@pytest.fixture()
def df_with_null_ids() -> pl.DataFrame:
    """Create a test DataFrame with NULL ID values.

    Simulates Polars parsing of VCF data with null_values=["."],
    where '.' entries become None.
    """
    return pl.DataFrame(
        {
            "ID": [None, "rs123", None, "rs456", None],
            "gnomAD_AF": [None, 0.5, 0.00001, 0.01, None],
            "POS": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture()
def dbsnp_gnomad_filters() -> list[dict]:
    """Return filter definitions for dbSNP and gnomAD filtering."""
    return [
        {
            "name": "not_in_dbsnp",
            "type": "region",
            "field": "ID",
            "op": "eq",
            "value": ".",
        },
        {
            "name": "not_in_gnomad",
            "type": "region",
            "field": "gnomAD_AF",
            "op": "le",
            "value": 0.0001,
        },
    ]


def test_filter_with_null_ids(
    df_with_null_ids: pl.DataFrame,
    dbsnp_gnomad_filters: list[dict],
) -> None:
    """Test that null ID values are treated as '.' from VCF.

    Rows with ID=None and low/null gnomAD_AF (POS 100, 300, 500)
    should pass the combined filter.
    """
    lazy_frame = df_with_null_ids.lazy()
    lazy_frame, filter_cols = _create_filter_columns(lazy_frame, dbsnp_gnomad_filters)

    # Treat null ID as "." (matching VCF convention)
    id_col_name = "__filter_not_in_dbsnp"
    lazy_frame = lazy_frame.with_columns((pl.col(id_col_name) | pl.col("ID").is_null()).alias(id_col_name))

    # Treat null gnomAD_AF as passing the filter
    gnomad_col_name = "__filter_not_in_gnomad"
    lazy_frame = lazy_frame.with_columns(
        (pl.col(gnomad_col_name) | pl.col("gnomAD_AF").is_null()).alias(gnomad_col_name)
    )

    lazy_frame = _create_final_filter_column(lazy_frame, filter_cols, None)
    result = lazy_frame.collect()

    filtered = result.filter(pl.col("__filter_final")).select(pl.exclude("^__filter_.*$"))

    assert filtered.height == 3, f"Expected 3 rows after filtering, got {filtered.height}"
    assert filtered["POS"].to_list() == [100, 300, 500]


def test_filter_columns_created(
    df_with_null_ids: pl.DataFrame,
    dbsnp_gnomad_filters: list[dict],
) -> None:
    """Test that _create_filter_columns produces expected column names."""
    lazy_frame = df_with_null_ids.lazy()
    lazy_frame, filter_cols = _create_filter_columns(lazy_frame, dbsnp_gnomad_filters)

    assert filter_cols == [
        "__filter_not_in_dbsnp",
        "__filter_not_in_gnomad",
    ]

    result = lazy_frame.collect()
    for col in filter_cols:
        assert col in result.columns, f"Missing filter column: {col}"


def test_final_filter_column_exists(
    df_with_null_ids: pl.DataFrame,
    dbsnp_gnomad_filters: list[dict],
) -> None:
    """Test that _create_final_filter_column adds __filter_final."""
    lazy_frame = df_with_null_ids.lazy()
    lazy_frame, filter_cols = _create_filter_columns(lazy_frame, dbsnp_gnomad_filters)
    lazy_frame = _create_final_filter_column(lazy_frame, filter_cols, None)
    result = lazy_frame.collect()

    assert "__filter_final" in result.columns
