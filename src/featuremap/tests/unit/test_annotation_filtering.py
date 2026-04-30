import polars as pl
from ugbio_featuremap.filter_dataframe import _create_filter_columns, _create_final_filter_column, _mask_for_rule


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1", "chr2", "chr2"],
            "POS": [100, 200, 300, 400, 500],
            "PCAWG": ["rs1", None, "rs3", None, "rs5"],
            "EXCLUDE_TRAINING": [None, "ex1", None, "ex2", None],
            "INCLUDE_INFERENCE": ["inc1", None, None, None, "inc5"],
        }
    )


def _apply_filters(frame: pl.DataFrame, filters: list[dict]) -> pl.DataFrame:
    lazy = frame.lazy()
    lazy, filter_cols = _create_filter_columns(lazy, filters)
    lazy = _create_final_filter_column(lazy, filter_cols, None)
    result = lazy.collect()
    return result.filter(pl.col("__filter_final")).select(pl.exclude("^__filter_.*$"))


def test_is_null_excludes_annotated():
    filters = [{"name": "not_pcawg", "type": "region", "field": "PCAWG", "op": "is_null"}]
    result = _apply_filters(_sample_frame(), filters)
    assert result.height == 2
    assert result["POS"].to_list() == [200, 400]


def test_is_not_null_includes_annotated():
    filters = [{"name": "has_include", "type": "region", "field": "INCLUDE_INFERENCE", "op": "is_not_null"}]
    result = _apply_filters(_sample_frame(), filters)
    assert result.height == 2
    assert result["POS"].to_list() == [100, 500]


def test_is_null_multiple_fields_and_logic():
    filters = [
        {"name": "not_pcawg", "type": "region", "field": "PCAWG", "op": "is_null"},
        {"name": "not_excluded", "type": "region", "field": "EXCLUDE_TRAINING", "op": "is_null"},
    ]
    result = _apply_filters(_sample_frame(), filters)
    assert result.height == 0


def test_any_not_null_or_logic():
    filters = [
        {"name": "in_inference", "type": "region", "op": "any_not_null", "fields": ["INCLUDE_INFERENCE", "PCAWG"]}
    ]
    result = _apply_filters(_sample_frame(), filters)
    assert result.height == 3
    assert result["POS"].to_list() == [100, 300, 500]


def test_any_not_null_single_field():
    filters = [{"name": "has_pcawg", "type": "region", "op": "any_not_null", "fields": ["PCAWG"]}]
    result = _apply_filters(_sample_frame(), filters)
    assert result.height == 3
    assert result["POS"].to_list() == [100, 300, 500]


def test_is_null_combined_with_any_not_null():
    filters = [
        {"name": "not_excluded", "type": "region", "field": "EXCLUDE_TRAINING", "op": "is_null"},
        {"name": "in_inference", "type": "region", "op": "any_not_null", "fields": ["INCLUDE_INFERENCE", "PCAWG"]},
    ]
    result = _apply_filters(_sample_frame(), filters)
    assert result.height == 3
    assert result["POS"].to_list() == [100, 300, 500]


def test_no_filters_returns_all():
    frame = _sample_frame()
    lazy = frame.lazy()
    lazy, filter_cols = _create_filter_columns(lazy, [])
    assert filter_cols == []
    assert lazy.collect().height == 5


def test_is_null_all_null_column():
    frame = pl.DataFrame({"POS": [1, 2], "FIELD": [None, None]})
    filters = [{"name": "test", "type": "region", "field": "FIELD", "op": "is_null"}]
    result = _apply_filters(frame, filters)
    assert result.height == 2


def test_is_not_null_all_null_column():
    frame = pl.DataFrame({"POS": [1, 2], "FIELD": [None, None]})
    filters = [{"name": "test", "type": "region", "field": "FIELD", "op": "is_not_null"}]
    result = _apply_filters(frame, filters)
    assert result.height == 0


def test_mask_for_rule_is_null():
    expr = _mask_for_rule({"field": "X", "op": "is_null", "type": "region"})
    frame = pl.DataFrame({"X": ["a", None, "b"]})
    result = frame.select(expr.alias("mask"))["mask"].to_list()
    assert result == [False, True, False]


def test_mask_for_rule_any_not_null():
    expr = _mask_for_rule({"op": "any_not_null", "type": "region", "fields": ["A", "B"]})
    frame = pl.DataFrame({"A": [None, "x", None], "B": [None, None, "y"]})
    result = frame.select(expr.alias("mask"))["mask"].to_list()
    assert result == [False, True, True]
