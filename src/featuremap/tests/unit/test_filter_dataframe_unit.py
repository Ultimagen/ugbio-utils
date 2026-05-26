"""Unit tests for filter_dataframe module - pure logic tests without file I/O.

Covers:
- _get_filter_name: name derivation from rule structures (regression for U5)
- _mask_for_rule: all operators including is_null, is_not_null, any_not_null
- _create_filter_columns: filter pipeline
- _create_downsample_column: head and random downsampling
- _create_final_filter_column: combining filters with/without downsample
- _calculate_statistics: with any_not_null rules (regression for U2)
- validate_filter_config: config validation
- _parse_value_based_on_operation: value parsing
- _merge_config_and_cli: config merging
- filter_parquet: end-to-end (with tmp parquet files)
- _validate_stats_dict / read_filtering_stats_json: stats validation
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap.filter_dataframe import (
    COL_FILTER_DOWNSAMPLE,
    COL_FILTER_FINAL,
    COL_PREFIX_FILTER,
    KEY_DOWNSAMPLE,
    KEY_FIELD,
    KEY_FIELDS,
    KEY_FILTERS,
    KEY_METHOD,
    KEY_NAME,
    KEY_OP,
    KEY_SEED,
    KEY_SIZE,
    KEY_TYPE,
    KEY_VALUE,
    KEY_VALUE_FIELD,
    METHOD_HEAD,
    METHOD_RANDOM,
    OP_ANY_NOT_NULL,
    OP_IS_NOT_NULL,
    OP_IS_NULL,
    TYPE_LABEL,
    TYPE_QUALITY,
    TYPE_REGION,
    _calculate_statistics,
    _create_downsample_column,
    _create_filter_columns,
    _create_final_filter_column,
    _get_filter_name,
    _mask_for_rule,
    _merge_config_and_cli,
    _parse_cli_downsample,
    _parse_cli_filter,
    _parse_value_based_on_operation,
    _validate_stats_dict,
    filter_parquet,
    read_filtering_stats_json,
    validate_filter_config,
)

pl.enable_string_cache()


# ──────────────────────── _get_filter_name ──────────────────────


class TestGetFilterName:
    """Test name derivation from filter rule structures (regression for U5)."""

    def test_explicit_name_used(self):
        rule = {KEY_NAME: "my_filter", KEY_FIELD: "QUAL", KEY_OP: "gt", KEY_VALUE: 30}
        assert _get_filter_name(rule) == "my_filter"

    def test_fallback_to_field_op(self):
        rule = {KEY_FIELD: "QUAL", KEY_OP: "gt", KEY_VALUE: 30}
        assert _get_filter_name(rule) == "QUAL_gt"

    def test_any_not_null_uses_fields(self):
        """Regression U5: any_not_null rules use KEY_FIELDS, not KEY_FIELD."""
        rule = {KEY_OP: OP_ANY_NOT_NULL, KEY_FIELDS: ["gnomAD_AF", "PCAWG"], KEY_TYPE: TYPE_REGION}
        name = _get_filter_name(rule)
        # Should join fields with underscore
        assert name == "gnomAD_AF_PCAWG_any_not_null"

    def test_single_field_in_fields_list(self):
        rule = {KEY_OP: OP_IS_NULL, KEY_FIELDS: ["my_col"], KEY_TYPE: TYPE_REGION}
        name = _get_filter_name(rule)
        assert name == "my_col_is_null"

    def test_empty_name_string_falls_back(self):
        rule = {KEY_NAME: "", KEY_FIELD: "DP", KEY_OP: "ge"}
        # Empty string is falsy, should fall back
        assert _get_filter_name(rule) == "DP_ge"

    def test_none_name_falls_back(self):
        rule = {KEY_NAME: None, KEY_FIELD: "DP", KEY_OP: "le"}
        assert _get_filter_name(rule) == "DP_le"

    def test_no_field_no_fields(self):
        """Edge case: rule with no field/fields key."""
        rule = {KEY_OP: "gt", KEY_VALUE: 10}
        name = _get_filter_name(rule)
        # Should use empty join of KEY_FIELDS default (empty list)
        assert name == "_gt"


# ──────────────────────── _mask_for_rule ──────────────────────────────


class TestMaskForRule:
    """Test boolean expression generation for all operators."""

    @pytest.fixture
    def sample_lf(self):
        """Create a sample lazy frame for testing."""
        return pl.DataFrame(
            {
                "score": [10, 20, 30, None, 50],
                "category": ["A", "B", "A", None, "C"],
                "field_a": [1.0, None, 3.0, None, 5.0],
                "field_b": [None, 2.0, None, 4.0, 5.0],
                "ref": ["A", "C", "G", "T", "A"],
                "alt": ["A", "G", "G", "T", "C"],
            }
        ).lazy()

    def test_eq(self, sample_lf):
        rule = {KEY_FIELD: "score", KEY_OP: "eq", KEY_VALUE: 20}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, True, False, None, False]

    def test_ne(self, sample_lf):
        rule = {KEY_FIELD: "score", KEY_OP: "ne", KEY_VALUE: 20}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, False, True, None, True]

    def test_lt(self, sample_lf):
        rule = {KEY_FIELD: "score", KEY_OP: "lt", KEY_VALUE: 25}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, True, False, None, False]

    def test_le(self, sample_lf):
        rule = {KEY_FIELD: "score", KEY_OP: "le", KEY_VALUE: 30}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, True, True, None, False]

    def test_gt(self, sample_lf):
        rule = {KEY_FIELD: "score", KEY_OP: "gt", KEY_VALUE: 20}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, False, True, None, True]

    def test_ge(self, sample_lf):
        rule = {KEY_FIELD: "score", KEY_OP: "ge", KEY_VALUE: 20}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, True, True, None, True]

    def test_in(self, sample_lf):
        rule = {KEY_FIELD: "category", KEY_OP: "in", KEY_VALUE: ["A", "B"]}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, True, True, None, False]

    def test_not_in(self, sample_lf):
        rule = {KEY_FIELD: "category", KEY_OP: "not_in", KEY_VALUE: ["A"]}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, True, False, None, True]

    def test_is_null(self, sample_lf):
        rule = {KEY_FIELD: "score", KEY_OP: OP_IS_NULL}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, False, False, True, False]

    def test_is_not_null(self, sample_lf):
        rule = {KEY_FIELD: "score", KEY_OP: OP_IS_NOT_NULL}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, True, True, False, True]

    def test_any_not_null_single_field(self, sample_lf):
        rule = {KEY_FIELDS: ["field_a"], KEY_OP: OP_ANY_NOT_NULL}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        # field_a: [1.0, None, 3.0, None, 5.0]
        assert result.to_list() == [True, False, True, False, True]

    def test_any_not_null_multiple_fields(self, sample_lf):
        """Regression U2: any_not_null with multiple fields."""
        rule = {KEY_FIELDS: ["field_a", "field_b"], KEY_OP: OP_ANY_NOT_NULL}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        # field_a: [1.0, None, 3.0, None, 5.0]
        # field_b: [None, 2.0, None, 4.0, 5.0]
        # OR:      [True, True, True, True, True]
        assert result.to_list() == [True, True, True, True, True]

    def test_any_not_null_all_null(self):
        lf = pl.DataFrame(
            {
                "x": [None, None, None],
                "y": [None, None, None],
            }
        ).lazy()
        rule = {KEY_FIELDS: ["x", "y"], KEY_OP: OP_ANY_NOT_NULL}
        expr = _mask_for_rule(rule)
        result = lf.select(expr).collect().to_series()
        assert result.to_list() == [False, False, False]

    def test_value_field_comparison(self, sample_lf):
        """Compare two columns using value_field."""
        rule = {KEY_FIELD: "ref", KEY_OP: "eq", KEY_VALUE_FIELD: "alt"}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        # ref: [A, C, G, T, A] vs alt: [A, G, G, T, C]
        # eq:  [T, F, T, T, F]
        assert result.to_list() == [True, False, True, True, False]

    def test_value_field_ne(self, sample_lf):
        rule = {KEY_FIELD: "ref", KEY_OP: "ne", KEY_VALUE_FIELD: "alt"}
        expr = _mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, True, False, False, True]

    def test_unsupported_op(self):
        rule = {KEY_FIELD: "x", KEY_OP: "unsupported_op", KEY_VALUE: 1}
        with pytest.raises(ValueError, match="Unsupported op"):
            _mask_for_rule(rule)


# ──────────────────────── _create_filter_columns ──────────────────────


class TestCreateFilterColumns:
    """Test binary column creation for filter pipeline."""

    def test_basic_filter_columns(self):
        lf = pl.DataFrame(
            {
                "score": [10, 20, 30, 40, 50],
                "category": ["A", "B", "A", "B", "C"],
            }
        ).lazy()

        filters = [
            {KEY_NAME: "high_score", KEY_FIELD: "score", KEY_OP: "gt", KEY_VALUE: 25, KEY_TYPE: TYPE_QUALITY},
            {KEY_NAME: "cat_a", KEY_FIELD: "category", KEY_OP: "eq", KEY_VALUE: "A", KEY_TYPE: TYPE_LABEL},
        ]

        result_lf, filter_cols = _create_filter_columns(lf, filters)
        assert len(filter_cols) == 2
        assert filter_cols[0] == f"{COL_PREFIX_FILTER}high_score"
        assert filter_cols[1] == f"{COL_PREFIX_FILTER}cat_a"

        result_df = result_lf.collect()
        assert result_df[filter_cols[0]].to_list() == [False, False, True, True, True]
        assert result_df[filter_cols[1]].to_list() == [True, False, True, False, False]

    def test_empty_filters(self):
        lf = pl.DataFrame({"x": [1, 2, 3]}).lazy()
        result_lf, filter_cols = _create_filter_columns(lf, [])
        assert filter_cols == []
        assert result_lf.collect().shape == (3, 1)

    def test_any_not_null_filter_column(self):
        """Test that any_not_null creates proper filter column."""
        lf = pl.DataFrame(
            {
                "field_a": [1.0, None, 3.0],
                "field_b": [None, 2.0, None],
            }
        ).lazy()

        filters = [
            {KEY_FIELDS: ["field_a", "field_b"], KEY_OP: OP_ANY_NOT_NULL, KEY_TYPE: TYPE_REGION},
        ]

        result_lf, filter_cols = _create_filter_columns(lf, filters)
        result_df = result_lf.collect()
        col_name = filter_cols[0]
        # At least one non-null in each row
        assert result_df[col_name].to_list() == [True, True, True]


# ──────────────────────── _calculate_statistics ──────────────────────


class TestCalculateStatistics:
    """Test statistics calculation including any_not_null regression (U2)."""

    def test_basic_stats(self):
        lf = pl.DataFrame(
            {
                "score": [10, 20, 30, 40, 50],
            }
        ).lazy()

        filters = [
            {KEY_NAME: "high_score", KEY_FIELD: "score", KEY_OP: "gt", KEY_VALUE: 25, KEY_TYPE: TYPE_QUALITY},
        ]
        lf, filter_cols = _create_filter_columns(lf, filters)
        cfg = {KEY_FILTERS: filters}

        stats = _calculate_statistics(lf, filter_cols, None, filters, total_rows=5, cfg=cfg)

        assert stats[KEY_FILTERS][0][KEY_NAME] == "raw"
        assert stats[KEY_FILTERS][0]["rows"] == 5
        assert stats[KEY_FILTERS][1][KEY_NAME] == "high_score"
        assert stats[KEY_FILTERS][1]["rows"] == 3  # 30, 40, 50

        assert stats["single_effect"]["high_score"] == 3
        assert "combinations" in stats

    def test_stats_with_any_not_null(self):
        """Regression U2: any_not_null rules should produce correct statistics."""
        lf = pl.DataFrame(
            {
                "field_a": [1.0, None, 3.0, None, 5.0],
                "field_b": [None, 2.0, None, None, 5.0],
            }
        ).lazy()

        filters = [
            {KEY_FIELDS: ["field_a", "field_b"], KEY_OP: OP_ANY_NOT_NULL, KEY_TYPE: TYPE_REGION},
        ]
        lf, filter_cols = _create_filter_columns(lf, filters)
        cfg = {KEY_FILTERS: filters}

        stats = _calculate_statistics(lf, filter_cols, None, filters, total_rows=5, cfg=cfg)

        # any_not_null: rows where at least one field is not null
        # field_a: [1,None,3,None,5], field_b: [None,2,None,None,5]
        # OR: [True, True, True, False, True] => 4 pass
        filter_name = _get_filter_name(filters[0])
        assert stats[KEY_FILTERS][1]["rows"] == 4
        assert stats["single_effect"][filter_name] == 4

    def test_stats_with_downsample(self):
        lf = pl.DataFrame(
            {
                "score": list(range(100)),
            }
        ).lazy()

        filters = [
            {KEY_NAME: "pass_all", KEY_FIELD: "score", KEY_OP: "ge", KEY_VALUE: 0, KEY_TYPE: TYPE_QUALITY},
        ]
        lf, filter_cols = _create_filter_columns(lf, filters)
        cfg = {KEY_FILTERS: filters, KEY_DOWNSAMPLE: {KEY_SIZE: 50, KEY_METHOD: METHOD_HEAD}}

        stats = _calculate_statistics(lf, filter_cols, None, filters, total_rows=100, cfg=cfg)

        # Downsample entry should be appended
        assert stats[KEY_FILTERS][-1][KEY_NAME] == "downsample"
        assert stats[KEY_FILTERS][-1]["rows"] == 50

    def test_combination_patterns(self):
        """Combination stats should have 2^n patterns."""
        lf = pl.DataFrame(
            {
                "a": [True, False, True, False],
                "b": [True, True, False, False],
            }
        ).lazy()

        filters = [
            {KEY_NAME: "filt_a", KEY_FIELD: "a", KEY_OP: "eq", KEY_VALUE: True, KEY_TYPE: TYPE_QUALITY},
            {KEY_NAME: "filt_b", KEY_FIELD: "b", KEY_OP: "eq", KEY_VALUE: True, KEY_TYPE: TYPE_QUALITY},
        ]
        lf, filter_cols = _create_filter_columns(lf, filters)
        cfg = {KEY_FILTERS: filters}

        stats = _calculate_statistics(lf, filter_cols, None, filters, total_rows=4, cfg=cfg)

        combos = stats["combinations"]
        # 2 filters => 4 possible patterns (00, 01, 10, 11)
        assert len(combos) == 4
        assert combos["11"] == 1  # row 0: both True
        assert combos["10"] == 1  # row 2: a=True, b=False
        assert combos["01"] == 1  # row 1: a=False, b=True
        assert combos["00"] == 1  # row 3: both False


# ──────────────────────── validate_filter_config ──────────────────────


class TestValidateFilterConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        cfg = {
            KEY_FILTERS: [
                {KEY_FIELD: "QUAL", KEY_OP: "gt", KEY_VALUE: 30, KEY_TYPE: TYPE_QUALITY},
            ]
        }
        validate_filter_config(cfg)  # Should not raise

    def test_valid_any_not_null(self):
        cfg = {
            KEY_FILTERS: [
                {KEY_FIELDS: ["gnomAD_AF", "PCAWG"], KEY_OP: OP_ANY_NOT_NULL, KEY_TYPE: TYPE_REGION},
            ]
        }
        validate_filter_config(cfg)  # Should not raise

    def test_missing_filters_key(self):
        with pytest.raises(ValueError, match="Configuration must contain 'filters' key"):
            validate_filter_config({})

    def test_filters_not_list(self):
        with pytest.raises(ValueError, match="'filters' must be a list"):
            validate_filter_config({KEY_FILTERS: "not_a_list"})

    def test_filter_not_dict(self):
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_filter_config({KEY_FILTERS: ["not_a_dict"]})

    def test_missing_op(self):
        with pytest.raises(ValueError, match="missing required 'op' key"):
            validate_filter_config({KEY_FILTERS: [{KEY_FIELD: "X", KEY_TYPE: TYPE_QUALITY}]})

    def test_missing_type(self):
        with pytest.raises(ValueError, match="missing required 'type' key"):
            validate_filter_config({KEY_FILTERS: [{KEY_FIELD: "X", KEY_OP: "gt", KEY_VALUE: 1}]})

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="invalid type"):
            validate_filter_config({KEY_FILTERS: [{KEY_FIELD: "X", KEY_OP: "gt", KEY_VALUE: 1, KEY_TYPE: "invalid"}]})

    def test_invalid_op(self):
        with pytest.raises(ValueError, match="unsupported operator"):
            validate_filter_config(
                {KEY_FILTERS: [{KEY_FIELD: "X", KEY_OP: "bad_op", KEY_VALUE: 1, KEY_TYPE: TYPE_QUALITY}]}
            )

    def test_missing_field_for_non_any_not_null(self):
        with pytest.raises(ValueError, match="missing required 'field' key"):
            validate_filter_config({KEY_FILTERS: [{KEY_OP: "gt", KEY_VALUE: 1, KEY_TYPE: TYPE_QUALITY}]})

    def test_missing_value_for_binary_op(self):
        with pytest.raises(ValueError, match="must have either"):
            validate_filter_config({KEY_FILTERS: [{KEY_FIELD: "X", KEY_OP: "gt", KEY_TYPE: TYPE_QUALITY}]})

    def test_any_not_null_missing_fields(self):
        with pytest.raises(ValueError, match="requires a non-empty 'fields' list"):
            validate_filter_config({KEY_FILTERS: [{KEY_OP: OP_ANY_NOT_NULL, KEY_TYPE: TYPE_REGION}]})

    def test_any_not_null_empty_fields(self):
        with pytest.raises(ValueError, match="requires a non-empty 'fields' list"):
            validate_filter_config({KEY_FILTERS: [{KEY_OP: OP_ANY_NOT_NULL, KEY_FIELDS: [], KEY_TYPE: TYPE_REGION}]})

    def test_valid_downsample(self):
        cfg = {
            KEY_FILTERS: [{KEY_FIELD: "X", KEY_OP: "gt", KEY_VALUE: 1, KEY_TYPE: TYPE_QUALITY}],
            KEY_DOWNSAMPLE: {KEY_SIZE: 100, KEY_METHOD: METHOD_RANDOM, KEY_SEED: 42},
        }
        validate_filter_config(cfg)  # Should not raise

    def test_invalid_downsample_size(self):
        cfg = {
            KEY_FILTERS: [],
            KEY_DOWNSAMPLE: {KEY_SIZE: -1},
        }
        with pytest.raises(ValueError, match="positive integer"):
            validate_filter_config(cfg)

    def test_invalid_downsample_method(self):
        cfg = {
            KEY_FILTERS: [],
            KEY_DOWNSAMPLE: {KEY_SIZE: 100, KEY_METHOD: "invalid_method"},
        }
        with pytest.raises(ValueError, match="Invalid downsample method"):
            validate_filter_config(cfg)

    def test_is_null_unary_no_value_needed(self):
        """is_null does not require value/values/value_field."""
        cfg = {KEY_FILTERS: [{KEY_FIELD: "X", KEY_OP: OP_IS_NULL, KEY_TYPE: TYPE_REGION}]}
        validate_filter_config(cfg)  # Should not raise

    def test_is_not_null_unary_no_value_needed(self):
        cfg = {KEY_FILTERS: [{KEY_FIELD: "X", KEY_OP: OP_IS_NOT_NULL, KEY_TYPE: TYPE_REGION}]}
        validate_filter_config(cfg)  # Should not raise


# ──────────────────────── _parse_value_based_on_operation ──────────────


class TestParseValue:
    def test_int(self):
        assert _parse_value_based_on_operation("42", "eq") == 42

    def test_float(self):
        assert _parse_value_based_on_operation("3.14", "le") == pytest.approx(3.14)

    def test_string(self):
        assert _parse_value_based_on_operation("chr1", "eq") == "chr1"

    def test_bool_true(self):
        assert _parse_value_based_on_operation("true", "eq") is True
        assert _parse_value_based_on_operation("True", "eq") is True
        assert _parse_value_based_on_operation("TRUE", "eq") is True

    def test_bool_false(self):
        assert _parse_value_based_on_operation("false", "eq") is False
        assert _parse_value_based_on_operation("False", "eq") is False

    def test_list_for_in_op(self):
        result = _parse_value_based_on_operation("1,2,3", "in")
        assert result == [1, 2, 3]

    def test_list_for_not_in_op(self):
        result = _parse_value_based_on_operation("chr1,chr2", "not_in")
        assert result == ["chr1", "chr2"]

    def test_scientific_notation(self):
        assert _parse_value_based_on_operation("1e-3", "lt") == pytest.approx(0.001)

    def test_zero(self):
        assert _parse_value_based_on_operation("0", "eq") == 0


# ──────────────────────── _merge_config_and_cli ──────────────────────


class TestMergeConfigAndCli:
    def test_cli_only(self, tmp_path):
        cli_filters = ["name=f1:field=X:op=gt:value=10:type=quality"]
        result = _merge_config_and_cli(None, cli_filters, None)
        assert len(result[KEY_FILTERS]) == 1
        assert result[KEY_FILTERS][0][KEY_NAME] == "f1"

    def test_config_only(self, tmp_path):
        cfg = {KEY_FILTERS: [{KEY_NAME: "c1", KEY_FIELD: "Y", KEY_OP: "lt", KEY_VALUE: 5, KEY_TYPE: TYPE_QUALITY}]}
        cfg_path = str(tmp_path / "cfg.json")
        Path(cfg_path).write_text(json.dumps(cfg))
        result = _merge_config_and_cli(cfg_path, None, None)
        assert len(result[KEY_FILTERS]) == 1
        assert result[KEY_FILTERS][0][KEY_NAME] == "c1"

    def test_combined(self, tmp_path):
        cfg = {KEY_FILTERS: [{KEY_NAME: "c1", KEY_FIELD: "Y", KEY_OP: "lt", KEY_VALUE: 5, KEY_TYPE: TYPE_QUALITY}]}
        cfg_path = str(tmp_path / "cfg.json")
        Path(cfg_path).write_text(json.dumps(cfg))
        cli_filters = ["name=f1:field=X:op=gt:value=10:type=quality"]
        result = _merge_config_and_cli(cfg_path, cli_filters, None)
        assert len(result[KEY_FILTERS]) == 2

    def test_cli_downsample_override(self, tmp_path):
        cfg = {
            KEY_FILTERS: [],
            KEY_DOWNSAMPLE: {KEY_SIZE: 100, KEY_METHOD: METHOD_RANDOM},
        }
        cfg_path = str(tmp_path / "cfg.json")
        Path(cfg_path).write_text(json.dumps(cfg))
        result = _merge_config_and_cli(cfg_path, None, "head:50")
        assert result[KEY_DOWNSAMPLE][KEY_METHOD] == METHOD_HEAD
        assert result[KEY_DOWNSAMPLE][KEY_SIZE] == 50


# ──────────────────────── _parse_cli_filter / _parse_cli_downsample ────


class TestParseCliFilter:
    def test_basic(self):
        result = _parse_cli_filter("name=f:field=X:op=gt:value=10:type=quality")
        assert result == {KEY_NAME: "f", KEY_FIELD: "X", KEY_OP: "gt", KEY_VALUE: 10, KEY_TYPE: TYPE_QUALITY}

    def test_value_field(self):
        result = _parse_cli_filter("name=cmp:field=REF:op=ne:value_field=ALT:type=label")
        assert result[KEY_VALUE_FIELD] == "ALT"
        assert KEY_VALUE not in result

    def test_too_few_parts_raises(self):
        with pytest.raises(ValueError, match="must have at least"):
            _parse_cli_filter("a:b:c")

    def test_missing_eq_raises(self):
        with pytest.raises(ValueError, match="key=value format"):
            _parse_cli_filter("name=f:field=X:opgt:value=10:type=quality")


class TestParseCliDownsample:
    def test_random_with_seed(self):
        result = _parse_cli_downsample("random:1000:42")
        assert result == {KEY_METHOD: "random", KEY_SIZE: 1000, KEY_SEED: 42}

    def test_head_no_seed(self):
        result = _parse_cli_downsample("head:500")
        assert result == {KEY_METHOD: "head", KEY_SIZE: 500}

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_cli_downsample("random:abc")

    def test_too_many_parts_raises(self):
        with pytest.raises(ValueError, match="must have 2-3 parts"):
            _parse_cli_downsample("random:100:42:extra")

    def test_invalid_seed_raises(self):
        with pytest.raises(ValueError, match="seed must be an integer"):
            _parse_cli_downsample("random:100:not_a_number")

    def test_too_few_parts_raises(self):
        with pytest.raises(ValueError, match="must have 2-3 parts"):
            _parse_cli_downsample("random")


# ──────────────────────── _create_downsample_column ──────────────────────


class TestCreateDownsampleColumn:
    """Test downsample column creation for head and random methods."""

    def test_no_downsample_returns_none(self):
        lf = pl.DataFrame({"score": [10, 20, 30]}).lazy()
        cfg = {KEY_FILTERS: []}
        result_lf, ds_col = _create_downsample_column(lf, [], cfg)
        assert ds_col is None

    def test_head_downsample(self):
        lf = pl.DataFrame({"score": list(range(10))}).lazy()
        filters = [
            {KEY_NAME: "pass_all", KEY_FIELD: "score", KEY_OP: "ge", KEY_VALUE: 0, KEY_TYPE: TYPE_QUALITY},
        ]
        lf, filter_cols = _create_filter_columns(lf, filters)
        cfg = {KEY_FILTERS: filters, KEY_DOWNSAMPLE: {KEY_SIZE: 5, KEY_METHOD: METHOD_HEAD}}

        result_lf, ds_col = _create_downsample_column(lf, filter_cols, cfg)
        assert ds_col == COL_FILTER_DOWNSAMPLE

        collected = result_lf.collect()
        ds_values = collected[ds_col].to_list()
        # First 5 should be True, rest False
        assert sum(v is True for v in ds_values) == 5
        assert sum(v is False for v in ds_values) == 5

    def test_random_downsample(self):
        lf = pl.DataFrame({"score": list(range(20))}).lazy()
        filters = [
            {KEY_NAME: "pass_all", KEY_FIELD: "score", KEY_OP: "ge", KEY_VALUE: 0, KEY_TYPE: TYPE_QUALITY},
        ]
        lf, filter_cols = _create_filter_columns(lf, filters)
        cfg = {KEY_FILTERS: filters, KEY_DOWNSAMPLE: {KEY_SIZE: 10, KEY_METHOD: METHOD_RANDOM, KEY_SEED: 42}}

        result_lf, ds_col = _create_downsample_column(lf, filter_cols, cfg)
        assert ds_col == COL_FILTER_DOWNSAMPLE

        collected = result_lf.collect()
        ds_values = collected[ds_col].to_list()
        # Exactly 10 should be True
        assert sum(v is True for v in ds_values) == 10

    def test_downsample_with_some_rows_filtered(self):
        """Only rows passing all filters should participate in downsample."""
        lf = pl.DataFrame({"score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}).lazy()
        filters = [
            {KEY_NAME: "high", KEY_FIELD: "score", KEY_OP: "gt", KEY_VALUE: 5, KEY_TYPE: TYPE_QUALITY},
        ]
        lf, filter_cols = _create_filter_columns(lf, filters)
        cfg = {KEY_FILTERS: filters, KEY_DOWNSAMPLE: {KEY_SIZE: 3, KEY_METHOD: METHOD_HEAD}}

        result_lf, ds_col = _create_downsample_column(lf, filter_cols, cfg)
        collected = result_lf.collect()
        ds_values = collected[ds_col].to_list()
        # Only 5 rows pass the filter (score > 5: 6,7,8,9,10), head 3 of those
        assert sum(v is True for v in ds_values) == 3
        # First 5 rows don't pass the filter, so their ds value is None
        assert all(v is None for v in ds_values[:5])


# ──────────────────────── _create_final_filter_column ──────────────────────


class TestCreateFinalFilterColumn:
    """Test final combined filter column creation."""

    def test_without_downsample(self):
        lf = pl.DataFrame(
            {
                "__filter_a": [True, False, True, False],
                "__filter_b": [True, True, False, False],
            }
        ).lazy()
        result_lf = _create_final_filter_column(lf, ["__filter_a", "__filter_b"], None)
        collected = result_lf.collect()
        assert collected[COL_FILTER_FINAL].to_list() == [True, False, False, False]

    def test_with_downsample(self):
        lf = pl.DataFrame(
            {
                "__filter_a": [True, True, True, True],
                COL_FILTER_DOWNSAMPLE: [True, False, True, None],
            }
        ).lazy()
        result_lf = _create_final_filter_column(lf, ["__filter_a"], COL_FILTER_DOWNSAMPLE)
        collected = result_lf.collect()
        # filter_a=True & downsample: [True, False, True, False(null filled)]
        assert collected[COL_FILTER_FINAL].to_list() == [True, False, True, False]

    def test_all_filters_pass(self):
        lf = pl.DataFrame(
            {
                "__filter_x": [True, True, True],
            }
        ).lazy()
        result_lf = _create_final_filter_column(lf, ["__filter_x"], None)
        collected = result_lf.collect()
        assert collected[COL_FILTER_FINAL].to_list() == [True, True, True]


# ──────────────────────── filter_parquet (integration) ──────────────────────


class TestFilterParquet:
    """Integration tests for the filter_parquet function with actual parquet files."""

    def test_basic_filter(self, tmp_path):
        """Test basic filtering writes correct output."""
        # Create input parquet
        in_frame = pl.DataFrame({"score": [10, 20, 30, 40, 50], "name": ["a", "b", "c", "d", "e"]})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        # Create config
        cfg = {
            KEY_FILTERS: [
                {KEY_NAME: "high_score", KEY_FIELD: "score", KEY_OP: "gt", KEY_VALUE: 25, KEY_TYPE: TYPE_QUALITY},
            ]
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        filter_parquet(in_path, out_path, None, cfg_path, stats_path)

        # Check output
        result = pl.read_parquet(out_path)
        assert len(result) == 3  # rows with score > 25: 30, 40, 50
        assert list(result["score"]) == [30, 40, 50]

        # Check stats
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats[KEY_FILTERS][0]["rows"] == 5
        assert stats[KEY_FILTERS][1]["rows"] == 3

    def test_filter_with_downsample_head(self, tmp_path):
        """Test filtering with head downsample."""
        in_frame = pl.DataFrame({"val": list(range(100))})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        cfg = {
            KEY_FILTERS: [
                {KEY_NAME: "all", KEY_FIELD: "val", KEY_OP: "ge", KEY_VALUE: 0, KEY_TYPE: TYPE_QUALITY},
            ],
            KEY_DOWNSAMPLE: {KEY_SIZE: 10, KEY_METHOD: METHOD_HEAD},
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        filter_parquet(in_path, out_path, None, cfg_path, stats_path)

        result = pl.read_parquet(out_path)
        assert len(result) == 10

    def test_filter_full_output(self, tmp_path):
        """Test that out_path_full contains all rows with filter columns."""
        in_frame = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        cfg = {
            KEY_FILTERS: [
                {KEY_NAME: "big", KEY_FIELD: "x", KEY_OP: "gt", KEY_VALUE: 3, KEY_TYPE: TYPE_QUALITY},
            ]
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_full_path = str(tmp_path / "full.parquet")
        stats_path = str(tmp_path / "stats.json")

        filter_parquet(in_path, None, out_full_path, cfg_path, stats_path)

        result = pl.read_parquet(out_full_path)
        # Should have all 5 rows
        assert len(result) == 5
        # Should have the filter column
        assert any(col.startswith(COL_PREFIX_FILTER) for col in result.columns)

    def test_filter_with_cli_filters(self, tmp_path):
        """Test using CLI filter specifications."""
        in_frame = pl.DataFrame({"score": [10, 20, 30, 40, 50]})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        cli_filters = ["name=high:field=score:op=gt:value=30:type=quality"]
        filter_parquet(in_path, out_path, None, None, stats_path, cli_filters=cli_filters)

        result = pl.read_parquet(out_path)
        assert len(result) == 2  # 40, 50


# ──────────────────────── _validate_stats_dict ──────────────────────────


class TestValidateStatsDict:
    """Test stats dictionary validation."""

    def test_valid_stats(self):
        data = {
            "filters": [
                {"name": "raw", "rows": 1000, "type": "raw"},
                {"name": "qual", "rows": 800, "type": "quality"},
            ],
            "single_effect": {"qual": 800},
            "combinations": {"1": 800, "0": 200},
        }
        # Should not raise
        _validate_stats_dict(data, where=" (test)")

    def test_missing_key(self):
        data = {"filters": [{"name": "raw", "rows": 100}], "single_effect": {}}
        with pytest.raises(ValueError, match="missing key"):
            _validate_stats_dict(data, where="")

    def test_empty_filters(self):
        data = {"filters": [], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="non-empty list"):
            _validate_stats_dict(data, where="")

    def test_filters_not_list(self):
        data = {"filters": "not_a_list", "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="non-empty list"):
            _validate_stats_dict(data, where="")

    def test_filter_not_dict(self):
        data = {"filters": ["not_a_dict"], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="is not an object"):
            _validate_stats_dict(data, where="")

    def test_filter_missing_name_or_rows(self):
        data = {"filters": [{"name": "raw"}], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="missing 'name' or 'rows'"):
            _validate_stats_dict(data, where="")

    def test_filter_negative_rows(self):
        data = {"filters": [{"name": "raw", "rows": -1}], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="non-negative integer"):
            _validate_stats_dict(data, where="")

    def test_filter_rows_not_int(self):
        data = {"filters": [{"name": "raw", "rows": 1.5}], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="non-negative integer"):
            _validate_stats_dict(data, where="")

    def test_first_filter_not_raw(self):
        data = {
            "filters": [{"name": "not_raw", "rows": 100}],
            "single_effect": {},
            "combinations": {},
        }
        with pytest.raises(ValueError, match="first filter must have name 'raw'"):
            _validate_stats_dict(data, where="")

    def test_single_effect_not_dict(self):
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": "wrong",
            "combinations": {},
        }
        with pytest.raises(ValueError, match="'single_effect' must be an object"):
            _validate_stats_dict(data, where="")

    def test_single_effect_negative_value(self):
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": {"f1": -5},
            "combinations": {},
        }
        with pytest.raises(ValueError, match="non-negative integers"):
            _validate_stats_dict(data, where="")

    def test_combinations_negative_value(self):
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": {},
            "combinations": {"11": -1},
        }
        with pytest.raises(ValueError, match="non-negative integers"):
            _validate_stats_dict(data, where="")

    def test_combinations_not_dict_ignored(self):
        """If combinations is not a dict, validation passes (no crash)."""
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": {},
            "combinations": "skipped_if_not_dict",
        }
        # Should not raise - the condition short-circuits
        _validate_stats_dict(data, where="")


# ──────────────────────── read_filtering_stats_json ──────────────────────


class TestReadFilteringStatsJson:
    def test_valid_file(self, tmp_path):
        data = {
            "filters": [
                {"name": "raw", "rows": 1000, "type": "raw"},
                {"name": "coverage", "rows": 900, "type": "region"},
            ],
            "single_effect": {"coverage": 900},
            "combinations": {"1": 900, "0": 100},
        }
        path = tmp_path / "stats.json"
        path.write_text(json.dumps(data))

        result = read_filtering_stats_json(str(path))
        assert result == data

    def test_invalid_file_raises(self, tmp_path):
        data = {"filters": [], "single_effect": {}, "combinations": {}}
        path = tmp_path / "bad_stats.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="non-empty list"):
            read_filtering_stats_json(str(path))


# ──────────────────────── additional _parse_cli_filter tests ──────────────


class TestParseCliFilterAdditional:
    def test_missing_required_keys_raises(self):
        """Missing name/field/op/type raises."""
        with pytest.raises(ValueError, match="Missing required keys"):
            _parse_cli_filter("field=X:op=gt:value=10:type=quality")  # missing name

    def test_missing_value_and_value_field_raises(self):
        with pytest.raises(ValueError, match="Must specify either"):
            _parse_cli_filter("name=f:field=X:op=gt:type=quality")

    def test_in_op_parses_list(self):
        result = _parse_cli_filter("name=f:field=X:op=in:value=a,b,c:type=quality")
        assert result[KEY_VALUE] == ["a", "b", "c"]
