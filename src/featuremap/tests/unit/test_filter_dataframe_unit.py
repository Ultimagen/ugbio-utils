"""Unit tests for filter_dataframe module - pure logic tests without file I/O.

Covers:
- _default_filter_name: name derivation from rule structures (regression for U5)
- _mask_for_rule: all operators including is_null, is_not_null, any_not_null
- _create_filter_columns: filter pipeline
- _calculate_statistics: with any_not_null rules (regression for U2)
- validate_filter_config: config validation
- _parse_value_based_on_operation: value parsing
- _merge_config_and_cli: config merging
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap.filter_dataframe import (
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
    _create_filter_columns,
    _default_filter_name,
    _mask_for_rule,
    _merge_config_and_cli,
    _parse_cli_downsample,
    _parse_cli_filter,
    _parse_value_based_on_operation,
    validate_filter_config,
)

pl.enable_string_cache()


# ──────────────────────── _default_filter_name ──────────────────────


class TestDefaultFilterName:
    """Test name derivation from filter rule structures (regression for U5)."""

    def test_explicit_name_used(self):
        rule = {KEY_NAME: "my_filter", KEY_FIELD: "QUAL", KEY_OP: "gt", KEY_VALUE: 30}
        assert _default_filter_name(rule) == "my_filter"

    def test_fallback_to_field_op(self):
        rule = {KEY_FIELD: "QUAL", KEY_OP: "gt", KEY_VALUE: 30}
        assert _default_filter_name(rule) == "QUAL_gt"

    def test_any_not_null_uses_fields(self):
        """Regression U5: any_not_null rules use KEY_FIELDS, not KEY_FIELD."""
        rule = {KEY_OP: OP_ANY_NOT_NULL, KEY_FIELDS: ["gnomAD_AF", "PCAWG"], KEY_TYPE: TYPE_REGION}
        name = _default_filter_name(rule)
        # Should join fields with underscore
        assert name == "gnomAD_AF_PCAWG_any_not_null"

    def test_single_field_in_fields_list(self):
        rule = {KEY_OP: OP_IS_NULL, KEY_FIELDS: ["my_col"], KEY_TYPE: TYPE_REGION}
        name = _default_filter_name(rule)
        assert name == "my_col_is_null"

    def test_empty_name_string_falls_back(self):
        rule = {KEY_NAME: "", KEY_FIELD: "DP", KEY_OP: "ge"}
        # Empty string is falsy, should fall back
        assert _default_filter_name(rule) == "DP_ge"

    def test_none_name_falls_back(self):
        rule = {KEY_NAME: None, KEY_FIELD: "DP", KEY_OP: "le"}
        assert _default_filter_name(rule) == "DP_le"

    def test_no_field_no_fields(self):
        """Edge case: rule with no field/fields key."""
        rule = {KEY_OP: "gt", KEY_VALUE: 10}
        name = _default_filter_name(rule)
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
        filter_name = _default_filter_name(filters[0])
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
