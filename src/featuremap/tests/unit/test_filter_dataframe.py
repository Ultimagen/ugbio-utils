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

import json
from pathlib import Path

import polars as pl
import pytest
import ugbio_featuremap.filter_dataframe as fd

pl.enable_string_cache()


# ────────────────────────────────────────────────────────


class TestGetFilterName:
    """Test name derivation from filter rule structures (regression for U5)."""

    def test_explicit_name_used(self):
        rule = {fd.KEY_NAME: "my_filter", fd.KEY_FIELD: "QUAL", fd.KEY_OP: "gt", fd.KEY_VALUE: 30}
        assert fd._get_filter_name(rule) == "my_filter"

    def test_fallback_to_field_op(self):
        rule = {fd.KEY_FIELD: "QUAL", fd.KEY_OP: "gt", fd.KEY_VALUE: 30}
        assert fd._get_filter_name(rule) == "QUAL_gt"

    def test_any_not_null_uses_fields(self):
        """Regression U5: any_not_null rules use KEY_FIELDS, not KEY_FIELD."""
        rule = {fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_FIELDS: ["gnomAD_AF", "PCAWG"], fd.KEY_TYPE: fd.TYPE_REGION}
        name = fd._get_filter_name(rule)
        # Should join fields with underscore
        assert name == "gnomAD_AF_PCAWG_any_not_null"

    def test_single_field_in_fields_list(self):
        rule = {fd.KEY_OP: fd.OP_IS_NULL, fd.KEY_FIELDS: ["my_col"], fd.KEY_TYPE: fd.TYPE_REGION}
        name = fd._get_filter_name(rule)
        assert name == "my_col_is_null"

    def test_empty_name_string_falls_back(self):
        rule = {fd.KEY_NAME: "", fd.KEY_FIELD: "DP", fd.KEY_OP: "ge"}
        # Empty string is falsy, should fall back
        assert fd._get_filter_name(rule) == "DP_ge"

    def test_none_name_falls_back(self):
        rule = {fd.KEY_NAME: None, fd.KEY_FIELD: "DP", fd.KEY_OP: "le"}
        assert fd._get_filter_name(rule) == "DP_le"

    def test_no_field_no_fields(self):
        """Edge case: rule with no field/fields key."""
        rule = {fd.KEY_OP: "gt", fd.KEY_VALUE: 10}
        name = fd._get_filter_name(rule)
        # Should use empty join of KEY_FIELDS default (empty list)
        assert name == "_gt"


# ────────────────────────────────────────────────────────


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
        rule = {fd.KEY_FIELD: "score", fd.KEY_OP: "eq", fd.KEY_VALUE: 20}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, True, False, None, False]

    def test_ne(self, sample_lf):
        rule = {fd.KEY_FIELD: "score", fd.KEY_OP: "ne", fd.KEY_VALUE: 20}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, False, True, None, True]

    def test_lt(self, sample_lf):
        rule = {fd.KEY_FIELD: "score", fd.KEY_OP: "lt", fd.KEY_VALUE: 25}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, True, False, None, False]

    def test_le(self, sample_lf):
        rule = {fd.KEY_FIELD: "score", fd.KEY_OP: "le", fd.KEY_VALUE: 30}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, True, True, None, False]

    def test_gt(self, sample_lf):
        rule = {fd.KEY_FIELD: "score", fd.KEY_OP: "gt", fd.KEY_VALUE: 20}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, False, True, None, True]

    def test_ge(self, sample_lf):
        rule = {fd.KEY_FIELD: "score", fd.KEY_OP: "ge", fd.KEY_VALUE: 20}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, True, True, None, True]

    def test_in(self, sample_lf):
        rule = {fd.KEY_FIELD: "category", fd.KEY_OP: "in", fd.KEY_VALUE: ["A", "B"]}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, True, True, None, False]

    def test_not_in(self, sample_lf):
        rule = {fd.KEY_FIELD: "category", fd.KEY_OP: "not_in", fd.KEY_VALUE: ["A"]}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, True, False, None, True]

    def test_is_null(self, sample_lf):
        rule = {fd.KEY_FIELD: "score", fd.KEY_OP: fd.OP_IS_NULL}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, False, False, True, False]

    def test_is_not_null(self, sample_lf):
        rule = {fd.KEY_FIELD: "score", fd.KEY_OP: fd.OP_IS_NOT_NULL}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [True, True, True, False, True]

    def test_any_not_null_single_field(self, sample_lf):
        rule = {fd.KEY_FIELDS: ["field_a"], fd.KEY_OP: fd.OP_ANY_NOT_NULL}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        # field_a: [1.0, None, 3.0, None, 5.0]
        assert result.to_list() == [True, False, True, False, True]

    def test_any_not_null_multiple_fields(self, sample_lf):
        """Regression U2: any_not_null with multiple fields."""
        rule = {fd.KEY_FIELDS: ["field_a", "field_b"], fd.KEY_OP: fd.OP_ANY_NOT_NULL}
        expr = fd._mask_for_rule(rule)
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
        rule = {fd.KEY_FIELDS: ["x", "y"], fd.KEY_OP: fd.OP_ANY_NOT_NULL}
        expr = fd._mask_for_rule(rule)
        result = lf.select(expr).collect().to_series()
        assert result.to_list() == [False, False, False]

    def test_value_field_comparison(self, sample_lf):
        """Compare two columns using value_field."""
        rule = {fd.KEY_FIELD: "ref", fd.KEY_OP: "eq", fd.KEY_VALUE_FIELD: "alt"}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        # ref: [A, C, G, T, A] vs alt: [A, G, G, T, C]
        # eq:  [T, F, T, T, F]
        assert result.to_list() == [True, False, True, True, False]

    def test_value_field_ne(self, sample_lf):
        rule = {fd.KEY_FIELD: "ref", fd.KEY_OP: "ne", fd.KEY_VALUE_FIELD: "alt"}
        expr = fd._mask_for_rule(rule)
        result = sample_lf.select(expr).collect().to_series()
        assert result.to_list() == [False, True, False, False, True]

    def test_unsupported_op(self):
        rule = {fd.KEY_FIELD: "x", fd.KEY_OP: "unsupported_op", fd.KEY_VALUE: 1}
        with pytest.raises(ValueError, match="Unsupported op"):
            fd._mask_for_rule(rule)


# ────────────────────────────────────────────────────────


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
            {
                fd.KEY_NAME: "high_score",
                fd.KEY_FIELD: "score",
                fd.KEY_OP: "gt",
                fd.KEY_VALUE: 25,
                fd.KEY_TYPE: fd.TYPE_QUALITY,
            },
            {
                fd.KEY_NAME: "cat_a",
                fd.KEY_FIELD: "category",
                fd.KEY_OP: "eq",
                fd.KEY_VALUE: "A",
                fd.KEY_TYPE: fd.TYPE_LABEL,
            },
        ]

        result_lf, filter_cols = fd._create_filter_columns(lf, filters)
        assert len(filter_cols) == 2
        assert filter_cols[0] == f"{fd.COL_PREFIX_FILTER}high_score"
        assert filter_cols[1] == f"{fd.COL_PREFIX_FILTER}cat_a"

        result_df = result_lf.collect()
        assert result_df[filter_cols[0]].to_list() == [False, False, True, True, True]
        assert result_df[filter_cols[1]].to_list() == [True, False, True, False, False]

    def test_empty_filters(self):
        lf = pl.DataFrame({"x": [1, 2, 3]}).lazy()
        result_lf, filter_cols = fd._create_filter_columns(lf, [])
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
            {fd.KEY_FIELDS: ["field_a", "field_b"], fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_TYPE: fd.TYPE_REGION},
        ]

        result_lf, filter_cols = fd._create_filter_columns(lf, filters)
        result_df = result_lf.collect()
        col_name = filter_cols[0]
        # At least one non-null in each row
        assert result_df[col_name].to_list() == [True, True, True]


# ────────────────────────────────────────────────────────


class TestCalculateStatistics:
    """Test statistics calculation including any_not_null regression (U2)."""

    def test_basic_stats(self):
        lf = pl.DataFrame(
            {
                "score": [10, 20, 30, 40, 50],
            }
        ).lazy()

        filters = [
            {
                fd.KEY_NAME: "high_score",
                fd.KEY_FIELD: "score",
                fd.KEY_OP: "gt",
                fd.KEY_VALUE: 25,
                fd.KEY_TYPE: fd.TYPE_QUALITY,
            },
        ]
        lf, filter_cols = fd._create_filter_columns(lf, filters)
        cfg = {fd.KEY_FILTERS: filters}

        stats = fd._calculate_statistics(lf, filter_cols, None, filters, total_rows=5, cfg=cfg)

        assert stats[fd.KEY_FILTERS][0][fd.KEY_NAME] == "raw"
        assert stats[fd.KEY_FILTERS][0]["rows"] == 5
        assert stats[fd.KEY_FILTERS][1][fd.KEY_NAME] == "high_score"
        assert stats[fd.KEY_FILTERS][1]["rows"] == 3  # 30, 40, 50

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
            {fd.KEY_FIELDS: ["field_a", "field_b"], fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_TYPE: fd.TYPE_REGION},
        ]
        lf, filter_cols = fd._create_filter_columns(lf, filters)
        cfg = {fd.KEY_FILTERS: filters}

        stats = fd._calculate_statistics(lf, filter_cols, None, filters, total_rows=5, cfg=cfg)

        # any_not_null: rows where at least one field is not null
        # field_a: [1,None,3,None,5], field_b: [None,2,None,None,5]
        # OR: [True, True, True, False, True] => 4 pass
        filter_name = fd._get_filter_name(filters[0])
        assert stats[fd.KEY_FILTERS][1]["rows"] == 4
        assert stats["single_effect"][filter_name] == 4

    def test_stats_with_downsample(self):
        lf = pl.DataFrame(
            {
                "score": list(range(100)),
            }
        ).lazy()

        filters = [
            {
                fd.KEY_NAME: "pass_all",
                fd.KEY_FIELD: "score",
                fd.KEY_OP: "ge",
                fd.KEY_VALUE: 0,
                fd.KEY_TYPE: fd.TYPE_QUALITY,
            },
        ]
        lf, filter_cols = fd._create_filter_columns(lf, filters)
        cfg = {fd.KEY_FILTERS: filters, fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 50, fd.KEY_METHOD: fd.METHOD_HEAD}}

        stats = fd._calculate_statistics(lf, filter_cols, None, filters, total_rows=100, cfg=cfg)

        # Downsample entry should be appended
        assert stats[fd.KEY_FILTERS][-1][fd.KEY_NAME] == "downsample"
        assert stats[fd.KEY_FILTERS][-1]["rows"] == 50

    def test_combination_patterns(self):
        """Combination stats should have 2^n patterns."""
        lf = pl.DataFrame(
            {
                "a": [True, False, True, False],
                "b": [True, True, False, False],
            }
        ).lazy()

        filters = [
            {
                fd.KEY_NAME: "filt_a",
                fd.KEY_FIELD: "a",
                fd.KEY_OP: "eq",
                fd.KEY_VALUE: True,
                fd.KEY_TYPE: fd.TYPE_QUALITY,
            },
            {
                fd.KEY_NAME: "filt_b",
                fd.KEY_FIELD: "b",
                fd.KEY_OP: "eq",
                fd.KEY_VALUE: True,
                fd.KEY_TYPE: fd.TYPE_QUALITY,
            },
        ]
        lf, filter_cols = fd._create_filter_columns(lf, filters)
        cfg = {fd.KEY_FILTERS: filters}

        stats = fd._calculate_statistics(lf, filter_cols, None, filters, total_rows=4, cfg=cfg)

        combos = stats["combinations"]
        # 2 filters => 4 possible patterns (00, 01, 10, 11)
        assert len(combos) == 4
        assert combos["11"] == 1  # row 0: both True
        assert combos["10"] == 1  # row 2: a=True, b=False
        assert combos["01"] == 1  # row 1: a=False, b=True
        assert combos["00"] == 1  # row 3: both False


# ────────────────────────────────────────────────────────


class TestValidateFilterConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        cfg = {
            fd.KEY_FILTERS: [
                {fd.KEY_FIELD: "QUAL", fd.KEY_OP: "gt", fd.KEY_VALUE: 30, fd.KEY_TYPE: fd.TYPE_QUALITY},
            ]
        }
        fd.validate_filter_config(cfg)  # Should not raise

    def test_valid_any_not_null(self):
        cfg = {
            fd.KEY_FILTERS: [
                {fd.KEY_FIELDS: ["gnomAD_AF", "PCAWG"], fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_TYPE: fd.TYPE_REGION},
            ]
        }
        fd.validate_filter_config(cfg)  # Should not raise

    def test_missing_filters_key(self):
        with pytest.raises(ValueError, match="Configuration must contain 'filters' key"):
            fd.validate_filter_config({})

    def test_filters_not_list(self):
        with pytest.raises(ValueError, match="'filters' must be a list"):
            fd.validate_filter_config({fd.KEY_FILTERS: "not_a_list"})

    def test_filter_not_dict(self):
        with pytest.raises(ValueError, match="must be a dictionary"):
            fd.validate_filter_config({fd.KEY_FILTERS: ["not_a_dict"]})

    def test_missing_op(self):
        with pytest.raises(ValueError, match="missing required 'op' key"):
            fd.validate_filter_config({fd.KEY_FILTERS: [{fd.KEY_FIELD: "X", fd.KEY_TYPE: fd.TYPE_QUALITY}]})

    def test_missing_type(self):
        with pytest.raises(ValueError, match="missing required 'type' key"):
            fd.validate_filter_config({fd.KEY_FILTERS: [{fd.KEY_FIELD: "X", fd.KEY_OP: "gt", fd.KEY_VALUE: 1}]})

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="invalid type"):
            fd.validate_filter_config(
                {fd.KEY_FILTERS: [{fd.KEY_FIELD: "X", fd.KEY_OP: "gt", fd.KEY_VALUE: 1, fd.KEY_TYPE: "invalid"}]}
            )

    def test_invalid_op(self):
        with pytest.raises(ValueError, match="unsupported operator"):
            fd.validate_filter_config(
                {
                    fd.KEY_FILTERS: [
                        {fd.KEY_FIELD: "X", fd.KEY_OP: "bad_op", fd.KEY_VALUE: 1, fd.KEY_TYPE: fd.TYPE_QUALITY}
                    ]
                }
            )

    def test_missing_field_for_non_any_not_null(self):
        with pytest.raises(ValueError, match="missing required 'field' key"):
            fd.validate_filter_config(
                {fd.KEY_FILTERS: [{fd.KEY_OP: "gt", fd.KEY_VALUE: 1, fd.KEY_TYPE: fd.TYPE_QUALITY}]}
            )

    def test_missing_value_for_binary_op(self):
        with pytest.raises(ValueError, match="must have either"):
            fd.validate_filter_config(
                {fd.KEY_FILTERS: [{fd.KEY_FIELD: "X", fd.KEY_OP: "gt", fd.KEY_TYPE: fd.TYPE_QUALITY}]}
            )

    def test_any_not_null_missing_fields(self):
        with pytest.raises(ValueError, match="requires a non-empty 'fields' list"):
            fd.validate_filter_config({fd.KEY_FILTERS: [{fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_TYPE: fd.TYPE_REGION}]})

    def test_any_not_null_empty_fields(self):
        with pytest.raises(ValueError, match="requires a non-empty 'fields' list"):
            fd.validate_filter_config(
                {fd.KEY_FILTERS: [{fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_FIELDS: [], fd.KEY_TYPE: fd.TYPE_REGION}]}
            )

    def test_valid_downsample(self):
        cfg = {
            fd.KEY_FILTERS: [{fd.KEY_FIELD: "X", fd.KEY_OP: "gt", fd.KEY_VALUE: 1, fd.KEY_TYPE: fd.TYPE_QUALITY}],
            fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 100, fd.KEY_METHOD: fd.METHOD_RANDOM, fd.KEY_SEED: 42},
        }
        fd.validate_filter_config(cfg)  # Should not raise

    def test_invalid_downsample_size(self):
        cfg = {
            fd.KEY_FILTERS: [],
            fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: -1},
        }
        with pytest.raises(ValueError, match="positive integer"):
            fd.validate_filter_config(cfg)

    def test_invalid_downsample_method(self):
        cfg = {
            fd.KEY_FILTERS: [],
            fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 100, fd.KEY_METHOD: "invalid_method"},
        }
        with pytest.raises(ValueError, match="Invalid downsample method"):
            fd.validate_filter_config(cfg)

    def test_is_null_unary_no_value_needed(self):
        """is_null does not require value/values/value_field."""
        cfg = {fd.KEY_FILTERS: [{fd.KEY_FIELD: "X", fd.KEY_OP: fd.OP_IS_NULL, fd.KEY_TYPE: fd.TYPE_REGION}]}
        fd.validate_filter_config(cfg)  # Should not raise

    def test_is_not_null_unary_no_value_needed(self):
        cfg = {fd.KEY_FILTERS: [{fd.KEY_FIELD: "X", fd.KEY_OP: fd.OP_IS_NOT_NULL, fd.KEY_TYPE: fd.TYPE_REGION}]}
        fd.validate_filter_config(cfg)  # Should not raise


# ────────────────────────────────────────────────────────


class TestParseValue:
    def test_int(self):
        assert fd._parse_value_based_on_operation("42", "eq") == 42

    def test_float(self):
        assert fd._parse_value_based_on_operation("3.14", "le") == pytest.approx(3.14)

    def test_string(self):
        assert fd._parse_value_based_on_operation("chr1", "eq") == "chr1"

    def test_bool_true(self):
        assert fd._parse_value_based_on_operation("true", "eq") is True
        assert fd._parse_value_based_on_operation("True", "eq") is True
        assert fd._parse_value_based_on_operation("TRUE", "eq") is True

    def test_bool_false(self):
        assert fd._parse_value_based_on_operation("false", "eq") is False
        assert fd._parse_value_based_on_operation("False", "eq") is False

    def test_list_for_in_op(self):
        result = fd._parse_value_based_on_operation("1,2,3", "in")
        assert result == [1, 2, 3]

    def test_list_for_not_in_op(self):
        result = fd._parse_value_based_on_operation("chr1,chr2", "not_in")
        assert result == ["chr1", "chr2"]

    def test_scientific_notation(self):
        assert fd._parse_value_based_on_operation("1e-3", "lt") == pytest.approx(0.001)

    def test_zero(self):
        assert fd._parse_value_based_on_operation("0", "eq") == 0


# ────────────────────────────────────────────────────────


class TestMergeConfigAndCli:
    def test_cli_only(self, tmp_path):
        cli_filters = ["name=f1:field=X:op=gt:value=10:type=quality"]
        result = fd._merge_config_and_cli(None, cli_filters, None)
        assert len(result[fd.KEY_FILTERS]) == 1
        assert result[fd.KEY_FILTERS][0][fd.KEY_NAME] == "f1"

    def test_config_only(self, tmp_path):
        cfg = {
            fd.KEY_FILTERS: [
                {fd.KEY_NAME: "c1", fd.KEY_FIELD: "Y", fd.KEY_OP: "lt", fd.KEY_VALUE: 5, fd.KEY_TYPE: fd.TYPE_QUALITY}
            ]
        }
        cfg_path = str(tmp_path / "cfg.json")
        Path(cfg_path).write_text(json.dumps(cfg))
        result = fd._merge_config_and_cli(cfg_path, None, None)
        assert len(result[fd.KEY_FILTERS]) == 1
        assert result[fd.KEY_FILTERS][0][fd.KEY_NAME] == "c1"

    def test_combined(self, tmp_path):
        cfg = {
            fd.KEY_FILTERS: [
                {fd.KEY_NAME: "c1", fd.KEY_FIELD: "Y", fd.KEY_OP: "lt", fd.KEY_VALUE: 5, fd.KEY_TYPE: fd.TYPE_QUALITY}
            ]
        }
        cfg_path = str(tmp_path / "cfg.json")
        Path(cfg_path).write_text(json.dumps(cfg))
        cli_filters = ["name=f1:field=X:op=gt:value=10:type=quality"]
        result = fd._merge_config_and_cli(cfg_path, cli_filters, None)
        assert len(result[fd.KEY_FILTERS]) == 2

    def test_cli_downsample_override(self, tmp_path):
        cfg = {
            fd.KEY_FILTERS: [],
            fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 100, fd.KEY_METHOD: fd.METHOD_RANDOM},
        }
        cfg_path = str(tmp_path / "cfg.json")
        Path(cfg_path).write_text(json.dumps(cfg))
        result = fd._merge_config_and_cli(cfg_path, None, "head:50")
        assert result[fd.KEY_DOWNSAMPLE][fd.KEY_METHOD] == fd.METHOD_HEAD
        assert result[fd.KEY_DOWNSAMPLE][fd.KEY_SIZE] == 50


# ────────────────────────────────────────────────────────


class TestParseCliFilter:
    def test_basic(self):
        result = fd._parse_cli_filter("name=f:field=X:op=gt:value=10:type=quality")
        assert result == {
            fd.KEY_NAME: "f",
            fd.KEY_FIELD: "X",
            fd.KEY_OP: "gt",
            fd.KEY_VALUE: 10,
            fd.KEY_TYPE: fd.TYPE_QUALITY,
        }

    def test_value_field(self):
        result = fd._parse_cli_filter("name=cmp:field=REF:op=ne:value_field=ALT:type=label")
        assert result[fd.KEY_VALUE_FIELD] == "ALT"
        assert fd.KEY_VALUE not in result

    def test_too_few_parts_raises(self):
        with pytest.raises(ValueError, match="must have at least"):
            fd._parse_cli_filter("a:b:c")

    def test_missing_eq_raises(self):
        with pytest.raises(ValueError, match="key=value format"):
            fd._parse_cli_filter("name=f:field=X:opgt:value=10:type=quality")


class TestParseCliDownsample:
    def test_random_with_seed(self):
        result = fd._parse_cli_downsample("random:1000:42")
        assert result == {fd.KEY_METHOD: "random", fd.KEY_SIZE: 1000, fd.KEY_SEED: 42}

    def test_head_no_seed(self):
        result = fd._parse_cli_downsample("head:500")
        assert result == {fd.KEY_METHOD: "head", fd.KEY_SIZE: 500}

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            fd._parse_cli_downsample("random:abc")

    def test_too_many_parts_raises(self):
        with pytest.raises(ValueError, match="must have 2-3 parts"):
            fd._parse_cli_downsample("random:100:42:extra")

    def test_invalid_seed_raises(self):
        with pytest.raises(ValueError, match="seed must be an integer"):
            fd._parse_cli_downsample("random:100:not_a_number")

    def test_too_few_parts_raises(self):
        with pytest.raises(ValueError, match="must have 2-3 parts"):
            fd._parse_cli_downsample("random")


# ────────────────────────────────────────────────────────


class TestCreateDownsampleColumn:
    """Test downsample column creation for head and random methods."""

    def test_no_downsample_returns_none(self):
        lf = pl.DataFrame({"score": [10, 20, 30]}).lazy()
        cfg = {fd.KEY_FILTERS: []}
        result_lf, ds_col = fd._create_downsample_column(lf, [], cfg)
        assert ds_col is None

    def test_head_downsample(self):
        lf = pl.DataFrame({"score": list(range(10))}).lazy()
        filters = [
            {
                fd.KEY_NAME: "pass_all",
                fd.KEY_FIELD: "score",
                fd.KEY_OP: "ge",
                fd.KEY_VALUE: 0,
                fd.KEY_TYPE: fd.TYPE_QUALITY,
            },
        ]
        lf, filter_cols = fd._create_filter_columns(lf, filters)
        cfg = {fd.KEY_FILTERS: filters, fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 5, fd.KEY_METHOD: fd.METHOD_HEAD}}

        result_lf, ds_col = fd._create_downsample_column(lf, filter_cols, cfg)
        assert ds_col == fd.COL_FILTER_DOWNSAMPLE

        collected = result_lf.collect()
        ds_values = collected[ds_col].to_list()
        # First 5 should be True, rest False
        assert sum(v is True for v in ds_values) == 5
        assert sum(v is False for v in ds_values) == 5

    def test_random_downsample(self):
        lf = pl.DataFrame({"score": list(range(20))}).lazy()
        filters = [
            {
                fd.KEY_NAME: "pass_all",
                fd.KEY_FIELD: "score",
                fd.KEY_OP: "ge",
                fd.KEY_VALUE: 0,
                fd.KEY_TYPE: fd.TYPE_QUALITY,
            },
        ]
        lf, filter_cols = fd._create_filter_columns(lf, filters)
        cfg = {
            fd.KEY_FILTERS: filters,
            fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 10, fd.KEY_METHOD: fd.METHOD_RANDOM, fd.KEY_SEED: 42},
        }

        result_lf, ds_col = fd._create_downsample_column(lf, filter_cols, cfg)
        assert ds_col == fd.COL_FILTER_DOWNSAMPLE

        collected = result_lf.collect()
        ds_values = collected[ds_col].to_list()
        # Exactly 10 should be True
        assert sum(v is True for v in ds_values) == 10

    def test_downsample_with_some_rows_filtered(self):
        """Only rows passing all filters should participate in downsample."""
        lf = pl.DataFrame({"score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}).lazy()
        filters = [
            {
                fd.KEY_NAME: "high",
                fd.KEY_FIELD: "score",
                fd.KEY_OP: "gt",
                fd.KEY_VALUE: 5,
                fd.KEY_TYPE: fd.TYPE_QUALITY,
            },
        ]
        lf, filter_cols = fd._create_filter_columns(lf, filters)
        cfg = {fd.KEY_FILTERS: filters, fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 3, fd.KEY_METHOD: fd.METHOD_HEAD}}

        result_lf, ds_col = fd._create_downsample_column(lf, filter_cols, cfg)
        collected = result_lf.collect()
        ds_values = collected[ds_col].to_list()
        # Only 5 rows pass the filter (score > 5: 6,7,8,9,10), head 3 of those
        assert sum(v is True for v in ds_values) == 3
        # First 5 rows don't pass the filter, so their ds value is None
        assert all(v is None for v in ds_values[:5])


# ────────────────────────────────────────────────────────


class TestCreateFinalFilterColumn:
    """Test final combined filter column creation."""

    def test_without_downsample(self):
        lf = pl.DataFrame(
            {
                "__filter_a": [True, False, True, False],
                "__filter_b": [True, True, False, False],
            }
        ).lazy()
        result_lf = fd._create_final_filter_column(lf, ["__filter_a", "__filter_b"], None)
        collected = result_lf.collect()
        assert collected[fd.COL_FILTER_FINAL].to_list() == [True, False, False, False]

    def test_with_downsample(self):
        lf = pl.DataFrame(
            {
                "__filter_a": [True, True, True, True],
                fd.COL_FILTER_DOWNSAMPLE: [True, False, True, None],
            }
        ).lazy()
        result_lf = fd._create_final_filter_column(lf, ["__filter_a"], fd.COL_FILTER_DOWNSAMPLE)
        collected = result_lf.collect()
        # filter_a=True & downsample: [True, False, True, False(null filled)]
        assert collected[fd.COL_FILTER_FINAL].to_list() == [True, False, True, False]

    def test_all_filters_pass(self):
        lf = pl.DataFrame(
            {
                "__filter_x": [True, True, True],
            }
        ).lazy()
        result_lf = fd._create_final_filter_column(lf, ["__filter_x"], None)
        collected = result_lf.collect()
        assert collected[fd.COL_FILTER_FINAL].to_list() == [True, True, True]


# ────────────────────────────────────────────────────────


class TestFilterParquet:
    """Integration tests for filter_parquet with actual parquet files."""

    def test_basic_filter(self, tmp_path):
        """Test basic filtering writes correct output."""
        # Create input parquet
        in_frame = pl.DataFrame({"score": [10, 20, 30, 40, 50], "name": ["a", "b", "c", "d", "e"]})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        # Create config
        cfg = {
            fd.KEY_FILTERS: [
                {
                    fd.KEY_NAME: "high_score",
                    fd.KEY_FIELD: "score",
                    fd.KEY_OP: "gt",
                    fd.KEY_VALUE: 25,
                    fd.KEY_TYPE: fd.TYPE_QUALITY,
                },
            ]
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        fd.filter_parquet(in_path, out_path, None, cfg_path, stats_path)

        # Check output
        result = pl.read_parquet(out_path)
        assert len(result) == 3  # rows with score > 25: 30, 40, 50
        assert list(result["score"]) == [30, 40, 50]

        # Check stats
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats[fd.KEY_FILTERS][0]["rows"] == 5
        assert stats[fd.KEY_FILTERS][1]["rows"] == 3

    def test_filter_with_downsample_head(self, tmp_path):
        """Test filtering with head downsample."""
        in_frame = pl.DataFrame({"val": list(range(100))})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        cfg = {
            fd.KEY_FILTERS: [
                {
                    fd.KEY_NAME: "all",
                    fd.KEY_FIELD: "val",
                    fd.KEY_OP: "ge",
                    fd.KEY_VALUE: 0,
                    fd.KEY_TYPE: fd.TYPE_QUALITY,
                },
            ],
            fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 10, fd.KEY_METHOD: fd.METHOD_HEAD},
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        fd.filter_parquet(in_path, out_path, None, cfg_path, stats_path)

        result = pl.read_parquet(out_path)
        assert len(result) == 10

    def test_filter_full_output(self, tmp_path):
        """Test that out_path_full contains all rows with filter columns."""
        in_frame = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        cfg = {
            fd.KEY_FILTERS: [
                {fd.KEY_NAME: "big", fd.KEY_FIELD: "x", fd.KEY_OP: "gt", fd.KEY_VALUE: 3, fd.KEY_TYPE: fd.TYPE_QUALITY},
            ]
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_full_path = str(tmp_path / "full.parquet")
        stats_path = str(tmp_path / "stats.json")

        fd.filter_parquet(in_path, None, out_full_path, cfg_path, stats_path)

        result = pl.read_parquet(out_full_path)
        # Should have all 5 rows
        assert len(result) == 5
        # Should have the filter column
        assert any(col.startswith(fd.COL_PREFIX_FILTER) for col in result.columns)

    def test_filter_with_cli_filters(self, tmp_path):
        """Test using CLI filter specifications."""
        in_frame = pl.DataFrame({"score": [10, 20, 30, 40, 50]})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        cli_filters = ["name=high:field=score:op=gt:value=30:type=quality"]
        fd.filter_parquet(in_path, out_path, None, None, stats_path, cli_filters=cli_filters)

        result = pl.read_parquet(out_path)
        assert len(result) == 2  # 40, 50


# ────────────────────────────────────────────────────────


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
        fd._validate_stats_dict(data, where=" (test)")

    def test_missing_key(self):
        data = {"filters": [{"name": "raw", "rows": 100}], "single_effect": {}}
        with pytest.raises(ValueError, match="missing key"):
            fd._validate_stats_dict(data, where="")

    def test_empty_filters(self):
        data = {"filters": [], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="non-empty list"):
            fd._validate_stats_dict(data, where="")

    def test_filters_not_list(self):
        data = {"filters": "not_a_list", "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="non-empty list"):
            fd._validate_stats_dict(data, where="")

    def test_filter_not_dict(self):
        data = {"filters": ["not_a_dict"], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="is not an object"):
            fd._validate_stats_dict(data, where="")

    def test_filter_missing_name_or_rows(self):
        data = {"filters": [{"name": "raw"}], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="missing 'name' or 'rows'"):
            fd._validate_stats_dict(data, where="")

    def test_filter_negative_rows(self):
        data = {"filters": [{"name": "raw", "rows": -1}], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="non-negative integer"):
            fd._validate_stats_dict(data, where="")

    def test_filter_rows_not_int(self):
        data = {"filters": [{"name": "raw", "rows": 1.5}], "single_effect": {}, "combinations": {}}
        with pytest.raises(ValueError, match="non-negative integer"):
            fd._validate_stats_dict(data, where="")

    def test_first_filter_not_raw(self):
        data = {
            "filters": [{"name": "not_raw", "rows": 100}],
            "single_effect": {},
            "combinations": {},
        }
        with pytest.raises(ValueError, match="first filter must have name 'raw'"):
            fd._validate_stats_dict(data, where="")

    def test_single_effect_not_dict(self):
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": "wrong",
            "combinations": {},
        }
        with pytest.raises(ValueError, match="'single_effect' must be an object"):
            fd._validate_stats_dict(data, where="")

    def test_single_effect_negative_value(self):
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": {"f1": -5},
            "combinations": {},
        }
        with pytest.raises(ValueError, match="non-negative integers"):
            fd._validate_stats_dict(data, where="")

    def test_combinations_negative_value(self):
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": {},
            "combinations": {"11": -1},
        }
        with pytest.raises(ValueError, match="non-negative integers"):
            fd._validate_stats_dict(data, where="")

    def test_combinations_not_dict_ignored(self):
        """If combinations is not a dict, validation passes (no crash)."""
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": {},
            "combinations": "skipped_if_not_dict",
        }
        # Should not raise - the condition short-circuits
        fd._validate_stats_dict(data, where="")


# ────────────────────────────────────────────────────────


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

        result = fd.read_filtering_stats_json(str(path))
        assert result == data

    def test_invalid_file_raises(self, tmp_path):
        data = {"filters": [], "single_effect": {}, "combinations": {}}
        path = tmp_path / "bad_stats.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="non-empty list"):
            fd.read_filtering_stats_json(str(path))


# ────────────────────────────────────────────────────────


class TestParseCliFilterAdditional:
    def test_missing_required_keys_raises(self):
        """Missing name/field/op/type raises."""
        with pytest.raises(ValueError, match="Missing required keys"):
            fd._parse_cli_filter("field=X:op=gt:value=10:type=quality")  # missing name

    def test_missing_value_and_value_field_raises(self):
        with pytest.raises(ValueError, match="Must specify either"):
            fd._parse_cli_filter("name=f:field=X:op=gt:type=quality")

    def test_in_op_parses_list(self):
        result = fd._parse_cli_filter("name=f:field=X:op=in:value=a,b,c:type=quality")
        assert result[fd.KEY_VALUE] == ["a", "b", "c"]


# ──────────────────────── additional _validate_downsample tests ──────────


class TestValidateDownsampleAdditional:
    """Additional validation tests for downsample configuration."""

    def test_downsample_not_dict_raises(self):
        cfg = {fd.KEY_FILTERS: [], fd.KEY_DOWNSAMPLE: "not_a_dict"}
        with pytest.raises(ValueError, match="must be a dictionary"):
            fd.validate_filter_config(cfg)

    def test_downsample_missing_size_raises(self):
        cfg = {fd.KEY_FILTERS: [], fd.KEY_DOWNSAMPLE: {fd.KEY_METHOD: fd.METHOD_RANDOM}}
        with pytest.raises(ValueError, match="must have 'size' key"):
            fd.validate_filter_config(cfg)

    def test_downsample_size_zero_raises(self):
        cfg = {fd.KEY_FILTERS: [], fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 0}}
        with pytest.raises(ValueError, match="positive integer"):
            fd.validate_filter_config(cfg)

    def test_downsample_size_string_raises(self):
        cfg = {fd.KEY_FILTERS: [], fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: "100"}}
        with pytest.raises(ValueError, match="positive integer"):
            fd.validate_filter_config(cfg)

    def test_downsample_head_method_valid(self):
        cfg = {
            fd.KEY_FILTERS: [{fd.KEY_FIELD: "X", fd.KEY_OP: "gt", fd.KEY_VALUE: 1, fd.KEY_TYPE: fd.TYPE_QUALITY}],
            fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 100, fd.KEY_METHOD: fd.METHOD_HEAD},
        }
        fd.validate_filter_config(cfg)  # Should not raise


# ──────────────────────── additional _try_to_convert tests ──────────────


class TestTryToConvertAdditional:
    """Additional tests for value conversion edge cases."""

    def test_negative_int(self):
        assert fd._parse_value_based_on_operation("-5", "eq") == -5

    def test_negative_float(self):
        assert fd._parse_value_based_on_operation("-1.5", "lt") == pytest.approx(-1.5)

    def test_empty_string(self):
        assert fd._parse_value_based_on_operation("", "eq") == ""

    def test_mixed_list(self):
        """In/not_in with mixed types in list."""
        result = fd._parse_value_based_on_operation("1,hello,3.14,true", "in")
        assert result == [1, "hello", pytest.approx(3.14), True]

    def test_false_variants(self):
        assert fd._parse_value_based_on_operation("FALSE", "eq") is False


# ──────────────────────── combination stats skip ──────────────────────────


class TestCalculateStatisticsAdditional:
    """Tests for skipping combination stats when too many filters."""

    def test_many_filters_skips_combinations(self):
        """When >MAX_COMBINATION_FILTERS filters, combinations should be empty."""
        from ugbio_featuremap.filter_dataframe import MAX_COMBINATION_FILTERS

        lf = pl.DataFrame({"score": list(range(10))}).lazy()

        # Create more than MAX_COMBINATION_FILTERS filters
        filters = []
        for i in range(MAX_COMBINATION_FILTERS + 1):
            filters.append(
                {
                    fd.KEY_NAME: f"f{i}",
                    fd.KEY_FIELD: "score",
                    fd.KEY_OP: "ge",
                    fd.KEY_VALUE: 0,
                    fd.KEY_TYPE: fd.TYPE_QUALITY,
                }
            )

        lf, filter_cols = fd._create_filter_columns(lf, filters)
        cfg = {fd.KEY_FILTERS: filters}

        stats = fd._calculate_statistics(lf, filter_cols, None, filters, total_rows=10, cfg=cfg)

        # With >MAX_COMBINATION_FILTERS filters, combinations should be empty dict
        assert stats["combinations"] == {}

    def test_zero_filters_empty_combinations(self):
        """When no filters, combinations should be empty."""
        lf = pl.DataFrame({"score": [1, 2, 3]}).lazy()
        cfg = {fd.KEY_FILTERS: []}

        stats = fd._calculate_statistics(lf, [], None, [], total_rows=3, cfg=cfg)
        assert stats["combinations"] == {}


# ────────────────────────────────────────────────────────


class TestFilterParquetAdditional:
    """Additional integration tests for filter_parquet."""

    def test_filter_with_random_downsample(self, tmp_path):
        """Test filtering with random downsample and seed."""
        in_frame = pl.DataFrame({"val": list(range(100))})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        cfg = {
            fd.KEY_FILTERS: [
                {
                    fd.KEY_NAME: "all",
                    fd.KEY_FIELD: "val",
                    fd.KEY_OP: "ge",
                    fd.KEY_VALUE: 0,
                    fd.KEY_TYPE: fd.TYPE_QUALITY,
                },
            ],
            fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 25, fd.KEY_METHOD: fd.METHOD_RANDOM, fd.KEY_SEED: 42},
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        fd.filter_parquet(in_path, out_path, None, cfg_path, stats_path)

        result = pl.read_parquet(out_path)
        assert len(result) == 25

        # Verify determinism: same seed same output
        out_path2 = str(tmp_path / "output2.parquet")
        stats_path2 = str(tmp_path / "stats2.json")
        fd.filter_parquet(in_path, out_path2, None, cfg_path, stats_path2)
        result2 = pl.read_parquet(out_path2)
        assert result["val"].to_list() == result2["val"].to_list()

    def test_filter_multiple_filters(self, tmp_path):
        """Test with multiple filters applied together."""
        in_frame = pl.DataFrame(
            {
                "score": [10, 20, 30, 40, 50],
                "category": ["A", "B", "A", "B", "A"],
            }
        )
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        cfg = {
            fd.KEY_FILTERS: [
                {
                    fd.KEY_NAME: "high",
                    fd.KEY_FIELD: "score",
                    fd.KEY_OP: "gt",
                    fd.KEY_VALUE: 15,
                    fd.KEY_TYPE: fd.TYPE_QUALITY,
                },
                {
                    fd.KEY_NAME: "cat_a",
                    fd.KEY_FIELD: "category",
                    fd.KEY_OP: "eq",
                    fd.KEY_VALUE: "A",
                    fd.KEY_TYPE: fd.TYPE_LABEL,
                },
            ]
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        fd.filter_parquet(in_path, out_path, None, cfg_path, stats_path)

        result = pl.read_parquet(out_path)
        # score > 15 AND category == "A": rows with (30, A) and (50, A)
        assert len(result) == 2
        assert list(result["score"]) == [30, 50]

    def test_filter_cli_downsample(self, tmp_path):
        """Test CLI downsample overrides config."""
        in_frame = pl.DataFrame({"val": list(range(50))})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        cli_filters = ["name=all:field=val:op=ge:value=0:type=quality"]
        fd.filter_parquet(in_path, out_path, None, None, stats_path, cli_filters=cli_filters, cli_downsample="head:10")

        result = pl.read_parquet(out_path)
        assert len(result) == 10

    def test_filter_value_field_comparison(self, tmp_path):
        """Test filter comparing two columns with value_field."""
        in_frame = pl.DataFrame(
            {
                "ref": ["A", "C", "G", "T", "A"],
                "alt": ["A", "G", "G", "C", "T"],
            }
        )
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        cfg = {
            fd.KEY_FILTERS: [
                {
                    fd.KEY_NAME: "ref_ne_alt",
                    fd.KEY_FIELD: "ref",
                    fd.KEY_OP: "ne",
                    fd.KEY_VALUE_FIELD: "alt",
                    fd.KEY_TYPE: fd.TYPE_LABEL,
                },
            ]
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_path = str(tmp_path / "output.parquet")
        stats_path = str(tmp_path / "stats.json")

        fd.filter_parquet(in_path, out_path, None, cfg_path, stats_path)

        result = pl.read_parquet(out_path)
        # ref != alt: rows 1(C!=G), 3(T!=C), 4(A!=T) = 3 rows
        assert len(result) == 3

    def test_both_outputs(self, tmp_path):
        """Test writing both filtered and full outputs."""
        in_frame = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        in_path = str(tmp_path / "input.parquet")
        in_frame.write_parquet(in_path)

        cfg = {
            fd.KEY_FILTERS: [
                {fd.KEY_NAME: "big", fd.KEY_FIELD: "x", fd.KEY_OP: "gt", fd.KEY_VALUE: 3, fd.KEY_TYPE: fd.TYPE_QUALITY},
            ]
        }
        cfg_path = str(tmp_path / "config.json")
        Path(cfg_path).write_text(json.dumps(cfg))

        out_path = str(tmp_path / "output.parquet")
        out_full_path = str(tmp_path / "full.parquet")
        stats_path = str(tmp_path / "stats.json")

        fd.filter_parquet(in_path, out_path, out_full_path, cfg_path, stats_path)

        # Filtered output: only rows passing filter
        filtered = pl.read_parquet(out_path)
        assert len(filtered) == 2

        # Full output: all rows with filter columns
        full = pl.read_parquet(out_full_path)
        assert len(full) == 5
        assert any(col.startswith(fd.COL_PREFIX_FILTER) for col in full.columns)


# ────────────────────────────────────────────────────────


class TestValidateStatsDictAdditional:
    """Additional edge case tests for stats validation."""

    def test_single_effect_float_value_fails(self):
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": {"f1": 5.5},
            "combinations": {},
        }
        with pytest.raises(ValueError, match="non-negative integers"):
            fd._validate_stats_dict(data, where="")

    def test_valid_empty_combinations(self):
        """Empty combinations dict should be valid."""
        data = {
            "filters": [{"name": "raw", "rows": 100}],
            "single_effect": {},
            "combinations": {},
        }
        fd._validate_stats_dict(data, where=" (test)")  # Should not raise

    def test_valid_multiple_filters(self):
        """Stats with multiple valid filters."""
        data = {
            "filters": [
                {"name": "raw", "rows": 1000, "type": "raw"},
                {"name": "coverage", "rows": 900, "type": "region"},
                {"name": "qual", "rows": 700, "type": "quality"},
                {"name": "label", "rows": 500, "type": "label"},
            ],
            "single_effect": {"coverage": 900, "qual": 700, "label": 500},
            "combinations": {"111": 500, "110": 200, "100": 100, "000": 100, "101": 50, "010": 30, "011": 20},
        }
        fd._validate_stats_dict(data, where="")  # Should not raise
