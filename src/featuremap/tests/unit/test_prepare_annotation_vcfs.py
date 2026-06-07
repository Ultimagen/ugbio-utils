"""Unit tests for prepare_annotation_vcfs module.

Covers:
- CLI argument parsing (_parse_args)
- _update_coverage_threshold
- _inject_exclusion_filter
- _create_inference_filters
- _build_inference_fields
- run() integration with file I/O
"""

import json
from pathlib import Path

from ugbio_featuremap.prepare_annotation_vcfs import (
    _build_inference_fields,
    _create_inference_filters,
    _inject_exclusion_filter,
    _parse_args,
    _update_coverage_threshold,
    run,
)

# ──────────────────────── _parse_args ──────────────────────────────────


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_minimal_args(self):
        args = _parse_args([])
        assert args.exclude_field is None
        assert args.include_field is None
        assert args.pcawg_field is None
        assert args.read_filters is None
        assert args.coverage_threshold is None
        assert args.output_dir == "."

    def test_all_args(self):
        args = _parse_args(
            [
                "--exclude-field",
                "EXCLUDE_TRAINING",
                "--include-field",
                "INCLUDE_INFERENCE",
                "--pcawg-field",
                "PCAWG",
                "--read-filters",
                "/path/to/filters.json",
                "--coverage-threshold",
                "42",
                "--output-dir",
                "/output",
            ]
        )
        assert args.exclude_field == "EXCLUDE_TRAINING"
        assert args.include_field == "INCLUDE_INFERENCE"
        assert args.pcawg_field == "PCAWG"
        assert args.read_filters == "/path/to/filters.json"
        assert args.coverage_threshold == 42
        assert args.output_dir == "/output"

    def test_coverage_threshold_is_int(self):
        args = _parse_args(["--coverage-threshold", "100"])
        assert isinstance(args.coverage_threshold, int)
        assert args.coverage_threshold == 100


# ──────────────────────── _update_coverage_threshold ──────────────────


class TestUpdateCoverageThreshold:
    """Test coverage threshold update in filter JSONs."""

    def test_updates_both_filter_sets(self):
        filters_json = {
            "filters_full_output": [
                {"name": "coverage_le_max", "value": 100},
                {"name": "other_filter", "value": 50},
            ],
            "filters_random_sample": [
                {"name": "coverage_le_max", "value": 100},
                {"name": "another_filter", "value": 30},
            ],
        }
        result = _update_coverage_threshold(filters_json, 42)
        assert result["filters_full_output"][0]["value"] == 42
        assert result["filters_random_sample"][0]["value"] == 42
        # Other filters unchanged
        assert result["filters_full_output"][1]["value"] == 50
        assert result["filters_random_sample"][1]["value"] == 30

    def test_no_matching_filter_name(self):
        """If coverage_le_max doesn't exist, nothing changes."""
        filters_json = {
            "filters_full_output": [
                {"name": "other_filter", "value": 50},
            ],
            "filters_random_sample": [
                {"name": "another_filter", "value": 30},
            ],
        }
        result = _update_coverage_threshold(filters_json, 42)
        assert result["filters_full_output"][0]["value"] == 50
        assert result["filters_random_sample"][0]["value"] == 30

    def test_missing_filter_set_keys(self):
        """If filter sets are missing, no error."""
        filters_json = {"some_other_key": []}
        result = _update_coverage_threshold(filters_json, 42)
        assert result == {"some_other_key": []}


# ──────────────────────── _inject_exclusion_filter ──────────────────


class TestInjectExclusionFilter:
    """Test exclusion filter injection."""

    def test_injects_into_both_sets(self):
        filters_json = {
            "filters_full_output": [{"name": "existing"}],
            "filters_random_sample": [{"name": "existing"}],
        }
        result = _inject_exclusion_filter(filters_json, "EXCLUDE_TRAINING")

        # Should append to both lists
        assert len(result["filters_full_output"]) == 2
        assert len(result["filters_random_sample"]) == 2

        injected = result["filters_full_output"][1]
        assert injected["name"] == "not_in_EXCLUDE_TRAINING"
        assert injected["type"] == "region"
        assert injected["field"] == "EXCLUDE_TRAINING"
        assert injected["op"] == "is_null"

    def test_missing_filter_sets(self):
        """If filter sets are missing, no error."""
        filters_json = {}
        result = _inject_exclusion_filter(filters_json, "MY_FIELD")
        assert result == {}

    def test_inject_pcawg(self):
        filters_json = {
            "filters_full_output": [],
            "filters_random_sample": [],
        }
        result = _inject_exclusion_filter(filters_json, "PCAWG")
        assert result["filters_full_output"][0]["name"] == "not_in_PCAWG"
        assert result["filters_full_output"][0]["field"] == "PCAWG"


# ──────────────────────── _create_inference_filters ──────────────────


class TestCreateInferenceFilters:
    """Test inference filter JSON generation."""

    def test_creates_file(self, tmp_path):
        output_path = str(tmp_path / "inference_filters.json")
        _create_inference_filters(["INCLUDE_INFERENCE", "PCAWG"], output_path)

        assert Path(output_path).exists()
        content = json.loads(Path(output_path).read_text())
        assert "filters_inference" in content
        assert len(content["filters_inference"]) == 1

        filt = content["filters_inference"][0]
        assert filt["name"] == "in_inference_set"
        assert filt["type"] == "region"
        assert filt["op"] == "any_not_null"
        assert filt["fields"] == ["INCLUDE_INFERENCE", "PCAWG"]

    def test_single_field(self, tmp_path):
        output_path = str(tmp_path / "inf.json")
        _create_inference_filters(["MY_FIELD"], output_path)

        content = json.loads(Path(output_path).read_text())
        assert content["filters_inference"][0]["fields"] == ["MY_FIELD"]


# ──────────────────────── _build_inference_fields ──────────────────────


class TestBuildInferenceFields:
    """Test inference fields list construction."""

    def test_both_fields(self):
        result = _build_inference_fields("INCLUDE", "PCAWG")
        assert result == ["INCLUDE", "PCAWG"]

    def test_include_only(self):
        result = _build_inference_fields("INCLUDE", None)
        assert result == ["INCLUDE"]

    def test_pcawg_only_no_include(self):
        """PCAWG alone without include_field yields empty list."""
        result = _build_inference_fields(None, "PCAWG")
        assert result == []

    def test_neither(self):
        result = _build_inference_fields(None, None)
        assert result == []


# ──────────────────────── run() integration ──────────────────────────


class TestRun:
    """Test the main run() function end-to-end."""

    def test_full_run(self, tmp_path):
        """Test run with all arguments provided."""
        # Create input read_filters JSON
        read_filters = {
            "filters_full_output": [
                {"name": "coverage_le_max", "value": 100},
                {"name": "some_filter", "value": 10},
            ],
            "filters_random_sample": [
                {"name": "coverage_le_max", "value": 100},
            ],
        }
        input_json = tmp_path / "read_filters.json"
        input_json.write_text(json.dumps(read_filters))

        output_dir = tmp_path / "output"

        run(
            [
                "--exclude-field",
                "EXCLUDE_TRAINING",
                "--include-field",
                "INCLUDE_INFERENCE",
                "--pcawg-field",
                "PCAWG",
                "--read-filters",
                str(input_json),
                "--coverage-threshold",
                "42",
                "--output-dir",
                str(output_dir),
            ]
        )

        # Check augmented read_filters output
        augmented_path = output_dir / "read_filters_with_max_coverage.json"
        assert augmented_path.exists()
        augmented = json.loads(augmented_path.read_text())
        assert augmented["filters_full_output"][0]["value"] == 42
        # Should have exclusion filters appended
        exclusion_names = [f["name"] for f in augmented["filters_full_output"] if "not_in" in f["name"]]
        assert "not_in_EXCLUDE_TRAINING" in exclusion_names
        assert "not_in_PCAWG" in exclusion_names

        # Check inference_filters output
        inference_path = output_dir / "inference_filters.json"
        assert inference_path.exists()
        inference = json.loads(inference_path.read_text())
        assert inference["filters_inference"][0]["fields"] == ["INCLUDE_INFERENCE", "PCAWG"]

    def test_no_read_filters(self, tmp_path):
        """Test run without read_filters - only inference output."""
        output_dir = tmp_path / "output"
        run(
            [
                "--include-field",
                "INCLUDE_INFERENCE",
                "--output-dir",
                str(output_dir),
            ]
        )

        # No augmented read_filters since none provided
        augmented_path = output_dir / "read_filters_with_max_coverage.json"
        assert not augmented_path.exists()

        # Inference filters should still be created
        inference_path = output_dir / "inference_filters.json"
        assert inference_path.exists()

    def test_no_include_field(self, tmp_path):
        """Test run without include_field - no inference output."""
        read_filters = {
            "filters_full_output": [{"name": "coverage_le_max", "value": 100}],
            "filters_random_sample": [{"name": "coverage_le_max", "value": 100}],
        }
        input_json = tmp_path / "read_filters.json"
        input_json.write_text(json.dumps(read_filters))

        output_dir = tmp_path / "output"
        run(
            [
                "--exclude-field",
                "EXCLUDE_TRAINING",
                "--read-filters",
                str(input_json),
                "--output-dir",
                str(output_dir),
            ]
        )

        # Augmented read_filters should exist
        augmented_path = output_dir / "read_filters_with_max_coverage.json"
        assert augmented_path.exists()

        # No inference filters since include_field is None
        inference_path = output_dir / "inference_filters.json"
        assert not inference_path.exists()

    def test_empty_run(self, tmp_path):
        """Test run with no arguments - nothing should be produced."""
        output_dir = tmp_path / "output"
        run(["--output-dir", str(output_dir)])
        # Output dir created but empty (except the dir itself)
        assert output_dir.exists()
        assert len(list(output_dir.iterdir())) == 0

    def test_output_dir_created(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nested" / "dir" / "output"
        run(
            [
                "--include-field",
                "MY_FIELD",
                "--output-dir",
                str(output_dir),
            ]
        )
        assert output_dir.exists()
