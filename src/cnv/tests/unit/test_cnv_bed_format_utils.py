import pandas as pd
import pytest
from ugbio_cnv.cnv_bed_format_utils import process_filter_columns


class TestProcessFilterColumns:
    """Tests for process_filter_columns function."""

    # Define test registries for use across all test methods
    test_filter_columns_registry = ["jalign_filter", "LCR_label_value"]
    test_filter_tag_registry = {
        "LowQual": ("LowQual", None, None, "Low quality filter", "FILTER"),
        "HighCoverage": ("HighCoverage", None, None, "High coverage filter", "FILTER"),
        "LCR": ("LCR", None, None, "Low complexity region", "FILTER"),
    }

    def test_process_filter_columns_single_filter(self):
        """Test processing a single filter value."""
        row = pd.Series({"jalign_filter": "LowQual", "LCR_label_value": "PASS"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "LowQual"

    def test_process_filter_columns_multiple_filters(self):
        """Test processing multiple filter values from different columns."""
        row = pd.Series({"jalign_filter": "LowQual", "LCR_label_value": "HighCoverage"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        # Should be sorted and comma-separated
        assert result == "HighCoverage,LowQual"

    def test_process_filter_columns_pipe_separated(self):
        """Test processing pipe-separated filter values in a single column."""
        row = pd.Series({"jalign_filter": "LowQual|HighCoverage", "LCR_label_value": "PASS"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "HighCoverage,LowQual"

    def test_process_filter_columns_semicolon_separated(self):
        """Test processing semicolon-separated filter values in a single column."""
        row = pd.Series({"jalign_filter": "LowQual;HighCoverage", "LCR_label_value": "PASS"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "HighCoverage,LowQual"

    def test_process_filter_columns_pass_only(self):
        """Test that PASS values are filtered out and result is PASS."""
        row = pd.Series({"jalign_filter": "PASS", "LCR_label_value": "PASS"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "PASS"

    def test_process_filter_columns_duplicates_removed(self):
        """Test that duplicate filter values are removed."""
        row = pd.Series({"jalign_filter": "LowQual|LowQual", "LCR_label_value": "LowQual"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "LowQual"

    def test_process_filter_columns_unknown_filter_raises_error(self):
        """Test that unknown filter values raise a ValueError."""
        row = pd.Series({"jalign_filter": "UnknownFilter", "LCR_label_value": "PASS"})
        with pytest.raises(ValueError, match="Unknown filter values found"):
            process_filter_columns(
                row,
                filter_columns_registry=self.test_filter_columns_registry,
                filter_tags_registry=self.test_filter_tag_registry,
            )
