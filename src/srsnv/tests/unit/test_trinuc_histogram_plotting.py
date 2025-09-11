"""Unit tests for trinucleotide histogram plotting functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from ugbio_srsnv.trinuc_histogram_plotting import (
    calc_and_plot_trinuc_hist,
    calc_trinuc_stats,
    plot_trinuc_hist,
    plot_trinuc_hist_and_qual_panels,
    plot_trinuc_hist_panels,
    plot_trinuc_qual,
    reverse_engineer_hist_stats,
)


class TestTrinucHistogramPlotting:
    """Test cases for trinucleotide histogram plotting functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Fix random seed for reproducibility
        rng = np.random.default_rng(42)

        trinuc_contexts = [
            "AAAC",
            "AAAG",
            "AAAT",
            "AACG",
            "AACT",
            "AAGT",
            "ACAC",
            "ACAG",
            "ACAT",
            "ACCG",
            "ACCT",
            "ACGT",
            "AGAC",
            "AGAG",
            "AGAT",
            "AGCG",
            "AGCT",
            "AGGT",
            "ATAC",
            "ATAG",
            "ATAT",
            "ATCG",
            "ATCT",
            "ATGT",
        ] * 50  # Repeat to get more data

        data = {
            "tcwa_fwd": rng.choice(trinuc_contexts, 1000),
            "label": rng.choice([False, True], 1000, p=[0.8, 0.2]),
            "is_forward": rng.choice([True, False], 1000),
            "SNVQ": rng.normal(35, 12, 1000).clip(5, 70),
            "is_mixed": rng.choice([True, False], 1000, p=[0.7, 0.3]),
            "is_cycle_skip": rng.choice([True, False], 1000, p=[0.3, 0.7]),
        }

        return pd.DataFrame(data)

    # =====================================
    # FAST TESTS: Data processing logic only
    # =====================================

    def test_calc_trinuc_stats_basic_structure(self, sample_data):
        """Test that calc_trinuc_stats returns correct data structure."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)

        # Check basic structure
        assert isinstance(stats_df, pd.DataFrame)
        assert stats_df.shape[0] == 96  # TRINUC_FORWARD_COUNT for collapsed mode
        assert len([col for col in stats_df.columns if ")" in col]) == 2  # Two label columns

    def test_calc_trinuc_stats_with_quality_columns(self, sample_data):
        """Test that quality columns are created when requested."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        # Check quality columns exist
        qual_cols = [col for col in stats_df.columns if "qual" in col]
        assert len(qual_cols) > 0

        # Check for mixed categories
        mixed_all_cols = [col for col in qual_cols if "mixed=all" in col]
        mixed_true_cols = [col for col in qual_cols if "mixed=True" in col]
        mixed_false_cols = [col for col in qual_cols if "mixed=False" in col]

        assert len(mixed_all_cols) > 0
        assert len(mixed_true_cols) > 0
        assert len(mixed_false_cols) > 0

    def test_calc_trinuc_stats_collapsed_vs_full_shapes(self, sample_data):
        """Test that collapsed and full modes return expected shapes."""
        stats_collapsed = calc_trinuc_stats(sample_data, labels=[False, True], collapsed=True)
        stats_full = calc_trinuc_stats(sample_data, labels=[False, True], collapsed=False)

        assert stats_collapsed.shape[0] == 96
        assert stats_full.shape[0] == 192
        assert stats_collapsed.shape[1] == stats_full.shape[1]  # Same number of columns

    def test_reverse_engineer_hist_stats_data_integrity(self, sample_data):
        """Test the reverse engineering function preserves data integrity."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)

        hist_cols = [col for col in stats_df.columns if ")" in col]
        hist_stats_df = stats_df[hist_cols]

        labels, hist_stats, total_snvs = reverse_engineer_hist_stats(hist_stats_df)

        assert len(labels) == 2
        assert len(hist_stats) == 2
        assert len(total_snvs) == 2
        assert all(isinstance(count, int) for count in total_snvs.values())

    def test_motif_orientation_parameter_validation(self, sample_data):
        """Test motif_orientation parameter validation."""
        # Valid values should work
        for orientation in ["seq_dir", "ref_dir", "fwd_only"]:
            stats_df = calc_trinuc_stats(
                sample_data, labels=[False, True], motif_orientation=orientation, collapsed=True
            )
            assert isinstance(stats_df, pd.DataFrame)
            assert stats_df.shape[0] == 96

        # Invalid value should raise error
        with pytest.raises(ValueError, match="motif_orientation"):
            calc_trinuc_stats(sample_data, labels=[False, True], motif_orientation="invalid_value", collapsed=True)

    def test_error_handling_empty_data(self):
        """Test error handling with empty data."""
        empty_df = pd.DataFrame(
            {
                "tcwa_fwd": pd.Series([], dtype="object"),
                "label": pd.Series([], dtype="object"),
                "is_forward": pd.Series([], dtype="bool"),
                "SNVQ": pd.Series([], dtype="float64"),
                "is_mixed": pd.Series([], dtype="bool"),
            }
        )

        # Should not crash, but return proper structure
        stats_df = calc_trinuc_stats(
            empty_df,
            labels=[False, True],
            include_quality=True,
            collapsed=True,
            motif_orientation="fwd_only",  # Avoid string processing on empty data
        )

        assert isinstance(stats_df, pd.DataFrame)
        assert stats_df.shape[0] == 96  # Should still have proper index

    # =====================================
    # PLOTTING TESTS: Mock matplotlib operations
    # =====================================

    @patch("matplotlib.pyplot.subplots")
    def test_plot_trinuc_hist_mock_rendering(self, mock_subplots, sample_data):
        """Test plotting logic without actual matplotlib rendering."""
        # Mock matplotlib objects
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Prepare data
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)
        labels, hist_stats, _ = reverse_engineer_hist_stats(stats_df[[col for col in stats_df.columns if ")" in col]])

        # Test the function - should not raise errors
        returned_ax, bars = plot_trinuc_hist(hist_stats, labels=labels, panel_num=0, ax=mock_ax)

        # Verify mock was called and basic return values
        assert returned_ax is mock_ax
        assert isinstance(bars, dict)
        assert len(bars) == len(labels)

    @patch("ugbio_srsnv.trinuc_histogram_plotting.plt")
    def test_plot_trinuc_qual_mock_rendering(self, mock_plt, sample_data):
        """Test quality plotting logic without actual rendering."""
        mock_ax = MagicMock()

        # Mock all the ax methods that might be called
        mock_line = MagicMock()
        mock_ax.step.return_value = [mock_line]
        mock_ax.fill_between.return_value = MagicMock()
        mock_ax.get_ylim.return_value = (0, 50)  # Mock y-limits
        mock_ax.get_xticklabels.return_value = [MagicMock() for _ in range(96)]

        # Don't need to patch subplots since we're providing ax directly
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        returned_ax, lines = plot_trinuc_qual(stats_df, panel_num=0, ax=mock_ax)

        assert returned_ax is mock_ax
        assert isinstance(lines, dict)

    # =====================================
    # INTEGRATION TESTS: End-to-end functionality
    # =====================================

    def test_calc_and_plot_trinuc_hist_with_quality_integration(self, sample_data):
        """Test the main calc_and_plot function with quality (minimal integration test)."""
        # Use pytest's tmp_path fixture would be better, but for now just test the function works
        fig, stats_df = calc_and_plot_trinuc_hist(
            sample_data,
            trinuc_col="tcwa_fwd",
            labels=[False, True],
            collapsed=True,
            include_quality=True,
            suptitle="Test Integration",
        )

        # Verify basic functionality without detailed plot inspection
        assert fig is not None  # Figure was created
        assert isinstance(stats_df, pd.DataFrame)
        assert len(fig.axes) == 2  # Quality + histogram panels

        # Clean up
        fig.clear()

    def test_calc_and_plot_trinuc_hist_histogram_only_integration(self, sample_data):
        """Test the main calc_and_plot function without quality."""
        fig, stats_df = calc_and_plot_trinuc_hist(
            sample_data,
            trinuc_col="tcwa_fwd",
            labels=[False, True],
            collapsed=True,
            include_quality=False,
            suptitle="Test Histogram Only",
        )

        assert fig is not None
        assert isinstance(stats_df, pd.DataFrame)
        assert len(fig.axes) == 1  # Only histogram panel

        # Clean up
        fig.clear()

    def test_plot_trinuc_hist_panels_integration(self, sample_data):
        """Test histogram-only panels function."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)
        hist_cols = [col for col in stats_df.columns if ")" in col]
        hist_stats_df = stats_df[hist_cols]

        fig = plot_trinuc_hist_panels(hist_stats_df, collapsed=True, suptitle="Test Histogram Panels")

        assert len(fig.axes) == 1  # Only one panel for collapsed mode

        # Clean up
        fig.clear()

    # =====================================
    # PARAMETER TESTING: Focused tests for specific parameter handling
    # =====================================

    @pytest.mark.parametrize("collapsed", [True, False])
    @pytest.mark.parametrize("include_quality", [True, False])
    def test_calc_and_plot_parameter_combinations(self, sample_data, collapsed, include_quality):
        """Test different parameter combinations work."""
        fig, stats_df = calc_and_plot_trinuc_hist(
            sample_data,
            trinuc_col="tcwa_fwd",
            labels=[False, True],
            collapsed=collapsed,
            include_quality=include_quality,
        )

        expected_panels = (2 if include_quality else 1) * (1 if collapsed else 2)
        assert len(fig.axes) == expected_panels
        assert isinstance(stats_df, pd.DataFrame)

        # Clean up
        fig.clear()

    @pytest.mark.parametrize("motif_orientation", ["seq_dir", "ref_dir", "fwd_only"])
    def test_motif_orientation_integration(self, sample_data, motif_orientation):
        """Test different motif orientations in integration."""
        fig, stats_df = calc_and_plot_trinuc_hist(
            sample_data,
            trinuc_col="tcwa_fwd",
            labels=[False, True],
            collapsed=True,
            include_quality=True,
            motif_orientation=motif_orientation,
        )

        assert fig is not None
        assert isinstance(stats_df, pd.DataFrame)

        # Clean up
        fig.clear()

    def test_height_ratio_and_layout_parameters(self, sample_data):
        """Test layout parameters work without errors."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        # Test different parameter combinations
        fig1 = plot_trinuc_hist_and_qual_panels(
            stats_df, collapsed=True, hist_to_qual_height_ratio=2.0, bottom_scale=1.0
        )
        fig2 = plot_trinuc_hist_and_qual_panels(
            stats_df, collapsed=True, hist_to_qual_height_ratio=3.0, bottom_scale=1.5
        )

        # Both should succeed without error
        assert len(fig1.axes) == 2
        assert len(fig2.axes) == 2

        # Clean up
        fig1.clear()
        fig2.clear()

    def test_legend_labels_tp_fp_content(self, sample_data):
        """Test that legend labels are properly formatted."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)
        fig = plot_trinuc_hist_and_qual_panels(stats_df, collapsed=True)

        # Check that legends exist
        legends = fig.legends
        assert len(legends) > 0

        # Check that at least one legend contains TP/FP labels
        legend_texts = []
        for legend in legends:
            legend_texts.extend([text.get_text() for text in legend.get_texts()])

        has_tp_fp = any("TP" in text or "FP" in text for text in legend_texts)
        assert has_tp_fp, f"Expected TP/FP in legend labels, got: {legend_texts}"

        # Clean up
        fig.clear()
