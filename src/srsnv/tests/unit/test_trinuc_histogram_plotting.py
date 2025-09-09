"""Unit tests for trinucleotide histogram plotting functionality."""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
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

    def test_calc_trinuc_stats_basic(self, sample_data):
        """Test basic functionality of calc_trinuc_stats."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)

        # Check basic structure
        assert isinstance(stats_df, pd.DataFrame)
        assert stats_df.shape[0] == 96  # TRINUC_FORWARD_COUNT for collapsed mode
        assert len([col for col in stats_df.columns if ")" in col]) == 2  # Two label columns

        # Check histogram columns exist
        hist_cols = [
            col
            for col in stats_df.columns
            if not col.endswith(("_median_qual", "_q1_qual", "_q2_qual")) and col != "is_cycle_skip"
        ]
        assert len(hist_cols) == 2  # False and True labels

    def test_calc_trinuc_stats_with_quality(self, sample_data):
        """Test calc_trinuc_stats with quality statistics."""
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

    def test_calc_trinuc_stats_collapsed_vs_full(self, sample_data):
        """Test difference between collapsed and full modes."""
        stats_collapsed = calc_trinuc_stats(sample_data, labels=[False, True], collapsed=True)

        stats_full = calc_trinuc_stats(sample_data, labels=[False, True], collapsed=False)

        assert stats_collapsed.shape[0] == 96
        assert stats_full.shape[0] == 192
        assert stats_collapsed.shape[1] == stats_full.shape[1]  # Same number of columns

    def test_plot_trinuc_hist_basic(self, sample_data):
        """Test basic histogram plotting functionality."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)

        # Reverse engineer to get hist_stats format
        labels, hist_stats, _ = reverse_engineer_hist_stats(stats_df[[col for col in stats_df.columns if ")" in col]])

        fig, ax = plt.subplots(figsize=(16, 3))
        returned_ax, bars = plot_trinuc_hist(hist_stats, labels=labels, panel_num=0, ax=ax)

        assert returned_ax is ax
        assert len(bars) == len(labels)
        plt.close(fig)

    def test_plot_trinuc_hist_annotations(self, sample_data):
        """Test annotation functionality in histogram plotting."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)

        labels, hist_stats, _ = reverse_engineer_hist_stats(stats_df[[col for col in stats_df.columns if ")" in col]])

        # Test with annotations
        fig1, ax1 = plt.subplots(figsize=(16, 3))
        plot_trinuc_hist(hist_stats, labels=labels, panel_num=0, ax=ax1, add_annotations=True)

        # Test without annotations
        fig2, ax2 = plt.subplots(figsize=(16, 3))
        plot_trinuc_hist(hist_stats, labels=labels, panel_num=0, ax=ax2, add_annotations=False)

        # Check that annotations were added in first case
        assert len(ax1.texts) > len(ax2.texts)

        plt.close(fig1)
        plt.close(fig2)

    def test_plot_trinuc_qual_basic(self, sample_data):
        """Test basic quality plotting functionality."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        fig, ax = plt.subplots(figsize=(16, 3))
        returned_ax, lines = plot_trinuc_qual(stats_df, panel_num=0, ax=ax)

        assert returned_ax is ax
        assert isinstance(lines, dict)
        assert len(lines) > 0  # Should have at least one quality line
        plt.close(fig)

    def test_plot_trinuc_qual_separators(self, sample_data):
        """Test that quality plots include separators."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        fig, ax = plt.subplots(figsize=(16, 3))
        plot_trinuc_qual(stats_df, panel_num=0, ax=ax)

        # Check that separator lines were added (5 separators expected)
        lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(lines) == 5

        plt.close(fig)

    def test_plot_trinuc_qual_annotations(self, sample_data):
        """Test annotation functionality in quality plotting."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        # Test with annotations
        fig1, ax1 = plt.subplots(figsize=(16, 3))
        plot_trinuc_qual(stats_df, panel_num=0, ax=ax1, add_annotations=True)

        # Test without annotations
        fig2, ax2 = plt.subplots(figsize=(16, 3))
        plot_trinuc_qual(stats_df, panel_num=0, ax=ax2, add_annotations=False)

        # Check that annotations were added in first case
        assert len(ax1.texts) > len(ax2.texts)

        plt.close(fig1)
        plt.close(fig2)

    def test_plot_trinuc_hist_and_qual_panels_collapsed(self, sample_data):
        """Test combined histogram and quality plotting in collapsed mode."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        fig = plot_trinuc_hist_and_qual_panels(stats_df, collapsed=True, suptitle="Test Collapsed Mode")

        # Should have 2 axes (quality + histogram)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_trinuc_hist_and_qual_panels_full(self, sample_data):
        """Test combined plotting in non-collapsed mode."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=False)

        fig = plot_trinuc_hist_and_qual_panels(stats_df, collapsed=False, suptitle="Test Full Mode")

        # Should have 4 axes (quality1 + hist1 + quality2 + hist2)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_height_ratio_parameter(self, sample_data):
        """Test configurable height ratio parameter."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        # Test default ratio
        fig1 = plot_trinuc_hist_and_qual_panels(stats_df, collapsed=True, hist_to_qual_height_ratio=2.0)

        # Test custom ratio
        fig2 = plot_trinuc_hist_and_qual_panels(stats_df, collapsed=True, hist_to_qual_height_ratio=3.0)

        # Both should succeed without error
        assert len(fig1.axes) == 2
        assert len(fig2.axes) == 2

        plt.close(fig1)
        plt.close(fig2)

    def test_bottom_scale_parameter(self, sample_data):
        """Test bottom_scale parameter for legend positioning."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        # Test with different bottom_scale values
        fig1 = plot_trinuc_hist_and_qual_panels(stats_df, collapsed=True, bottom_scale=1.0)

        fig2 = plot_trinuc_hist_and_qual_panels(stats_df, collapsed=True, bottom_scale=1.5)

        # Both should succeed without error
        assert len(fig1.axes) == 2
        assert len(fig2.axes) == 2

        plt.close(fig1)
        plt.close(fig2)

    def test_legend_labels_tp_fp(self, sample_data):
        """Test that legend labels show TP/FP instead of True/False."""
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

        plt.close(fig)

    def test_calc_and_plot_trinuc_hist_with_quality(self, sample_data):
        """Test the main calc_and_plot function with quality."""
        fig, stats_df = calc_and_plot_trinuc_hist(
            sample_data,
            trinuc_col="tcwa_fwd",
            labels=[False, True],
            collapsed=True,
            include_quality=True,
            suptitle="Test Integration",
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(stats_df, pd.DataFrame)
        assert len(fig.axes) == 2  # Quality + histogram panels

        plt.close(fig)

    def test_calc_and_plot_trinuc_hist_histogram_only(self, sample_data):
        """Test the main calc_and_plot function without quality."""
        fig, stats_df = calc_and_plot_trinuc_hist(
            sample_data,
            trinuc_col="tcwa_fwd",
            labels=[False, True],
            collapsed=True,
            include_quality=False,
            suptitle="Test Histogram Only",
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(stats_df, pd.DataFrame)
        assert len(fig.axes) == 1  # Only histogram panel

        plt.close(fig)

    def test_height_ratio_in_calc_and_plot(self, sample_data):
        """Test height ratio parameter in main function."""
        fig, _ = calc_and_plot_trinuc_hist(
            sample_data,
            trinuc_col="tcwa_fwd",
            labels=[False, True],
            collapsed=True,
            include_quality=True,
            hist_to_qual_height_ratio=3.0,
        )

        assert len(fig.axes) == 2
        plt.close(fig)

    def test_reverse_engineer_hist_stats(self, sample_data):
        """Test the reverse engineering function."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)

        hist_cols = [col for col in stats_df.columns if ")" in col]
        hist_stats_df = stats_df[hist_cols]

        labels, hist_stats, total_snvs = reverse_engineer_hist_stats(hist_stats_df)

        assert len(labels) == 2
        assert len(hist_stats) == 2
        assert len(total_snvs) == 2
        assert all(isinstance(count, int) for count in total_snvs.values())

    def test_plot_trinuc_hist_panels(self, sample_data):
        """Test histogram-only panels function."""
        stats_df = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=False, collapsed=True)

        hist_cols = [col for col in stats_df.columns if ")" in col]
        hist_stats_df = stats_df[hist_cols]

        fig = plot_trinuc_hist_panels(hist_stats_df, collapsed=True, suptitle="Test Histogram Panels")

        assert len(fig.axes) == 1  # Only one panel for collapsed mode
        plt.close(fig)

    def test_error_handling_empty_data(self):
        """Test error handling with empty data."""
        empty_df = pd.DataFrame(
            {
                "tcwa_fwd": pd.Series([], dtype="object"),  # Ensure proper dtype for string operations
                "label": pd.Series([], dtype="object"),
                "is_forward": pd.Series([], dtype="bool"),
                "SNVQ": pd.Series([], dtype="float64"),
                "is_mixed": pd.Series([], dtype="bool"),
            }
        )

        # Should not crash, but return empty or minimal stats
        # Use fwd_only to avoid string processing complications with empty data
        stats_df = calc_trinuc_stats(
            empty_df,
            labels=[False, True],
            include_quality=True,
            collapsed=True,
            motif_orientation="fwd_only",  # Avoid string processing on empty data
        )

        assert isinstance(stats_df, pd.DataFrame)
        assert stats_df.shape[0] == 96  # Should still have proper index

    def test_quality_mixed_category_detection(self, sample_data):
        """Test automatic detection of mixed categories in quality plotting."""
        # Test with mixed=True/False data
        stats_df_mixed = calc_trinuc_stats(sample_data, labels=[False, True], include_quality=True, collapsed=True)

        fig, ax = plt.subplots(figsize=(16, 3))
        _, lines_mixed = plot_trinuc_qual(stats_df_mixed, panel_num=0, ax=ax)

        # Should detect both mixed=True and mixed=False, so plot only those
        mixed_keys = list(lines_mixed.keys())
        has_mixed_specific = any("mixed=True" in key or "mixed=False" in key for key in mixed_keys)
        has_mixed_all = any("mixed=all" in key for key in mixed_keys)

        # Should prioritize mixed=True/False over mixed=all
        assert has_mixed_specific or has_mixed_all

        plt.close(fig)

    def test_motif_orientation_parameter(self, sample_data):
        """Test the new motif_orientation parameter."""
        # Test fwd_only
        stats_df_fwd = calc_trinuc_stats(
            sample_data, labels=[False, True], motif_orientation="fwd_only", collapsed=True
        )

        # Test seq_dir (default)
        stats_df_seq = calc_trinuc_stats(sample_data, labels=[False, True], motif_orientation="seq_dir", collapsed=True)

        # Test ref_dir
        stats_df_ref = calc_trinuc_stats(sample_data, labels=[False, True], motif_orientation="ref_dir", collapsed=True)

        # All should produce valid DataFrames
        assert isinstance(stats_df_fwd, pd.DataFrame)
        assert isinstance(stats_df_seq, pd.DataFrame)
        assert isinstance(stats_df_ref, pd.DataFrame)

        # All should have same structure
        assert stats_df_fwd.shape[0] == 96
        assert stats_df_seq.shape[0] == 96
        assert stats_df_ref.shape[0] == 96

    def test_motif_orientation_invalid_value(self, sample_data):
        """Test that invalid motif_orientation values raise errors."""
        with pytest.raises(ValueError, match="motif_orientation"):
            calc_trinuc_stats(sample_data, labels=[False, True], motif_orientation="invalid_value", collapsed=True)

    def test_calc_and_plot_with_motif_orientation(self, sample_data):
        """Test the main function with motif_orientation parameter."""
        fig, stats_df = calc_and_plot_trinuc_hist(
            sample_data,
            trinuc_col="tcwa_fwd",
            labels=[False, True],
            collapsed=True,
            include_quality=True,
            motif_orientation="fwd_only",
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(stats_df, pd.DataFrame)
        plt.close(fig)
