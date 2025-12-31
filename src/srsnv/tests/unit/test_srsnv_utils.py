"""Unit tests for srsnv_utils module, specifically the HandlePPMSeqTagsInFeatureMapDataFrame class."""

import numpy as np
import pandas as pd
import pytest
from ugbio_ppmseq.ppmSeq_utils import (
    MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
    MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
    STRAND_RATIO_LOWER_THRESH,
    STRAND_RATIO_UPPER_THRESH,
    PpmseqCategories,
)
from ugbio_srsnv.srsnv_utils import AE, AS, ET, ET_FILLNA, ST, ST_FILLNA, TE, TS, HandlePPMSeqTagsInFeatureMapDataFrame


class TestHandlePPMSeqTagsInFeatureMapDataFrame:
    """Test class for HandlePPMSeqTagsInFeatureMapDataFrame."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data with comprehensive edge cases."""
        return pd.DataFrame(
            {
                AS: [0, 4, 2, 1, 4, 1, 3, 1, 5, np.nan, 2, 0, 2],  # A's in start
                TS: [4, 0, 2, 4, 1, 3, 1, 1, 5, 1, np.nan, 0, 6],  # T's in start
                AE: [2, 1, 1, 2, 1, 1, 1, np.nan, 1, 1, 1, 3, 1],  # A's in end
                TE: [2, 3, 3, 2, 3, 3, 5, 1, 2, 2, 2, 1, 7],  # T's in end
            }
        )

    @pytest.fixture
    def expected_st_results(self):
        """Expected results for the start tag (ST) column."""
        return [
            PpmseqCategories.MINUS.value,  # AS=0, TS=4, total=4 → MINUS
            PpmseqCategories.PLUS.value,  # AS=4, TS=0, total=4 → PLUS
            PpmseqCategories.MIXED.value,  # AS=2, TS=2, total=4, ratio=0.5 → MIXED
            PpmseqCategories.UNDETERMINED.value,  # AS=1, TS=4, total=5, ratio=0.8 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=4, TS=1, total=5, ratio=0.2 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=1, TS=3, total=4, ratio=0.75 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=3, TS=1, total=4, ratio=0.25 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=1, TS=1, total=2 (out of range) → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=5, TS=5, total=10 (out of range) → UNDETERMINED
            np.nan,  # AS=NaN, TS=1 → NaN
            np.nan,  # AS=2, TS=NaN → NaN
            PpmseqCategories.UNDETERMINED.value,  # AS=0, TS=0, total=0 (out of range) → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=2, TS=6, total=8 (in range), ratio=0.75 → UNDETERMINED
        ]

    @pytest.fixture
    def expected_et_results(self):
        """Expected results for the end tag (ET) column."""
        return [
            PpmseqCategories.MIXED.value,  # AE=2, TE=2, total=4, ratio=0.5 → MIXED
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=3, total=4, ratio=0.75 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=3, total=4, ratio=0.75 → UNDETERMINED
            PpmseqCategories.MIXED.value,  # AE=2, TE=2, total=4, ratio=0.5 → MIXED
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=3, total=4, ratio=0.75 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=3, total=4, ratio=0.75 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=5, total=6, ratio=0.83 → UNDETERMINED
            np.nan,  # AE=NaN, TE=1 → NaN
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=2, total=3 (out of range) → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=2, total=3 (out of range) → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=2, total=3 (out of range) → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AE=3, TE=1, total=4, ratio=0.25 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AE=1, TE=7, total=8, ratio=0.875 → UNDETERMINED
        ]

    def test_add_ppmseq_tags_to_featuremap_basic(self, sample_data, expected_st_results, expected_et_results):
        """Test basic functionality of _add_ppmseq_tags_to_featuremap."""
        # Create handler instance
        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=sample_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        # Call the function
        handler._add_ppmseq_tags_to_featuremap()

        # Check that ST and ET columns were added
        assert ST in handler.featuremap_df.columns
        assert ET in handler.featuremap_df.columns

        # Verify ST results
        for i, expected in enumerate(expected_st_results):
            actual = handler.featuremap_df.iloc[i][ST]
            if pd.isna(expected):
                assert pd.isna(actual), f"Row {i}: Expected NaN for ST, got {actual}"
            else:
                assert actual == expected, f"Row {i}: Expected {expected} for ST, got {actual}"

        # Verify ET results
        for i, expected in enumerate(expected_et_results):
            actual = handler.featuremap_df.iloc[i][ET]
            if pd.isna(expected):
                assert pd.isna(actual), f"Row {i}: Expected NaN for ET, got {actual}"
            else:
                assert actual == expected, f"Row {i}: Expected {expected} for ET, got {actual}"

    def test_add_ppmseq_tags_to_featuremap_custom_parameters(self, sample_data):
        """Test _add_ppmseq_tags_to_featuremap with custom parameters."""
        # Create handler instance
        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=sample_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        # Call the function with custom parameters
        custom_sr_lower = 0.3
        custom_sr_upper = 0.7
        custom_min_total = 3
        custom_max_total = 9

        handler._add_ppmseq_tags_to_featuremap(
            sr_lower=custom_sr_lower,
            sr_upper=custom_sr_upper,
            min_total_hmer=custom_min_total,
            max_total_hmer=custom_max_total,
        )

        # Check that ST and ET columns were added
        assert ST in handler.featuremap_df.columns
        assert ET in handler.featuremap_df.columns

        # Verify that the custom parameters affected the results
        # Row 7: AS=1, TS=1, total=2 should now be in valid range (>= 3 is false, so still UNDETERMINED)
        assert handler.featuremap_df.iloc[7][ST] == PpmseqCategories.UNDETERMINED.value

        # Row 3: AS=1, TS=4, total=5, ratio=0.8 should still be UNDETERMINED (outside 0.3-0.7)
        assert handler.featuremap_df.iloc[3][ST] == PpmseqCategories.UNDETERMINED.value

    def test_strand_ratio_edge_cases(self):
        """Test specific edge cases for strand ratio calculations."""
        test_cases = pd.DataFrame(
            {
                AS: [0, 4, 2, 1, 4, 1, 3],
                TS: [4, 0, 2, 1, 1, 3, 1],
                AE: [0, 0, 0, 0, 0, 0, 0],  # Dummy values for ET calculation
                TE: [4, 4, 4, 4, 4, 4, 4],  # Dummy values for ET calculation
            }
        )

        expected_st = [
            PpmseqCategories.MINUS.value,  # AS=0, TS=4, total=4 → MINUS
            PpmseqCategories.PLUS.value,  # AS=4, TS=0, total=4 → PLUS
            PpmseqCategories.MIXED.value,  # AS=2, TS=2, total=4, ratio=0.5 → MIXED
            PpmseqCategories.MIXED.value,  # AS=1, TS=1, total=2 (out of range) → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=4, TS=1, total=5, ratio=0.2 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=1, TS=3, total=4, ratio=0.75 → UNDETERMINED
            PpmseqCategories.UNDETERMINED.value,  # AS=3, TS=1, total=4, ratio=0.25 → UNDETERMINED
        ]

        # Fix the expected result for the case where total is out of range
        expected_st[3] = PpmseqCategories.UNDETERMINED.value  # AS=1, TS=1, total=2 (out of range) → UNDETERMINED

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=test_cases.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        handler._add_ppmseq_tags_to_featuremap()

        for i, expected in enumerate(expected_st):
            actual = handler.featuremap_df.iloc[i][ST]
            assert actual == expected, f"Row {i}: Expected {expected} for ST, got {actual}"

    def test_nan_handling(self):
        """Test proper handling of NaN values."""
        test_data = pd.DataFrame(
            {
                AS: [np.nan, 2, np.nan, 1],
                TS: [2, np.nan, np.nan, 2],
                AE: [1, 1, 1, 1],
                TE: [3, 3, 3, 3],
            }
        )

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=test_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        handler._add_ppmseq_tags_to_featuremap()

        # All rows with NaN in AS or TS should result in NaN for ST
        assert pd.isna(handler.featuremap_df.iloc[0][ST])  # AS=NaN
        assert pd.isna(handler.featuremap_df.iloc[1][ST])  # TS=NaN
        assert pd.isna(handler.featuremap_df.iloc[2][ST])  # Both NaN

        # Row with valid values should not be NaN
        assert not pd.isna(handler.featuremap_df.iloc[3][ST])

    def test_zero_total_count(self):
        """Test handling of zero total count."""
        test_data = pd.DataFrame(
            {
                AS: [0, 0],
                TS: [0, 4],
                AE: [1, 1],
                TE: [3, 3],
            }
        )

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=test_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        handler._add_ppmseq_tags_to_featuremap()

        # AS=0, TS=0, total=0 (out of range) → UNDETERMINED
        assert handler.featuremap_df.iloc[0][ST] == PpmseqCategories.UNDETERMINED.value

        # AS=0, TS=4, total=4 (in range, AS=0) → MINUS
        assert handler.featuremap_df.iloc[1][ST] == PpmseqCategories.MINUS.value

    @pytest.mark.parametrize(
        "sr_lower,sr_upper,min_total,max_total",
        [
            (0.2, 0.8, 3, 9),
            (0.3, 0.6, 5, 7),
            (0.1, 0.9, 2, 10),
        ],
    )
    def test_parameter_variations(self, sample_data, sr_lower, sr_upper, min_total, max_total):
        """Test function with various parameter combinations."""
        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=sample_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        # Should not raise any exceptions
        handler._add_ppmseq_tags_to_featuremap(
            sr_lower=sr_lower, sr_upper=sr_upper, min_total_hmer=min_total, max_total_hmer=max_total
        )

        # Check that columns were added
        assert ST in handler.featuremap_df.columns
        assert ET in handler.featuremap_df.columns

        # Check that all values are valid category values or NaN
        valid_categories = {
            PpmseqCategories.MIXED.value,
            PpmseqCategories.MINUS.value,
            PpmseqCategories.PLUS.value,
            PpmseqCategories.UNDETERMINED.value,
        }

        for value in handler.featuremap_df[ST]:
            assert pd.isna(value) or value in valid_categories

        for value in handler.featuremap_df[ET]:
            assert pd.isna(value) or value in valid_categories

    def test_default_parameters_match_constants(self, sample_data):
        """Test that default parameters match the imported constants."""
        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=sample_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        # Call with defaults
        handler._add_ppmseq_tags_to_featuremap()
        result_default = handler.featuremap_df.copy()

        # Reset and call with explicit constants
        handler.featuremap_df = sample_data.copy()
        handler._add_ppmseq_tags_to_featuremap(
            sr_lower=STRAND_RATIO_LOWER_THRESH,
            sr_upper=STRAND_RATIO_UPPER_THRESH,
            min_total_hmer=MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
            max_total_hmer=MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
        )
        result_explicit = handler.featuremap_df.copy()

        # Results should be identical
        pd.testing.assert_frame_equal(result_default, result_explicit)

    def test_boundary_strand_ratios(self):
        """Test edge cases exactly at the boundary thresholds."""
        # Create test cases with exact boundary ratios
        # sr_lower = 0.27, sr_upper = 0.73
        # For total = 4: 0.27 * 4 = 1.08, 0.73 * 4 = 2.92
        # For TS=1, ratio=0.25 (< 0.27) → UNDETERMINED
        # For TS=2, ratio=0.50 (in [0.27, 0.73]) → MIXED
        # For TS=3, ratio=0.75 (> 0.73) → UNDETERMINED
        test_data = pd.DataFrame(
            {
                AS: [3, 2, 1, 1, 3],  # Corresponding to TS values below
                TS: [1, 2, 3, 1, 1],  # Different ratios: 0.25, 0.5, 0.75, 0.5, 0.25
                AE: [1, 1, 1, 1, 1],  # Dummy values
                TE: [3, 3, 3, 3, 3],  # Dummy values
            }
        )

        # Totals: 4, 4, 4, 2, 4
        # Ratios: 0.25, 0.5, 0.75, 0.5, 0.25
        # Valid ranges: yes, yes, yes, no, yes
        expected_st = [
            PpmseqCategories.UNDETERMINED.value,  # ratio=0.25 < 0.27
            PpmseqCategories.MIXED.value,  # ratio=0.5 in [0.27, 0.73]
            PpmseqCategories.UNDETERMINED.value,  # ratio=0.75 > 0.73
            PpmseqCategories.UNDETERMINED.value,  # total=2 out of range
            PpmseqCategories.UNDETERMINED.value,  # ratio=0.25 < 0.27
        ]

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=test_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        handler._add_ppmseq_tags_to_featuremap()

        for i, expected in enumerate(expected_st):
            actual = handler.featuremap_df.iloc[i][ST]
            assert actual == expected, f"Row {i}: Expected {expected} for ST, got {actual}"

    def test_exact_boundary_thresholds(self):
        """Test cases exactly at the strand ratio thresholds (0.27 and 0.73)."""
        # For a total of 100, we can get exact ratios
        test_data = pd.DataFrame(
            {
                AS: [73, 27, 50],  # Will give ratios of exactly 0.27, 0.73, 0.5
                TS: [27, 73, 50],  # Total = 100 for each
                AE: [1, 1, 1],  # Dummy values
                TE: [3, 3, 3],  # Dummy values
            }
        )

        # All totals are 100 (out of range for default 4-8), so all should be UNDETERMINED
        expected_st = [
            PpmseqCategories.UNDETERMINED.value,  # total=100 out of range
            PpmseqCategories.UNDETERMINED.value,  # total=100 out of range
            PpmseqCategories.UNDETERMINED.value,  # total=100 out of range
        ]

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=test_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        handler._add_ppmseq_tags_to_featuremap()

        for i, expected in enumerate(expected_st):
            actual = handler.featuremap_df.iloc[i][ST]
            assert actual == expected, f"Row {i}: Expected {expected} for ST, got {actual}"

        # Test with extended range that includes 100
        handler.featuremap_df = test_data.copy()
        handler._add_ppmseq_tags_to_featuremap(min_total_hmer=4, max_total_hmer=100)

        # Now ratios should be evaluated
        expected_st_in_range = [
            PpmseqCategories.MIXED.value,  # ratio=0.27 (at boundary)
            PpmseqCategories.MIXED.value,  # ratio=0.73 (at boundary)
            PpmseqCategories.MIXED.value,  # ratio=0.5 (in range)
        ]

        for i, expected in enumerate(expected_st_in_range):
            actual = handler.featuremap_df.iloc[i][ST]
            assert actual == expected, f"Row {i}: Expected {expected} for ST with extended range, got {actual}"

    def test_empty_dataframe(self):
        """Test function with empty DataFrame."""
        empty_data = pd.DataFrame({AS: [], TS: [], AE: [], TE: []})

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=empty_data.copy(), categorical_features_names=[], ppmseq_adapter_version="legacy_v5"
        )

        # Should not raise an exception
        handler._add_ppmseq_tags_to_featuremap()

        # Should add empty ST and ET columns
        assert ST in handler.featuremap_df.columns
        assert ET in handler.featuremap_df.columns
        assert len(handler.featuremap_df) == 0

    def test_fill_nan_tags_with_tm_column(self):
        """Test fill_nan_tags function with TM column present."""
        # Import TM constant
        from ugbio_srsnv.srsnv_utils import TM

        # Create test data with NaN values in ST and ET columns
        test_data = pd.DataFrame(
            {
                AS: [2, 3, 1, 4],
                TS: [3, 1, 4, 1],
                AE: [1, 2, 3, 2],
                TE: [2, 4, 1, 3],
                ST: ["MIXED", np.nan, "PLUS", np.nan],
                ET: [np.nan, "MINUS", np.nan, np.nan],
                TM: ["ABC", "XYZ", "ADE", "FGH"],  # positions 0,2 contain 'A', positions 1,3 don't
            }
        )

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=test_data,
            categorical_features_names=[],
            ppmseq_adapter_version="v1",
            start_tag_col=ST,
            end_tag_col=ET,
        )

        # Fill NaN values
        handler.fill_nan_tags()

        # Verify ST_FILLNA column: all NaN should be UNDETERMINED
        assert handler.featuremap_df.loc[1, ST_FILLNA] == PpmseqCategories.UNDETERMINED.value
        assert handler.featuremap_df.loc[3, ST_FILLNA] == PpmseqCategories.UNDETERMINED.value

        # Verify ET_FILLNA column: based on TM content
        assert handler.featuremap_df.loc[0, ET_FILLNA] == PpmseqCategories.UNDETERMINED.value  # tm='ABC' contains 'A'
        assert handler.featuremap_df.loc[2, ET_FILLNA] == PpmseqCategories.UNDETERMINED.value  # tm='ADE' contains 'A'
        assert handler.featuremap_df.loc[3, ET_FILLNA] == PpmseqCategories.END_UNREACHED.value  # tm='FGH' no 'A'

        # Verify no NaN values remain in fillna columns
        assert handler.featuremap_df[ST_FILLNA].isna().sum() == 0
        assert handler.featuremap_df[ET_FILLNA].isna().sum() == 0

        # Verify existing values are preserved in fillna columns
        assert handler.featuremap_df.loc[0, ST_FILLNA] == "MIXED"
        assert handler.featuremap_df.loc[2, ST_FILLNA] == "PLUS"
        assert handler.featuremap_df.loc[1, ET_FILLNA] == "MINUS"

        # Verify original columns are unchanged
        assert pd.isna(handler.featuremap_df.loc[1, ST])
        assert pd.isna(handler.featuremap_df.loc[3, ST])
        assert pd.isna(handler.featuremap_df.loc[0, ET])
        assert pd.isna(handler.featuremap_df.loc[2, ET])
        assert pd.isna(handler.featuremap_df.loc[3, ET])

    def test_fill_nan_tags_without_tm_column(self):
        """Test fill_nan_tags function without TM column present."""
        # Create test data without TM column
        test_data = pd.DataFrame(
            {
                AS: [2, 3, 1, 4],
                TS: [3, 1, 4, 1],
                AE: [1, 2, 3, 2],
                TE: [2, 4, 1, 3],
                ST: ["MIXED", np.nan, "PLUS", np.nan],
                ET: [np.nan, "MINUS", np.nan, "MIXED"],
            }
        )

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=test_data,
            categorical_features_names=[],
            ppmseq_adapter_version="v1",
            start_tag_col=ST,
            end_tag_col=ET,
        )

        # Fill NaN values (should log warning about missing TM column)
        handler.fill_nan_tags()

        # Verify all NaN values are filled with UNDETERMINED in fillna columns
        assert handler.featuremap_df.loc[1, ST_FILLNA] == PpmseqCategories.UNDETERMINED.value
        assert handler.featuremap_df.loc[3, ST_FILLNA] == PpmseqCategories.UNDETERMINED.value
        assert handler.featuremap_df.loc[0, ET_FILLNA] == PpmseqCategories.UNDETERMINED.value
        assert handler.featuremap_df.loc[2, ET_FILLNA] == PpmseqCategories.UNDETERMINED.value

        # Verify no NaN values remain in fillna columns
        assert handler.featuremap_df[ST_FILLNA].isna().sum() == 0
        assert handler.featuremap_df[ET_FILLNA].isna().sum() == 0

        # Verify existing values are preserved in fillna columns
        assert handler.featuremap_df.loc[0, ST_FILLNA] == "MIXED"
        assert handler.featuremap_df.loc[2, ST_FILLNA] == "PLUS"
        assert handler.featuremap_df.loc[1, ET_FILLNA] == "MINUS"
        assert handler.featuremap_df.loc[3, ET_FILLNA] == "MIXED"

        # Verify original columns are unchanged
        assert pd.isna(handler.featuremap_df.loc[1, ST])
        assert pd.isna(handler.featuremap_df.loc[3, ST])
        assert pd.isna(handler.featuremap_df.loc[0, ET])
        assert pd.isna(handler.featuremap_df.loc[2, ET])

    def test_fill_nan_tags_without_tag_columns_setup(self):
        """Test fill_nan_tags function when tag columns are not set up."""
        test_data = pd.DataFrame(
            {
                "some_col": [1, 2],
                "other_col": [3, 4],
            }
        )

        handler = HandlePPMSeqTagsInFeatureMapDataFrame(
            featuremap_df=test_data,
            categorical_features_names=[],
            ppmseq_adapter_version="v1",
            start_tag_col=None,  # Not set up
            end_tag_col=None,  # Not set up
        )

        # Should log warning and return early
        handler.fill_nan_tags()

        # The fillna columns should not be created since start_tag_col and end_tag_col are None
        assert ST_FILLNA not in handler.featuremap_df.columns
        assert ET_FILLNA not in handler.featuremap_df.columns


class TestGetFilterRatio:
    """Test class for get_filter_ratio function."""

    @pytest.fixture
    def sample_filters(self):
        """Create sample filter list for testing."""
        return [
            {"name": "raw", "type": "raw", "funnel": 1000},
            {"name": "coverage_ge_min", "type": "region", "funnel": 950},
            {"name": "coverage_le_max", "type": "region", "funnel": 900},
            {"name": "mapq_ge_60", "type": "quality", "funnel": 800},
            {"name": "no_adj_ref_diff", "type": "quality", "funnel": 700},
            {"name": "bcsq_gt_40", "type": "quality", "funnel": 600},
            {"name": "ref_eq_alt", "type": "label", "funnel": 500},
            {"name": "downsample", "type": "downsample", "funnel": 100},
        ]

    def test_default_behavior(self, sample_filters):
        """Test default behavior: before 'label' / 'raw'."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        ratio = get_filter_ratio(sample_filters)
        # Default: last before first 'label' (index 6) / raw (index 0)
        # = filters[5]['funnel'] / filters[0]['funnel'] = 600 / 1000 = 0.6
        assert ratio == 0.6

    def test_by_name_numerator_and_denominator(self, sample_filters):
        """Test specifying both filters by name."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        # Last before 'ref_eq_alt' (index 6) / last before 'mapq_ge_60' (index 3)
        # = filters[5]['funnel'] / filters[2]['funnel'] = 600 / 900
        ratio = get_filter_ratio(sample_filters, numerator_filter="ref_eq_alt", denominator_filter="mapq_ge_60")
        assert ratio == pytest.approx(600 / 900)

    def test_by_type_numerator_and_denominator(self, sample_filters):
        """Test specifying both filters by type."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        # Last before first 'quality' (index 3) / 'raw' (index 0)
        # = filters[2]['funnel'] / filters[0]['funnel'] = 900 / 1000
        ratio = get_filter_ratio(sample_filters, numerator_type="quality", denominator_type="raw")
        assert ratio == 0.9

    def test_mixed_name_and_type(self, sample_filters):
        """Test mixing name and type specifications."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        # By name for numerator, by type for denominator
        ratio = get_filter_ratio(sample_filters, numerator_filter="ref_eq_alt", denominator_type="quality")
        # = filters[5]['funnel'] / filters[2]['funnel'] = 600 / 900
        assert ratio == pytest.approx(600 / 900)

    def test_raw_special_case_numerator(self, sample_filters):
        """Test 'raw' as numerator uses raw filter itself."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        ratio = get_filter_ratio(sample_filters, numerator_filter="raw", denominator_filter="raw")
        # Both use filters[0]['funnel'] / filters[0]['funnel'] = 1.0
        assert ratio == 1.0

    def test_raw_special_case_type(self, sample_filters):
        """Test 'raw' type behaves like 'raw' name."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        ratio = get_filter_ratio(sample_filters, numerator_type="raw", denominator_type="raw")
        assert ratio == 1.0

    def test_reproduce_base_recall_behavior(self, sample_filters):
        """Test that it can reproduce get_base_recall_from_filters behavior."""
        from ugbio_srsnv.srsnv_utils import get_base_recall_from_filters, get_filter_ratio

        # get_base_recall_from_filters: last before 'ref_eq_alt' / last before first 'quality'
        ratio_new = get_filter_ratio(sample_filters, numerator_filter="ref_eq_alt", denominator_type="quality")
        ratio_old = get_base_recall_from_filters(sample_filters)
        assert ratio_new == pytest.approx(ratio_old)

    def test_error_filter_not_found_by_name(self, sample_filters):
        """Test error when filter name is not found."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        with pytest.raises(ValueError, match="Filter with name 'nonexistent' not found"):
            get_filter_ratio(sample_filters, numerator_filter="nonexistent")

    def test_error_filter_not_found_by_type(self, sample_filters):
        """Test error when filter type is not found."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        with pytest.raises(ValueError, match="Filter with type 'nonexistent_type' not found"):
            get_filter_ratio(sample_filters, numerator_type="nonexistent_type")

    def test_error_empty_filter_list(self):
        """Test error with empty filter list."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        with pytest.raises(ValueError, match="Filter list is empty"):
            get_filter_ratio([])

    def test_error_denominator_zero_rows(self):
        """Test error when denominator has zero rows."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        filters = [
            {"name": "raw", "type": "raw", "funnel": 0},  # Raw has 0 rows
            {"name": "filter2", "type": "quality", "funnel": 100},
            {"name": "filter3", "type": "label", "funnel": 50},
        ]
        with pytest.raises(ValueError, match="Denominator filter has 0 rows"):
            # Denominator uses 'raw' which has 0 rows
            get_filter_ratio(filters, numerator_filter="filter3", denominator_type="raw")

    def test_single_filter_raw(self):
        """Test with a single 'raw' filter."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        filters = [{"name": "raw", "type": "raw", "funnel": 500}]
        ratio = get_filter_ratio(filters, numerator_type="raw", denominator_type="raw")
        assert ratio == 1.0

    def test_multiple_filters_same_type(self, sample_filters):
        """Test behavior when multiple filters have the same type."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        # There are multiple 'quality' filters. Should use first one.
        # First quality is at index 3 (mapq_ge_60)
        # Last before it is index 2 (coverage_le_max) with rows=900
        ratio = get_filter_ratio(sample_filters, numerator_type="quality", denominator_type="raw")
        assert ratio == 0.9  # 900 / 1000

    def test_downsample_ratio(self, sample_filters):
        """Test calculating downsample ratio."""
        from ugbio_srsnv.srsnv_utils import get_filter_ratio

        # Last before 'downsample' (index 7) / last before first 'label' (index 6)
        # When numerator_filter is specified, it takes precedence over numerator_type
        # = filters[6]['funnel'] / filters[5]['funnel'] = 500 / 600
        ratio = get_filter_ratio(sample_filters, numerator_filter="downsample", denominator_type="label")
        assert ratio == pytest.approx(500 / 600)
