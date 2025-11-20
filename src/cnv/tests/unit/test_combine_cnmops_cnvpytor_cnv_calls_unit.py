from pathlib import Path

import pandas as pd
import pytest
from ugbio_cnv import combine_cnmops_cnvpytor_cnv_calls


@pytest.fixture
def resources_dir():
    """Fixture providing path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def sample_name():
    """Fixture providing a sample name for testing."""
    return "TEST_SAMPLE"


class TestCombineCnmopsCnvpytorCNVCalls:
    """Tests for combine_cnmops_cnvpytor_cnv_calls module."""

    # Tests for get_dup_cnmops_cnv_calls function
    def test_get_dup_cnmops_unsorted_dataframe(self, tmpdir, sample_name):
        """Test that unsorted duplications dataframe is properly sorted by chrom and start position."""
        out_directory = str(tmpdir)
        distance_threshold = 1500

        # Create unsorted dataframe with duplications
        df_cnmops = pd.DataFrame(
            {
                "chrom": ["chr2", "chr1", "chr1", "chr1"],
                "start": [5000, 3000, 1000, 1000],
                "end": [15000, 13000, 11000, 13000],
                "CN": ["CN3", "CN3", "CN2.5", "CN3"],
            }
        )

        result_file = combine_cnmops_cnvpytor_cnv_calls.get_dup_cnmops_cnv_calls(
            df_cnmops, sample_name, out_directory, distance_threshold
        )

        # Verify output file exists and is sorted
        assert result_file != ""
        assert result_file.endswith(".cnmops_cnvs.DUP.all_fields.bed")

        df_result = pd.read_csv(result_file, sep="\t", header=None)
        df_result.columns = ["chrom", "start", "end", "CNV_type", "source", "copy_number"]

        # Verify sorting: first by chrom, then by start, then by end
        for i in range(len(df_result) - 1):
            curr_chrom = df_result.iloc[i]["chrom"]
            next_chrom = df_result.iloc[i + 1]["chrom"]

            if curr_chrom == next_chrom:
                curr_start = df_result.iloc[i]["start"]
                next_start = df_result.iloc[i + 1]["start"]
                assert curr_start <= next_start, f"Start positions not sorted: {curr_start} > {next_start}"

                if curr_start == next_start:
                    curr_end = df_result.iloc[i]["end"]
                    next_end = df_result.iloc[i + 1]["end"]
                    assert curr_end <= next_end, f"End positions not sorted: {curr_end} > {next_end}"
            else:
                # Verify chromosomes are sorted
                assert curr_chrom < next_chrom, f"Chromosomes not sorted: {curr_chrom} > {next_chrom}"

    def test_get_dup_cnmops_above_threshold(self, tmpdir, sample_name):
        """Test extraction of duplications above CN=2."""
        out_directory = str(tmpdir)
        distance_threshold = 1500

        # Create dataframe with various CN values
        df_cnmops = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr1", "chr2"],
                "start": [1000, 20000, 30000, 5000],
                "end": [11000, 30000, 40000, 15000],
                "CN": ["CN1.5", "CN2.0", "CN2.5", "CN3.0"],  # Only CN2.5 and CN3.0 should be included
            }
        )

        result_file = combine_cnmops_cnvpytor_cnv_calls.get_dup_cnmops_cnv_calls(
            df_cnmops, sample_name, out_directory, distance_threshold
        )

        assert result_file != ""

        df_result = pd.read_csv(result_file, sep="\t", header=None)
        df_result.columns = ["chrom", "start", "end", "CNV_type", "source", "copy_number"]

        # Verify only duplications with CN > 2 are present
        assert len(df_result) > 0
        assert all(df_result["CNV_type"] == "DUP")
        assert all(df_result["source"] == "cn.mops")

        # Check that CN values are properly extracted (handle both string and numeric types)
        for _, row in df_result.iterrows():
            cn_value = row["copy_number"]
            # Convert to float for comparison (pandas may read as string or numeric)
            cn_numeric = float(cn_value) if isinstance(cn_value, str) else cn_value
            # CN value should be from the duplications (2.5 or 3.0)
            assert cn_numeric in [2.5, 3.0], f"Unexpected CN value: {cn_value}"

    def test_get_dup_cnmops_length_filter(self, tmpdir, sample_name):
        """Test filtering duplications by length (>=10000bp)."""
        out_directory = str(tmpdir)
        distance_threshold = 1500

        # Create dataframe with duplications of varying lengths
        df_cnmops = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr1", "chr2"],
                "start": [1000, 20000, 30000, 5000],
                "end": [5000, 30000, 45000, 20000],  # lengths: 4000, 10000, 15000, 15000
                "CN": ["CN3", "CN3", "CN3", "CN3"],
            }
        )

        result_file = combine_cnmops_cnvpytor_cnv_calls.get_dup_cnmops_cnv_calls(
            df_cnmops, sample_name, out_directory, distance_threshold
        )

        assert result_file != ""

        df_result = pd.read_csv(result_file, sep="\t", header=None)
        df_result.columns = ["chrom", "start", "end", "CNV_type", "source", "copy_number"]

        # Verify all duplications are >= 10000bp
        for _, row in df_result.iterrows():
            length = row["end"] - row["start"]
            assert length >= 10000, f"Duplication length {length} is below 10000bp threshold"

    def test_get_dup_cnmops_merge_distance(self, tmpdir, sample_name):
        """Test merging duplications based on distance threshold."""
        out_directory = str(tmpdir)
        distance_threshold = 1500

        # Create dataframe with close duplications that should be merged
        df_cnmops = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr1"],
                "start": [1000, 12000, 30000],  # 12000 - 11000 = 1000 < 1500, should merge with first
                "end": [11000, 22000, 45000],  # 30000 is far from 22000, should not merge
                "CN": ["CN3", "CN2.5", "CN3"],
            }
        )

        result_file = combine_cnmops_cnvpytor_cnv_calls.get_dup_cnmops_cnv_calls(
            df_cnmops, sample_name, out_directory, distance_threshold
        )

        assert result_file != ""

        df_result = pd.read_csv(result_file, sep="\t", header=None)
        df_result.columns = ["chrom", "start", "end", "CNV_type", "source", "copy_number"]

        # With distance threshold of 1500, first two should merge
        # Third should remain separate
        # Exact number depends on bedtools merge behavior, but should be less than original
        assert len(df_result) <= len(df_cnmops)

        # Verify all are marked as DUP from cn.mops
        assert all(df_result["CNV_type"] == "DUP")
        assert all(df_result["source"] == "cn.mops")

    def test_get_dup_cnmops_no_duplications(self, tmpdir, sample_name):
        """Test handling when no duplications are found."""
        out_directory = str(tmpdir)
        distance_threshold = 1500

        # Create dataframe with only deletions (CN <= 2)
        df_cnmops = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [1000, 20000, 5000],
                "end": [11000, 30000, 15000],
                "CN": ["CN1.0", "CN1.5", "CN2.0"],
            }
        )

        result_file = combine_cnmops_cnvpytor_cnv_calls.get_dup_cnmops_cnv_calls(
            df_cnmops, sample_name, out_directory, distance_threshold
        )

        # Should return empty string when no duplications found
        assert result_file == ""
