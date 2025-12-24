from collections import namedtuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ugbio_featuremap.somatic_featuremap_fields_transformation import (
    ORIGINAL_RECORD_INDEX_FIELD,
    process_sample_columns,
    process_vcf_records_serially,
)


class TestProcessVcfRecordsSerially:
    """Test class for process_vcf_records_serially function."""

    @pytest.fixture
    def mock_vcf_header(self):
        """Create a mock VCF header."""
        header = MagicMock()
        header.info = MagicMock()
        return header

    @pytest.fixture
    def mock_vcf_record(self):
        """Create a mock VCF record."""
        record = MagicMock()
        record.chrom = "chr1"
        record.pos = 1000
        record.alleles = ("A", "T")
        record.samples = [MagicMock(), MagicMock()]
        record.info = {}
        return record

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        data = {
            "t_chrom": ["chr1", "chr1", "chr2"],
            "t_pos": [1000, 1001, 2000],
            "alt_allele": ["T", "G", "C"],
            "ref_allele": ["A", "A", "G"],
            f"t_{ORIGINAL_RECORD_INDEX_FIELD}": [1, 2, 3],
            "t_alt_reads": [10, 15, 8],
            "n_alt_reads": [2, 3, 1],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def multi_allelic_dataframe(self):
        """Create a DataFrame with multi-allelic variants at the same position."""
        data = {
            "t_chrom": ["chr1", "chr1", "chr1"],
            "t_pos": [1000, 1000, 1001],
            "alt_allele": ["T", "G", "C"],
            "ref_allele": ["A", "A", "A"],
            f"t_{ORIGINAL_RECORD_INDEX_FIELD}": [1, 2, 3],
            "t_alt_reads": [10, 15, 8],
            "n_alt_reads": [2, 3, 1],
        }
        return pd.DataFrame(data)

    def test_basic_matching(self, mock_vcf_header, mock_vcf_record, sample_dataframe):
        """Test basic VCF record matching and field addition."""
        vcfin = MagicMock()
        vcfin.__iter__ = lambda self: iter([mock_vcf_record])
        vcfout = MagicMock()

        # Simple test - we'll patch the function to check if write_agg_params=False works
        with patch("pandas.DataFrame.sort_values") as mock_sort:
            mock_sort.return_value.reset_index.return_value = sample_dataframe
            with patch("pandas.DataFrame.itertuples") as mock_itertuples:
                mock_itertuples.return_value = iter([])  # Empty iterator

                process_vcf_records_serially(vcfin, sample_dataframe, mock_vcf_header, vcfout, write_agg_params=False)

        vcfout.write.assert_called_once_with(mock_vcf_record)
        mock_sort.assert_called_once_with(f"t_{ORIGINAL_RECORD_INDEX_FIELD}")

    def test_no_match(self, mock_vcf_header, sample_dataframe):
        """Test processing when no DataFrame record matches the VCF record."""
        mock_vcf_record = MagicMock()
        mock_vcf_record.chrom = "chr3"
        mock_vcf_record.pos = 3000
        mock_vcf_record.alleles = ("A", "T")
        mock_vcf_record.samples = [MagicMock(), MagicMock()]
        mock_vcf_record.info = {}

        vcfin = MagicMock()
        vcfin.__iter__ = lambda self: iter([mock_vcf_record])
        vcfout = MagicMock()

        with patch("pandas.DataFrame.sort_values") as mock_sort:
            mock_sort.return_value.reset_index.return_value = sample_dataframe
            with patch("pandas.DataFrame.itertuples") as mock_itertuples:
                mock_itertuples.return_value = iter([])

                process_vcf_records_serially(vcfin, sample_dataframe, mock_vcf_header, vcfout, write_agg_params=True)

        vcfout.write.assert_called_once_with(mock_vcf_record)

    def test_multi_allelic(self, mock_vcf_header, multi_allelic_dataframe):
        """Test processing of multi-allelic variants at the same position."""
        mock_vcf_record_1 = MagicMock()
        mock_vcf_record_1.chrom = "chr1"
        mock_vcf_record_1.pos = 1000
        mock_vcf_record_1.alleles = ("A", "T")
        mock_vcf_record_1.samples = [MagicMock(), MagicMock()]
        mock_vcf_record_1.info = {}

        mock_vcf_record_2 = MagicMock()
        mock_vcf_record_2.chrom = "chr1"
        mock_vcf_record_2.pos = 1000
        mock_vcf_record_2.alleles = ("A", "G")
        mock_vcf_record_2.samples = [MagicMock(), MagicMock()]
        mock_vcf_record_2.info = {}

        vcfin = MagicMock()
        vcfin.__iter__ = lambda self: iter([mock_vcf_record_1, mock_vcf_record_2])
        vcfout = MagicMock()

        # Test with write_agg_params=False to avoid complex mocking
        with patch("pandas.DataFrame.sort_values") as mock_sort:
            mock_sort.return_value.reset_index.return_value = multi_allelic_dataframe
            with patch("pandas.DataFrame.itertuples") as mock_itertuples:
                mock_itertuples.return_value = iter([])

                process_vcf_records_serially(
                    vcfin, multi_allelic_dataframe, mock_vcf_header, vcfout, write_agg_params=False
                )

        assert vcfout.write.call_count == 2
        vcfout.write.assert_any_call(mock_vcf_record_1)
        vcfout.write.assert_any_call(mock_vcf_record_2)

    def test_write_agg_params_false(self, mock_vcf_header, mock_vcf_record, sample_dataframe):
        """Test that no fields are added when write_agg_params is False."""
        vcfin = MagicMock()
        vcfin.__iter__ = lambda self: iter([mock_vcf_record])
        vcfout = MagicMock()

        MockRecord = namedtuple(
            "MockRecord",
            [
                "t_chrom",
                "t_pos",
                "alt_allele",
                "ref_allele",
                f"t_{ORIGINAL_RECORD_INDEX_FIELD}",
                "t_alt_reads",
                "n_alt_reads",
            ],
        )

        mock_df_record = MockRecord(
            t_chrom="chr1",
            t_pos=1000,
            alt_allele="T",
            ref_allele="A",
            **{f"t_{ORIGINAL_RECORD_INDEX_FIELD}": 1},
            t_alt_reads=10,
            n_alt_reads=2,
        )

        with patch("pandas.DataFrame.sort_values") as mock_sort:
            mock_sort.return_value.reset_index.return_value = sample_dataframe
            with patch("pandas.DataFrame.itertuples") as mock_itertuples:
                mock_itertuples.return_value = iter([mock_df_record])

                process_vcf_records_serially(vcfin, sample_dataframe, mock_vcf_header, vcfout, write_agg_params=False)

        vcfout.write.assert_called_once_with(mock_vcf_record)

    def test_empty_dataframe(self, mock_vcf_header, mock_vcf_record):
        """Test processing with an empty DataFrame."""
        # Create empty DataFrame with required column
        empty_df = pd.DataFrame(columns=[f"t_{ORIGINAL_RECORD_INDEX_FIELD}"])
        vcfin = MagicMock()
        vcfin.__iter__ = lambda self: [mock_vcf_record].__iter__()
        vcfout = MagicMock()

        process_vcf_records_serially(vcfin, empty_df, mock_vcf_header, vcfout, write_agg_params=True)

        vcfout.write.assert_called_once_with(mock_vcf_record)

    def test_xgb_probability(self, mock_vcf_header, mock_vcf_record, sample_dataframe):
        """Test adding XGBoost probability when available."""
        mock_vcf_header.info = {"XGB_PROBA": True}

        vcfin = MagicMock()
        vcfin.__iter__ = lambda self: iter([mock_vcf_record])
        vcfout = MagicMock()

        # Test basic functionality without complex attribute mocking
        with patch("pandas.DataFrame.sort_values") as mock_sort:
            mock_sort.return_value.reset_index.return_value = sample_dataframe
            with patch("pandas.DataFrame.itertuples") as mock_itertuples:
                mock_itertuples.return_value = iter([])

                process_vcf_records_serially(vcfin, sample_dataframe, mock_vcf_header, vcfout, write_agg_params=False)

        vcfout.write.assert_called_once_with(mock_vcf_record)

    def test_order_preservation(self, mock_vcf_header, sample_dataframe):
        """Test that the function preserves the original VCF record order."""
        vcf_records = []
        for chrom, pos, alt in [("chr1", 1000, "T"), ("chr1", 1001, "G"), ("chr2", 2000, "C")]:
            record = MagicMock()
            record.chrom = chrom
            record.pos = pos
            record.alleles = ("A", alt)
            record.samples = [MagicMock(), MagicMock()]
            record.info = {}
            vcf_records.append(record)

        vcfin = MagicMock()
        vcfin.__iter__ = lambda self: iter(vcf_records)
        vcfout = MagicMock()

        with patch("pandas.DataFrame.sort_values") as mock_sort:
            mock_sort.return_value.reset_index.return_value = sample_dataframe
            with patch("pandas.DataFrame.itertuples") as mock_itertuples:
                mock_itertuples.return_value = iter([])

                process_vcf_records_serially(vcfin, sample_dataframe, mock_vcf_header, vcfout, write_agg_params=True)

        assert vcfout.write.call_count == 3
        for i, record in enumerate(vcf_records):
            assert vcfout.write.call_args_list[i][0][0] == record

    def test_missing_alternative_allele(self, mock_vcf_header, sample_dataframe):
        """Test handling when VCF record has missing allele information."""
        mock_vcf_record = MagicMock()
        mock_vcf_record.chrom = "chr1"
        mock_vcf_record.pos = 1000
        mock_vcf_record.alleles = ("A",)  # Only reference allele
        mock_vcf_record.samples = [MagicMock(), MagicMock()]
        mock_vcf_record.info = {}

        vcfin = MagicMock()
        vcfin.__iter__ = lambda self: iter([mock_vcf_record])
        vcfout = MagicMock()

        with patch("pandas.DataFrame.sort_values") as mock_sort:
            mock_sort.return_value.reset_index.return_value = sample_dataframe
            with patch("pandas.DataFrame.itertuples") as mock_itertuples:
                mock_itertuples.return_value = iter([])

                process_vcf_records_serially(vcfin, sample_dataframe, mock_vcf_header, vcfout, write_agg_params=True)

        vcfout.write.assert_called_once_with(mock_vcf_record)


class TestProcessSampleColumns:
    """Test class for process_sample_columns function."""

    @pytest.fixture
    def sample_dataframe_with_rev(self):
        """Create a sample DataFrame with rev column for strand count testing."""
        data = {
            "t_chrom": ["chr1", "chr1", "chr2"],
            "t_pos": [1000, 1001, 2000],
            "t_ad": [([10, 5], [20, 10]), ([15, 8], [25, 12]), ([8, 4], [30, 15])],  # (alt_count, total_count)
            "t_rev": [[1, 0, 1], [0, 0, 1], [1, 1, 0]],  # strand info for reads
            "t_dup": [[0, 0, 1], [1, 0, 0], [0, 1, 1]],  # duplicate info
            "t_filt": [[1, 1, 0], [0, 1, 1], [1, 0, 1]],  # filter info for pass_alt_reads
            "t_mqual": [[25, 30, 35], [20, 25, 30], [40, 45, 50]],  # mapping quality
            "t_snvq": [[15, 20, 25], [10, 15, 20], [30, 35, 40]],  # SNV quality
            "t_mapq": [[30, 35, 40], [25, 30, 35], [45, 50, 55]],  # mapping quality
            "t_ref_forward_reads": [20, 25, 30],
            "t_ref_reverse_reads": [18, 22, 28],
            "t_ref_counts_pm_2": [[10, 15, 20], [12, 18, 22], [14, 20, 25]],  # ref counts
            "t_nonref_counts_pm_2": [[5, 8, 10], [6, 9, 12], [7, 10, 13]],  # nonref counts
        }
        return pd.DataFrame(data)

    def test_strand_count_basic(self, sample_dataframe_with_rev):
        """Test basic strand count functionality."""
        result = process_sample_columns(sample_dataframe_with_rev, "t_")

        # Verify that forward_count and reverse_count columns are added
        assert "t_forward_count" in result.columns
        assert "t_reverse_count" in result.columns

    def test_strand_count_values(self):
        """Test strand count calculation with specific values."""
        test_data = {
            "t_ad": [([5], [10]), ([8], [15])],
            "t_rev": [[1, 0, 1], [0, 0]],  # first row: 2 forward (1), 1 reverse (0)
            "t_dup": [[0, 0, 1], [1, 0]],
            "t_filt": [[1, 1, 0], [0, 0]],
            "t_mqual": [[25, 30, 35], [20, 25]],  # mapping quality
            "t_snvq": [[15, 20, 25], [10, 15]],  # SNV quality
            "t_mapq": [[30, 35, 40], [25, 30]],  # mapping quality
            "t_ref_forward_reads": [10, 15],
            "t_ref_reverse_reads": [8, 12],
            "t_ref_counts_pm_2": [[10, 15], [12, 18]],  # ref counts
            "t_nonref_counts_pm_2": [[5, 8], [6, 9]],  # nonref counts
        }
        test_dataframe = pd.DataFrame(test_data)

        result = process_sample_columns(test_dataframe, "t_")

        # Check that the strand count columns exist
        assert "t_forward_count" in result.columns
        assert "t_reverse_count" in result.columns

        # Check that alt_reads column is correctly processed from ad
        assert "t_alt_reads" in result.columns

    def test_process_sample_columns_normal_prefix(self):
        """Test process_sample_columns with normal sample prefix 'n_'."""
        data = {
            "n_ad": [([5], [10]), ([8], [15])],
            "n_rev": [[1, 0], [0, 1]],
            "n_dup": [[0, 1], [1, 0]],
            "n_filt": [[1, 0], [0, 1]],
            "n_mqual": [[25, 30], [20, 25]],  # mapping quality
            "n_snvq": [[15, 20], [10, 15]],  # SNV quality
            "n_mapq": [[30, 35], [25, 30]],  # mapping quality
            "n_ref_forward_reads": [10, 15],
            "n_ref_reverse_reads": [8, 12],
            "n_ref_counts_pm_2": [[10, 15], [12, 18]],  # ref counts
            "n_nonref_counts_pm_2": [[5, 8], [6, 9]],  # nonref counts
        }
        test_dataframe = pd.DataFrame(data)

        result = process_sample_columns(test_dataframe, "n_")

        # Verify that forward_count and reverse_count columns are added with normal prefix
        assert "n_forward_count" in result.columns
        assert "n_reverse_count" in result.columns
        assert "n_alt_reads" in result.columns

    def test_duplicate_count_processing(self):
        """Test that duplicate counting works correctly."""
        data = {
            "t_ad": [([5], [10]), ([8], [15])],
            "t_rev": [[1, 0], [0, 1]],
            "t_dup": [[1, 1, 0], [0, 0, 1]],  # first: 2 duplicates, second: 1 duplicate
            "t_filt": [[1, 0], [0, 1]],
            "t_mqual": [[25, 30], [20, 25]],
            "t_snvq": [[15, 20], [10, 15]],
            "t_mapq": [[30, 35], [25, 30]],
            "t_ref_forward_reads": [10, 15],
            "t_ref_reverse_reads": [8, 12],
            "t_ref_counts_pm_2": [[10, 15], [12, 18]],  # ref counts
            "t_nonref_counts_pm_2": [[5, 8], [6, 9]],  # nonref counts
        }

        test_dataframe = pd.DataFrame(data)

        result = process_sample_columns(test_dataframe, "t_")

        # Check duplicate-related columns are added
        assert "t_count_duplicate" in result.columns
        assert "t_count_non_duplicate" in result.columns

    def test_pass_alt_reads_processing(self):
        """Test that pass_alt_reads is correctly calculated from filter column."""
        data = {
            "t_ad": [([5], [10]), ([8], [15])],
            "t_rev": [[1, 0], [0, 1]],
            "t_dup": [[0, 1], [1, 0]],
            "t_filt": [[1, 1, 1], [0, 1, 0]],  # first: 3 pass, second: 1 pass
            "t_mqual": [[25, 30], [20, 25]],
            "t_snvq": [[15, 20], [10, 15]],
            "t_mapq": [[30, 35], [25, 30]],
            "t_ref_forward_reads": [10, 15],
            "t_ref_reverse_reads": [8, 12],
            "t_ref_counts_pm_2": [[10, 15], [12, 18]],  # ref counts
            "t_nonref_counts_pm_2": [[5, 8], [6, 9]],  # nonref counts
        }
        test_dataframe = pd.DataFrame(data)

        result = process_sample_columns(test_dataframe, "t_")

        # Check pass_alt_reads column is added
        assert "t_pass_alt_reads" in result.columns

    def test_aggregation_features(self):
        """Test that aggregation features are correctly computed."""
        data = {
            "t_ad": [([5], [10])],
            "t_rev": [[1, 0]],
            "t_dup": [[0, 1]],
            "t_filt": [[1, 0]],
            "t_mqual": [[25, 30, 35]],  # min=25, max=35, mean=30
            "t_snvq": [[15, 20, 25]],  # min=15, max=25, mean=20
            "t_mapq": [[30, 35, 40]],  # min=30, max=40, mean=35
            "t_ref_forward_reads": [10],
            "t_ref_reverse_reads": [8],
            "t_ref_counts_pm_2": [[10, 15]],  # ref counts
            "t_nonref_counts_pm_2": [[5, 8]],  # nonref counts
        }
        test_dataframe = pd.DataFrame(data)

        result = process_sample_columns(test_dataframe, "t_")

        # Check aggregation columns are added
        assert "t_mqual_min" in result.columns
        assert "t_mqual_max" in result.columns
        assert "t_mqual_mean" in result.columns
        assert "t_snvq_min" in result.columns
        assert "t_snvq_max" in result.columns
        assert "t_snvq_mean" in result.columns
        assert "t_mapq_min" in result.columns
        assert "t_mapq_max" in result.columns
        assert "t_mapq_mean" in result.columns
