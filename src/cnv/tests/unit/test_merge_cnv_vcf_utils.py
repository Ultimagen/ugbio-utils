"""Tests for merge_cnv_vcf_utils module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pysam
import pytest
from ugbio_cnv import merge_cnv_vcf_utils
from ugbio_core import test_utils


@pytest.fixture
def resources_dir():
    """Fixture providing path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def cnv_vcf_header():
    """Fixture providing a standard CNV VCF header with all required fields."""
    header = pysam.VariantHeader()
    header.add_line("##fileformat=VCFv4.2")
    header.add_line("##contig=<ID=chr1,length=248956422>")
    header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">')
    header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="SV length">')
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">')
    header.add_line('##INFO=<ID=CollapseId,Number=1,Type=Integer,Description="Collapse ID">')
    header.add_line('##INFO=<ID=CNMOPS_COV_MEAN,Number=1,Type=Float,Description="Mean coverage">')
    header.add_line('##INFO=<ID=CNMOPS_COV_STDEV,Number=1,Type=Float,Description="Coverage stdev">')
    header.add_line('##INFO=<ID=CNMOPS_COHORT_MEAN,Number=1,Type=Float,Description="Cohort mean">')
    header.add_line('##INFO=<ID=CNMOPS_COHORT_STDEV,Number=1,Type=Float,Description="Cohort stdev">')
    header.add_line('##INFO=<ID=CopyNumber,Number=1,Type=Float,Description="Copy number">')
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    header.add_line('##FORMAT=<ID=CN,Number=1,Type=Float,Description="Copy number">')
    header.add_sample("test_sample")
    return header


@pytest.fixture
def tmp_vcf_with_cnvs(tmp_path, cnv_vcf_header):
    """Create a temporary VCF file with CNV records for testing."""
    vcf_path = tmp_path / "test_cnvs.vcf.gz"

    # Add chr2 contig to header
    cnv_vcf_header.add_line("##contig=<ID=chr2,length=242193529>")

    # Create VCF with CNV records
    with pysam.VariantFile(str(vcf_path), "w", header=cnv_vcf_header) as vcf:
        # CNV 1: DEL on chr1, 1000-2000 (length 1000)
        record1 = vcf.new_record()
        record1.contig = "chr1"
        record1.pos = 1000
        record1.stop = 2000
        record1.alleles = ("N", "<DEL>")
        record1.info["SVLEN"] = (1000,)
        record1.info["SVTYPE"] = "DEL"
        record1.info["CNMOPS_COV_MEAN"] = 10.5
        record1.info["CNMOPS_COV_STDEV"] = 2.1
        record1.info["CNMOPS_COHORT_MEAN"] = 20.0
        record1.info["CNMOPS_COHORT_STDEV"] = 3.5
        record1.info["CopyNumber"] = 1.0
        record1.samples["test_sample"]["GT"] = (0, 1)
        record1.samples["test_sample"]["CN"] = 1.0
        vcf.write(record1)

        # CNV 2: DEL on chr1, 2500-3500 (length 1000, close to CNV1)
        record2 = vcf.new_record()
        record2.contig = "chr1"
        record2.pos = 2500
        record2.stop = 3500
        record2.alleles = ("N", "<DEL>")
        record2.info["SVLEN"] = (1000,)
        record2.info["SVTYPE"] = "DEL"
        record2.info["CNMOPS_COV_MEAN"] = 11.0
        record2.info["CNMOPS_COV_STDEV"] = 2.3
        record2.info["CNMOPS_COHORT_MEAN"] = 20.5
        record2.info["CNMOPS_COHORT_STDEV"] = 3.6
        record2.info["CopyNumber"] = 1.1
        record2.samples["test_sample"]["GT"] = (0, 1)
        record2.samples["test_sample"]["CN"] = 1.1
        vcf.write(record2)

        # CNV 3: DUP on chr1, 10000-12000 (length 2000, far from others)
        record3 = vcf.new_record()
        record3.contig = "chr1"
        record3.pos = 10000
        record3.stop = 12000
        record3.alleles = ("N", "<DUP>")
        record3.info["SVLEN"] = (2000,)
        record3.info["SVTYPE"] = "DUP"
        record3.info["CNMOPS_COV_MEAN"] = 30.0
        record3.info["CNMOPS_COV_STDEV"] = 4.5
        record3.info["CNMOPS_COHORT_MEAN"] = 20.0
        record3.info["CNMOPS_COHORT_STDEV"] = 3.5
        record3.info["CopyNumber"] = 3.0
        record3.samples["test_sample"]["GT"] = (1, 1)
        record3.samples["test_sample"]["CN"] = 3.0
        vcf.write(record3)

    # Index the VCF
    pysam.tabix_index(str(vcf_path), preset="vcf", force=True)

    return vcf_path


class TestMergeCnvsInVcf:
    """Tests for merge_cnvs_in_vcf function."""

    @patch("ugbio_cnv.merge_cnv_vcf_utils.mu.cleanup_temp_files")
    @patch("ugbio_cnv.merge_cnv_vcf_utils.vcftools.get_vcf_df")
    @patch("ugbio_cnv.merge_cnv_vcf_utils.VcfUtils")
    def test_merge_cnvs_basic_workflow(
        self, mock_vcf_utils_class, mock_get_vcf_df, mock_cleanup, tmp_path, cnv_vcf_header
    ):
        """Test basic workflow of merge_cnvs_in_vcf function."""
        # Setup
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"
        collapse_vcf = tmp_path / "output.vcf.gz.collapse.tmp.vcf.gz"  # Must match function's naming
        removed_vcf = tmp_path / "removed.vcf.gz"

        # Create mock VcfUtils instance
        mock_vu = MagicMock()
        mock_vu.collapse_vcf.return_value = removed_vcf
        mock_vcf_utils_class.return_value = mock_vu

        # Create mock dataframe for removed records
        mock_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "pos": [1000, 2500],
                "svlen": [(1000,), (1000,)],
                "matchid": [(1.0,), (1.0,)],  # Same matchid means they were collapsed together
                "cnmops_cov_mean": [10.5, 11.0],
                "cnmops_cov_stdev": [2.1, 2.3],
                "cnmops_cohort_mean": [20.0, 20.5],
                "cnmops_cohort_stdev": [3.5, 3.6],
                "copynumber": [1.0, 1.1],
            }
        )
        mock_get_vcf_df.return_value = mock_df

        # Create a mock collapsed VCF file using the fixture header
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            # Merged record with CollapseId
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 1000
            record.stop = 3500
            record.alleles = ("N", "<DEL>")
            record.info["CollapseId"] = 1.0
            record.info["SVLEN"] = (2500,)
            record.info["SVTYPE"] = "DEL"
            record.info["CNMOPS_COV_MEAN"] = 10.5
            record.info["CNMOPS_COV_STDEV"] = 2.1
            record.info["CNMOPS_COHORT_MEAN"] = 20.0
            record.info["CNMOPS_COHORT_STDEV"] = 3.5
            record.info["CopyNumber"] = 1.0
            record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record)

        # Create dummy input file (not read by test)
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass

        # Execute
        merge_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000)

        # Verify collapse_vcf was called with correct parameters
        mock_vu.collapse_vcf.assert_called_once_with(
            vcf=str(input_vcf),
            output_vcf=str(output_vcf) + ".collapse.tmp.vcf.gz",
            refdist=1000,
            pctseq=0.0,
            pctsize=0.0,
            ignore_filter=True,
            ignore_type=False,
            erase_removed=False,
        )

        # Verify get_vcf_df was called
        expected_fields = [
            "CNMOPS_COV_MEAN",
            "CNMOPS_COV_STDEV",
            "CNMOPS_COHORT_MEAN",
            "CNMOPS_COHORT_STDEV",
            "CopyNumber",
            "SVLEN",
            "MatchId",
        ]
        mock_get_vcf_df.assert_called_once_with(str(removed_vcf), custom_info_fields=expected_fields)

        # Verify cleanup was called
        mock_cleanup.assert_called_once()
        cleanup_args = mock_cleanup.call_args[0][0]
        assert str(output_vcf) + ".collapse.tmp.vcf.gz" in cleanup_args
        assert str(removed_vcf) in cleanup_args

        # Verify output file was created
        assert output_vcf.exists()

    @patch("ugbio_cnv.merge_cnv_vcf_utils.mu.cleanup_temp_files")
    @patch("ugbio_cnv.merge_cnv_vcf_utils.vcftools.get_vcf_df")
    @patch("ugbio_cnv.merge_cnv_vcf_utils.VcfUtils")
    def test_merge_cnvs_weighted_average_calculation(
        self, mock_vcf_utils_class, mock_get_vcf_df, mock_cleanup, tmp_path, cnv_vcf_header
    ):
        """Test that weighted averages are calculated correctly based on SVLEN."""
        # Setup
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"
        collapse_vcf = tmp_path / "output.vcf.gz.collapse.tmp.vcf.gz"  # Must match function's naming
        removed_vcf = tmp_path / "removed.vcf.gz"

        # Create mock VcfUtils instance
        mock_vu = MagicMock()
        mock_vu.collapse_vcf.return_value = removed_vcf
        mock_vcf_utils_class.return_value = mock_vu

        # Create mock dataframe with two records that were merged
        # Record 1: pos=1000, length 1000, end=1999 (1000+1000-1), value 10.0
        # Record 2: pos=2500, length 2000, end=4499 (2500+2000-1), value 20.0
        # Expected weighted average: (10*1000 + 20*2000) / (1000+2000) = 50000/3000 = 16.667
        mock_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "pos": [1000, 2500],
                "svlen": [(1000,), (2000,)],
                "matchid": [(1.0,), (1.0,)],
                "cnmops_cov_mean": [10.0, 20.0],
                "cnmops_cov_stdev": [1.0, 2.0],
                "cnmops_cohort_mean": [15.0, 25.0],
                "cnmops_cohort_stdev": [2.0, 3.0],
                "copynumber": [1.0, 2.0],
            }
        )
        mock_get_vcf_df.return_value = mock_df

        # Create collapsed VCF with a merged record using the fixture header
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 1000
            record.stop = 3500
            record.alleles = ("N", "<DEL>")
            record.info["CollapseId"] = 1.0
            record.info["SVLEN"] = (1500,)  # Original merged length
            record.info["SVTYPE"] = "DEL"
            record.info["CNMOPS_COV_MEAN"] = 15.0  # Will be replaced by weighted avg
            record.info["CNMOPS_COV_STDEV"] = 1.5
            record.info["CNMOPS_COHORT_MEAN"] = 20.0
            record.info["CNMOPS_COHORT_STDEV"] = 2.5
            record.info["CopyNumber"] = 1.5
            record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record)

        # Create dummy input file
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass

        # Execute
        merge_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000)

        # Read output and verify weighted averages
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1

            record = records[0]

            # Calculate expected weighted average for CNMOPS_COV_MEAN
            # Values: [10.0, 20.0, 15.0] (from removed records + collapsed record)
            # Lengths: [1000, 2000, 1500]
            # Weighted avg = (10*1000 + 20*2000 + 15*1500) / (1000+2000+1500) = 72500/4500 = 16.111
            expected_mean = (10.0 * 1000 + 20.0 * 2000 + 15.0 * 1500) / (1000 + 2000 + 1500)
            assert abs(record.info["CNMOPS_COV_MEAN"] - round(expected_mean, 3)) < 0.001

            # Verify SVLEN was updated to record.stop - record.start
            assert record.info["SVLEN"] == (record.stop - record.start,)

            # Verify END was updated to max of merged records
            # max end = 4499 (from pos=2500 + svlen=2000 - 1)
            assert record.stop == 4499


class TestMergeCnvsInVcfIntegration:
    """Integration tests for merge_cnvs_in_vcf using real data."""

    def test_merge_cnvs_real_data(self, resources_dir, tmp_path):
        """Test merge_cnvs_in_vcf with real HG002 data files."""
        # Input and expected output files
        input_vcf = resources_dir / "merge_cnv_input.vcf.gz"
        expected_output_vcf = resources_dir / "merge_cnv_output.vcf.gz"

        # Create output file in tmp directory
        output_vcf = tmp_path / "test_output.vcf.gz"

        merge_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1500)
        # Verify output matches expected
        test_utils.compare_vcfs(str(expected_output_vcf), str(output_vcf))
