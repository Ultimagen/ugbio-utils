from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pysam
import pytest
import ugbio_core.test_utils as test_utils
from ugbio_cnv import combine_cnv_vcf_utils


@pytest.fixture
def resources_dir():
    """Fixture providing path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


class TestCombineVcfHeadersForCnv:
    """Tests for combine_vcf_headers_for_cnv function."""

    def test_combine_vcf_headers_basic(self, resources_dir):
        """Test that combine_vcf_headers_for_cnv enforces SVLEN and SVTYPE specifications"""
        vcf1_path = resources_dir / "sample1_test.500.CNV.vcf.gz"

        # Skip if test files don't exist
        if not vcf1_path.exists():
            pytest.skip("Test VCF files not available")

        # Read headers from both files
        with pysam.VariantFile(str(vcf1_path), "r") as vcf1:
            header1 = vcf1.header

        # Create a second header for testing with matching sample name
        # Get the sample name from header1
        sample_name = list(header1.samples)[0] if header1.samples else "sample1"

        header2 = pysam.VariantHeader()
        header2.add_line("##contig=<ID=chr1,length=248956422>")
        header2.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
        header2.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type. can be DUP or DEL">')
        header2.add_line('##INFO=<ID=CopyNumber,Number=1,Type=Float,Description="Copy number">')
        header2.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header2.add_sample(sample_name)

        # Combine the headers (header1 has Number=1 for SVLEN, should be enforced to ".")
        combined_header = combine_cnv_vcf_utils.combine_vcf_headers_for_cnv(header1, header2)

        # Verify enforced specifications
        assert "SVLEN" in combined_header.info
        assert combined_header.info["SVLEN"].number == "."  # Enforced to "."
        assert combined_header.info["SVLEN"].type == "Integer"

        assert "SVTYPE" in combined_header.info
        assert combined_header.info["SVTYPE"].number == 1  # Enforced to 1
        assert combined_header.info["SVTYPE"].type == "String"

        # Verify other fields from both headers are present
        assert "CopyNumber" in combined_header.info

        # Verify contigs
        assert "chr1" in combined_header.contigs

        # Verify samples from both headers
        assert len(combined_header.samples) >= 1

    def test_combine_vcf_headers_collision_compatible(self):
        """Test that compatible collisions (same type and number) are handled correctly"""
        # Create two headers with the same INFO field and same sample name
        header1 = pysam.VariantHeader()
        header1.add_line("##contig=<ID=chr1,length=248956422>")
        header1.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">')
        header1.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header1.add_sample("sample1")

        header2 = pysam.VariantHeader()
        header2.add_line("##contig=<ID=chr1,length=248956422>")
        header2.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Different Description">')
        header2.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header2.add_sample("sample1")  # Same sample name

        # Should not raise an error - same type and number
        combined_header = combine_cnv_vcf_utils.combine_vcf_headers_for_cnv(header1, header2)

        # Verify the first definition is used
        assert "DP" in combined_header.info
        assert combined_header.info["DP"].type == "Integer"
        assert combined_header.info["DP"].number == 1
        assert combined_header.info["DP"].description == "Total Depth"  # From header1

        # Verify sample is present
        assert "sample1" in combined_header.samples

    def test_combine_vcf_headers_collision_different_type(self):
        """Test that collisions with different types raise RuntimeError"""
        # Create two headers with conflicting INFO types
        header1 = pysam.VariantHeader()
        header1.add_line("##contig=<ID=chr1,length=248956422>")
        header1.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">')

        header2 = pysam.VariantHeader()
        header2.add_line("##contig=<ID=chr1,length=248956422>")
        header2.add_line('##INFO=<ID=DP,Number=1,Type=Float,Description="Total Depth">')

        # Should raise RuntimeError due to different types
        with pytest.raises(RuntimeError, match="INFO field 'DP' has conflicting types"):
            combine_cnv_vcf_utils.combine_vcf_headers_for_cnv(header1, header2)

    def test_combine_vcf_headers_collision_different_number(self):
        """Test that collisions with different numbers raise RuntimeError"""
        # Create two headers with conflicting INFO numbers
        header1 = pysam.VariantHeader()
        header1.add_line("##contig=<ID=chr1,length=248956422>")
        header1.add_line('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">')

        header2 = pysam.VariantHeader()
        header2.add_line("##contig=<ID=chr1,length=248956422>")
        header2.add_line('##INFO=<ID=AF,Number=1,Type=Float,Description="Allele Frequency">')

        # Should raise RuntimeError due to different numbers
        with pytest.raises(RuntimeError, match="INFO field 'AF' has conflicting numbers"):
            combine_cnv_vcf_utils.combine_vcf_headers_for_cnv(header1, header2)

    def test_combine_vcf_headers_enforced_svlen_svtype(self):
        """Test that SVLEN and SVTYPE specifications are enforced even with conflicts"""
        # Create headers with conflicting SVLEN/SVTYPE specifications
        header1 = pysam.VariantHeader()
        header1.add_line("##contig=<ID=chr1,length=248956422>")
        header1.add_line('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Wrong spec for SVLEN">')
        header1.add_line('##INFO=<ID=SVTYPE,Number=.,Type=String,Description="Wrong spec for SVTYPE">')

        header2 = pysam.VariantHeader()
        header2.add_line("##contig=<ID=chr1,length=248956422>")
        header2.add_line('##INFO=<ID=SVLEN,Number=A,Type=Float,Description="Another wrong spec">')
        header2.add_line('##INFO=<ID=SVTYPE,Number=R,Type=Integer,Description="Another wrong spec">')

        # Should not raise an error because enforcement overrides conflicts
        combined_header = combine_cnv_vcf_utils.combine_vcf_headers_for_cnv(header1, header2)

        # Verify enforced specifications are applied
        assert "SVLEN" in combined_header.info
        assert combined_header.info["SVLEN"].number == "."
        assert combined_header.info["SVLEN"].type == "Integer"

        assert "SVTYPE" in combined_header.info
        assert combined_header.info["SVTYPE"].number == 1
        assert combined_header.info["SVTYPE"].type == "String"

    def test_combine_vcf_headers_filters_kept_when_requested(self):
        """Test that FILTER fields are combined when keep_filters=True"""
        # Create headers with FILTER fields
        header1 = pysam.VariantHeader()
        header1.add_line("##contig=<ID=chr1,length=248956422>")
        header1.add_line('##FILTER=<ID=PASS,Description="All filters passed">')
        header1.add_line('##FILTER=<ID=LowQual,Description="Low quality">')
        header1.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">')

        header2 = pysam.VariantHeader()
        header2.add_line("##contig=<ID=chr1,length=248956422>")
        header2.add_line('##FILTER=<ID=PASS,Description="All filters passed">')
        header2.add_line('##FILTER=<ID=HighCoverage,Description="High coverage region">')
        header2.add_line('##INFO=<ID=AF,Number=1,Type=Float,Description="Allele Frequency">')

        # Combine with keep_filters=True
        combined_header = combine_cnv_vcf_utils.combine_vcf_headers_for_cnv(header1, header2, keep_filters=True)

        # Verify FILTER fields from both headers are present
        assert "LowQual" in combined_header.filters
        assert "HighCoverage" in combined_header.filters
        assert combined_header.filters["LowQual"].description == "Low quality"
        assert combined_header.filters["HighCoverage"].description == "High coverage region"

        # Verify INFO fields are still present
        assert "DP" in combined_header.info
        assert "AF" in combined_header.info


class TestWriteVcfRecordsWithSource:
    """Tests for write_vcf_records_with_source function."""

    def test_write_vcf_records_with_source_basic(self, resources_dir, tmp_path):
        """Test that write_vcf_records_with_source adds CNV_SOURCE and clears filters"""
        # Use an existing test VCF file
        vcf_path = resources_dir / "sample1_test.500.CNV.vcf.gz"

        # Skip if test file doesn't exist
        if not vcf_path.exists():
            pytest.skip("Test VCF file not available")

        # Create a combined header that includes CNV_SOURCE
        combined_header = pysam.VariantHeader()
        combined_header.add_line("##fileformat=VCFv4.2")
        combined_header.add_line("##contig=<ID=chr1,length=248956422>")
        combined_header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">')
        combined_header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="SV length">')
        combined_header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">')
        combined_header.add_line('##INFO=<ID=CNV_SOURCE,Number=.,Type=String,Description="Source of CNV call">')
        combined_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        combined_header.add_line('##FORMAT=<ID=CN,Number=1,Type=Float,Description="Copy number">')
        combined_header.add_sample("sample1_test")

        # Open input and output VCF files
        output_vcf_path = tmp_path / "output.vcf.gz"
        with pysam.VariantFile(str(vcf_path), "r") as vcf_in:
            with pysam.VariantFile(str(output_vcf_path), "w", header=combined_header) as vcf_out:
                # Write first 5 records with source annotation
                source_name = "CNVpytor"
                record_count = 0
                for record in vcf_in:
                    if record_count >= 5:
                        break
                    combine_cnv_vcf_utils.write_vcf_records_with_source(vcf_in, vcf_out, combined_header, source_name)
                    record_count += 1
                    break  # write_vcf_records_with_source processes all records internally

        # Read the output file and verify
        with pysam.VariantFile(str(output_vcf_path), "r") as vcf_result:
            records = list(vcf_result)
            assert len(records) > 0

            # Check that CNV_SOURCE is present in all records
            for record in records:
                assert "CNV_SOURCE" in record.info
                assert record.info["CNV_SOURCE"] == ("CNVpytor",)

                # Check that filters are cleared (should be empty or only PASS)
                # In pysam, an empty filter set means PASS
                assert len(record.filter) == 0 or record.filter.keys() == ["PASS"]

    def test_write_vcf_records_with_source_preserves_existing_source(self, tmp_path):
        """Test that write_vcf_records_with_source doesn't overwrite existing CNV_SOURCE"""
        # Create a test VCF with existing CNV_SOURCE
        input_header = pysam.VariantHeader()
        input_header.add_line("##fileformat=VCFv4.2")
        input_header.add_line("##contig=<ID=chr1,length=248956422>")
        input_header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">')
        input_header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="SV length">')
        input_header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">')
        input_header.add_line('##INFO=<ID=CNV_SOURCE,Number=.,Type=String,Description="Source of CNV call">')
        input_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        input_header.add_sample("test_sample")

        # Create input VCF with records that already have CNV_SOURCE
        input_vcf_path = tmp_path / "input_with_source.vcf.gz"
        with pysam.VariantFile(str(input_vcf_path), "w", header=input_header) as vcf_in:
            # Add a record with existing CNV_SOURCE
            record = vcf_in.new_record()
            record.contig = "chr1"
            record.pos = 1000
            record.stop = 2000  # END is a reserved attribute, use record.stop
            record.alleles = ("N", "<DEL>")
            record.info["SVLEN"] = 1000
            record.info["SVTYPE"] = "DEL"
            record.info["CNV_SOURCE"] = ("ExistingCaller",)
            record.samples["test_sample"]["GT"] = (1, 1)
            vcf_in.write(record)

        # Process the file
        output_vcf_path = tmp_path / "output_preserve_source.vcf.gz"
        with pysam.VariantFile(str(input_vcf_path), "r") as vcf_in:
            with pysam.VariantFile(str(output_vcf_path), "w", header=input_header) as vcf_out:
                combine_cnv_vcf_utils.write_vcf_records_with_source(vcf_in, vcf_out, input_header, "NewCaller")

        # Verify that existing CNV_SOURCE is preserved
        with pysam.VariantFile(str(output_vcf_path), "r") as vcf_result:
            records = list(vcf_result)
            assert len(records) == 1
            # Original CNV_SOURCE should be preserved
            assert "CNV_SOURCE" in records[0].info
            assert records[0].info["CNV_SOURCE"] == ("ExistingCaller",)

    def test_write_vcf_records_with_source_clears_filters(self, tmp_path):
        """Test that write_vcf_records_with_source clears all FILTER values"""
        # Create a test VCF with various filter values
        input_header = pysam.VariantHeader()
        input_header.add_line("##fileformat=VCFv4.2")
        input_header.add_line("##contig=<ID=chr1,length=248956422>")
        input_header.add_line('##FILTER=<ID=PASS,Description="All filters passed">')
        input_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')
        input_header.add_line('##FILTER=<ID=HighCoverage,Description="High coverage">')
        input_header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">')
        input_header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">')
        input_header.add_line('##INFO=<ID=CNV_SOURCE,Number=.,Type=String,Description="Source of CNV call">')
        input_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        input_header.add_sample("test_sample")

        # Create combined header without FILTER definitions (as per function behavior)
        combined_header = pysam.VariantHeader()
        combined_header.add_line("##fileformat=VCFv4.2")
        combined_header.add_line("##contig=<ID=chr1,length=248956422>")
        combined_header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">')
        combined_header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">')
        combined_header.add_line('##INFO=<ID=CNV_SOURCE,Number=.,Type=String,Description="Source of CNV call">')
        combined_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        combined_header.add_sample("test_sample")

        # Create input VCF with various filter states
        input_vcf_path = tmp_path / "input_with_filters.vcf.gz"
        with pysam.VariantFile(str(input_vcf_path), "w", header=input_header) as vcf_in:
            # Record with PASS filter
            record1 = vcf_in.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 2000  # END is a reserved attribute, use record.stop
            record1.alleles = ("N", "<DEL>")
            record1.info["SVTYPE"] = "DEL"
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (1, 1)
            vcf_in.write(record1)

            # Record with LowQual filter
            record2 = vcf_in.new_record()
            record2.contig = "chr1"
            record2.pos = 5000
            record2.stop = 6000  # END is a reserved attribute, use record.stop
            record2.alleles = ("N", "<DUP>")
            record2.info["SVTYPE"] = "DUP"
            record2.filter.add("LowQual")
            record2.samples["test_sample"]["GT"] = (0, 1)
            vcf_in.write(record2)

            # Record with multiple filters
            record3 = vcf_in.new_record()
            record3.contig = "chr1"
            record3.pos = 10000
            record3.stop = 11000  # END is a reserved attribute, use record.stop
            record3.alleles = ("N", "<DEL>")
            record3.info["SVTYPE"] = "DEL"
            record3.filter.add("LowQual")
            record3.filter.add("HighCoverage")
            record3.samples["test_sample"]["GT"] = (1, 1)
            vcf_in.write(record3)

        # Process the file
        output_vcf_path = tmp_path / "output_cleared_filters.vcf.gz"
        with pysam.VariantFile(str(input_vcf_path), "r") as vcf_in:
            with pysam.VariantFile(str(output_vcf_path), "w", header=combined_header) as vcf_out:
                combine_cnv_vcf_utils.write_vcf_records_with_source(vcf_in, vcf_out, combined_header, "TestCaller")

        # Verify that all filters are cleared
        with pysam.VariantFile(str(output_vcf_path), "r") as vcf_result:
            records = list(vcf_result)
            assert len(records) == 3

            for record in records:
                # All filters should be cleared (empty filter set means PASS in pysam)
                assert len(record.filter) == 0 or record.filter.keys() == ["PASS"]

                # CNV_SOURCE should be added
                assert "CNV_SOURCE" in record.info
                assert record.info["CNV_SOURCE"] == ("TestCaller",)


@pytest.fixture
def cnv_vcf_header():
    """Fixture providing a standard CNV VCF header with all required fields."""
    header = pysam.VariantHeader()
    header.add_line("##fileformat=VCFv4.2")
    header.add_line("##contig=<ID=chr1,length=248956422>")
    header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">')
    header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="SV length">')
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">')
    header.add_line('##INFO=<ID=CollapseId,Number=1,Type=String,Description="Truvari collapse ID">')
    header.add_line('##INFO=<ID=CNMOPS_SAMPLE_MEAN,Number=1,Type=Float,Description="Mean coverage">')
    header.add_line('##INFO=<ID=CNMOPS_SAMPLE_STDEV,Number=1,Type=Float,Description="Coverage stdev">')
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
        record1.info["CNMOPS_SAMPLE_MEAN"] = 10.5
        record1.info["CNMOPS_SAMPLE_STDEV"] = 2.1
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
        record2.info["CNMOPS_SAMPLE_MEAN"] = 11.0
        record2.info["CNMOPS_SAMPLE_STDEV"] = 2.3
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
        record3.info["CNMOPS_SAMPLE_MEAN"] = 30.0
        record3.info["CNMOPS_SAMPLE_STDEV"] = 4.5
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

    @patch("ugbio_cnv.combine_cnv_vcf_utils.mu.cleanup_temp_files")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.vcftools.get_vcf_df")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.VcfUtils")
    def test_merge_cnvs_basic_workflow(
        self, mock_vcf_utils_class, mock_get_vcf_df, mock_cleanup, tmp_path, cnv_vcf_header
    ):
        """Test basic workflow of merge_cnvs_in_vcf function."""
        # Setup
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"
        collapse_vcf = tmp_path / "output.vcf.gz.collapse.tmp.vcf.gz"  # Must match function's naming
        collapse_sorted_vcf = tmp_path / "output.vcf.gz.collapse.sort.tmp.vcf.gz"  # After sorting
        removed_vcf = tmp_path / "removed.vcf.gz"

        # Create mock VcfUtils instance
        mock_vu = MagicMock()
        mock_vu.collapse_vcf.return_value = removed_vcf

        # Mock sort_vcf to create the output file
        def mock_sort_vcf(input_path, output_path):
            # Copy the unsorted file to the output path
            with pysam.VariantFile(str(input_path), "r") as vcf_in:
                with pysam.VariantFile(str(output_path), "w", header=vcf_in.header) as vcf_out:
                    for record in vcf_in:
                        vcf_out.write(record)

        mock_vu.sort_vcf.side_effect = mock_sort_vcf
        mock_vcf_utils_class.return_value = mock_vu

        # Create mock dataframe for removed records
        mock_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "pos": [1000, 2500],
                "svlen": [(1000,), (1000,)],
                "filter": ["PASS", "PASS"],
                "matchid": [(1.0,), (1.0,)],  # Same matchid means they were collapsed together
                "cnmops_sample_mean": [10.5, 11.0],
                "cnmops_sample_stdev": [2.1, 2.3],
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
            record.info["CollapseId"] = "1"
            record.info["SVLEN"] = (2500,)
            record.info["SVTYPE"] = "DEL"
            record.info["CNMOPS_SAMPLE_MEAN"] = 10.5
            record.info["CNMOPS_SAMPLE_STDEV"] = 2.1
            record.info["CNMOPS_COHORT_MEAN"] = 20.0
            record.info["CNMOPS_COHORT_STDEV"] = 3.5
            record.info["CopyNumber"] = 1.0
            record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record)

        # Create the sorted VCF file (copy of collapsed VCF for this test)
        with pysam.VariantFile(str(collapse_vcf), "r") as vcf_in:
            with pysam.VariantFile(str(collapse_sorted_vcf), "w", header=vcf_in.header) as vcf_out:
                for record in vcf_in:
                    vcf_out.write(record)

        # Create dummy input file (not read by test)
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass

        # Execute
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000)

        # Verify collapse_vcf was called with correct parameters
        mock_vu.collapse_vcf.assert_called_once_with(
            vcf=str(input_vcf),
            output_vcf=str(output_vcf) + ".collapse.tmp.vcf.gz",
            refdist=1000,
            pctseq=0.0,
            pctsize=0.0,
            maxsize=-1,
            ignore_filter=True,
            ignore_sv_type=False,
            pick_best=False,
            erase_removed=False,
        )

        # Verify get_vcf_df was called
        expected_fields = [
            "CNMOPS_SAMPLE_MEAN",
            "CNMOPS_SAMPLE_STDEV",
            "CNMOPS_COHORT_MEAN",
            "CNMOPS_COHORT_STDEV",
            "CopyNumber",
            "GAP_PERCENTAGE",
            "JALIGN_DEL_SUPPORT",
            "JALIGN_DUP_SUPPORT",
            "JALIGN_DEL_SUPPORT_STRONG",
            "JALIGN_DUP_SUPPORT_STRONG",
            "TREE_SCORE",
            "CNV_SOURCE",
            "CIPOS",
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

    @patch("ugbio_cnv.combine_cnv_vcf_utils.mu.cleanup_temp_files")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.vcftools.get_vcf_df")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.VcfUtils")
    def test_merge_cnvs_weighted_average_calculation(
        self, mock_vcf_utils_class, mock_get_vcf_df, mock_cleanup, tmp_path, cnv_vcf_header
    ):
        """Test that weighted averages are calculated correctly based on SVLEN."""
        # Setup
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"
        collapse_vcf = tmp_path / "output.vcf.gz.collapse.tmp.vcf.gz"  # Must match function's naming
        collapse_sorted_vcf = tmp_path / "output.vcf.gz.collapse.sort.tmp.vcf.gz"  # After sorting
        removed_vcf = tmp_path / "removed.vcf.gz"

        # Create mock VcfUtils instance
        mock_vu = MagicMock()
        mock_vu.collapse_vcf.return_value = removed_vcf

        # Mock sort_vcf to create the output file
        def mock_sort_vcf(input_path, output_path):
            # Copy the unsorted file to the output path
            with pysam.VariantFile(str(input_path), "r") as vcf_in:
                with pysam.VariantFile(str(output_path), "w", header=vcf_in.header) as vcf_out:
                    for record in vcf_in:
                        vcf_out.write(record)

        mock_vu.sort_vcf.side_effect = mock_sort_vcf
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
                "filter": ["PASS", "PASS"],
                "matchid": [(1.0,), (1.0,)],
                "cnmops_sample_mean": [10.0, 20.0],
                "cnmops_sample_stdev": [1.0, 2.0],
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
            record.info["CollapseId"] = "1"
            record.info["SVLEN"] = (1500,)  # Original merged length
            record.info["SVTYPE"] = "DEL"
            record.filter.add("PASS")
            record.info["CNMOPS_SAMPLE_MEAN"] = 15.0  # Will be replaced by weighted avg
            record.info["CNMOPS_SAMPLE_STDEV"] = 1.5
            record.info["CNMOPS_COHORT_MEAN"] = 20.0
            record.info["CNMOPS_COHORT_STDEV"] = 2.5
            record.info["CopyNumber"] = 1.5
            record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record)

        # Create the sorted VCF file (copy of collapsed VCF for this test)
        with pysam.VariantFile(str(collapse_vcf), "r") as vcf_in:
            with pysam.VariantFile(str(collapse_sorted_vcf), "w", header=vcf_in.header) as vcf_out:
                for record in vcf_in:
                    vcf_out.write(record)

        # Create dummy input file
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass

        # Execute
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000)

        # Read output and verify weighted averages
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1

            record = records[0]

            # Calculate expected weighted average for CNMOPS_SAMPLE_MEAN
            # Values: [10.0, 20.0, 15.0] (from removed records + collapsed record)
            # Lengths: [1000, 2000, 1500]
            # Weighted avg = (10*1000 + 20*2000 + 15*1500) / (1000+2000+1500) = 72500/4500 = 16.111
            expected_mean = (10.0 * 1000 + 20.0 * 2000 + 15.0 * 1500) / (1000 + 2000 + 1500)
            assert abs(record.info["CNMOPS_SAMPLE_MEAN"] - round(expected_mean, 3)) < 0.001

            # Verify SVLEN was updated to record.stop - record.start
            assert record.info["SVLEN"] == (record.stop - record.start,)

            # Verify END was updated to max of merged records
            # max end = 4499 (from pos=2500 + svlen=2000 - 1)
            assert record.stop == 4499

    @patch("ugbio_cnv.combine_cnv_vcf_utils.mu.cleanup_temp_files")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.vcftools.get_vcf_df")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.VcfUtils")
    def test_merge_cnvs_cipos_aggregation(
        self, mock_vcf_utils_class, mock_get_vcf_df, mock_cleanup, tmp_path, cnv_vcf_header
    ):
        """Test CIPOS aggregation using the minlength strategy.

        The CIPOS confidence interval is selected as the tightest
        (shortest-width) interval across the merged records, rather
        than by taking an element-wise minimum.
        """
        # Add CIPOS to header
        cnv_vcf_header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS">')

        # Setup
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"
        collapse_vcf = tmp_path / "output.vcf.gz.collapse.tmp.vcf.gz"
        removed_vcf = tmp_path / "removed.vcf.gz"

        # Create mock VcfUtils instance
        mock_vu = MagicMock()
        mock_vu.collapse_vcf.return_value = removed_vcf

        def mock_sort_vcf(input_path, output_path):
            with pysam.VariantFile(str(input_path), "r") as vcf_in:
                with pysam.VariantFile(str(output_path), "w", header=vcf_in.header) as vcf_out:
                    for record in vcf_in:
                        vcf_out.write(record)

        mock_vu.sort_vcf.side_effect = mock_sort_vcf
        mock_vcf_utils_class.return_value = mock_vu

        # Create mock dataframe with two records that have CIPOS values
        # Record 1: CIPOS=(-250, 251) with length = 251 - (-250) = 501
        # Record 2: CIPOS=(-300, 301) with length = 301 - (-300) = 601
        # Expected merged CIPOS: tuple with min length = (-250, 251)
        mock_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "pos": [1000, 2500],
                "svlen": [(1000,), (1000,)],
                "matchid": [(1.0,), (1.0,)],
                "cipos": [(-250, 251), (-300, 301)],
                "cnmops_sample_mean": [10.0, 11.0],
            }
        )
        mock_get_vcf_df.return_value = mock_df

        # Create collapsed VCF with CIPOS
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 1000
            record.stop = 3500
            record.alleles = ("N", "<DEL>")
            record.info["CollapseId"] = "1.0"
            record.info["SVLEN"] = (2500,)
            record.info["SVTYPE"] = "DEL"
            record.info["CIPOS"] = (-200, 200)  # Will be replaced by min-length aggregation
            record.info["CNMOPS_SAMPLE_MEAN"] = 10.5
            record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record)

        # Create dummy input file
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass

        # Execute
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000)

        # Read output and verify CIPOS aggregation
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1

            record = records[0]

            # Verify CIPOS was aggregated correctly using minlength
            # Input values: [(-250, 251), (-300, 301), (-200, 200)]
            # Lengths: [501, 601, 400]
            # Expected: tuple with min length = (-200, 200)
            assert "CIPOS" in record.info, "CIPOS should be present in merged record"
            cipos = record.info["CIPOS"]
            assert cipos == (-200, 200), f"Expected CIPOS=(-200, 200), got {cipos}"


def make_cnv_record(vcf, contig, pos, stop, record_id, svtype="DEL", svlen=None, qual=50.0, filter_val="PASS", **info):
    """
    Helper function to create CNV records with minimal boilerplate.

    Parameters
    ----------
    vcf : pysam.VariantFile
        The VCF file object to create the record from
    contig : str
        Chromosome name
    pos : int
        Start position (1-based)
    stop : int
        Stop position (1-based)
    record_id : str
        Variant ID
    svtype : str, optional
        SV type (DEL, DUP, etc.), by default "DEL"
    svlen : int, optional
        SV length, calculated from stop-pos if not provided
    qual : float, optional
        Quality score, by default 50.0
    filter_val : str, optional
        Filter value ("PASS", "LowQual", etc.), by default "PASS"
    **info : dict
        Additional INFO fields (e.g., CNMOPS_SAMPLE_MEAN=2.5)

    Returns
    -------
    pysam.VariantRecord
        The created variant record
    """
    record = vcf.new_record()
    record.contig = contig
    record.pos = pos
    record.stop = stop
    record.id = record_id
    record.alleles = ("N", f"<{svtype}>")
    record.info["SVTYPE"] = svtype
    record.info["SVLEN"] = (svlen if svlen is not None else stop - pos,)
    record.qual = qual
    if filter_val:
        record.filter.add(filter_val)
    record.samples["test_sample"]["GT"] = (0, 1)

    # Add any additional INFO fields
    for key, value in info.items():
        record.info[key] = value

    return record


class TestMergeCnvsInVcfTwoStage:
    """Integration tests for two-stage merge with ignore_filter=False."""

    def test_two_stage_pass_variants_only(self, tmp_path, cnv_vcf_header):
        """Test two-stage merge with only PASS variants."""
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"

        # Create input VCF with 3 overlapping PASS DEL variants
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            vcf.write(make_cnv_record(vcf, "chr1", 1000, 2000, "CNV1", qual=50.0, CNMOPS_SAMPLE_MEAN=2.5))
            vcf.write(make_cnv_record(vcf, "chr1", 1500, 2500, "CNV2", qual=45.0, CNMOPS_SAMPLE_MEAN=3.0))
            vcf.write(make_cnv_record(vcf, "chr1", 2000, 3000, "CNV3", qual=40.0, CNMOPS_SAMPLE_MEAN=2.8))

        # Index the VCF file
        pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

        # Execute with ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)

        # Verify: 3 PASS variants should merge into 1
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert records[0].filter.keys() == ["PASS"]
            # Verify aggregation occurred
            assert "CNMOPS_SAMPLE_MEAN" in records[0].info

    def test_two_stage_filtered_variants_only(self, tmp_path, cnv_vcf_header):
        """Test two-stage merge with only filtered variants - only first variant kept."""
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"

        # Add FILTER line to header
        cnv_vcf_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')

        # Create input VCF with 3 overlapping filtered variants
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            vcf.write(make_cnv_record(vcf, "chr1", 1000, 2000, "CNV1", svtype="DUP", qual=15.0, filter_val="LowQual"))
            vcf.write(make_cnv_record(vcf, "chr1", 1500, 2500, "CNV2", svtype="DUP", qual=12.0, filter_val="LowQual"))
            vcf.write(make_cnv_record(vcf, "chr1", 2000, 3000, "CNV3", svtype="DUP", qual=10.0, filter_val="LowQual"))

        # Index the VCF file
        pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

        # Execute with ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)

        # Verify: Only first variant should remain (merged)
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert records[0].filter.keys() == ["LowQual"]

    def test_two_stage_mixed_non_overlapping(self, tmp_path, cnv_vcf_header):
        """Test two-stage merge with PASS and non-overlapping filtered variants."""
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"

        # Add FILTER line to header
        cnv_vcf_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')

        # Create input VCF
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            # PASS variant 1
            record1 = vcf.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 2000
            record1.id = "CNV1"
            record1.alleles = ("N", "<DEL>")
            record1.info["SVLEN"] = (1000,)
            record1.info["SVTYPE"] = "DEL"
            record1.qual = 50.0
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record1)

            # PASS variant 2 (overlapping with variant 1)
            record2 = vcf.new_record()
            record2.contig = "chr1"
            record2.pos = 1500
            record2.stop = 2500
            record2.id = "CNV2"
            record2.alleles = ("N", "<DEL>")
            record2.info["SVLEN"] = (1000,)
            record2.info["SVTYPE"] = "DEL"
            record2.qual = 45.0
            record2.filter.add("PASS")
            record2.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record2)

            # Filtered variant (far away, distance > 1000)
            record3 = vcf.new_record()
            record3.contig = "chr1"
            record3.pos = 10000
            record3.stop = 11000
            record3.id = "CNV3"
            record3.alleles = ("N", "<DEL>")
            record3.info["SVLEN"] = (1000,)
            record3.info["SVTYPE"] = "DEL"
            record3.qual = 20.0
            record3.filter.add("LowQual")
            record3.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record3)

        # Index the VCF file
        pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

        # Execute with ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)

        # Verify: 2 records (1 merged PASS, 1 filtered preserved)
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 2
            # Check we have one PASS and one filtered
            filters = [rec.filter.keys() for rec in records]
            assert ["PASS"] in filters
            assert ["LowQual"] in filters

    def test_two_stage_overlapping_filtered_removed(self, tmp_path, cnv_vcf_header):
        """Test that overlapping filtered variants with PASS are removed (BIOIN-2648 core)."""
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"

        # Add FILTER line to header
        cnv_vcf_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')

        # Create input VCF with overlapping PASS and filtered
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            vcf.write(make_cnv_record(vcf, "chr1", 1000, 2000, "CNV1", qual=50.0))
            vcf.write(make_cnv_record(vcf, "chr1", 1500, 2500, "CNV2", qual=20.0, filter_val="LowQual"))

        # Index the VCF file
        pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

        # Execute with ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)

        # Verify: Only PASS variant should be in output (filtered removed)
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert records[0].filter.keys() == ["PASS"]

    def test_two_stage_multiple_filtered_one_pass(self, tmp_path, cnv_vcf_header):
        """Test multiple filtered variants overlapping with one PASS."""
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"

        # Add FILTER lines to header
        cnv_vcf_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')
        cnv_vcf_header.add_line('##FILTER=<ID=HighCov,Description="High coverage">')

        # Create input VCF
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            # PASS variant
            record1 = vcf.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 2000
            record1.id = "CNV1"
            record1.alleles = ("N", "<DEL>")
            record1.info["SVLEN"] = (1000,)
            record1.info["SVTYPE"] = "DEL"
            record1.qual = 50.0
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record1)

            # Filtered variant 1 overlapping
            record2 = vcf.new_record()
            record2.contig = "chr1"
            record2.pos = 1200
            record2.stop = 2200
            record2.id = "CNV2"
            record2.alleles = ("N", "<DEL>")
            record2.info["SVLEN"] = (1000,)
            record2.info["SVTYPE"] = "DEL"
            record2.qual = 20.0
            record2.filter.add("LowQual")
            record2.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record2)

            # Filtered variant 3 overlapping (position 1400)
            record4 = vcf.new_record()
            record4.contig = "chr1"
            record4.pos = 1400
            record4.stop = 1900
            record4.id = "CNV4"
            record4.alleles = ("N", "<DEL>")
            record4.info["SVLEN"] = (500,)
            record4.info["SVTYPE"] = "DEL"
            record4.qual = 18.0
            record4.filter.add("LowQual")
            record4.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record4)

            # Filtered variant 2 overlapping (position 1800)
            record3 = vcf.new_record()
            record3.contig = "chr1"
            record3.pos = 1800
            record3.stop = 2800
            record3.id = "CNV3"
            record3.alleles = ("N", "<DEL>")
            record3.info["SVLEN"] = (1000,)
            record3.info["SVTYPE"] = "DEL"
            record3.qual = 15.0
            record3.filter.add("HighCov")
            record3.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record3)

        # Index the VCF file
        pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

        # Execute with ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)

        # Verify: Only PASS should remain (all filtered removed)
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert records[0].filter.keys() == ["PASS"]

    def test_two_stage_filtered_preserved(self, tmp_path, cnv_vcf_header):
        """Test that non-overlapping filtered variants are preserved."""
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"

        # Add FILTER line to header
        cnv_vcf_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')

        # Create input VCF
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            # PASS variant 1
            record1 = vcf.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 2000
            record1.id = "CNV1"
            record1.alleles = ("N", "<DEL>")
            record1.info["SVLEN"] = (1000,)
            record1.info["SVTYPE"] = "DEL"
            record1.qual = 50.0
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record1)

            # PASS variant 2 (overlapping with variant 1)
            record2 = vcf.new_record()
            record2.contig = "chr1"
            record2.pos = 1500
            record2.stop = 2500
            record2.id = "CNV2"
            record2.alleles = ("N", "<DEL>")
            record2.info["SVLEN"] = (1000,)
            record2.info["SVTYPE"] = "DEL"
            record2.qual = 45.0
            record2.filter.add("PASS")
            record2.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record2)

            # Filtered variant (far away - should be preserved)
            record3 = vcf.new_record()
            record3.contig = "chr1"
            record3.pos = 5000
            record3.stop = 6000
            record3.id = "CNV3"
            record3.alleles = ("N", "<DEL>")
            record3.info["SVLEN"] = (1000,)
            record3.info["SVTYPE"] = "DEL"
            record3.qual = 20.0
            record3.filter.add("LowQual")
            record3.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record3)

        # Index the VCF file
        pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

        # Execute with ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)

        # Verify: 2 records (1 merged PASS, 1 filtered preserved)
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 2
            # Check filters
            filters = [rec.filter.keys() for rec in records]
            assert ["PASS"] in filters
            assert ["LowQual"] in filters
            # Verify the filtered record is at position 5000
            filtered_record = [rec for rec in records if rec.filter.keys() == ["LowQual"]][0]
            assert filtered_record.pos == 5000

    def test_two_stage_empty_vcf(self, tmp_path, cnv_vcf_header):
        """Test two-stage merge with empty VCF."""
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"

        # Create empty VCF (header only)
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass  # No records

        # Index the VCF file
        pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

        # Execute with ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)

        # Verify: Output should also be empty
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 0

    def test_two_stage_same_id_different_positions_not_filtered(self, tmp_path, cnv_vcf_header):
        """Test that CNVs with same ID but different positions are not incorrectly filtered (BIOIN-2648).

        This is a regression test for a bug where variants were incorrectly removed based on ID alone
        without considering position. The fix uses (id, chrom, pos) tuple instead of just id.
        """
        input_vcf = tmp_path / "input.vcf.gz"
        output_vcf = tmp_path / "output.vcf.gz"

        # Add FILTER line to header
        cnv_vcf_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')

        # Create input VCF with two CNVs that have the same ID but different positions
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            # PASS variant at position 1000
            record1 = vcf.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 2000
            record1.id = "CNV_000000001"  # Same ID
            record1.alleles = ("N", "<DEL>")
            record1.info["SVLEN"] = (1000,)
            record1.info["SVTYPE"] = "DEL"
            record1.qual = 50.0
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (0, 1)
            record1.samples["test_sample"]["CN"] = 1.0
            vcf.write(record1)

            # Filtered variant at a different position with same ID
            record2 = vcf.new_record()
            record2.contig = "chr1"
            record2.pos = 5000
            record2.stop = 6000
            record2.id = "CNV_000000001"  # Same ID as record1
            record2.alleles = ("N", "<DEL>")
            record2.info["SVLEN"] = (1000,)
            record2.info["SVTYPE"] = "DEL"
            record2.qual = 15.0
            record2.filter.add("LowQual")
            record2.samples["test_sample"]["GT"] = (0, 1)
            record2.samples["test_sample"]["CN"] = 1.0
            vcf.write(record2)

            # PASS variant at yet another position with same ID
            record3 = vcf.new_record()
            record3.contig = "chr1"
            record3.pos = 10000
            record3.stop = 11000
            record3.id = "CNV_000000001"  # Same ID again
            record3.alleles = ("N", "<DUP>")
            record3.info["SVLEN"] = (1000,)
            record3.info["SVTYPE"] = "DUP"
            record3.qual = 60.0
            record3.filter.add("PASS")
            record3.samples["test_sample"]["GT"] = (0, 1)
            record3.samples["test_sample"]["CN"] = 3.0
            vcf.write(record3)

        # Index the VCF file
        pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

        # Execute with ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)

        # Verify: All 3 records should be in output (2 PASS, 1 filtered)
        # The filtered one should NOT be removed just because it shares the same ID
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 3, f"Expected 3 records, got {len(records)}"

            # Verify we have the correct positions
            positions = sorted([rec.pos for rec in records])
            assert positions == [1000, 5000, 10000], f"Expected positions [1000, 5000, 10000], got {positions}"

            # Verify filters: 2 PASS, 1 LowQual
            filters = [rec.filter.keys() for rec in records]
            pass_count = sum(1 for f in filters if f == ["PASS"])
            lowqual_count = sum(1 for f in filters if f == ["LowQual"])
            assert pass_count == 2, f"Expected 2 PASS records, got {pass_count}"
            assert lowqual_count == 1, f"Expected 1 LowQual record, got {lowqual_count}"

            # Verify the filtered record is at the correct position
            filtered_records = [rec for rec in records if rec.filter.keys() == ["LowQual"]]
            assert len(filtered_records) == 1
            assert (
                filtered_records[0].pos == 5000
            ), f"Expected filtered record at pos 5000, got {filtered_records[0].pos}"


class TestMergeCnvsInVcfIntegration:
    """Integration tests for merge_cnvs_in_vcf using real data."""

    def test_merge_cnvs_real_data(self, resources_dir, tmp_path):
        """Test merge_cnvs_in_vcf with real HG002 data files."""
        # Input and expected output files
        input_vcf = resources_dir / "merge_cnv_input.vcf.gz"
        expected_output_vcf = resources_dir / "merge_cnv_output.vcf.gz"

        # Create output file in tmp directory
        output_vcf = tmp_path / "test_output.vcf.gz"

        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1500)
        # Verify output matches expected
        test_utils.compare_vcfs(str(expected_output_vcf), str(output_vcf))
