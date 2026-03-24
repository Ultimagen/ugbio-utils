from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pysam
import pytest
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

    def test_write_vcf_records_with_source_preserves_filters_when_clear_filters_false(self, tmp_path):
        """Test that write_vcf_records_with_source preserves FILTER values when clear_filters=False."""
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

        # Create combined header that includes all FILTER definitions
        combined_header = pysam.VariantHeader()
        combined_header.add_line("##fileformat=VCFv4.2")
        combined_header.add_line("##contig=<ID=chr1,length=248956422>")
        combined_header.add_line('##FILTER=<ID=PASS,Description="All filters passed">')
        combined_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')
        combined_header.add_line('##FILTER=<ID=HighCoverage,Description="High coverage">')
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
            record1.stop = 2000
            record1.alleles = ("N", "<DEL>")
            record1.info["SVTYPE"] = "DEL"
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (1, 1)
            vcf_in.write(record1)

            # Record with LowQual filter
            record2 = vcf_in.new_record()
            record2.contig = "chr1"
            record2.pos = 5000
            record2.stop = 6000
            record2.alleles = ("N", "<DUP>")
            record2.info["SVTYPE"] = "DUP"
            record2.filter.add("LowQual")
            record2.samples["test_sample"]["GT"] = (0, 1)
            vcf_in.write(record2)

            # Record with multiple filters
            record3 = vcf_in.new_record()
            record3.contig = "chr1"
            record3.pos = 10000
            record3.stop = 11000
            record3.alleles = ("N", "<DEL>")
            record3.info["SVTYPE"] = "DEL"
            record3.filter.add("LowQual")
            record3.filter.add("HighCoverage")
            record3.samples["test_sample"]["GT"] = (1, 1)
            vcf_in.write(record3)

        # Process the file with clear_filters=False
        output_vcf_path = tmp_path / "output_preserved_filters.vcf.gz"
        with pysam.VariantFile(str(input_vcf_path), "r") as vcf_in:
            with pysam.VariantFile(str(output_vcf_path), "w", header=combined_header) as vcf_out:
                combine_cnv_vcf_utils.write_vcf_records_with_source(
                    vcf_in, vcf_out, combined_header, "TestCaller", clear_filters=False
                )

        # Verify that all filters are preserved
        with pysam.VariantFile(str(output_vcf_path), "r") as vcf_result:
            records = list(vcf_result)
            assert len(records) == 3

            # Record 1: Should have PASS filter preserved
            assert records[0].filter.keys() == ["PASS"]
            assert "CNV_SOURCE" in records[0].info
            assert records[0].info["CNV_SOURCE"] == ("TestCaller",)

            # Record 2: Should have LowQual filter preserved
            assert records[1].filter.keys() == ["LowQual"]
            assert "CNV_SOURCE" in records[1].info
            assert records[1].info["CNV_SOURCE"] == ("TestCaller",)

            # Record 3: Should have both LowQual and HighCoverage filters preserved
            assert set(records[2].filter.keys()) == {"LowQual", "HighCoverage"}
            assert "CNV_SOURCE" in records[2].info
            assert records[2].info["CNV_SOURCE"] == ("TestCaller",)


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
    header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS">')
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

        # Create mock dataframes - one for removed.vcf, one for collapsed.vcf
        # removed.vcf contains ONLY the record at pos=1000 (not the representative)
        removed_df = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [1000],
                "svlen": [(1000,)],
                "filter": ["PASS"],
                "matchid": [(1.0,)],
                "cnmops_sample_mean": [10.5],
                "cnmops_sample_stdev": [2.1],
                "cnmops_cohort_mean": [20.0],
                "cnmops_cohort_stdev": [3.5],
                "copynumber": [1.0],
                "cipos": [(-250, 251)],
            }
        )

        # collapsed.vcf contains the representative at pos=2500
        collapsed_df = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [2500],
                "svlen": [(1000,)],
                "filter": ["PASS"],
                "matchid": [(1.0,)],
                "cnmops_sample_mean": [11.0],
                "cnmops_sample_stdev": [2.3],
                "cnmops_cohort_mean": [20.5],
                "cnmops_cohort_stdev": [3.6],
                "copynumber": [1.1],
                "cipos": [(-250, 251)],
            }
        )

        # Return different DataFrames based on which file is being read
        def mock_get_vcf_df_side_effect(path, **kwargs):
            if "removed" in str(path):
                return removed_df
            else:  # collapsed.vcf
                return collapsed_df

        mock_get_vcf_df.side_effect = mock_get_vcf_df_side_effect

        # Create a mock collapsed VCF file - representative is at pos=2500
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            # Merged record with CollapseId (representative at pos=2500)
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 2500
            record.stop = 3500
            record.alleles = ("N", "<DEL>")
            record.info["CollapseId"] = "1"
            record.info["SVLEN"] = (1000,)
            record.info["SVTYPE"] = "DEL"
            record.info["CNMOPS_SAMPLE_MEAN"] = 11.0
            record.info["CNMOPS_SAMPLE_STDEV"] = 2.3
            record.info["CNMOPS_COHORT_MEAN"] = 20.5
            record.info["CNMOPS_COHORT_STDEV"] = 3.6
            record.info["CopyNumber"] = 1.1
            record.samples["test_sample"]["GT"] = (0, 1)
            record.info["CIPOS"] = (-250, 251)
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

        # Verify get_vcf_df was called (twice: once for removed.vcf, once for collapsed.vcf)
        assert mock_get_vcf_df.call_count == 2

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

        # Create mock dataframes - one for removed.vcf, one for collapsed.vcf
        # removed.vcf contains records at pos=1000 and pos=2000 (NOT pos=2500)
        # Record 1: pos=1000, length 1000, value 10.0
        # Record 2: pos=2000, length 2000, value 20.0
        removed_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "pos": [1000, 2000],
                "svlen": [(1000,), (2000,)],
                "filter": ["PASS", "PASS"],
                "matchid": [(1.0,), (1.0,)],
                "cnmops_sample_mean": [10.0, 20.0],
                "cnmops_sample_stdev": [1.0, 2.0],
                "cnmops_cohort_mean": [15.0, 25.0],
                "cnmops_cohort_stdev": [2.0, 3.0],
                "copynumber": [1.0, 2.0],
                "cipos": [(-250, 251), (-250, 251)],
            }
        )

        # collapsed.vcf contains the representative at pos=2500
        # Record 3: pos=2500, length 1500, value 15.0
        collapsed_df = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [2500],
                "svlen": [(1500,)],
                "filter": ["PASS"],
                "matchid": [(1.0,)],
                "cnmops_sample_mean": [15.0],
                "cnmops_sample_stdev": [1.5],
                "cnmops_cohort_mean": [20.0],
                "cnmops_cohort_stdev": [2.5],
                "copynumber": [1.5],
                "cipos": [(-250, 251)],
            }
        )

        # Return different DataFrames based on which file is being read
        def mock_get_vcf_df_side_effect(path, **kwargs):
            if "removed" in str(path):
                return removed_df
            else:  # collapsed.vcf
                return collapsed_df

        mock_get_vcf_df.side_effect = mock_get_vcf_df_side_effect

        # Create collapsed VCF - representative is at pos=2500
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 2500
            record.stop = 4000
            record.alleles = ("N", "<DEL>")
            record.info["CollapseId"] = "1"
            record.info["SVLEN"] = (1500,)
            record.info["SVTYPE"] = "DEL"
            record.filter.add("PASS")
            record.info["CNMOPS_SAMPLE_MEAN"] = 15.0
            record.info["CNMOPS_SAMPLE_STDEV"] = 1.5
            record.info["CNMOPS_COHORT_MEAN"] = 20.0
            record.info["CNMOPS_COHORT_STDEV"] = 2.5
            record.info["CopyNumber"] = 1.5
            record.info["CIPOS"] = (-250, 251)
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
            # Removed records: pos=1000 (val=10.0, len=1000), pos=2000 (val=20.0, len=2000)
            # Collapsed record: pos=2500 (val=15.0, len=1500)
            # Weighted avg = (10*1000 + 20*2000 + 15*1500) / (1000+2000+1500) = 72500/4500 = 16.111
            expected_mean = (10.0 * 1000 + 20.0 * 2000 + 15.0 * 1500) / (1000 + 2000 + 1500)
            assert abs(record.info["CNMOPS_SAMPLE_MEAN"] - round(expected_mean, 3)) < 0.001

            # Verify SVLEN was updated to record.stop - record.start
            assert record.info["SVLEN"] == (record.stop - record.start - 1,)

            # Verify END was updated to max of merged records
            # Removed: pos=1000 (end=1999), pos=2000 (end=3999)
            # Collapsed: pos=2500 (end=3999)
            # max end = 3999, so record.stop (END+1, exclusive) = 4000
            assert record.stop == 4000

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

        # Mock dataframe represents removed.vcf - contains only one removed record
        # The other record (at pos=2500) is the representative in collapsed.vcf
        # This removed record has CIPOS=(-250, 251) with length = 501
        mock_df = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [1000],
                "svlen": [(1000,)],
                "matchid": [(1.0,)],
                "cipos": [(-250, 251)],
                "cnmops_sample_mean": [10.0],
            }
        )
        mock_get_vcf_df.return_value = mock_df

        # Collapsed VCF contains representative (another record at pos=2500)
        # After boundary update, this will be widened to include the removed record
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 2500  # Representative starts at pos=2500
            record.stop = 3500
            record.alleles = ("N", "<DEL>")
            record.info["CollapseId"] = "1.0"
            record.info["SVLEN"] = (1000,)
            record.info["SVTYPE"] = "DEL"
            record.info["CIPOS"] = (-300, 301)  # Will be replaced with tighter (-250, 251)
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

            # Verification:
            # 1. Boundary update: 2500-3500 → 1000-3500 (widened to include removed record)
            # 2. CIPOS candidates from removed.vcf: Record at pos=1000 with CIPOS=(-250,251)
            # 3. Window filter: |1000-1000|=0 ≤ 2500 → INCLUDED
            # 4. Result: CIPOS=(-250,251) - the only candidate (tighter than representative's)
            assert "CIPOS" in record.info, "CIPOS should be present in merged record"
            cipos = record.info["CIPOS"]
            assert cipos == (-250, 251), f"Expected CIPOS=(-250, 251), got {cipos}"

    @patch("ugbio_cnv.combine_cnv_vcf_utils.mu.cleanup_temp_files")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.vcftools.get_vcf_df")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.VcfUtils")
    def test_merge_cnvs_cipos_aggregation_large_vs_small_outside_window(
        self, mock_vcf_utils_class, mock_get_vcf_df, mock_cleanup, tmp_path, cnv_vcf_header
    ):
        """Test CIPOS aggregation when small deletion is outside window of boundaries.

        When a small deletion with tight CIPOS is far from both boundaries (>2500bp),
        it should be filtered out and not contribute to CIPOS aggregation.
        """
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

        # Mock dataframe for removed.vcf - contains ONLY the large deletion.
        # The representative record (small deletion) is loaded separately from
        # collapsed.vcf in _prepare_update_dataframe.
        removed_df = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [1000],
                "end": [100000],
                "qual": [3],
                "svlen": [(99000,)],
                "matchid": [(1.0,)],
                "cipos": [(-200, 200)],
                "cnmops_sample_mean": [10.0],
            }
        )

        # Mock dataframe for collapsed.vcf - contains ONLY the representative.
        collapsed_df = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [50000],
                "end": [50100],
                "qual": [300],
                "svlen": [(100,)],
                "matchid": [(1.0,)],
                "cipos": [(-1, 1)],
                "cnmops_sample_mean": [10.5],
            }
        )
        mock_get_vcf_df.side_effect = [removed_df, collapsed_df]

        # Collapsed VCF contains representative (small deletion with higher QUAL=300)
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 50000  # Representative starts at small deletion position
            record.stop = 50100
            record.alleles = ("N", "<DEL>")
            record.info["CollapseId"] = "1.0"
            record.info["SVLEN"] = (100,)
            record.info["SVTYPE"] = "DEL"
            record.qual = 300  # Higher QUAL - why it was chosen as representative
            record.info["CIPOS"] = (-1, 1)  # Will be replaced during aggregation
            record.info["CNMOPS_SAMPLE_MEAN"] = 10.5
            record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record)

        # Create dummy input file
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass

        # Execute
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=50000)

        # Read output and verify CIPOS aggregation
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1

            record = records[0]

            # Verification:
            # 1. Boundary update: 50000-50100 → 1000-100000 (widened to removed record's bounds)
            # 2. CIPOS candidates from removed.vcf: Large deletion at pos=1000, CIPOS=(-200,200)
            # 3. Window filter: |1000-1000|=0 ≤ 2500 → INCLUDED
            # 4. Result: CIPOS=(-200,200) - the only candidate within window
            assert "CIPOS" in record.info, "CIPOS should be present in merged record"
            cipos = record.info["CIPOS"]
            assert cipos == (-200, 200), f"Expected CIPOS=(-200, 200), got {cipos}"

    @patch("ugbio_cnv.combine_cnv_vcf_utils.mu.cleanup_temp_files")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.vcftools.get_vcf_df")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.VcfUtils")
    def test_merge_cnvs_cipos_aggregation_large_vs_small_within_window(
        self, mock_vcf_utils_class, mock_get_vcf_df, mock_cleanup, tmp_path, cnv_vcf_header
    ):
        """Test CIPOS aggregation when small deletion is within window of start boundary.

        When a small deletion with tight CIPOS is within 2500bp of a boundary,
        its tight CIPOS should be selected.
        """
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

        # Mock dataframe for removed.vcf - contains ONLY the small deletion.
        # The representative record (large deletion) is loaded separately from
        # collapsed.vcf in _prepare_update_dataframe.
        removed_df = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [2000],
                "end": [2100],
                "qual": [300],
                "svlen": [(100,)],
                "matchid": [(1.0,)],
                "cipos": [(-1, 1)],
                "cnmops_sample_mean": [11.0],
            }
        )

        # Mock dataframe for collapsed.vcf - contains ONLY the representative.
        collapsed_df = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [1000],
                "end": [100000],
                "qual": [3],
                "svlen": [(99000,)],
                "matchid": [(1.0,)],
                "cipos": [(-200, 200)],
                "cnmops_sample_mean": [10.5],
            }
        )
        mock_get_vcf_df.side_effect = [removed_df, collapsed_df]

        # Collapsed VCF contains representative (large deletion at pos=1000)
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 1000  # Representative starts at large deletion position
            record.stop = 100000
            record.alleles = ("N", "<DEL>")
            record.info["CollapseId"] = "1.0"
            record.info["SVLEN"] = (99000,)
            record.info["SVTYPE"] = "DEL"
            record.qual = 3  # Lower QUAL but was chosen as representative
            record.info["CIPOS"] = (-200, 200)  # Will be replaced during aggregation
            record.info["CNMOPS_SAMPLE_MEAN"] = 10.5
            record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(record)

        # Create dummy input file
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass

        # Execute
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=50000)

        # Read output and verify CIPOS aggregation
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1

            record = records[0]

            # Verification:
            # 1. Boundary update: 1000-100000 → 1000-100000 (no change, already at widest)
            # 2. CIPOS candidates from removed.vcf: Small deletion at pos=2000, CIPOS=(-1,1)
            # 3. Window filter: |2000-1000|=1000 ≤ 2500 → INCLUDED (within window of start)
            # 4. Result: CIPOS=(-1,1) - tightest interval from the only candidate
            assert "CIPOS" in record.info, "CIPOS should be present in merged record"
            cipos = record.info["CIPOS"]
            assert cipos == (-1, 1), f"Expected CIPOS=(-1, 1), got {cipos}"

    @patch("ugbio_cnv.combine_cnv_vcf_utils.mu.cleanup_temp_files")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.vcftools.get_vcf_df")
    @patch("ugbio_cnv.combine_cnv_vcf_utils.VcfUtils")
    def test_merge_cnvs_svtype_and_alt_allele_consistency(
        self, mock_vcf_utils_class, mock_get_vcf_df, mock_cleanup, tmp_path, cnv_vcf_header
    ):
        """Test that ALT allele is updated to match aggregated SVTYPE.

        When SVTYPE changes due to weighted majority voting, the ALT allele
        and genotype should be updated to match the new SVTYPE.
        """
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

        # Create mock dataframe with mixed SVTYPEs:
        # Record 1: DEL with SVLEN=500
        # Record 2: DUP with SVLEN=2000 (DUP wins with weighted majority)
        # Record 3: DUP with SVLEN=1500
        # Total: DUP=3500 > DEL=1000 (500 from update + 500 from collapsed)
        mock_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr1"],
                "pos": [1000, 1500, 2000],
                "end": [1500, 3500, 3500],
                "svlen": [(500,), (2000,), (1500,)],
                "svtype": ["DEL", "DUP", "DUP"],
                "matchid": [(1.0,), (1.0,), (1.0,)],
                "cipos": [(-100, 100), (-50, 50), (-50, 50)],
                "cnmops_sample_mean": [10.0, 11.0, 12.0],
            }
        )
        mock_get_vcf_df.return_value = mock_df

        # Create collapsed VCF with initial SVTYPE=DEL and ALT=<DEL>
        with pysam.VariantFile(str(collapse_vcf), "w", header=cnv_vcf_header) as vcf:
            record = vcf.new_record()
            record.contig = "chr1"
            record.pos = 1000
            record.stop = 1500
            record.alleles = ("N", "<DEL>")  # Initial ALT is DEL
            record.info["CollapseId"] = "1.0"
            record.info["SVLEN"] = (500,)  # Collapsed record has SVLEN=500
            record.info["SVTYPE"] = "DEL"  # Initial SVTYPE is DEL
            record.info["CIPOS"] = (-100, 100)
            record.info["CNMOPS_SAMPLE_MEAN"] = 10.5
            record.samples["test_sample"]["GT"] = (0, 1)  # Initial GT for DEL
            vcf.write(record)

        # Create dummy input file
        with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf:
            pass

        # Execute
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=5000)

        # Read output and verify SVTYPE, ALT allele, and genotype consistency
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1

            record = records[0]

            # Verify SVTYPE changed to DUP (weighted majority: 3500 > 1000)
            assert "SVTYPE" in record.info, "SVTYPE should be present in merged record"
            assert record.info["SVTYPE"] == "DUP", f"Expected SVTYPE=DUP, got {record.info['SVTYPE']}"

            # Verify ALT allele was updated to match SVTYPE
            assert record.alts is not None, "ALT alleles should be present"
            assert len(record.alts) == 1, f"Expected 1 ALT allele, got {len(record.alts)}"
            assert record.alts[0] == "<DUP>", f"Expected ALT=<DUP>, got {record.alts[0]}"

            # Verify genotype was updated to match DUP
            assert "test_sample" in record.samples, "Sample should be present"
            gt = record.samples["test_sample"]["GT"]
            assert gt == (None, 1), f"Expected GT=(None, 1) for DUP, got {gt}"

    def test_select_breakpoints_by_cipos_window_prefers_tightest_interval(self):
        """Test that breakpoint selection prefers records with the tightest CIPOS."""

        update_records = pd.DataFrame(
            {
                "pos": [1000, 1200, 1300],
                "end": [5000, 4800, 5200],
                "cipos": [(-500, 500), (-10, 10), (-150, 150)],
            }
        )

        new_start, new_end = combine_cnv_vcf_utils._select_breakpoints_by_cipos_window(
            update_records,
            window=2500,
        )

        assert new_start == 1200
        assert new_end == 4800

    def test_select_breakpoints_by_cipos_window_tie_breaks_by_position(self):
        """Test tie-breaks: lowest start and highest end when CIPOS lengths are equal."""
        update_records = pd.DataFrame(
            {
                "pos": [1000, 1200, 1300],
                "end": [5000, 5200, 4800],
                "cipos": [(-10, 10), (-20, 0), (-5, 15)],
            }
        )

        new_start, new_end = combine_cnv_vcf_utils._select_breakpoints_by_cipos_window(
            update_records,
            window=2500,
        )

        assert new_start == 1000
        assert new_end == 5200


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
    record.info["CIPOS"] = (-250, 251)  # Default CIPOS for testing
    record.qual = qual
    if filter_val:
        record.filter.add(filter_val)
    record.samples["test_sample"]["GT"] = (0, 1)

    # Add any additional INFO fields
    for key, value in info.items():
        record.info[key] = value

    return record


def test_merge_cnvs_duplicate_ids_fail(tmp_path, cnv_vcf_header):
    """Test that merge_cnvs_in_vcf fails when input VCF has duplicate IDs."""
    input_vcf = tmp_path / "input_with_duplicates.vcf.gz"
    output_vcf = tmp_path / "output.vcf.gz"

    # Create a VCF with duplicate IDs
    with pysam.VariantFile(str(input_vcf), "w", header=cnv_vcf_header) as vcf_out:
        # First record with ID "dup_id"
        rec1 = vcf_out.new_record(
            contig="chr1",
            start=1000,
            stop=2000,
            alleles=("N", "<DEL>"),
            id="dup_id",
        )
        rec1.info["SVTYPE"] = "DEL"
        rec1.info["SVLEN"] = (-1000,)
        vcf_out.write(rec1)

        # Second record with same ID "dup_id"
        rec2 = vcf_out.new_record(
            contig="chr1",
            start=5000,
            stop=6000,
            alleles=("N", "<DEL>"),
            id="dup_id",
        )
        rec2.info["SVTYPE"] = "DEL"
        rec2.info["SVLEN"] = (-1000,)
        vcf_out.write(rec2)

        # Third record with unique ID
        rec3 = vcf_out.new_record(
            contig="chr1",
            start=10000,
            stop=11000,
            alleles=("N", "<DEL>"),
            id="unique_id",
        )
        rec3.info["SVTYPE"] = "DEL"
        rec3.info["SVLEN"] = (-1000,)
        vcf_out.write(rec3)

    pysam.tabix_index(str(input_vcf), preset="vcf", force=True)

    # Test that merge_cnvs_in_vcf raises ValueError for duplicate IDs
    with pytest.raises(ValueError, match="duplicate variant IDs.*dup_id"):
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(
            input_vcf=str(input_vcf),
            output_vcf=str(output_vcf),
            distance=1000,
        )


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
            record1.info["CIPOS"] = (-250, 251)
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
            record2.info["CIPOS"] = (-250, 251)
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
            record3.info["CIPOS"] = (-1, 1)
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
            record1.info["CIPOS"] = (-250, 251)
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
            record2.info["CIPOS"] = (-250, 251)
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
            record3.info["CIPOS"] = (-1, 1)
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

    def test_remove_overlapping_filtered_variants_keeps_non_overlapping_filtered(self, tmp_path, cnv_vcf_header):
        """Test second-stage filter keeps a filtered record when no PASS overlap exists."""
        original_vcf = tmp_path / "original.vcf.gz"
        merged_vcf = tmp_path / "merged.vcf.gz"

        # Add FILTER line to header
        cnv_vcf_header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')

        with pysam.VariantFile(str(original_vcf), "w", header=cnv_vcf_header) as vcf:
            # PASS record
            pass_record = vcf.new_record()
            pass_record.contig = "chr1"
            pass_record.pos = 1000
            pass_record.stop = 2000
            pass_record.id = "PASS1"
            pass_record.alleles = ("N", "<DEL>")
            pass_record.info["SVLEN"] = (1000,)
            pass_record.info["SVTYPE"] = "DEL"
            pass_record.info["CIPOS"] = (-250, 251)
            pass_record.filter.add("PASS")
            pass_record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(pass_record)

            # Non-overlapping filtered record
            filtered_record = vcf.new_record()
            filtered_record.contig = "chr1"
            filtered_record.pos = 10000
            filtered_record.stop = 11000
            filtered_record.id = "LOW1"
            filtered_record.alleles = ("N", "<DEL>")
            filtered_record.info["SVLEN"] = (1000,)
            filtered_record.info["SVTYPE"] = "DEL"
            filtered_record.info["CIPOS"] = (-250, 251)
            filtered_record.filter.add("LowQual")
            filtered_record.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(filtered_record)

        # Use identical content as merged input for direct helper validation
        with (
            pysam.VariantFile(str(original_vcf)) as vcf_in,
            pysam.VariantFile(str(merged_vcf), "w", header=vcf_in.header) as vcf_out,
        ):
            for rec in vcf_in:
                vcf_out.write(rec)

        pysam.tabix_index(str(original_vcf), preset="vcf", force=True)
        pysam.tabix_index(str(merged_vcf), preset="vcf", force=True)

        filtered_merged_vcf = combine_cnv_vcf_utils._remove_overlapping_filtered_variants(
            original_vcf=str(original_vcf),
            merged_vcf=str(merged_vcf),
            distance=1000,
            ignore_sv_type=False,
            pick_best=False,
        )

        with pysam.VariantFile(str(filtered_merged_vcf)) as vcf:
            records = list(vcf)

        assert len(records) == 2
        record_ids = {rec.id for rec in records}
        assert "PASS1" in record_ids
        assert "LOW1" in record_ids

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
        # Execute with ignore_filter=False - should fail due to duplicate IDs
        with pytest.raises(ValueError, match="duplicate variant IDs"):
            combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1000, ignore_filter=False)


class TestWeightedMajorityVoting:
    """Tests for SVLEN-weighted majority voting for SVTYPE."""

    def test_aggregate_weighted_majority_del_wins(self):
        """Test that DEL wins when total DEL length > DUP length."""
        # Create mock record and DataFrame
        header = pysam.VariantHeader()
        header.add_line("##contig=<ID=chr1>")
        header.info.add("SVLEN", ".", "Integer", "Length")
        header.info.add("SVTYPE", "1", "String", "Type")
        header.add_sample("test_sample")

        record = header.new_record()
        record.contig = "chr1"
        record.pos = 1000
        record.info["SVLEN"] = (1000,)
        record.info["SVTYPE"] = "DEL"

        # DataFrame includes collapsed record (SVLEN=1000, SVTYPE=DEL) + removed records
        # Collapsed: DEL 1000, Removed: DEL 1000, DEL 500, DUP 200
        update_df = pd.DataFrame({"svlen": [(1000,), (1000,), (500,), (200,)], "svtype": ["DEL", "DEL", "DEL", "DUP"]})

        # Values list (from _collect_field_values - all from update_df)
        values = ["DEL", "DEL", "DEL", "DUP"]

        # Call aggregation
        combine_cnv_vcf_utils._aggregate_weighted_majority(record, update_df, "SVTYPE", values)

        # Verify DEL won: 1000+1000+500 = 2500 > 200
        assert record.info["SVTYPE"] == "DEL"

    def test_aggregate_weighted_majority_dup_wins(self):
        """Test that DUP wins when total DUP length > DEL length."""
        header = pysam.VariantHeader()
        header.add_line("##contig=<ID=chr1>")
        header.info.add("SVLEN", ".", "Integer", "Length")
        header.info.add("SVTYPE", "1", "String", "Type")
        header.add_sample("test_sample")

        record = header.new_record()
        record.contig = "chr1"
        record.pos = 1000
        record.info["SVLEN"] = (100,)
        record.info["SVTYPE"] = "DEL"

        # DataFrame includes collapsed record (SVLEN=100, SVTYPE=DEL) + removed records
        # Collapsed: DEL 100, Removed: DUP 800, DUP 800
        update_df = pd.DataFrame({"svlen": [(100,), (800,), (800,)], "svtype": ["DEL", "DUP", "DUP"]})

        values = ["DEL", "DUP", "DUP"]

        combine_cnv_vcf_utils._aggregate_weighted_majority(record, update_df, "SVTYPE", values)

        # Verify DUP won: 800+800 = 1600 > 100
        assert record.info["SVTYPE"] == "DUP"

    def test_aggregate_weighted_majority_all_same(self):
        """Test that same SVTYPE is preserved when all are identical."""
        header = pysam.VariantHeader()
        header.add_line("##contig=<ID=chr1>")
        header.info.add("SVLEN", ".", "Integer", "Length")
        header.info.add("SVTYPE", "1", "String", "Type")

        record = header.new_record()
        record.contig = "chr1"
        record.pos = 1000
        record.info["SVLEN"] = (1000,)
        record.info["SVTYPE"] = "DEL"

        # DataFrame includes collapsed record (SVLEN=1000, SVTYPE=DEL) + removed records
        # Collapsed: DEL 1000, Removed: DEL 500, DEL 300
        update_df = pd.DataFrame({"svlen": [(1000,), (500,), (300,)], "svtype": ["DEL", "DEL", "DEL"]})

        values = ["DEL", "DEL", "DEL"]

        combine_cnv_vcf_utils._aggregate_weighted_majority(record, update_df, "SVTYPE", values)

        assert record.info["SVTYPE"] == "DEL"

    def test_update_genotype_from_svtype_del(self):
        """Test that DEL SVTYPE sets GT to (0,1)."""
        header = pysam.VariantHeader()
        header.add_line("##contig=<ID=chr1>")
        header.info.add("SVTYPE", "1", "String", "Type")
        header.formats.add("GT", "1", "String", "Genotype")
        header.add_sample("test_sample")

        record = header.new_record()
        record.contig = "chr1"
        record.pos = 1000
        record.alleles = ("N", "<DEL>")  # Need to set alleles before GT
        record.info["SVTYPE"] = "DEL"
        record.samples["test_sample"]["GT"] = (None, 1)  # Initial GT

        combine_cnv_vcf_utils._update_genotype_from_svtype(record)

        assert record.samples["test_sample"]["GT"] == (0, 1)

    def test_update_genotype_from_svtype_dup(self):
        """Test that DUP SVTYPE sets GT to (None,1)."""
        header = pysam.VariantHeader()
        header.add_line("##contig=<ID=chr1>")
        header.info.add("SVTYPE", "1", "String", "Type")
        header.formats.add("GT", "1", "String", "Genotype")
        header.add_sample("test_sample")

        record = header.new_record()
        record.contig = "chr1"
        record.pos = 1000
        record.alleles = ("N", "<DUP>")  # Need to set alleles before GT
        record.info["SVTYPE"] = "DUP"
        record.samples["test_sample"]["GT"] = (0, 1)  # Initial GT

        combine_cnv_vcf_utils._update_genotype_from_svtype(record)

        assert record.samples["test_sample"]["GT"] == (None, 1)

    def test_update_genotype_no_svtype(self):
        """Test that missing SVTYPE doesn't crash."""
        header = pysam.VariantHeader()
        header.add_line("##contig=<ID=chr1>")
        header.formats.add("GT", "1", "String", "Genotype")
        header.add_sample("test_sample")

        record = header.new_record()
        record.contig = "chr1"
        record.pos = 1000
        record.alleles = ("N", "<DEL>")  # Need to set alleles before GT
        record.samples["test_sample"]["GT"] = (0, 1)

        # Should not crash
        combine_cnv_vcf_utils._update_genotype_from_svtype(record)

        # GT should be unchanged
        assert record.samples["test_sample"]["GT"] == (0, 1)
