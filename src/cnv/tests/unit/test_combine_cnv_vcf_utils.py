from pathlib import Path

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

        # Create a second header for testing
        header2 = pysam.VariantHeader()
        header2.add_line("##contig=<ID=chr1,length=248956422>")
        header2.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
        header2.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type. can be DUP or DEL">')
        header2.add_line('##INFO=<ID=CopyNumber,Number=1,Type=Float,Description="Copy number">')
        header2.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header2.add_sample("sample2")

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
        # Create two headers with the same INFO field
        header1 = pysam.VariantHeader()
        header1.add_line("##contig=<ID=chr1,length=248956422>")
        header1.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">')
        header1.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header1.add_sample("sample1")

        header2 = pysam.VariantHeader()
        header2.add_line("##contig=<ID=chr1,length=248956422>")
        header2.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Different Description">')
        header2.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header2.add_sample("sample2")

        # Should not raise an error - same type and number
        combined_header = combine_cnv_vcf_utils.combine_vcf_headers_for_cnv(header1, header2)

        # Verify the first definition is used
        assert "DP" in combined_header.info
        assert combined_header.info["DP"].type == "Integer"
        assert combined_header.info["DP"].number == 1
        assert combined_header.info["DP"].description == "Total Depth"  # From header1

        # Verify both samples are present
        assert "sample1" in combined_header.samples
        assert "sample2" in combined_header.samples

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
