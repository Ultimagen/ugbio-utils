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
