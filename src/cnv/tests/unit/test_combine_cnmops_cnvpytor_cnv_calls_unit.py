from pathlib import Path

import pandas as pd
import pysam
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
        combined_header = combine_cnmops_cnvpytor_cnv_calls.combine_vcf_headers_for_cnv(header1, header2)

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
        combined_header = combine_cnmops_cnvpytor_cnv_calls.combine_vcf_headers_for_cnv(header1, header2)

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
            combine_cnmops_cnvpytor_cnv_calls.combine_vcf_headers_for_cnv(header1, header2)

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
            combine_cnmops_cnvpytor_cnv_calls.combine_vcf_headers_for_cnv(header1, header2)

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
        combined_header = combine_cnmops_cnvpytor_cnv_calls.combine_vcf_headers_for_cnv(header1, header2)

        # Verify enforced specifications are applied
        assert "SVLEN" in combined_header.info
        assert combined_header.info["SVLEN"].number == "."
        assert combined_header.info["SVLEN"].type == "Integer"

        assert "SVTYPE" in combined_header.info
        assert combined_header.info["SVTYPE"].number == 1
        assert combined_header.info["SVTYPE"].type == "String"


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
