import os
from pathlib import Path

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
        combined_header = combine_cnmops_cnvpytor_cnv_calls.combine_vcf_headers_for_cnv(
            header1, header2, keep_filters=True
        )

        # Verify FILTER fields from both headers are present
        assert "LowQual" in combined_header.filters
        assert "HighCoverage" in combined_header.filters
        assert combined_header.filters["LowQual"].description == "Low quality"
        assert combined_header.filters["HighCoverage"].description == "High coverage region"

        # Verify INFO fields are still present
        assert "DP" in combined_header.info
        assert "AF" in combined_header.info


def create_test_fasta(fasta_path: str, sequences: dict[str, str]) -> None:
    """Create a test FASTA file with specified sequences.

    Parameters
    ----------
    fasta_path : str
        Path to create the FASTA file
    sequences : dict[str, str]
        Dictionary mapping chromosome names to sequences
    """
    with open(fasta_path, "w") as f:
        for chrom, seq in sequences.items():
            f.write(f">{chrom}\n")
            # Write sequence in 50-char lines
            for i in range(0, len(seq), 50):
                f.write(seq[i : i + 50] + "\n")

    # Create index
    from pyfaidx import Fasta

    Fasta(fasta_path)  # This creates the .fai file


def create_test_vcf_for_gap_perc(vcf_path: str, records: list[dict], contigs: dict[str, int]) -> None:
    """Create a test VCF file with CNV records.

    Parameters
    ----------
    vcf_path : str
        Path to create the VCF file
    records : list[dict]
        List of record dicts with keys: chrom, start, stop, alleles, svtype
    contigs : dict[str, int]
        Dictionary mapping contig names to lengths
    """
    header = pysam.VariantHeader()
    header.add_meta("fileformat", value="VCFv4.2")

    for contig, length in contigs.items():
        header.contigs.add(contig, length=length)

    header.info.add("SVTYPE", number=1, type="String", description="Type of structural variant")
    header.info.add("SVLEN", number=".", type="Integer", description="Length of structural variant")
    header.add_sample("test_sample")

    with pysam.VariantFile(vcf_path, "w", header=header) as vcf:
        for rec_data in records:
            rec = vcf.new_record(
                contig=rec_data["chrom"],
                start=rec_data["start"],
                stop=rec_data["stop"],
                alleles=rec_data.get("alleles", ("N", "<DEL>")),
            )
            if "svtype" in rec_data:
                rec.info["SVTYPE"] = rec_data["svtype"]
            vcf.write(rec)

    pysam.tabix_index(vcf_path, preset="vcf", force=True)


class TestAnnotateVcfWithGapPerc:
    """Tests for annotate_vcf_with_gap_perc function."""

    def test_gap_perc_all_ns(self, tmp_path):
        """Test that a region with 100% N bases returns GAP_PERC = 1.0."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference with all N's in the region 0-100
        create_test_fasta(fasta_path, {"chr1": "N" * 200})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 100, "svtype": "DEL"}],
            {"chr1": 200},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "GAP_PERC" in records[0].info
            # region_len = stop - start + 1 = 100 - 0 + 1 = 101
            # seq = genome[0:101] = 101 N's
            # gap_perc = 101/101 = 1.0
            assert records[0].info["GAP_PERC"] == 1.0

    def test_gap_perc_no_ns(self, tmp_path):
        """Test that a region with 0% N bases returns GAP_PERC = 0.0."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference with all A's (no N's)
        create_test_fasta(fasta_path, {"chr1": "A" * 200})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 100, "svtype": "DEL"}],
            {"chr1": 200},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "GAP_PERC" in records[0].info
            assert records[0].info["GAP_PERC"] == 0.0

    def test_gap_perc_mixed_bases(self, tmp_path):
        """Test that a region with mixed N's and bases returns correct fraction."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference: 50 N's followed by 51 A's = region has 50 N's in first 50 positions
        # Region 0:100 extracts 101 bases (positions 0-100 inclusive)
        # seq = 50 N's + 51 A's = 50 N's out of 101 bases
        create_test_fasta(fasta_path, {"chr1": "N" * 50 + "A" * 51 + "G" * 100})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 100, "svtype": "DEL"}],
            {"chr1": 201},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            # region_len = 101, n_count = 50
            expected_gap_perc = 50 / 101
            assert records[0].info["GAP_PERC"] == pytest.approx(expected_gap_perc, rel=1e-4)

    def test_gap_perc_lowercase_ns_counted(self, tmp_path):
        """Test that lowercase 'n' bases are also counted as gaps."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference with lowercase n's
        create_test_fasta(fasta_path, {"chr1": "n" * 200})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 100, "svtype": "DEL"}],
            {"chr1": 200},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            # Lowercase n's should be counted as N's after .upper() conversion
            # region_len = 101, n_count = 101
            assert records[0].info["GAP_PERC"] == 1.0

    def test_gap_perc_multiple_records(self, tmp_path):
        """Test that multiple VCF records are all annotated correctly."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference with different gap patterns in different regions
        # Position 0-50: all N's (51 bases)
        # Position 51-100: all A's (50 bases)
        # Position 101-150: 25 N's + 25 A's
        seq = "N" * 51 + "A" * 50 + "N" * 25 + "A" * 25 + "G" * 50
        create_test_fasta(fasta_path, {"chr1": seq})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [
                {"chrom": "chr1", "start": 0, "stop": 50, "svtype": "DEL"},  # All N's
                {"chrom": "chr1", "start": 51, "stop": 100, "svtype": "DUP"},  # All A's
                {"chrom": "chr1", "start": 101, "stop": 150, "svtype": "DEL"},  # 50% N's
            ],
            {"chr1": 201},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 3

            # Region 0-50: 51 N's, region_len = 51, gap_perc = 1.0
            assert records[0].info["GAP_PERC"] == 1.0

            # Region 51-100: 50 A's, region_len = 50, gap_perc = 0.0
            assert records[1].info["GAP_PERC"] == 0.0

            # Region 101-150: 25 N's + 25 A's, region_len = 50, gap_perc = 0.5
            assert records[2].info["GAP_PERC"] == 0.5

    def test_gap_perc_header_added(self, tmp_path):
        """Test that GAP_PERC INFO field is added to the VCF header."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_fasta(fasta_path, {"chr1": "A" * 100})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 50, "svtype": "DEL"}],
            {"chr1": 100},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            assert "GAP_PERC" in vcf.header.info
            gap_perc_info = vcf.header.info["GAP_PERC"]
            assert gap_perc_info.type == "Float"
            assert gap_perc_info.number == 1

    def test_gap_perc_output_vcf_indexed(self, tmp_path):
        """Test that output VCF is properly indexed after annotation."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_fasta(fasta_path, {"chr1": "A" * 100})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 50, "svtype": "DEL"}],
            {"chr1": 100},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        # Check that index file exists (.tbi or .csi)
        assert os.path.exists(output_vcf + ".tbi") or os.path.exists(output_vcf + ".csi")
