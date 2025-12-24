from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pysam
import pytest
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.vcf_utils import VcfUtils


@pytest.fixture
def mock_logger(mocker):
    """Fixture for mock logger"""
    return mocker.Mock()


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestVcfUtils:
    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_filter_vcf_with_include_expression(self, mock_execute, tmp_path):
        """Test filter_vcf with include expression"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.filter_vcf(
            input_vcf=input_vcf, output_vcf=output_vcf, filter_name="LowQual", include_expression="QUAL>=30"
        )

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools filter -i 'QUAL>=30' --threads 1 -s LowQual -m + -O z -o {output_vcf} {input_vcf}"
        mock_execute.assert_called_once_with(expected_cmd)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_filter_vcf_with_exclude_expression(self, mock_execute, tmp_path):
        """Test filter_vcf with exclude expression"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.filter_vcf(
            input_vcf=input_vcf, output_vcf=output_vcf, filter_name="LowDepth", exclude_expression="DP<10"
        )

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools filter -e 'DP<10' --threads 1 -s LowDepth -m + -O z -o {output_vcf} {input_vcf}"
        mock_execute.assert_called_once_with(expected_cmd)

    def test_filter_vcf_validation_errors(self):
        """Test filter_vcf validation errors"""
        vcf_utils = VcfUtils()

        # Test error when neither expression is provided
        with pytest.raises(
            ValueError, match="At least one of include_expression or exclude_expression must" " be provided"
        ):
            vcf_utils.filter_vcf("input.vcf", "output.vcf", "TestFilter")

        # Test error when both expressions are provided
        with pytest.raises(
            ValueError, match="Only one of include_expression or exclude_expression " "can be provided at a time"
        ):
            vcf_utils.filter_vcf(
                "input.vcf", "output.vcf", "TestFilter", include_expression="QUAL>=30", exclude_expression="QUAL<30"
            )

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_filter_vcf_with_simple_pipeline(self, mock_execute, tmp_path):
        """Test filter_vcf works with SimplePipeline"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        sp = SimplePipeline(0, 10)
        vcf_utils = VcfUtils(sp)
        vcf_utils.filter_vcf(
            input_vcf=input_vcf,
            output_vcf=output_vcf,
            filter_name="ComplexFilter",
            exclude_expression="TYPE!='snp' | QUAL<20",
        )

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools filter -e 'TYPE!='snp' | QUAL<20' --threads 1 -s ComplexFilter -m + -O z -o {output_vcf} {input_vcf}"  # noqa: E501
        mock_execute.assert_called_once_with(expected_cmd)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_view_vcf_basic(self, mock_execute, tmp_path):
        """Test basic view_vcf functionality"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.view_vcf(input_vcf=input_vcf, output_vcf=output_vcf)

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools view --threads 1 -O z -o {output_vcf} {input_vcf}"
        mock_execute.assert_called_once_with(expected_cmd)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_view_vcf_with_threads(self, mock_execute, tmp_path):
        """Test view_vcf with custom number of threads"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.view_vcf(input_vcf=input_vcf, output_vcf=output_vcf, n_threads=4)

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools view --threads 4 -O z -o {output_vcf} {input_vcf}"
        mock_execute.assert_called_once_with(expected_cmd)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_view_vcf_with_extra_args(self, mock_execute, tmp_path):
        """Test view_vcf with extra arguments"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.view_vcf(input_vcf=input_vcf, output_vcf=output_vcf, n_threads=2, extra_args="-H -s sample1,sample2")

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools view --threads 2 -H -s sample1,sample2 -O z -o {output_vcf} {input_vcf}"
        mock_execute.assert_called_once_with(expected_cmd)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_view_vcf_with_simple_pipeline(self, mock_execute, tmp_path):
        """Test view_vcf works with SimplePipeline"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        sp = SimplePipeline(0, 10)
        vcf_utils = VcfUtils(sp)
        vcf_utils.view_vcf(input_vcf=input_vcf, output_vcf=output_vcf, n_threads=8, extra_args="-r chr1:1000-2000")

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools view --threads 8 -r chr1:1000-2000 -O z -o {output_vcf} {input_vcf}"
        mock_execute.assert_called_once_with(expected_cmd)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_remove_filter_annotations_default_threads(self, mock_execute, tmp_path):
        """Test remove_filter_annotations with default thread count"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.remove_filter_annotations(input_vcf=input_vcf, output_vcf=output_vcf)

        # Verify the correct bcftools commands were called
        expected_annotate_cmd = f"bcftools annotate -x FILTER --threads 1 -o {output_vcf} -O z {input_vcf}"
        expected_index_cmd = f"bcftools index -tf {output_vcf}"

        expected_calls = [mock.call(expected_annotate_cmd), mock.call(expected_index_cmd)]
        mock_execute.assert_has_calls(expected_calls)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_remove_filter_annotations_custom_threads(self, mock_execute, tmp_path):
        """Test remove_filter_annotations with custom thread count"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")
        n_threads = 4

        vcf_utils = VcfUtils()
        vcf_utils.remove_filter_annotations(input_vcf=input_vcf, output_vcf=output_vcf, n_threads=n_threads)

        # Verify the correct bcftools commands were called with custom threads
        expected_annotate_cmd = f"bcftools annotate -x FILTER --threads {n_threads} -o {output_vcf} -O z {input_vcf}"
        expected_index_cmd = f"bcftools index -tf {output_vcf}"

        expected_calls = [mock.call(expected_annotate_cmd), mock.call(expected_index_cmd)]
        mock_execute.assert_has_calls(expected_calls)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_remove_filter_annotations_with_simple_pipeline(self, mock_execute, tmp_path):
        """Test remove_filter_annotations works with SimplePipeline"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        sp = SimplePipeline(0, 10)
        vcf_utils = VcfUtils(sp)
        vcf_utils.remove_filter_annotations(input_vcf=input_vcf, output_vcf=output_vcf, n_threads=2)

        # Verify the correct bcftools commands were called with SimplePipeline
        expected_annotate_cmd = f"bcftools annotate -x FILTER --threads 2 -o {output_vcf} -O z {input_vcf}"
        expected_index_cmd = f"bcftools index -tf {output_vcf}"

        expected_calls = [mock.call(expected_annotate_cmd), mock.call(expected_index_cmd)]
        mock_execute.assert_has_calls(expected_calls)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_update_vcf_contigs_from_fai_basic(self, mock_execute, tmp_path):
        """Test update_vcf_contigs_from_fai with basic parameters"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")
        fasta_fai = str(tmp_path / "reference.fa.fai")

        vcf_utils = VcfUtils()
        vcf_utils.update_vcf_contigs_from_fai(input_vcf=input_vcf, output_vcf=output_vcf, fasta_fai=fasta_fai)

        # Verify the correct bcftools commands were called
        expected_reheader_cmd = f"bcftools reheader -f {fasta_fai} -o {output_vcf} {input_vcf}"
        expected_index_cmd = f"bcftools index -tf {output_vcf}"

        expected_calls = [mock.call(expected_reheader_cmd), mock.call(expected_index_cmd)]
        mock_execute.assert_has_calls(expected_calls)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_update_vcf_contigs_from_fai_with_simple_pipeline(self, mock_execute, tmp_path):
        """Test update_vcf_contigs_from_fai works with SimplePipeline"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")
        fasta_fai = str(tmp_path / "reference.fa.fai")

        sp = SimplePipeline(0, 10)
        vcf_utils = VcfUtils(sp)
        vcf_utils.update_vcf_contigs_from_fai(input_vcf=input_vcf, output_vcf=output_vcf, fasta_fai=fasta_fai)

        # Verify the correct bcftools commands were called with SimplePipeline
        expected_reheader_cmd = f"bcftools reheader -f {fasta_fai} -o {output_vcf} {input_vcf}"
        expected_index_cmd = f"bcftools index -tf {output_vcf}"

        expected_calls = [mock.call(expected_reheader_cmd), mock.call(expected_index_cmd)]
        mock_execute.assert_has_calls(expected_calls)

    def test_update_vcf_contigs_from_fai_integration(self, tmp_path):
        """Integration test for update_vcf_contigs_from_fai with real VCF and FAI files"""
        # Use the real test files from cnv/tests/resources
        cnv_resources = Path(__file__).parent.parent.parent.parent / "cnv" / "tests" / "resources"
        input_vcf = cnv_resources / "sample1_test.500.CNV.vcf.gz"
        fasta_fai = cnv_resources / "Homo_sapiens_assembly38.fasta.fai"

        # Skip if test files don't exist
        if not input_vcf.exists() or not fasta_fai.exists():
            pytest.skip("Test resources not available")

        output_vcf = tmp_path / "output_with_updated_contigs.vcf.gz"

        # Run the function
        vcf_utils = VcfUtils()
        vcf_utils.update_vcf_contigs_from_fai(
            input_vcf=str(input_vcf), output_vcf=str(output_vcf), fasta_fai=str(fasta_fai)
        )

        # Verify output file was created and indexed
        assert output_vcf.exists(), "Output VCF file was not created"
        assert (output_vcf.parent / f"{output_vcf.name}.tbi").exists(), "Output VCF index was not created"

        # Read the FAI file to get expected contigs
        fai_contigs = {}
        with open(fasta_fai) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    contig_name = parts[0]
                    contig_length = int(parts[1])
                    fai_contigs[contig_name] = contig_length

        # Read the output VCF header and verify contigs
        with pysam.VariantFile(str(output_vcf), "r") as vcf:
            vcf_contigs = {}
            for contig_name in vcf.header.contigs:
                contig_info = vcf.header.contigs[contig_name]
                vcf_contigs[contig_name] = contig_info.length

            # Verify that all contigs from FAI are in the VCF header
            for contig_name, expected_length in fai_contigs.items():
                assert contig_name in vcf_contigs, f"Contig {contig_name} from FAI not found in VCF header"
                assert vcf_contigs[contig_name] == expected_length, (
                    f"Contig {contig_name} has length {vcf_contigs[contig_name]} in VCF "
                    f"but expected {expected_length} from FAI"
                )

            # Verify the number of contigs matches
            assert len(vcf_contigs) == len(
                fai_contigs
            ), f"Number of contigs in VCF ({len(vcf_contigs)}) does not match FAI ({len(fai_contigs)})"

    @patch("os.unlink")
    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_concat_vcf_basic(self, mock_execute, mock_unlink, tmp_path):
        """Test concat_vcf with basic parameters"""
        input_files = [
            str(tmp_path / "input1.vcf.gz"),
            str(tmp_path / "input2.vcf.gz"),
            str(tmp_path / "input3.vcf.gz"),
        ]
        output_vcf = str(tmp_path / "output.vcf.gz")
        tmp_file = str(tmp_path / "output.vcf.gz.tmp.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.concat_vcf(input_files=input_files, output_file=output_vcf)

        # Verify the correct sequence of commands
        input_files_str = " ".join(input_files)
        expected_calls = [
            mock.call(f"bcftools concat -a -o {tmp_file} -O z {input_files_str}"),
            mock.call(f"bcftools sort -o {output_vcf} -O z {tmp_file} -T {tmp_path}/"),
            mock.call(f"bcftools index -tf {output_vcf}"),
        ]
        mock_execute.assert_has_calls(expected_calls)
        assert mock_execute.call_count == 3

        # Verify temporary file is cleaned up
        mock_unlink.assert_called_once_with(tmp_file)

    @patch("os.unlink")
    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_concat_vcf_single_file(self, mock_execute, mock_unlink, tmp_path):
        """Test concat_vcf with a single input file"""
        input_files = [str(tmp_path / "input1.vcf.gz")]
        output_vcf = str(tmp_path / "output.vcf.gz")
        tmp_file = str(tmp_path / "output.vcf.gz.tmp.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.concat_vcf(input_files=input_files, output_file=output_vcf)

        # Verify commands are still executed even with single file
        expected_calls = [
            mock.call(f"bcftools concat -a -o {tmp_file} -O z {input_files[0]}"),
            mock.call(f"bcftools sort -o {output_vcf} -O z {tmp_file} -T {tmp_path}/"),
            mock.call(f"bcftools index -tf {output_vcf}"),
        ]
        mock_execute.assert_has_calls(expected_calls)

        # Verify temporary file is cleaned up
        mock_unlink.assert_called_once_with(tmp_file)

    @patch("os.unlink")
    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_concat_vcf_many_files(self, mock_execute, mock_unlink, tmp_path):
        """Test concat_vcf with many input files"""
        # Create a list of 10 input files
        input_files = [str(tmp_path / f"input{i}.vcf.gz") for i in range(1, 11)]
        output_vcf = str(tmp_path / "output.vcf.gz")
        tmp_file = str(tmp_path / "output.vcf.gz.tmp.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.concat_vcf(input_files=input_files, output_file=output_vcf)

        # Verify the concat command includes all files
        input_files_str = " ".join(input_files)
        expected_calls = [
            mock.call(f"bcftools concat -a -o {tmp_file} -O z {input_files_str}"),
            mock.call(f"bcftools sort -o {output_vcf} -O z {tmp_file} -T {tmp_path}/"),
            mock.call(f"bcftools index -tf {output_vcf}"),
        ]
        mock_execute.assert_has_calls(expected_calls)

        # Verify temporary file is cleaned up
        mock_unlink.assert_called_once_with(tmp_file)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_remove_filters_all(self, mock_execute, tmp_path):
        """Test removing all filters from VCF."""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")

        vcf_utils = VcfUtils()
        vcf_utils.remove_filters(input_vcf=input_vcf, output_vcf=output_vcf)

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools annotate -x FILTER -o {output_vcf} {input_vcf}"
        mock_execute.assert_called_once_with(expected_cmd)

    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_remove_filters_specific(self, mock_execute, tmp_path):
        """Test removing specific filters from VCF."""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")
        filters = ["LowQual", "LowDP"]

        vcf_utils = VcfUtils()
        vcf_utils.remove_filters(input_vcf=input_vcf, output_vcf=output_vcf, filters_to_remove=filters)

        # Verify the correct bcftools command was called
        expected_cmd = f"bcftools annotate -x FILTER/LowQual,LowDP -o {output_vcf} {input_vcf}"
        mock_execute.assert_called_once_with(expected_cmd)

    @patch("os.unlink")
    @patch("ugbio_core.vcf_utils.VcfUtils._VcfUtils__execute")
    def test_collapse_vcf_pick_best(self, mock_execute, mock_unlink, tmp_path, mock_logger):
        """Test collapse_vcf with pick_best parameter"""
        input_vcf = str(tmp_path / "input.vcf.gz")
        output_vcf = str(tmp_path / "output.vcf.gz")
        removed_vcf_path = str(tmp_path / "tmp.vcf")

        # Test with pick_best=True
        vcf_utils = VcfUtils(logger=mock_logger)
        vcf_utils.collapse_vcf(vcf=input_vcf, output_vcf=output_vcf, pick_best=True)

        # Verify the command includes --keep maxqual
        call_args = mock_execute.call_args[0][0]
        assert "--keep maxqual" in call_args
        assert "--keep first" not in call_args
        assert "truvari collapse" in call_args

        # Verify temporary file cleanup
        mock_unlink.assert_called_once_with(removed_vcf_path)

        # Reset mocks
        mock_execute.reset_mock()
        mock_unlink.reset_mock()

        # Test with pick_best=False (default)
        vcf_utils.collapse_vcf(vcf=input_vcf, output_vcf=output_vcf, pick_best=False)

        # Verify the command includes --keep first
        call_args = mock_execute.call_args[0][0]
        assert "--keep first" in call_args
        assert "--keep maxqual" not in call_args

        # Verify temporary file cleanup again
        mock_unlink.assert_called_once_with(removed_vcf_path)
