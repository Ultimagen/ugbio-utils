from os.path import exists
from os.path import join as pjoin
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.vcf_utils import VcfUtils


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@patch("subprocess.call")
def test_intersect_bed_files(mock_subprocess_call, tmp_path, resources_dir):
    bed1 = pjoin(resources_dir, "bed1.bed")
    bed2 = pjoin(resources_dir, "bed2.bed")
    output_path = pjoin(tmp_path, "output.bed")

    # Test with simple pipeline
    sp = SimplePipeline(0, 10)
    VcfUtils(sp).intersect_bed_files(bed1, bed2, output_path)
    # TBD also add test for sp
    # mock_subprocess_call.assert_called_once_with(
    #     " ".join(["bedtools", "intersect", "-a", bed1, "-b", bed2]), stdout=mock.ANY, shell=True
    # )

    VcfUtils().intersect_bed_files(bed1, bed2, output_path)
    mock_subprocess_call.assert_called_once_with(
        " ".join(["bedtools", "intersect", "-a", bed1, "-b", bed2]), stdout=mock.ANY, shell=True
    )
    assert exists(output_path)


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
