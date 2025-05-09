from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ugbio_comparison.sv_comparison_pipeline import SVComparison


@pytest.fixture
def mock_simple_pipeline():
    return MagicMock()


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestSVComparison:
    @pytest.fixture(autouse=True)
    def setup(self, mock_simple_pipeline):
        self.sv_comparison = SVComparison(simple_pipeline=mock_simple_pipeline)

    @patch("os.path.exists")
    @patch("shutil.rmtree")
    @patch("..sv_comparison_pipeline.print_and_execute")
    def test_run_truvari(self, mock_print_and_execute, mock_rmtree, mock_exists, tmpdir, resources_dir):
        mock_exists.return_value = True
        calls = f"{resources_dir}/calls.vcf"
        gt = f"{resources_dir}/ground_truth.vcf"
        outdir = f"{tmpdir}/truvari_output"
        bed = f"{resources_dir}/regions.bed"
        pctseq = 0.9
        pctsize = 0.8

        self.sv_comparison.run_truvari(
            calls=calls,
            gt=gt,
            outdir=outdir,
            bed=bed,
            pctseq=pctseq,
            pctsize=pctsize,
            erase_outdir=True,
        )

        mock_rmtree.assert_called_once_with(outdir)
        expected_command = (
            f"truvari bench -b {gt} -c {calls} -o {outdir} --passonly "
            f"--includebed {bed} --pctseq {pctseq} --pctsize {pctsize}"
        )
        mock_print_and_execute.assert_called_once_with(
            expected_command, output_file=outdir, simple_pipeline=self.sv_comparison.sp, module_name="__main__"
        )

    @patch("os.path.exists")
    @patch("..sv_comparison_pipeline.print_and_execute")
    def test_run_truvari_no_erase(self, mock_print_and_execute, mock_exists, tmpdir, resources_dir):
        mock_exists.return_value = False
        calls = f"{resources_dir}/calls.vcf"
        gt = f"{resources_dir}/ground_truth.vcf"
        outdir = f"{tmpdir}/truvari_output"

        self.sv_comparison.run_truvari(
            calls=calls,
            gt=gt,
            outdir=outdir,
            erase_outdir=False,
        )

        mock_exists.assert_called_once_with(outdir)
        mock_print_and_execute.assert_called_once_with(
            f"truvari bench -b {gt} -c {calls} -o {outdir} --passonly",
            output_file=outdir,
            simple_pipeline=self.sv_comparison.sp,
            module_name="__main__",
        )
