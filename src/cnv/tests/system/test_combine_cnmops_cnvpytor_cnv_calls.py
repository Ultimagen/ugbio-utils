import os
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_cnv import combine_cnmops_cnvpytor_cnv_calls
from ugbio_core.test_utils import compare_vcfs


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestCombineCnmopsCnvpytorCnvCalls:
    def test_combine_cnmops_cnvpytor_cnv_calls(self, tmpdir, resources_dir):
        input_cnmops_file = pjoin(resources_dir, "NA24385.cnmops.cnvs.chr1_1-12950000.vcf.gz")
        input_cnvpytor_file = pjoin(resources_dir, "HG002_full_sample.pytor.bin500.CNVs.chr1_1-12950000.vcf.gz")
        input_fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        expected_out_combined_vcf = pjoin(resources_dir, "expected_test_HG002.cnv.vcf.gz")

        combine_cnmops_cnvpytor_cnv_calls.run(
            [
                "cnv_results_to_vcf",
                "concat",
                "--cnmops_vcf",
                input_cnmops_file,
                "--cnvpytor_vcf",
                input_cnvpytor_file,
                "--fasta_index",
                input_fasta_index_file,
                "--out_directory",
                f"{tmpdir}/",
                "--output_vcf",
                f"{tmpdir}/combined.cnmops.cnvpytor.vcf.gz",
            ]
        )

        out_combined_vcf = pjoin(tmpdir, "combined.cnmops.cnvpytor.vcf.gz")
        assert os.path.exists(out_combined_vcf)
        out_combined_vcf_idx = pjoin(tmpdir, "combined.cnmops.cnvpytor.vcf.gz.tbi")
        assert os.path.exists(out_combined_vcf_idx)
        compare_vcfs(out_combined_vcf, expected_out_combined_vcf)
