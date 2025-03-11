import filecmp
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_cnv import combine_cnmops_cnvpytor_cnv_calls


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestCombineCnmopsCnvpytorCnvCalls:
    def test_combine_cnmops_cnvpytor_cnv_calls(self, tmpdir, resources_dir):
        input_cnmops_file = pjoin(resources_dir, "NA24385.cnmops.cnvs.chr1-2.bed")
        input_cnvpytor_file = pjoin(resources_dir, "NA24385.cnvpytor.cnvs.chr1-2.bed")
        input_jalign_file = pjoin(resources_dir, "NA24385.cnmops500mod_cnvpytor500.DEL.jalign.chr1-2.bed")
        input_ug_cnv_lcr_file = pjoin(resources_dir, "ug_cnv_lcr.chr1-2.bed")
        input_fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.chr1-2.fasta.fai")
        sample_name = "test_HG002"
        expected_out_combined_bed = pjoin(
            resources_dir, "expected_HG002.cnmops_cnvpytor.cnvs.combined.UG-CNV-LCR_annotate.chr1-2.bed"
        )

        combine_cnmops_cnvpytor_cnv_calls.run(
            [
                "cnv_results_to_vcf",
                "--cnmops_cnv_calls",
                input_cnmops_file,
                "--cnvpytor_cnv_calls",
                input_cnvpytor_file,
                "--del_jalign_merged_results",
                input_jalign_file,
                "--ug_cnv_lcr",
                input_ug_cnv_lcr_file,
                "--fasta_index",
                input_fasta_index_file,
                "--out_directory",
                f"{tmpdir}/",
                "--sample_name",
                sample_name,
            ]
        )

        out_combined_bed = pjoin(tmpdir, f"{sample_name}.cnmops_cnvpytor.cnvs.combined.UG-CNV-LCR_annotate.bed")
        assert filecmp.cmp(out_combined_bed, expected_out_combined_bed)
