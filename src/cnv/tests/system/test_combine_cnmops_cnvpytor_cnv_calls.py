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
        input_cnmops_file = pjoin(resources_dir, "NA24385.cnmops500mod.full.cnvs.chr19.bed")
        input_cnvpytor_file = pjoin(resources_dir, "HG002_full_sample.pytor.bin500.CNVs.chr19.tsv")
        input_jalign_file = pjoin(resources_dir, "HG002_full_sample.TEST.DEL.jalign.chr19.bed")
        input_ug_cnv_lcr_file = pjoin(resources_dir, "ug_cnv_lcr.chr19.bed")
        input_ref_fasta_file = pjoin(resources_dir, "chr19.fasta")
        input_fasta_index_file = pjoin(resources_dir, "chr19.fasta.fai")
        sample_name = "test_HG002"
        expected_out_combined_bed = pjoin(
            resources_dir, "expected_TEST_HG002_chr19.cnmops_cnvpytor.cnvs.combined.bed.annotate.bed"
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
                "--ref_fasta",
                input_ref_fasta_file,
                "--fasta_index",
                input_fasta_index_file,
                "--out_directory",
                f"{tmpdir}/",
                "--sample_name",
                sample_name,
            ]
        )

        out_combined_bed = pjoin(
            tmpdir, f"{sample_name}.cnmops_cnvpytor.cnvs.combined.jalign_annotate.UG-CNV-LCR_annotate.bed"
        )
        assert filecmp.cmp(out_combined_bed, expected_out_combined_bed)
