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

    def test_filter_cnmops_dups(self, tmpdir, resources_dir):
        """Test filtering of short cn.mops duplications with merging of adjacent segments."""
        input_vcf = pjoin(resources_dir, "HG002.full_sample.combined.step1.vcf.gz")
        expected_output_vcf = pjoin(resources_dir, "HG002.full_sample.combined.step2.vcf.gz")
        output_vcf = pjoin(tmpdir, "filtered_dups.vcf.gz")

        combine_cnmops_cnvpytor_cnv_calls.run(
            [
                "cnv_results_to_vcf",
                "filter_cnmops_dups",
                "--combined_calls",
                input_vcf,
                "--combined_calls_annotated",
                output_vcf,
                "--filtered_length",
                "10000",
                "--distance_threshold",
                "1500",
            ]
        )

        assert os.path.exists(output_vcf)
        assert os.path.exists(f"{output_vcf}.tbi")
        compare_vcfs(output_vcf, expected_output_vcf)

    def test_annotate_vcf_with_regions(self, tmpdir, resources_dir):
        """Integration test for annotating VCF with region annotations from BED file."""
        input_vcf = pjoin(resources_dir, "HG002.full_sample.combined.step1.chr5.vcf.gz")
        annotation_bed = pjoin(resources_dir, "ug_cnv_lcr.chr5.bed")
        genome = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        output_vcf = pjoin(tmpdir, "annotated_regions.vcf.gz")
        expected_output_vcf = pjoin(resources_dir, "HG002.full_sample.combined.step1.chr5.annotated.vcf.gz")

        combine_cnmops_cnvpytor_cnv_calls.run(
            [
                "cnv_results_to_vcf",
                "annotate_regions",
                "--input_vcf",
                input_vcf,
                "--annotation_bed",
                annotation_bed,
                "--output_vcf",
                output_vcf,
                "--overlap_fraction",
                "0.5",
                "--genome",
                genome,
            ]
        )

        # Check output files exist
        assert os.path.exists(output_vcf)
        assert os.path.exists(f"{output_vcf}.tbi")

        # Compare output with expected output for consistency
        compare_vcfs(output_vcf, expected_output_vcf)
