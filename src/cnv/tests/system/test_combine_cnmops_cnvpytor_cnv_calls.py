import os
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_cnv import combine_cnmops_cnvpytor_cnv_calls, combine_cnv_vcf_utils
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


class TestMergeCnvsInVcfIntegration:
    """Integration tests for merge_cnvs_in_vcf using real data."""

    def test_merge_cnvs_real_data(self, resources_dir, tmp_path):
        """Test merge_cnvs_in_vcf with real HG002 data files."""
        # Input and expected output files
        input_vcf = resources_dir / "merge_cnv_input.vcf.gz"
        expected_output_vcf = resources_dir / "merge_cnv_output.vcf.gz"

        # Create output file in tmp directory
        output_vcf = tmp_path / "test_output.vcf.gz"

        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(input_vcf), str(output_vcf), distance=1500)
        # Verify output matches expected
        compare_vcfs(str(expected_output_vcf), str(output_vcf))

    def test_merge_cnvs_with_smoothing(self, resources_dir, tmp_path):
        """Test merge_cnvs_in_vcf with smoothing enabled on real HG002 data."""
        # Input and expected output files
        input_vcf = resources_dir / "merge_cnv_input.vcf.gz"
        expected_output_vcf = resources_dir / "merge_cnv_output_smoothed.vcf.gz"

        # Create output file in tmp directory
        output_vcf = tmp_path / "test_output_smoothed.vcf.gz"

        # Run merge with smoothing enabled
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(
            str(input_vcf),
            str(output_vcf),
            distance=1500,
            enable_smoothing=True,
            max_gap_absolute=50000,
            gap_scale_fraction=0.05,
            cipos_threshold=50,
            ignore_sv_type=False,
            ignore_filter=True,
        )

        # Verify output matches expected baseline
        compare_vcfs(str(expected_output_vcf), str(output_vcf))

    def test_merge_cnvs_with_smoothing_filter_enabled(self, resources_dir, tmp_path):
        """Test merge with smoothing and filter checking enabled (ignore_filter=False)."""
        input_vcf = resources_dir / "merge_cnv_input.vcf.gz"
        output_vcf = tmp_path / "test_output_smoothed_filtered.vcf.gz"

        # This should NOT crash - tests the code path that failed in omics
        # The key difference from test_merge_cnvs_with_smoothing is ignore_filter=False
        # which triggers _remove_overlapping_filtered_variants() requiring indexed VCF
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(
            str(input_vcf),
            str(output_vcf),
            distance=1500,
            enable_smoothing=True,
            max_gap_absolute=50000,
            gap_scale_fraction=0.05,
            cipos_threshold=50,
            ignore_sv_type=False,
            ignore_filter=False,  # Key difference from existing test
        )

        # Verify output exists (no crash from missing index)
        assert output_vcf.exists()
