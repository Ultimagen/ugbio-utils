import os
import random
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_cnv import combine_cnmops_cnvpytor_cnv_calls, combine_cnv_vcf_utils
from ugbio_core.test_utils import compare_vcfs
from ugbio_core.vcf_utils import VcfUtils


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def create_low_score_input_vcf(input_vcf: Path, output_vcf: Path, seed: int = 7) -> set[str]:
    """Create a deterministic test VCF with half the variants marked LOW_SCORE."""
    random_generator = random.Random(seed)

    with pysam.VariantFile(str(input_vcf)) as vcf_in:
        header = vcf_in.header.copy()
        if "LOW_SCORE" not in header.filters:
            header.add_line('##FILTER=<ID=LOW_SCORE,Description="Synthetic low-score filter for integration testing">')

        records = list(vcf_in)
        filtered_indices = set(random_generator.sample(range(len(records)), k=len(records) // 2))
        filtered_ids = {
            record_id
            for index in filtered_indices
            for record_id in [records[index].id]
            if record_id is not None and record_id != "."
        }

        with pysam.VariantFile(str(output_vcf), "w", header=header) as vcf_out:
            for index, record in enumerate(records):
                new_record = VcfUtils.copy_vcf_record(record, header)
                new_record.filter.clear()
                if index in filtered_indices:
                    new_record.filter.add("LOW_SCORE")
                else:
                    new_record.filter.add("PASS")
                vcf_out.write(new_record)

    pysam.tabix_index(str(output_vcf), preset="vcf", force=True)
    return filtered_ids


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
        """Test merge with smoothing and filter checking enabled (ignore_filter=False).

        Uses pre-generated VCF with 80% LOW_SCORE variants (8974 LOW_SCORE, 2244 PASS).
        Expects LOW_SCORE variants to be filtered out when ignore_filter=False.
        """
        input_vcf = resources_dir / "merge_cnv_input_with_low_score.vcf.gz"
        expected_output_vcf = resources_dir / "merge_cnv_output_smoothed_filtered.vcf.gz"
        output_vcf = tmp_path / "test_output_smoothed_filtered.vcf.gz"

        # Run merge with smoothing enabled and ignore_filter=False
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(
            str(input_vcf),
            str(output_vcf),
            distance=1500,
            enable_smoothing=True,
            max_gap_absolute=50000,
            gap_scale_fraction=0.05,
            cipos_threshold=50,
            ignore_sv_type=False,
            ignore_filter=False,  # Triggers _remove_overlapping_filtered_variants()
        )

        # Verify output was created successfully (no crash from missing index)
        assert output_vcf.exists()

        # Compare with golden file
        compare_vcfs(str(expected_output_vcf), str(output_vcf))

        # Check LOW_SCORE variants in output
        with pysam.VariantFile(str(output_vcf)) as vcf_in:
            low_score_count = sum(1 for record in vcf_in if "LOW_SCORE" in record.filter.keys())

        assert low_score_count >= 100, f"Expected at least 100 LOW_SCORE variants in output; got {low_score_count}"

    def test_merge_cnvs_weighted_majority_voting(self, tmp_path):
        """Test that merged SVTYPE is determined by SVLEN-weighted majority vote."""
        # Create test VCF with mixed SVTYPEs that will merge
        vcf_path = tmp_path / "mixed_svtype.vcf.gz"

        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line("##contig=<ID=chr1,length=248956422>")
        header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Length">')
        header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">')
        header.add_line('##INFO=<ID=CNV_SOURCE,Number=.,Type=String,Description="Source">')
        header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_sample("test_sample")

        with pysam.VariantFile(str(vcf_path), "w", header=header) as vcf:
            # Two DELs: 1000bp + 1000bp = 2000bp total
            r1 = vcf.new_record(contig="chr1", start=1000, alleles=("N", "<DEL>"), id="DEL1")
            r1.stop = 2000
            r1.info["SVLEN"] = (1000,)
            r1.info["SVTYPE"] = "DEL"
            r1.info["CNV_SOURCE"] = ("test",)
            r1.info["CIPOS"] = (-50, 50)
            r1.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(r1)

            r2 = vcf.new_record(contig="chr1", start=1500, alleles=("N", "<DEL>"), id="DEL2")
            r2.stop = 2500
            r2.info["SVLEN"] = (1000,)
            r2.info["SVTYPE"] = "DEL"
            r2.info["CNV_SOURCE"] = ("test",)
            r2.info["CIPOS"] = (-50, 50)
            r2.samples["test_sample"]["GT"] = (0, 1)
            vcf.write(r2)

            # One DUP: 200bp total
            r3 = vcf.new_record(contig="chr1", start=2000, alleles=("N", "<DUP>"), id="DUP1")
            r3.stop = 2200
            r3.info["SVLEN"] = (200,)
            r3.info["SVTYPE"] = "DUP"
            r3.info["CNV_SOURCE"] = ("test",)
            r3.info["CIPOS"] = (-50, 50)
            r3.samples["test_sample"]["GT"] = (None, 1)
            vcf.write(r3)

        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)

        # Merge with large distance to force merging
        output_vcf = tmp_path / "merged.vcf.gz"
        combine_cnv_vcf_utils.merge_cnvs_in_vcf(str(vcf_path), str(output_vcf), distance=10000, ignore_sv_type=True)

        # Verify merged output
        with pysam.VariantFile(str(output_vcf)) as vcf:
            records = list(vcf)
            assert len(records) == 1  # Should merge into 1

            merged = records[0]
            # DEL should win: 2000bp > 200bp
            assert merged.info["SVTYPE"] == "DEL"
            # Genotype should match DEL
            assert merged.samples["test_sample"]["GT"] == (0, 1)
