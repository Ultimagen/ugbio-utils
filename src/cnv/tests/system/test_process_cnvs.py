import os
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_cnv import process_cnvs
from ugbio_core.test_utils import compare_vcfs


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestProcessCnvsIntegration:
    """Integration tests for the complete process_cnvs pipeline."""

    def test_process_cnvs_full_pipeline(self, tmpdir, resources_dir):
        """Test the complete process_cnvs pipeline with all inputs."""
        # Input files
        input_bed_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.bed")
        cnv_lcr_file = pjoin(resources_dir, "ug_cnv_lcr.chr5.bed")
        sample_norm_coverage_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cov.bed")
        cohort_avg_coverage_file = pjoin(resources_dir, "coverage.cohort.bed")
        fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        sample_name = "test_sample"

        # Expected output
        expected_vcf_file = pjoin(resources_dir, "expected_test_sample.cnv.filtered.vcf.gz")

        # Run the pipeline
        process_cnvs.run(
            [
                "process_cnvs",
                "--input_bed_file",
                input_bed_file,
                "--cnv_lcr_file",
                cnv_lcr_file,
                "--out_directory",
                f"{tmpdir}/",
                "--sample_norm_coverage_file",
                sample_norm_coverage_file,
                "--cohort_avg_coverage_file",
                cohort_avg_coverage_file,
                "--fasta_index_file",
                fasta_index_file,
                "--intersection_cutoff",
                "0.5",
                "--sample_name",
                sample_name,
                "--min_cnv_length",
                "10000",
            ]
        )

        # Check that output files exist
        out_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz")
        out_vcf_idx_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz.tbi")

        assert os.path.exists(out_vcf_file), f"Output VCF file not found: {out_vcf_file}"
        assert os.path.exists(out_vcf_idx_file), f"Output VCF index file not found: {out_vcf_idx_file}"

        # Compare output VCF with expected VCF
        compare_vcfs(out_vcf_file, expected_vcf_file)

    def test_process_cnvs_without_lcr_filtering(self, tmpdir, resources_dir):
        """Test the process_cnvs pipeline without LCR filtering."""
        # Input files
        input_bed_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.bed")
        sample_norm_coverage_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cov.bed")
        cohort_avg_coverage_file = pjoin(resources_dir, "coverage.cohort.bed")
        fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        sample_name = "test_sample_no_lcr"

        # Run the pipeline without cnv_lcr_file
        process_cnvs.run(
            [
                "process_cnvs",
                "--input_bed_file",
                input_bed_file,
                "--out_directory",
                f"{tmpdir}/",
                "--sample_norm_coverage_file",
                sample_norm_coverage_file,
                "--cohort_avg_coverage_file",
                cohort_avg_coverage_file,
                "--fasta_index_file",
                fasta_index_file,
                "--intersection_cutoff",
                "0.5",
                "--sample_name",
                sample_name,
            ]
        )

        # Check that output files exist
        out_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz")
        out_vcf_idx_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz.tbi")

        assert os.path.exists(out_vcf_file), f"Output VCF file not found: {out_vcf_file}"
        assert os.path.exists(out_vcf_idx_file), f"Output VCF index file not found: {out_vcf_idx_file}"

        # Verify VCF can be opened and has records
        vcf = pysam.VariantFile(out_vcf_file)
        records = list(vcf)
        assert len(records) > 0, "Output VCF should contain variants"

        # Verify that variants have LEN filter (length filtering applied) but no LCR filter
        for rec in records:
            filter_keys = list(rec.filter.keys())
            # Should have either PASS or LEN filter, but not UG-CNV-LCR
            assert "UG-CNV-LCR" not in filter_keys, f"Unexpected UG-CNV-LCR filter: {filter_keys}"

    def test_process_cnvs_no_filtering(self, tmpdir, resources_dir):
        """Test the process_cnvs pipeline without any filtering."""
        # Input files
        input_bed_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.bed")
        sample_norm_coverage_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cov.bed")
        cohort_avg_coverage_file = pjoin(resources_dir, "coverage.cohort.bed")
        fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        sample_name = "test_sample_no_filter"

        # Run the pipeline without any filtering (no cnv_lcr_file and min_cnv_length=None)
        process_cnvs.run(
            [
                "process_cnvs",
                "--input_bed_file",
                input_bed_file,
                "--out_directory",
                f"{tmpdir}/",
                "--sample_norm_coverage_file",
                sample_norm_coverage_file,
                "--cohort_avg_coverage_file",
                cohort_avg_coverage_file,
                "--fasta_index_file",
                fasta_index_file,
                "--intersection_cutoff",
                "0.5",
                "--sample_name",
                sample_name,
                "--min_cnv_length",
                "0",  # Disable length filtering
            ]
        )

        # Check that output files exist (no filtering, just sorted)
        out_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz")
        out_vcf_idx_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz.tbi")

        assert os.path.exists(out_vcf_file), f"Output VCF file not found: {out_vcf_file}"
        assert os.path.exists(out_vcf_idx_file), f"Output VCF index file not found: {out_vcf_idx_file}"

        # Verify VCF can be opened and has records
        vcf = pysam.VariantFile(out_vcf_file)
        records = list(vcf)
        assert len(records) > 0, "Output VCF should contain variants"

        # Verify that all variants have PASS filter (no filtering applied)
        for rec in records:
            filter_keys = list(rec.filter.keys())
            # Should have PASS filter only, no LEN or UG-CNV-LCR
            assert "PASS" in filter_keys or len(filter_keys) == 0, f"Expected PASS filter, got: {filter_keys}"
            assert "UG-CNV-LCR" not in filter_keys, f"Unexpected UG-CNV-LCR filter: {filter_keys}"
            assert "LEN" not in filter_keys, f"Unexpected LEN filter: {filter_keys}"

    def test_process_cnvs_vcf_ids(self, tmpdir, resources_dir):
        """Test that VCF IDs are generated in the correct format: cnmops_<svtype>_<count>."""
        # Input files
        input_bed_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.bed")
        sample_norm_coverage_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cov.bed")
        cohort_avg_coverage_file = pjoin(resources_dir, "coverage.cohort.bed")
        fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        sample_name = "test_sample_ids"

        # Run the pipeline
        process_cnvs.run(
            [
                "process_cnvs",
                "--input_bed_file",
                input_bed_file,
                "--out_directory",
                f"{tmpdir}/",
                "--sample_norm_coverage_file",
                sample_norm_coverage_file,
                "--cohort_avg_coverage_file",
                cohort_avg_coverage_file,
                "--fasta_index_file",
                fasta_index_file,
                "--intersection_cutoff",
                "0.5",
                "--sample_name",
                sample_name,
                "--min_cnv_length",
                "0",
            ]
        )

        # Check the VCF
        out_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz")
        vcf = pysam.VariantFile(out_vcf_file)
        records = list(vcf)

        # Track counts per SVTYPE
        svtype_counts = {"dup": 0, "del": 0, "neutral": 0}

        # Verify all records have IDs in the correct format
        for rec in records:
            assert rec.id is not None, f"Record at {rec.chrom}:{rec.start} has no ID"

            svtype = rec.info.get("SVTYPE")
            assert svtype is not None, f"Record at {rec.chrom}:{rec.start} has no SVTYPE"

            svtype_lower = svtype.lower()
            svtype_counts[svtype_lower] += 1

            expected_id = f"cnmops_{svtype_lower}_{svtype_counts[svtype_lower]}"
            assert (
                rec.id == expected_id
            ), f"Record at {rec.chrom}:{rec.start} has ID '{rec.id}', expected '{expected_id}'"

        # Verify we have some variants of each type
        assert svtype_counts["dup"] > 0, "Expected at least one DUP variant"
        assert svtype_counts["del"] > 0, "Expected at least one DEL variant"

    def test_process_cnvs_with_window_size(self, tmpdir, resources_dir):
        """Test that CIPOS is added when window_size parameter is provided."""
        # Input files
        input_bed_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.bed")
        sample_norm_coverage_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cov.bed")
        cohort_avg_coverage_file = pjoin(resources_dir, "coverage.cohort.bed")
        fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        sample_name = "test_sample_window_size"
        window_size = 500

        # Run the pipeline with window_size
        process_cnvs.run(
            [
                "process_cnvs",
                "--input_bed_file",
                input_bed_file,
                "--out_directory",
                f"{tmpdir}/",
                "--sample_norm_coverage_file",
                sample_norm_coverage_file,
                "--cohort_avg_coverage_file",
                cohort_avg_coverage_file,
                "--fasta_index_file",
                fasta_index_file,
                "--sample_name",
                sample_name,
                "--min_cnv_length",
                "0",
                "--window_size",
                str(window_size),
            ]
        )

        # Check the VCF
        out_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz")
        vcf = pysam.VariantFile(out_vcf_file)

        # Verify header has CIPOS INFO field
        assert "CIPOS" in vcf.header.info
        assert vcf.header.info["CIPOS"].number == 2
        assert vcf.header.info["CIPOS"].type == "Integer"

        # Verify all records have CIPOS
        records = list(vcf)
        assert len(records) > 0, "Output VCF should contain variants"

        for rec in records:
            assert "CIPOS" in rec.info, f"Record at {rec.chrom}:{rec.start} missing CIPOS"
            cipos = rec.info["CIPOS"]
            assert len(cipos) == 2, f"CIPOS should have 2 values, got {len(cipos)}"

            # Verify CIPOS values match expected calculation: (-window_size/2, window_size/2+1)
            expected_cipos = (round(-window_size / 2), round(window_size / 2 + 1))
            assert (
                cipos == expected_cipos
            ), f"Record at {rec.chrom}:{rec.start} has CIPOS {cipos}, expected {expected_cipos}"
