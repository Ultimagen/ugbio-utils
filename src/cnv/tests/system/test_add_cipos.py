"""System tests for add_cipos_to_vcf functionality."""

import os
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_cnv import process_cnvs
from ugbio_cnv.add_cipos import add_cipos_to_vcf


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestAddCiposIntegration:
    """Integration tests for add_cipos_to_vcf functionality."""

    def test_add_cipos_to_vcf_basic(self, tmpdir, resources_dir):
        """Test adding CIPOS to a VCF file."""
        # First create a basic VCF using process_cnvs
        input_bed_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.bed")
        sample_norm_coverage_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cov.bed")
        cohort_avg_coverage_file = pjoin(resources_dir, "coverage.cohort.bed")
        fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        sample_name = "test_sample_cipos"
        window_size = 500

        # Run the pipeline to create a base VCF
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
            ]
        )

        # Add CIPOS using the dedicated function
        base_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz")
        out_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.with_cipos.vcf.gz")
        add_cipos_to_vcf(base_vcf_file, out_vcf_file, window_size)

        # Verify the output file exists
        assert os.path.exists(out_vcf_file), f"Output VCF file not created: {out_vcf_file}"

        # Check the VCF with CIPOS
        with pysam.VariantFile(out_vcf_file) as vcf:
            # Verify header has CIPOS INFO field
            assert "CIPOS" in vcf.header.info, "CIPOS not found in VCF header"
            assert vcf.header.info["CIPOS"].number == 2, "CIPOS should have 2 values"
            assert vcf.header.info["CIPOS"].type == "Integer", "CIPOS should be Integer type"

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

    def test_add_cipos_with_different_window_sizes(self, tmpdir, resources_dir):
        """Test adding CIPOS with various window sizes."""
        # First create a basic VCF using process_cnvs
        input_bed_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.bed")
        sample_norm_coverage_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cov.bed")
        cohort_avg_coverage_file = pjoin(resources_dir, "coverage.cohort.bed")
        fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        sample_name = "test_sample_cipos_windows"

        # Run the pipeline to create a base VCF
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
            ]
        )

        base_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz")

        # Test with different window sizes
        window_sizes = [100, 500, 1000, 2000]

        for window_size in window_sizes:
            out_vcf_file = pjoin(tmpdir, f"cipos_window_{window_size}.vcf.gz")
            add_cipos_to_vcf(base_vcf_file, out_vcf_file, window_size)

            # Verify CIPOS values
            vcf = pysam.VariantFile(out_vcf_file)
            records = list(vcf)
            assert len(records) > 0, "Output VCF should contain variants"

            expected_cipos = (round(-window_size / 2), round(window_size / 2 + 1))

            for rec in records:
                cipos = rec.info["CIPOS"]
                assert (
                    cipos == expected_cipos
                ), f"Window size {window_size}: expected CIPOS {expected_cipos}, got {cipos}"

    def test_add_cipos_preserves_all_fields(self, tmpdir, resources_dir):
        """Test that add_cipos_to_vcf preserves all existing VCF fields."""
        # Create a base VCF
        input_bed_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.bed")
        sample_norm_coverage_file = pjoin(resources_dir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cov.bed")
        cohort_avg_coverage_file = pjoin(resources_dir, "coverage.cohort.bed")
        fasta_index_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
        sample_name = "test_sample_preserve"

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
            ]
        )

        base_vcf_file = pjoin(tmpdir, "005499-X0040_MAPQ0.MAPQ0.bam.chr5.cnvs.annotate.vcf.gz")
        out_vcf_file = pjoin(tmpdir, "with_cipos_preserve.vcf.gz")

        # Read original VCF records
        base_vcf = pysam.VariantFile(base_vcf_file)
        base_records = list(base_vcf)
        base_vcf.close()

        # Add CIPOS
        add_cipos_to_vcf(base_vcf_file, out_vcf_file, window_size=500)

        # Read modified VCF records
        out_vcf = pysam.VariantFile(out_vcf_file)
        out_records = list(out_vcf)
        out_vcf.close()

        # Verify same number of records
        assert len(base_records) == len(out_records), "Number of records should be preserved"

        # Verify all original fields are preserved (except CIPOS which is added)
        for base_rec, out_rec in zip(base_records, out_records):
            # Basic fields
            assert base_rec.chrom == out_rec.chrom
            assert base_rec.start == out_rec.start
            assert base_rec.stop == out_rec.stop
            assert base_rec.ref == out_rec.ref
            assert base_rec.alts == out_rec.alts

            # INFO fields (excluding CIPOS which is new)
            for info_key in base_rec.info.keys():
                if info_key != "CIPOS":
                    assert info_key in out_rec.info, f"INFO field {info_key} missing in output"
                    assert base_rec.info[info_key] == out_rec.info[info_key], f"INFO field {info_key} value changed"

            # Verify CIPOS was added
            assert "CIPOS" in out_rec.info, "CIPOS should be added to output"
