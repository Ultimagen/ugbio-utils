import os
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_cnv import process_cnmops_cnvs


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def compare_vcfs(vcf1_file, vcf2_file):
    """Compare two VCF files for equality."""
    vcf1 = pysam.VariantFile(vcf1_file)
    vcf2 = pysam.VariantFile(vcf2_file)

    records1 = list(vcf1)
    records2 = list(vcf2)

    assert len(records1) == len(
        records2
    ), f"VCF files have different numbers of variants: {len(records1)} != {len(records2)}"

    for i, (rec1, rec2) in enumerate(zip(records1, records2)):
        assert rec1.chrom == rec2.chrom, f"Variant {i}: Chromosome mismatch: {rec1.chrom} != {rec2.chrom}"
        assert rec1.pos == rec2.pos, f"Variant {i}: Position mismatch: {rec1.pos} != {rec2.pos}"
        assert rec1.ref == rec2.ref, f"Variant {i}: Reference mismatch: {rec1.ref} != {rec2.ref}"
        assert rec1.alts == rec2.alts, f"Variant {i}: Alternate mismatch: {rec1.alts} != {rec2.alts}"
        assert rec1.qual == rec2.qual, f"Variant {i}: Quality mismatch: {rec1.qual} != {rec2.qual}"
        assert set(rec1.filter.keys()) == set(
            rec2.filter.keys()
        ), f"Variant {i}: Filter field mismatch: {rec1.filter.keys()} != {rec2.filter.keys()}"

        # Compare INFO fields
        assert (
            rec1.info.keys() == rec2.info.keys()
        ), f"Variant {i}: INFO fields mismatch: {rec1.info.keys()} != {rec2.info.keys()}"
        for key in rec1.info.keys():
            assert (
                rec1.info[key] == rec2.info[key]
            ), f"Variant {i}: INFO mismatch in {key}: {rec1.info[key]} != {rec2.info[key]}"


class TestProcessCnmopsCnvsIntegration:
    """Integration tests for the complete process_cnmops_cnvs pipeline."""

    def test_process_cnmops_cnvs_full_pipeline(self, tmpdir, resources_dir):
        """Test the complete process_cnmops_cnvs pipeline with all inputs."""
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
        process_cnmops_cnvs.run(
            [
                "process_cnmops_cnvs",
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
