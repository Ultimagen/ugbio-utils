import filecmp
import os
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_cnv import combine_cnmops_cnvpytor_cnv_calls


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def compare_vcfs(vcf1_file, vcf2_file):
    vcf1 = pysam.VariantFile(vcf1_file)
    vcf2 = pysam.VariantFile(vcf2_file)

    records1 = list(vcf1)
    records2 = list(vcf2)

    assert len(records1) == len(records2), "VCF files have different numbers of variants"

    for rec1, rec2 in zip(records1, records2):
        assert rec1.chrom == rec2.chrom, f"Chromosome mismatch: {rec1.chrom} != {rec2.chrom}"
        assert rec1.pos == rec2.pos, f"Position mismatch: {rec1.pos} != {rec2.pos}"
        assert rec1.ref == rec2.ref, f"Reference mismatch: {rec1.ref} != {rec2.ref}"
        assert rec1.alts == rec2.alts, f"Alternate mismatch: {rec1.alts} != {rec2.alts}"
        assert rec1.qual == rec2.qual, f"Quality mismatch: {rec1.qual} != {rec2.qual}"
        assert set(rec1.filter.keys()) == set(rec2.filter.keys()), "Filter field mismatch"

        # Compare INFO fields
        assert rec1.info.keys() == rec2.info.keys(), "INFO fields mismatch"
        for key in rec1.info.keys():
            assert rec1.info[key] == rec2.info[key], f"INFO mismatch in {key}: {rec1.info[key]} != {rec2.info[key]}"


class TestCombineCnmopsCnvpytorCnvCalls:
    def test_combine_cnmops_cnvpytor_cnv_calls(self, tmpdir, resources_dir):
        input_cnmops_file = pjoin(resources_dir, "NA24385.cnmops500mod.full.cnvs.chr19.bed")
        input_cnvpytor_file = pjoin(resources_dir, "HG002_full_sample.pytor.bin500.CNVs.chr19.tsv")
        input_jalign_file = pjoin(resources_dir, "HG002_full_sample.TEST.DEL.jalign.chr19.bed")
        input_ug_cnv_lcr_file = pjoin(resources_dir, "ug_cnv_lcr.chr19.bed")
        input_ref_fasta_file = pjoin(resources_dir, "chr19.fasta")
        input_fasta_index_file = pjoin(resources_dir, "chr19.fasta.fai")
        sample_name = "test_HG002"
        expected_out_combined_bed = pjoin(resources_dir, "expected_test_HG002.cnv.bed")
        expected_out_combined_vcf = pjoin(resources_dir, "expected_test_HG002.cnv.vcf.gz")

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

        out_combined_bed = pjoin(tmpdir, f"{sample_name}.cnv.bed")
        assert os.path.exists(out_combined_bed)
        out_combined_vcf = pjoin(tmpdir, f"{sample_name}.cnv.vcf.gz")
        assert os.path.exists(out_combined_vcf)
        out_combined_vcf_idx = pjoin(tmpdir, f"{sample_name}.cnv.vcf.gz.tbi")
        assert os.path.exists(out_combined_vcf_idx)
        assert filecmp.cmp(out_combined_bed, expected_out_combined_bed, shallow=False)
        compare_vcfs(out_combined_vcf, expected_out_combined_vcf)
