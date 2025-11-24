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
