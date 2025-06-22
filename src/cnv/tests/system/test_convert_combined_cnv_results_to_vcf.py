from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_cnv import convert_combined_cnv_results_to_vcf


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
        assert rec1.filter.keys() == rec2.filter.keys(), "Filter field mismatch"

        # Compare INFO fields
        assert rec1.info.keys() == rec2.info.keys(), "INFO fields mismatch"
        for key in rec1.info.keys():
            assert rec1.info[key] == rec2.info[key], f"INFO mismatch in {key}: {rec1.info[key]} != {rec2.info[key]}"


class TestConvertCombinedCnvResultsToVcf:
    def test_write_vcf(self, tmpdir, resources_dir):
        sample_name = "TEST_HG002_chr19"
        cnv_annotated_bed_file = pjoin(
            resources_dir, "expected_TEST_HG002_chr19.cnmops_cnvpytor.cnvs.combined.bed.annotate.bed"
        )
        fasta_index_file = pjoin(resources_dir, "chr19.fasta.fai")
        outfile = pjoin(tmpdir, f"{sample_name}.cnv.vcf.gz")
        convert_combined_cnv_results_to_vcf.write_combined_vcf(
            outfile, cnv_annotated_bed_file, sample_name, fasta_index_file
        )

        expected_vcf_file = pjoin(resources_dir, "TEST_HG002_chr19.cnv.vcf.gz")
        compare_vcfs(expected_vcf_file, outfile)
