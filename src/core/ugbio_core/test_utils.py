"""Shared test utilities for ugbio_utils test suites.

This module provides common test helper functions that can be reused across
different modules in the ugbio_utils workspace.
"""
# ruff: noqa: S101

import pysam


def compare_vcfs(vcf1_file, vcf2_file):
    """Compare two VCF files for equality.

    Performs a comprehensive comparison of two VCF files, checking:
    - Number of variants
    - Chromosome, position, reference, and alternate alleles
    - Quality scores
    - Filter fields
    - INFO fields and their values

    Args:
        vcf1_file: Path to the first VCF file
        vcf2_file: Path to the second VCF file

    Raises:
        AssertionError: If any differences are found between the VCF files
    """
    vcf1 = pysam.VariantFile(vcf1_file)
    vcf2 = pysam.VariantFile(vcf2_file)

    records1 = list(vcf1)
    records2 = list(vcf2)

    assert len(records1) == len(
        records2
    ), f"VCF files have different numbers of variants: {len(records1)} != {len(records2)}"

    for i, (rec1, rec2) in enumerate(zip(records1, records2, strict=False)):
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
