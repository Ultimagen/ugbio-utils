"""Shared test utilities for ugbio_utils test suites.

This module provides common test helper functions that can be reused across
different modules in the ugbio_utils workspace.
"""
# ruff: noqa: S101

import numpy as np
import pysam


def compare_vcfs(vcf1_file, vcf2_file):
    """Compare two VCF files for equality, accounting for sort order at same position.

    Performs a comprehensive comparison of two VCF files, checking:
    - Number of variants
    - Chromosome, position, reference, and alternate alleles
    - Quality scores
    - Filter fields
    - INFO fields and their values

    Groups variants by (chrom, pos) and compares them as sets to handle
    bcftools sort instability for records at identical positions.

    Args:
        vcf1_file: Path to the first VCF file
        vcf2_file: Path to the second VCF file

    Raises:
        AssertionError: If any differences are found between the VCF files
    """
    from collections import defaultdict

    vcf1 = pysam.VariantFile(vcf1_file)
    vcf2 = pysam.VariantFile(vcf2_file)

    records1 = list(vcf1)
    records2 = list(vcf2)

    assert len(records1) == len(
        records2
    ), f"VCF files have different numbers of variants: {len(records1)} != {len(records2)}"

    # Group records by (chrom, pos) to handle sort order instability
    def group_by_position(records):
        grouped = defaultdict(list)
        for rec in records:
            key = (rec.chrom, rec.pos)
            grouped[key].append(rec)
        return grouped

    groups1 = group_by_position(records1)
    groups2 = group_by_position(records2)

    # Compare positions
    positions1 = set(groups1.keys())
    positions2 = set(groups2.keys())

    assert positions1 == positions2, f"Position mismatch: {positions1.symmetric_difference(positions2)}"

    # For each position, compare records (order-agnostic within position)
    for pos in sorted(positions1):
        group1 = groups1[pos]
        group2 = groups2[pos]

        assert len(group1) == len(
            group2
        ), f"Position {pos}: different number of records: {len(group1)} != {len(group2)}"

        # Sort within position by ID and ALT to ensure stable comparison
        def sort_key(rec):
            return (rec.id or "", str(rec.alts))

        group1_sorted = sorted(group1, key=sort_key)
        group2_sorted = sorted(group2, key=sort_key)

        for rec1, rec2 in zip(group1_sorted, group2_sorted, strict=False):
            assert rec1.chrom == rec2.chrom, f"{pos} ID {rec1.id}: Chromosome mismatch: {rec1.chrom} != {rec2.chrom}"
            assert rec1.pos == rec2.pos, f"{pos} ID {rec1.id}: Position mismatch: {rec1.pos} != {rec2.pos}"
            assert rec1.ref == rec2.ref, f"{pos} ID {rec1.id}: Reference mismatch: {rec1.ref} != {rec2.ref}"
            assert rec1.alts == rec2.alts, f"{pos} ID {rec1.id}: Alternate mismatch: {rec1.alts} != {rec2.alts}"
            assert rec1.qual == rec2.qual, f"{pos} ID {rec1.id}: Quality mismatch: {rec1.qual} != {rec2.qual}"
            assert set(rec1.filter.keys()) == set(
                rec2.filter.keys()
            ), f"{pos} ID {rec1.id}: Filter field mismatch: {rec1.filter.keys()} != {rec2.filter.keys()}"

            # Compare INFO fields
            assert (
                rec1.info.keys() == rec2.info.keys()
            ), f"{pos} ID {rec1.id}: INFO fields mismatch: {rec1.info.keys()} != {rec2.info.keys()}"
            for key in rec1.info.keys():
                assert (rec1.info[key] == rec2.info[key]) or (
                    np.isnan(rec1.info[key]) and np.isnan(rec2.info[key])
                ), f"{pos} ID {rec1.id}: INFO mismatch in {key}: {rec1.info[key]} != {rec2.info[key]}"
