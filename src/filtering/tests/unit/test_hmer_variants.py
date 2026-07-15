#!/usr/bin/env python
"""
Test script for hmer-based variant classification.

Tests the logic for classifying nearby variants as insertions, deletions, or SNPs
based on homopolymer size changes.
"""

import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Create a mock reference object for testing
class MockFasta:
    """Mock Fasta object for testing without actual genome files."""

    def __init__(self, sequences: dict):
        """Initialize with chromosome -> sequence mapping.

        Args:
            sequences: Dictionary mapping chromosome names to sequence strings
        """
        self.sequences = sequences

    def __getitem__(self, chrom):
        """Get sequence accessor for a chromosome."""
        return MockChromosome(self.sequences[chrom])


class MockChromosome:
    """Mock chromosome sequence accessor."""

    def __init__(self, seq: str):
        self.seq = seq

    def __getitem__(self, key):
        """Support slicing and single position access."""
        return self.seq[key]


def test_variant_classification():
    """Test variant classification with synthetic reference sequence."""

    # Reference structure: 30 N's padding + 6 C's + 12 A's + 30 N's padding
    # This provides sufficient context for apply_variant (SEQUENCE_CONTEXT_SIZE = 30)
    # Positions 30-35: CCCCCC (6 C's)
    # Positions 36-47: AAAAAAAAAAAA (12 A's)
    # Rest: N's for padding
    ref_seq = "N" * 30 + "C" * 6 + "A" * 12 + "N" * 30

    print(f"Reference sequence (relevant region): {ref_seq[20:60]}")
    print(f"                                       {''.join(str(i % 10) for i in range(20, 60))}")
    print(f"Full reference length: {len(ref_seq)} bp")
    print()

    # Simulate VCF record structure
    # rec.pos = 30 (start of C region, after 30 N's)
    # X_HIL = 6 (6 C's in the homopolymer region)
    # offset_pos = rec.pos + X_HIL//2 = 30 + 3 = 33 (middle of C region at position 33)
    rec_pos = 30
    x_hil = 6
    offset_pos = rec_pos + x_hil // 2
    print(f"Simulated VCF record: rec.pos={rec_pos}, X_HIL={x_hil}")
    print(f"Offset position (rec.pos + X_HIL//2): {offset_pos}")
    print(f"Reference at offset_pos: {ref_seq[offset_pos]}")
    print()

    mock_fasta = MockFasta({"chr1": ref_seq})

    # Import the functions we're testing
    from ugbio_filtering.vcf_hmer_update import (
        apply_variant,
        does_variant_affect_hmer_size,
        get_ref_hmer_size,
    )

    print("=" * 80)
    print("TEST 1: Reference hmer size at offset position")
    print("=" * 80)

    # Test reference hmer at the offset position
    # offset_pos = 3 (middle of C-region)
    # The reference hmer_size is calculated at THIS offset position for all evaluations
    hmer_at_offset = get_ref_hmer_size(mock_fasta, "chr1", offset_pos)
    ref_nuc = ref_seq[offset_pos]
    print(f"Position {offset_pos} (offset_pos = rec.pos + X_HIL//2): nucleotide={ref_nuc}, ref_hmer={hmer_at_offset}")
    print("  Expected: 6 (all C's from positions 0-5)")
    assert hmer_at_offset == 6, f"Expected 6 but got {hmer_at_offset}"
    print("  ✓ PASS\n")

    print("=" * 80)
    print("TEST 2: Understanding hmer_size calculation for nearby variants")
    print("=" * 80)
    print(f"\nKey principle: hmer_size is ALWAYS calculated at the offset position ({offset_pos})")
    print(f"For all nearby variants, we use the SAME reference hmer_size = {hmer_at_offset}")
    print(f"\nPositions 30-31 have different positions but same reference hmer_size of {hmer_at_offset}")
    print("when calculated at the offset position perspective.\n")

    print("=" * 80)
    print("TEST 3: Variant testing at different positions")
    print("=" * 80)
    print(f"\nKey principle: After applying variant, check nucleotide at offset_pos={offset_pos}")
    print("Count hmer of the nucleotide that ends up at that position")
    print(f"Reference hmer = {hmer_at_offset}\n")

    # Test variants according to user specification
    # Positions are absolute genome positions (30-35 for C region, 36+ for A region)
    test_cases = [
        # (position, ref_allele, alt_allele, expected_hmer_after, expected_type)
        (35, "C", "A", 5, "deletion"),  # C->A: C region shrinks from 6 to 5
        (36, "A", "C", 7, "insertion"),  # A->C: C region extends from 6 to 7
        (30, "C", "T", 5, "deletion"),  # C->T: C region becomes positions 31-35 = 5
        (31, "C", "T", 4, "deletion"),  # C->T: C region becomes positions 32-35 = 4
        (32, "C", "T", 3, "deletion"),  # C->T: C region becomes positions 33-35 = 3
        (33, "C", "T", 1, "deletion"),  # C->T: offset_pos=33 is now T, count T's = 1
        (34, "C", "T", 4, "deletion"),  # C->T: C region becomes positions 30-33 = 4
        (35, "C", "T", 5, "deletion"),  # C->T: C region becomes positions 30-34 = 5
    ]

    test_results = []
    for test_pos, ref_allele, alt_allele, expected_hmer, expected_type in test_cases:
        print(f"--- Variant {ref_allele}→{alt_allele} at position {test_pos} ---")
        print(f"  Reference at pos {test_pos}: {ref_seq[test_pos]}")

        affects_hmer = does_variant_affect_hmer_size(
            mock_fasta, "chr1", test_pos, ref_allele, alt_allele, hmer_at_offset, check_pos=offset_pos
        )
        hmer_after, seq = apply_variant(
            mock_fasta, ["chr1", test_pos], [test_pos, ref_allele, alt_allele], check_pos=offset_pos
        )

        # Determine expected_affects_hmer based on expected_type
        expected_affects_hmer = expected_type != "snp"

        print(f"  Variant: {ref_allele}→{alt_allele} at position {test_pos}")
        print(f"  Checked hmer at position {offset_pos} (offset)")
        print(f"  Ref hmer (at offset {offset_pos}): {hmer_at_offset}")
        print(f"  Hmer after at offset: {hmer_after}")
        print(f"  Affects hmer: {affects_hmer} (expected: {expected_affects_hmer})")

        assert (
            hmer_after == expected_hmer
        ), f"Position {test_pos}: expected hmer_after={expected_hmer} but got {hmer_after}"
        assert (
            affects_hmer == expected_affects_hmer
        ), f"Position {test_pos}: expected affects_hmer={expected_affects_hmer} but got {affects_hmer}"
        print("  ✓ PASS\n")

        test_results.append((test_pos, expected_type, hmer_at_offset, hmer_after))

    print("✓ All variants correctly classified\n")

    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_variant_classification()
