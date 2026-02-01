import warnings

from ugbio_core import dna_sequence_utils

warnings.filterwarnings("ignore")


class TestDnaUtils:
    def test_revcomp(self):
        seq = "ATGCAGCTGTGTTACGCGAT"

        assert dna_sequence_utils.revcomp(seq) == "ATCGCGTAACACAGCTGCAT"

    def test_revcom_arr(self):
        seq = ["A", "T", "G", "C", "A", "G", "C", "T", "G", "T", "G", "T", "T", "A", "C", "G", "C", "G", "A", "T"]
        assert dna_sequence_utils.revcomp(list(seq)) == [
            "A",
            "T",
            "C",
            "G",
            "C",
            "G",
            "T",
            "A",
            "A",
            "C",
            "A",
            "C",
            "A",
            "G",
            "C",
            "T",
            "G",
            "C",
            "A",
            "T",
        ]

    def test_parse_cigar_string(self):
        # Test simple CIGAR string - M=0, S=4
        assert dna_sequence_utils.parse_cigar_string("50M30S") == [(0, 50), (4, 30)]

        # Test CIGAR string with left soft clip
        assert dna_sequence_utils.parse_cigar_string("30S50M") == [(4, 30), (0, 50)]

        # Test complex CIGAR string - M=0, D=2, S=4
        assert dna_sequence_utils.parse_cigar_string("10S40M2D20M10S") == [(4, 10), (0, 40), (2, 2), (0, 20), (4, 10)]

        # Test CIGAR with all operations - H=5, S=4, M=0, I=1, D=2
        assert dna_sequence_utils.parse_cigar_string("5H10S20M5I15M3D10M5S5H") == [
            (5, 5),
            (4, 10),
            (0, 20),
            (1, 5),
            (0, 15),
            (2, 3),
            (0, 10),
            (4, 5),
            (5, 5),
        ]

    def test_get_reference_alignment_end(self):
        # Test simple match - 50 bases consumed on reference
        assert dna_sequence_utils.get_reference_alignment_end(100, "50M") == 150

        # Test with soft clip at start - soft clips don't consume reference
        assert dna_sequence_utils.get_reference_alignment_end(100, "30S50M") == 150

        # Test with deletion - deletions consume reference bases
        assert dna_sequence_utils.get_reference_alignment_end(100, "50M2D30M") == 182

        # Test with insertion - insertions don't consume reference bases
        assert dna_sequence_utils.get_reference_alignment_end(100, "50M10I30M") == 180

        # Test complex CIGAR with multiple operations
        # 10S (no ref) + 40M (40 ref) + 2D (2 ref) + 20M (20 ref) + 10S (no ref)
        assert dna_sequence_utils.get_reference_alignment_end(0, "10S40M2D20M10S") == 62

        # Test with hard clips - hard clips don't consume reference
        assert dna_sequence_utils.get_reference_alignment_end(100, "5H50M5H") == 150
