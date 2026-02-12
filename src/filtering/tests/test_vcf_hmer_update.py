"""Tests for vcf_hmer_update module."""

import os

# Import functions to test
import sys
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ugbio_filtering.vcf_hmer_update import (
    apply_variant,
    calc_exp,
    calc_parameters,
    combine_scores,
    direction_score,
    fill_direction_results_with_error,
    filter_reads,
    get_cell,
    get_hmer_qualities_from_pileup_element,
    get_max_nuc,
    load_bed_intervals,
    pos_in_bed,
    process_reads,
)


class TestApplyVariant:
    """Test apply_variant function."""

    def test_apply_variant_basic(self):
        """Test basic variant application."""
        # Mock fasta file
        mock_fasta = {"chr1": Mock()}
        mock_fasta["chr1"].__getitem__ = Mock(return_value="AAABBBCCCAAABBBCCC")

        result = apply_variant(mock_fasta, ["chr1", 10], [5, "A", "AA"])
        assert result[0] >= -1  # Should return valid hmer size or -1


class TestLoadBedIntervals:
    """Test load_bed_intervals function."""

    def test_load_bed_intervals(self):
        """Test BED file loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
            f.write("chr1\t100\t200\n")
            f.write("chr1\t300\t400\n")
            f.write("chr2\t500\t600\n")
            f.flush()

            result = load_bed_intervals(f.name)
            assert "chr1" in result
            assert "chr2" in result
            assert len(result["chr1"]) == 2
            assert result["chr1"][0] == (100, 201)  # note: BED intervals are [start, end+1)

            os.unlink(f.name)

    def test_load_bed_intervals_merges(self):
        """Test that overlapping intervals are merged."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
            f.write("chr1\t100\t200\n")
            f.write("chr1\t150\t250\n")  # Overlaps with previous
            f.flush()

            result = load_bed_intervals(f.name)
            assert len(result["chr1"]) == 1  # Should be merged
            assert result["chr1"][0] == (100, 251)

            os.unlink(f.name)


class TestPosInBed:
    """Test pos_in_bed function."""

    def test_pos_in_bed_true(self):
        """Test position found in bed."""
        intervals = {"chr1": [(100, 200)]}
        assert pos_in_bed(intervals, "chr1", 150) is True

    def test_pos_in_bed_false(self):
        """Test position not found in bed."""
        intervals = {"chr1": [(100, 200)]}
        assert pos_in_bed(intervals, "chr1", 50) is False
        assert pos_in_bed(intervals, "chr1", 250) is False

    def test_pos_in_bed_empty_chrom(self):
        """Test empty chromosome."""
        intervals = {"chr1": [(100, 200)]}
        assert pos_in_bed(intervals, "chr2", 150) is False


class TestCalcExp:
    """Test calc_exp function."""

    def test_calc_exp_basic(self):
        """Test expectation calculation."""
        frequencies = [0.1, 0.2, 0.3, 0.4]
        result = calc_exp(frequencies)
        expected = 0 * 0.1 + 1 * 0.2 + 2 * 0.3 + 3 * 0.4
        assert pytest.approx(result) == expected

    def test_calc_exp_uniform(self):
        """Test with uniform distribution."""
        frequencies = [0.25, 0.25, 0.25, 0.25]
        result = calc_exp(frequencies)
        assert pytest.approx(result) == 1.5


class TestGetMaxNuc:
    """Test get_max_nuc function."""

    def test_get_max_nuc(self):
        """Test most common nucleotide."""
        nuc_list = ["A", "A", "A", "T", "G"]
        result = get_max_nuc(nuc_list)
        assert result == "A"

    def test_get_max_nuc_empty(self):
        """Test empty list."""
        result = get_max_nuc([])
        assert result is None

    def test_get_max_nuc_tie(self):
        """Test with tie - should return one of them."""
        nuc_list = ["A", "A", "T", "T"]
        result = get_max_nuc(nuc_list)
        assert result in ["A", "T"]


class TestFilterReads:
    """Test filter_reads function."""

    def test_filter_reads_basic(self):
        """Test read filtering."""
        results = [
            ("A", None, True, None, False, True),  # nuc=A, is_edge=False, is_not_dup=True, strand=True
            ("A", None, False, None, False, True),  # nuc=A, is_edge=False, is_not_dup=True, strand=False
            ("T", None, True, None, False, True),  # Different nuc
            ("A", None, True, None, True, True),  # is_edge=True (filtered out)
            ("A", None, True, None, False, False),  # is_not_dup=False (filtered out)
        ]

        filtered = filter_reads(results, "A", 1)  # strand=1 (True)
        assert len(filtered) == 1
        assert filtered[0][0] == "A"


class TestProcessReads:
    """Test process_reads function."""

    def test_process_reads(self):
        """Test read processing."""
        reads = [
            ("A", np.array([0.1, 0.8, 0.1]), True, 10, False),
            ("A", np.array([0.2, 0.7, 0.1]), False, 20, True),
        ]

        result = process_reads(reads)
        assert len(result) == 2
        assert len(result[0]) == 5  # (expect, max_conf, high_conf, pos, cycle)


class TestDirectionScore:
    """Test direction_score function."""

    def test_direction_score_basic(self):
        """Test direction score calculation."""
        result = direction_score(normal_score=10.0, normal_mixture=0.1, tumor_score=20.0, tumor_mixture=0.5)
        assert isinstance(result, (int | float))
        assert result >= 0

    def test_direction_score_zero(self):
        """Test with zero scores."""
        result = direction_score(0, 0, 0, 0)
        assert result == 0


class TestCombineScores:
    """Test combine_scores function."""

    def test_combine_scores(self):
        """Test score combination."""
        result = combine_scores(
            ttest_score=5.0,
            likely_score=3.0,
            likely_mixture=0.2,
            normal_fw_score=2.0,
            normal_fw_mixture=0.1,
            tumor_fw_score=8.0,
            tumor_fw_mixture=0.3,
        )
        assert isinstance(result, (int | float))


class TestFillDirectionResultsWithError:
    """Test fill_direction_results_with_error function."""

    def test_fill_direction_results(self):
        """Test filling error results."""
        direction_results = {}
        prefixes = ["fw_", "bw_"]

        fill_direction_results_with_error(direction_results, prefixes)

        assert len(direction_results) == 2
        assert 0 in direction_results
        assert 1 in direction_results
        assert all(v == -1 for v in direction_results[0].values())
        assert all(v == -1 for v in direction_results[1].values())


class TestGetHmerQualitiesFromPileupElement:
    """Test get_hmer_qualities_from_pileup_element function."""

    @patch("ugbio_core.flow_format.flow_based_read.generate_key_from_sequence")
    @patch("ugbio_core.math_utils.unphred")
    def test_get_hmer_qualities_basic(self, mock_unphred, mock_generate_key):
        """Test hmer quality extraction from pileup."""
        # Setup mocks
        mock_alignment = Mock()
        mock_alignment.query_sequence = "AAABBBAAA"
        mock_alignment.query_qualities = bytes([30, 30, 30, 20, 20, 20, 30, 30, 30])
        mock_alignment.query_length = 9
        mock_alignment.is_reverse = False
        mock_alignment.is_duplicate = False
        mock_alignment.get_tag = Mock(return_value=[0, 0, 0, 1, 1, 1, 0, 0, 0])
        mock_alignment.cigartuples = None

        mock_pileup_read = Mock()
        mock_pileup_read.query_position_or_next = 4
        mock_pileup_read.alignment = mock_alignment

        mock_generate_key.return_value = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        mock_unphred.return_value = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        result = get_hmer_qualities_from_pileup_element(mock_pileup_read)

        assert len(result) == 6
        assert result[0] == "B"  # nucleotide
        assert isinstance(result[1], np.ndarray)  # probabilities
        assert isinstance(result[2], bool)  # is_forward
        assert isinstance(result[3], (int | np.integer))  # cycle
        assert isinstance(result[4], bool)  # is_edge
        assert isinstance(result[5], bool)  # is_not_duplicate


class TestCalcParameters:
    """Test calc_parameters function."""

    def test_calc_parameters_basic(self):
        """Test parameter calculation."""
        reads = [
            (1.1, 0.8, 0.7, 10, False),
            (2.05, 0.9, 0.8, 20, False),
            (3.0, 0.85, 0.75, 30, False),
        ] * 200  # Enough reads to trigger high cvg

        result = calc_parameters(reads)
        assert len(result) == 2
        assert isinstance(result[0], tuple)  # exp_split
        assert isinstance(result[1], tuple)  # (high_conf, cycle)


class TestGetCell:
    """Test get_cell function."""

    def test_get_cell(self):
        """Test cell index calculation."""
        read = (1.5, 0.8, 0.7, 10, False)
        params = [(0.1, 0.5, 0.9), (0.8, 500)]  # exp_split, (high_conf, cycle)
        cell_shift = 10

        result = get_cell(read, params, cell_shift)
        assert isinstance(result, (int | np.integer))
        assert result >= 0


class TestIntegration:
    """Integration tests."""

    def test_flow_basic_pipeline(self):
        """Test basic pipeline flow."""
        # Test that functions can be called in sequence
        nuc_list = ["A", "A", "T", "G", "A"]
        nuc = get_max_nuc(nuc_list)
        assert nuc == "A"

        frequencies = [0.1, 0.2, 0.3, 0.4]
        exp = calc_exp(frequencies)
        assert exp > 0

        score = direction_score(5.0, 0.1, 10.0, 0.2)
        assert score >= 0


class TestErrorHandling:
    """Test error handling."""

    def test_apply_variant_invalid_fasta(self):
        """Test apply_variant with invalid FASTA."""
        mock_fasta = {"chr1": Mock(side_effect=IndexError)}
        result = apply_variant(mock_fasta, ["chr1", 10], [5, "A", "AA"])
        assert result == (-1, None)

    def test_pos_in_bed_edge_cases(self):
        """Test pos_in_bed with edge cases."""
        intervals = {"chr1": [(100, 200), (300, 400)]}

        # At boundaries
        assert pos_in_bed(intervals, "chr1", 100) is True
        assert pos_in_bed(intervals, "chr1", 199) is True
        assert pos_in_bed(intervals, "chr1", 200) is False

        # Between intervals
        assert pos_in_bed(intervals, "chr1", 250) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
