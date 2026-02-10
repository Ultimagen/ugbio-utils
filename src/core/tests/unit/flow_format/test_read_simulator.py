"""
Unit tests for the FlowBasedReadSimulator class.
"""

import numpy as np
import pytest
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugbio_core.flow_format.flow_based_read import FlowBasedRead
from ugbio_core.flow_format.read_simulator import FlowBasedReadSimulator, simulate_reads_from_region


class TestFlowBasedReadSimulator:
    """Test cases for FlowBasedReadSimulator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reference_genome = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        self.start_position = 10
        self.length = 20
        self.flow_cycle = DEFAULT_FLOW_ORDER
        self.noise_std = 0.01

    def test_initialization(self):
        """Test simulator initialization."""
        simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome,
            start_position=self.start_position,
            length=self.length,
            flow_cycle=self.flow_cycle,
            noise_std=self.noise_std,
        )

        assert simulator.reference_genome == self.reference_genome.upper()
        assert simulator.start_position == self.start_position
        assert simulator.length == self.length
        assert simulator.flow_cycle == self.flow_cycle
        assert simulator.noise_std == self.noise_std
        assert len(simulator.target_sequence) == self.length

    def test_target_sequence_extraction(self):
        """Test correct extraction of target sequence."""
        simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome, start_position=self.start_position, length=self.length
        )

        expected_sequence = self.reference_genome[self.start_position : self.start_position + self.length].upper()
        assert simulator.target_sequence == expected_sequence

    def test_invalid_sequence_bounds(self):
        """Test error handling for invalid sequence bounds."""
        with pytest.raises(ValueError, match="exceeds reference genome length"):
            FlowBasedReadSimulator(
                reference_genome=self.reference_genome, start_position=len(self.reference_genome) - 5, length=10
            )

    def test_flow_matrix_generation(self):
        """Test flow matrix generation and properties."""
        simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome,
            start_position=self.start_position,
            length=self.length,
            max_hmer_size=12,
        )

        # Check matrix dimensions
        expected_rows = 12 + 1  # max_hmer_size + 1
        expected_cols = len(simulator.flow_key)
        assert simulator.flow_matrix.shape == (expected_rows, expected_cols)

        # Check that probabilities sum to 1 for each flow
        column_sums = np.sum(simulator.flow_matrix, axis=0)
        np.testing.assert_allclose(column_sums, 1.0, rtol=1e-10)

        # Check that all probabilities are positive
        assert np.all(simulator.flow_matrix >= 0)

    def test_flow_matrix_peak_positions(self):
        """Test that flow matrix peaks are at correct positions."""
        simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome,
            start_position=self.start_position,
            length=self.length,
            noise_std=0.001,  # Very low noise for clear peaks
        )

        # Most probable calls should match the true flow key
        most_probable = simulator.get_most_probable_calls()

        # Account for clipping at max_hmer_size
        expected_calls = np.minimum(simulator.flow_key, simulator.max_hmer_size)
        np.testing.assert_array_equal(most_probable, expected_calls)

    def test_simulate_read(self):
        """Test read simulation."""
        simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome, start_position=self.start_position, length=self.length
        )

        # Test with default read name
        read = simulator.simulate_read()
        assert isinstance(read, FlowBasedRead)
        assert read.seq == simulator.target_sequence
        assert hasattr(read, "_flow_matrix")
        np.testing.assert_array_equal(read._flow_matrix, simulator.flow_matrix)

        # Test with custom read name
        custom_name = "test_read_001"
        read_custom = simulator.simulate_read(read_name=custom_name)
        assert read_custom.read_name == custom_name

    def test_flow_statistics(self):
        """Test flow statistics generation."""
        simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome,
            start_position=self.start_position,
            length=self.length,
            max_hmer_size=10,
        )

        stats = simulator.get_flow_statistics()

        assert stats["n_flows"] == len(simulator.flow_key)
        assert stats["max_hmer_size"] == 10
        assert stats["target_sequence_length"] == self.length
        assert stats["flow_key_length"] == len(simulator.flow_key)
        assert stats["flow_cycle"] == self.flow_cycle
        assert stats["noise_std"] == self.noise_std
        assert isinstance(stats["mean_hmer_call"], int | float | np.number)
        assert isinstance(stats["max_hmer_call"], int | np.integer)

    def test_call_probabilities(self):
        """Test call probability extraction."""
        simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome, start_position=self.start_position, length=self.length
        )

        call_probs = simulator.get_call_probabilities()

        # Should have one probability per flow
        assert len(call_probs) == len(simulator.flow_key)

        # All probabilities should be between 0 and 1
        assert np.all(call_probs >= 0)
        assert np.all(call_probs <= 1)

        # Should match maximum values in each column
        expected_probs = np.max(simulator.flow_matrix, axis=0)
        np.testing.assert_array_equal(call_probs, expected_probs)

    def test_different_flow_cycles(self):
        """Test simulator with different flow cycles."""
        custom_flow_cycle = "ACGT"

        simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome,
            start_position=self.start_position,
            length=self.length,
            flow_cycle=custom_flow_cycle,
        )

        assert simulator.flow_cycle == custom_flow_cycle

        # Flow key should be different from default
        default_simulator = FlowBasedReadSimulator(
            reference_genome=self.reference_genome,
            start_position=self.start_position,
            length=self.length,
            flow_cycle=DEFAULT_FLOW_ORDER,
        )

        # Keys might be different depending on the sequence
        # At minimum, the flow cycles should be different
        assert simulator.flow_cycle != default_simulator.flow_cycle

    def test_homopolymer_sequences(self):
        """Test simulator with homopolymer sequences."""
        # Create a sequence with long homopolymers
        homo_sequence = "AAAAAATTTTTCCCCCGGGGG"

        simulator = FlowBasedReadSimulator(
            reference_genome=homo_sequence, start_position=0, length=len(homo_sequence), max_hmer_size=8
        )

        # Should handle long homopolymers correctly
        assert len(simulator.flow_key) > 0
        assert simulator.flow_matrix.shape[1] == len(simulator.flow_key)

        # Check that long homopolymers are clipped to max_hmer_size
        assert np.all(simulator.get_most_probable_calls() <= simulator.max_hmer_size)


class TestSimulateReadsFromRegion:
    """Test cases for simulate_reads_from_region function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reference_genome = "A" * 1000  # Long reference for multiple reads

    def test_multiple_reads_generation(self):
        """Test generation of multiple reads from a region."""
        start_pos = 100
        end_pos = 400
        read_length = 50
        overlap = 10

        reads = simulate_reads_from_region(
            reference_genome=self.reference_genome,
            start_position=start_pos,
            end_position=end_pos,
            read_length=read_length,
            overlap=overlap,
        )

        # Check that we got the expected number of reads
        step_size = read_length - overlap
        expected_reads = (end_pos - start_pos - read_length) // step_size + 1
        assert len(reads) == expected_reads

        # Check that all reads are FlowBasedRead objects
        for read in reads:
            assert isinstance(read, FlowBasedRead)
            assert len(read.seq) == read_length

    def test_read_positioning(self):
        """Test that reads are positioned correctly."""
        start_pos = 0
        end_pos = 200
        read_length = 50
        overlap = 20

        reads = simulate_reads_from_region(
            reference_genome=self.reference_genome,
            start_position=start_pos,
            end_position=end_pos,
            read_length=read_length,
            overlap=overlap,
        )

        step_size = read_length - overlap

        # Check read names contain position information
        for i, read in enumerate(reads):
            expected_start = start_pos + i * step_size
            assert str(expected_start) in read.read_name
            assert str(expected_start + read_length) in read.read_name

    def test_empty_region(self):
        """Test handling of regions too small for any reads."""
        reads = simulate_reads_from_region(
            reference_genome=self.reference_genome,
            start_position=100,
            end_position=120,  # Only 20 bases
            read_length=50,  # Longer than region
            overlap=10,
        )

        # Should return empty list
        assert len(reads) == 0

    def test_sam_record_conversion(self):
        """Test conversion to SAM record format."""
        reference = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

        simulator = FlowBasedReadSimulator(reference_genome=reference, start_position=0, length=30)

        read = simulator.simulate_read("test_sam_read")

        # Convert to SAM record (this tests integration with existing code)
        sam_record = read.to_record()

        assert sam_record.query_name == "test_sam_read"
        assert sam_record.query_sequence == read.seq
        assert sam_record.has_tag("tp")  # Should have tp tag (modern format)
