"""
Flow-based read simulator for generating synthetic reads with flow matrix probabilities.

This module provides functionality to simulate Flow-based reads from reference sequences,
generating probability matrices where each hmer call follows a normal distribution
as specified in the UG CRAM format documentation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugbio_core.flow_format.flow_based_read import FlowBasedRead, generate_key_from_sequence


class FlowBasedReadSimulator:
    """
    Simulator for generating Flow-based reads with realistic probability matrices.

    This class takes a reference genome sequence and generates simulated Flow-based reads
    with probability matrices where each hmer call follows a normal distribution centered
    on the true hmer call with configurable standard deviation.

    Attributes
    ----------
    reference_genome : str
        Reference genome sequence
    start_position : int
        Starting position in the reference
    length : int
        Length of sequence to simulate
    flow_cycle : str
        Four-letter flow cycle (default: TGCA)
    noise_std : float
        Standard deviation for normal distribution around true hmer calls
    max_hmer_size : int
        Maximum hmer size to consider in probability matrix
    """

    def __init__(
        self,
        reference_genome: str,
        start_position: int,
        length: int,
        flow_cycle: str = DEFAULT_FLOW_ORDER,
        noise_std: float = 0.01,
        max_hmer_size: int = 20,
    ):
        """
        Initialize the Flow-based read simulator.

        Parameters
        ----------
        reference_genome : str
            Reference genome sequence
        start_position : int
            Starting position in the reference (0-based)
        length : int
            Length of sequence to simulate
        flow_cycle : str, optional
            Four-letter flow cycle, by default DEFAULT_FLOW_ORDER
        noise_std : float, optional
            Standard deviation for probability distributions, by default 0.01
        max_hmer_size : int, optional
            Maximum hmer size for probability matrix, by default 12
        """
        self.reference_genome = reference_genome.upper()
        self.start_position = start_position
        self.length = length
        self.flow_cycle = flow_cycle
        self.noise_std = noise_std
        self.max_hmer_size = max_hmer_size

        # Extract the target sequence
        self.target_sequence = self._extract_target_sequence()

        # Convert to flow space (key space)
        self.flow_key = generate_key_from_sequence(self.target_sequence, flow_order=self.flow_cycle)

        # Generate the flow matrix
        self.flow_matrix = self._generate_flow_matrix()

    def _extract_target_sequence(self) -> str:
        """
        Extract the target sequence from the reference genome.

        Returns
        -------
        str
            Target sequence of specified length starting at start_position

        Raises
        ------
        ValueError
            If start_position + length exceeds reference genome length
        """
        if self.start_position + self.length > len(self.reference_genome):
            raise ValueError(
                f"Requested sequence (start={self.start_position}, length={self.length}) "
                f"exceeds reference genome length ({len(self.reference_genome)})"
            )

        return self.reference_genome[self.start_position : self.start_position + self.length]

    def _generate_flow_matrix(self) -> np.ndarray:
        """
        Generate the probability flow matrix with normal distributions.

        Creates an (max_hmer_size + 1) × n_flows matrix where each column represents
        the probability distribution for a flow, with probabilities following a
        normal distribution centered on the true hmer call.

        Returns
        -------
        np.ndarray
            Flow matrix of shape (max_hmer_size + 1, n_flows)
        """
        n_flows = len(self.flow_key)
        flow_matrix = np.zeros((self.max_hmer_size + 1, n_flows))

        # For each flow, create a normal distribution around the true hmer call
        for flow_idx, true_hmer in enumerate(self.flow_key):
            # Clip true_hmer to max_hmer_size to avoid index errors
            true_hmer_clipped = min(true_hmer, self.max_hmer_size)

            # Create probability distribution for all possible hmer values
            hmer_values = np.arange(self.max_hmer_size + 1)

            # Generate probabilities using normal distribution
            # Center on true hmer call with specified standard deviation
            probabilities = stats.norm.pdf(hmer_values, loc=true_hmer_clipped, scale=self.noise_std)

            # Ensure minimum probability for numerical stability
            min_prob = 1e-6
            probabilities = np.maximum(probabilities, min_prob)

            # Normalize to sum to 1
            probabilities = probabilities / np.sum(probabilities)

            # Store in flow matrix
            flow_matrix[:, flow_idx] = probabilities

        return flow_matrix

    def simulate_read(self, read_name: str | None = None) -> FlowBasedRead:
        """
        Generate a simulated FlowBasedRead object.

        Parameters
        ----------
        read_name : str, optional
            Name for the simulated read. If None, generates a default name.

        Returns
        -------
        FlowBasedRead
            Simulated flow-based read with probability matrix
        """
        if read_name is None:
            read_name = f"sim_read_{self.start_position}_{self.length}"

        # Create FlowBasedRead using the from_tuple constructor
        flow_read = FlowBasedRead.from_tuple(
            read_name=read_name, read=self.target_sequence, flow_order=self.flow_cycle, max_hmer_size=self.max_hmer_size
        )

        # Set the simulated flow matrix
        flow_read._flow_matrix = self.flow_matrix

        return flow_read

    def add_sequencing_errors(self, error_rate: float = 0.01) -> None:
        """
        Add realistic sequencing errors to the flow matrix.

        This method modifies the existing flow matrix to simulate sequencing
        errors by redistributing some probability mass to neighboring hmer calls.

        Parameters
        ----------
        error_rate : float, optional
            Rate of sequencing errors to introduce, by default 0.01
        """
        if not (0 <= error_rate <= 1):
            raise ValueError("Error rate must be between 0 and 1")

        # For each flow, redistribute some probability to neighboring hmers
        for flow_idx in range(self.flow_matrix.shape[1]):
            true_hmer = min(self.flow_key[flow_idx], self.max_hmer_size)

            # Current probabilities
            probs = self.flow_matrix[:, flow_idx].copy()

            # Amount of probability to redistribute
            error_prob = probs[true_hmer] * error_rate

            # Reduce probability of true call
            probs[true_hmer] -= error_prob

            # Redistribute to neighboring positions
            neighbors = []
            if true_hmer > 0:
                neighbors.append(true_hmer - 1)
            if true_hmer < self.max_hmer_size:
                neighbors.append(true_hmer + 1)

            if neighbors:
                error_per_neighbor = error_prob / len(neighbors)
                for neighbor in neighbors:
                    probs[neighbor] += error_per_neighbor

            # Ensure probabilities sum to 1 (handle numerical precision)
            probs = probs / np.sum(probs)

            # Update flow matrix
            self.flow_matrix[:, flow_idx] = probs

    def get_flow_statistics(self) -> dict:
        """
        Get statistics about the generated flow matrix.

        Returns
        -------
        dict
            Dictionary containing flow matrix statistics
        """
        return {
            "n_flows": self.flow_matrix.shape[1],
            "max_hmer_size": self.max_hmer_size,
            "target_sequence_length": len(self.target_sequence),
            "flow_key_length": len(self.flow_key),
            "mean_hmer_call": np.mean(self.flow_key),
            "max_hmer_call": np.max(self.flow_key),
            "flow_cycle": self.flow_cycle,
            "noise_std": self.noise_std,
        }

    def get_most_probable_calls(self) -> np.ndarray:
        """
        Get the most probable hmer call for each flow.

        Returns
        -------
        np.ndarray
            Array of most probable hmer calls for each flow
        """
        return np.argmax(self.flow_matrix, axis=0)

    def get_call_probabilities(self) -> np.ndarray:
        """
        Get the probability of the most likely call for each flow.

        Returns
        -------
        np.ndarray
            Array of probabilities for the most likely call in each flow
        """
        return np.max(self.flow_matrix, axis=0)


def simulate_reads_from_region(
    reference_genome: str,
    start_position: int,
    end_position: int,
    read_length: int = 150,
    overlap: int = 50,
    flow_cycle: str = DEFAULT_FLOW_ORDER,
    noise_std: float = 0.01,
    max_hmer_size: int = 12,
) -> list[FlowBasedRead]:
    """
    Generate multiple overlapping simulated reads from a genomic region.

    Parameters
    ----------
    reference_genome : str
        Reference genome sequence
    start_position : int
        Start position of the region
    end_position : int
        End position of the region
    read_length : int, optional
        Length of each simulated read, by default 150
    overlap : int, optional
        Overlap between consecutive reads, by default 50
    flow_cycle : str, optional
        Flow cycle to use, by default DEFAULT_FLOW_ORDER
    noise_std : float, optional
        Standard deviation for probability distributions, by default 0.01
    max_hmer_size : int, optional
        Maximum hmer size, by default 12

    Returns
    -------
    list[FlowBasedRead]
        List of simulated FlowBasedRead objects covering the region
    """
    reads = []
    step_size = read_length - overlap

    current_pos = start_position
    read_idx = 0

    while current_pos + read_length <= end_position:
        simulator = FlowBasedReadSimulator(
            reference_genome=reference_genome,
            start_position=current_pos,
            length=read_length,
            flow_cycle=flow_cycle,
            noise_std=noise_std,
            max_hmer_size=max_hmer_size,
        )

        read_name = f"sim_read_{read_idx}_{current_pos}_{current_pos + read_length}"
        simulated_read = simulator.simulate_read(read_name)
        reads.append(simulated_read)

        current_pos += step_size
        read_idx += 1

    return reads
