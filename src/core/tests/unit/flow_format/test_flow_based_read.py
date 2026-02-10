import pickle
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pysam
import pytest
import ugbio_core.flow_format.flow_based_read as fbr
from ugbio_core.consts import DEFAULT_FLOW_ORDER


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent.parent / "resources"


def test_matrix_from_qual_tp(resources_dir):
    data = list(pysam.AlignmentFile(pjoin(resources_dir, "chr9.sample.bam")))
    expected = pickle.load(open(pjoin(resources_dir, "matrices.trim.pkl"), "rb"))
    fbrs = [fbr.FlowBasedRead.from_sam_record(x, flow_order=DEFAULT_FLOW_ORDER, max_hmer_size=12) for x in data]
    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert np.allclose(expected[i], rec._flow_matrix)


def test_matrix_from_qual_tp_no_trim(resources_dir):
    data = list(pysam.AlignmentFile(pjoin(resources_dir, "chr9.sample.bam")))
    expected = pickle.load(open(pjoin(resources_dir, "matrices.pkl"), "rb"))
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(x, flow_order=DEFAULT_FLOW_ORDER, max_hmer_size=12, spread_edge_probs=False)
        for x in data
    ]

    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert np.allclose(expected[i], rec._flow_matrix)


# test that spread probabilities on the first and last non-zero flow produces flat probabilities
# Since we read hmers 0-20 we expect P(0)=...P(20) = 1/21
def test_matrix_from_trimmed_read(resources_dir):
    data = list(pysam.AlignmentFile(pjoin(resources_dir, "trimmed_read.bam")))
    flow_based_read = fbr.FlowBasedRead.from_sam_record(
        data[0], flow_order=DEFAULT_FLOW_ORDER, max_hmer_size=20, spread_edge_probs=True
    )

    np.testing.assert_array_almost_equal(flow_based_read._flow_matrix[:, 2], np.ones(21) / 21, 0.0001)
    np.testing.assert_array_almost_equal(flow_based_read._flow_matrix[:, -1], np.ones(21) / 21, 0.0001)


@pytest.fixture
def flow_order():
    return "ACGT"


def test_generate_key_from_sequence_standard_nucleotides(flow_order):
    sequence = "AAGGTTCC"
    result = fbr.generate_key_from_sequence(sequence, flow_order)
    expected = np.array([2, 0, 2, 2, 0, 2])
    assert np.array_equal(result, expected)


def test_generate_key_from_sequence_non_standard_nucleotides(flow_order):
    sequence = "AAGGTTCCNN"
    result = fbr.generate_key_from_sequence(sequence, flow_order, non_standard_as_a=True)
    expected = np.array([2, 0, 2, 2, 0, 2, 0, 0, 2])
    assert np.array_equal(result, expected)


def test_generate_key_from_sequence_non_standard_nucleotides_exception(flow_order):
    sequence = "AAGGTTCCNN"
    with pytest.raises(ValueError):
        fbr.generate_key_from_sequence(sequence, flow_order, non_standard_as_a=False)


def test_generate_key_from_sequence_truncate(flow_order):
    sequence = "AAGGTTCC"

    result = fbr.generate_key_from_sequence(sequence, flow_order, truncate=1)
    expected = np.array([1, 0, 1, 1, 0, 1])

    assert np.array_equal(result, expected)


def test_generate_key_from_sequence_empty_sequence(flow_order):
    sequence = ""
    result = fbr.generate_key_from_sequence(sequence, flow_order)
    expected = np.array([])
    assert np.array_equal(result, expected)


def test_get_flow_matrix_column_for_base(resources_dir):
    data = list(pysam.AlignmentFile(pjoin(resources_dir, "chr9.sample.bam")))
    fbrs = [fbr.FlowBasedRead.from_sam_record(x, flow_order=DEFAULT_FLOW_ORDER, max_hmer_size=12) for x in data]
    for rec in fbrs:
        for i in range(len(str(rec.record.query_sequence))):
            assert rec.get_flow_matrix_column_for_base(i)[0] == str(rec.record.query_sequence)[i]
            if i > 12 and i < len(str(rec.record.query_sequence)) - 12:
                assert (
                    rec.key[rec.base_to_flow_mapping[i]] == np.argmax(rec.get_flow_matrix_column_for_base(i)[1])
                ) or np.max(rec.get_flow_matrix_column_for_base(i)[1]) < 0.9


def test_from_tuple_basic_functionality():
    """Test basic functionality of FlowBasedRead.from_tuple method."""
    # Test data
    read_name = "test_read_001"
    sequence = "ATCGATCGATCG"
    flow_order = DEFAULT_FLOW_ORDER
    max_hmer_size = 12

    # Create FlowBasedRead using from_tuple
    flow_read = fbr.FlowBasedRead.from_tuple(
        read_name=read_name, read=sequence, flow_order=flow_order, max_hmer_size=max_hmer_size
    )

    # Test basic attributes
    assert flow_read.read_name == read_name
    assert flow_read.seq == sequence.upper()
    assert flow_read.is_reverse is False
    assert flow_read.forward_seq == sequence.upper()
    assert flow_read._max_hmer == max_hmer_size
    assert flow_read.flow_order.startswith(flow_order)  # flow_order gets expanded to match key length
    assert flow_read.direction == "synthesis"

    # Test that key is generated correctly
    expected_key = fbr.generate_key_from_sequence(sequence.upper(), flow_order=flow_order)
    assert np.array_equal(flow_read.key, expected_key)

    # Test that flow2base is computed
    assert hasattr(flow_read, "flow2base")
    assert len(flow_read.flow2base) == len(flow_read.key)

    # Test that the read is valid
    assert flow_read.is_valid()

    # Test with different parameters
    custom_flow_order = "ACGT"
    custom_max_hmer = 8

    flow_read_custom = fbr.FlowBasedRead.from_tuple(
        read_name="custom_read",
        read="aaggttcc",  # lowercase to test uppercasing
        flow_order=custom_flow_order,
        max_hmer_size=custom_max_hmer,
    )

    assert flow_read_custom.seq == "AAGGTTCC"
    assert flow_read_custom._max_hmer == custom_max_hmer
    assert flow_read_custom.flow_order.startswith(custom_flow_order)  # flow_order gets expanded to match key length
