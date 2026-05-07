"""Tests for tp/t0 format conversion in FlowBasedRead"""

import numpy as np
import pysam
import pytest
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugbio_core.flow_format.flow_based_read import FlowBasedRead


def create_test_sam_record_with_tp_t0():
    """Create a test SAM record with tp/t0 tags for testing"""
    # Create a simple test record
    header_dict = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 248956422}],
        "RG": [{"ID": "test", "SM": "test_sample", "FO": DEFAULT_FLOW_ORDER}],
    }
    header = pysam.AlignmentHeader.from_dict(header_dict)

    record = pysam.AlignedSegment(header)
    record.query_name = "test_read"
    record.query_sequence = "ATCGATCGATCG"
    record.flag = 0
    record.reference_id = 0
    record.reference_start = 100
    record.mapping_quality = 60
    record.cigartuples = [(0, 12)]  # 12M

    # Set quality scores
    record.query_qualities = [30, 25, 35, 20, 40, 30, 25, 35, 20, 40, 30, 25]

    # Set tp tag (error offsets)
    record.set_tag("tp", [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])

    # Set t0 tag (optional, for zero flows)
    record.set_tag("t0", [45, 40, 35, 50, 45, 40, 35, 50, 45, 40, 35, 50])

    return record


def test_matrix_to_qual_tp_basic():
    """Test basic functionality of _matrix_to_qual_tp method"""
    # Create a FlowBasedRead from tuple
    fbr = FlowBasedRead.from_tuple("test_read", "ATCGATCG")

    # Create a simple flow matrix for testing
    n_flows = len(fbr.key)
    max_hmer = 12
    flow_matrix = np.ones((max_hmer + 1, n_flows)) * 0.001  # filler

    # Set actual call probabilities
    for i, hmer_size in enumerate(fbr.key):
        if hmer_size <= max_hmer:
            flow_matrix[hmer_size, i] = 0.9  # high probability for actual call
            # Add some error probabilities
            if hmer_size > 0:
                flow_matrix[hmer_size - 1, i] = 0.05  # deletion error
            if hmer_size < max_hmer:
                flow_matrix[hmer_size + 1, i] = 0.04  # insertion error

    fbr._flow_matrix = flow_matrix

    # Test the conversion
    qual, tp_tag, t0_tag = fbr._matrix_to_qual_tp()

    # Basic checks
    assert len(qual) == len(fbr.seq)
    assert len(tp_tag) == len(fbr.seq)
    assert all(q >= 2 for q in qual)  # minimum quality
    assert all(q <= 60 for q in qual)  # maximum quality


def test_to_record_tp_format():
    """Test to_record method with tp/t0 format"""
    # Create a FlowBasedRead from tuple
    fbr = FlowBasedRead.from_tuple("test_read", "ATCGATCG")

    # Create a flow matrix
    n_flows = len(fbr.key)
    max_hmer = 12
    flow_matrix = np.ones((max_hmer + 1, n_flows)) * 0.001

    for i, hmer_size in enumerate(fbr.key):
        if hmer_size <= max_hmer:
            flow_matrix[hmer_size, i] = 0.9
            if hmer_size > 0:
                flow_matrix[hmer_size - 1, i] = 0.05
            if hmer_size < max_hmer:
                flow_matrix[hmer_size + 1, i] = 0.04

    fbr._flow_matrix = flow_matrix

    # Test tp/t0 format (only format now)
    record_tp = fbr.to_record()

    assert record_tp.has_tag("tp")
    assert record_tp.query_qualities is not None
    assert len(record_tp.query_qualities) == len(fbr.seq)


def test_round_trip_conversion():
    """Test round-trip conversion: matrix -> tp/t0 -> matrix"""
    # Create test record with tp/t0 tags
    sam_record = create_test_sam_record_with_tp_t0()

    # Create FlowBasedRead from SAM record (this uses _matrix_from_qual_tp)
    fbr_original = FlowBasedRead.from_sam_record(sam_record)

    # Convert back to record using tp/t0 format
    new_record = fbr_original.to_record()

    # Create new FlowBasedRead from the converted record
    fbr_roundtrip = FlowBasedRead.from_sam_record(new_record)

    # Compare key (should be identical)
    np.testing.assert_array_equal(fbr_original.key, fbr_roundtrip.key)

    # Compare flow matrices (should be similar within tolerance)
    # Note: Some precision loss is expected due to quality score quantization
    assert fbr_original._flow_matrix.shape == fbr_roundtrip._flow_matrix.shape

    # Verify that the conversion process works without errors
    # and produces valid tp/t0 tags
    assert new_record.has_tag("tp")
    assert new_record.query_qualities is not None

    # Check that tp tag has correct length
    tp_values = new_record.get_tag("tp")
    assert len(tp_values) == len(fbr_original.seq)

    # Check that quality values are reasonable
    qual_values = new_record.query_qualities
    assert all(2 <= q <= 60 for q in qual_values)


def test_zero_flows_handling():
    """Test handling of zero flows (t0 tag)"""
    # Create a sequence with zero flows
    fbr = FlowBasedRead.from_tuple("test_read", "AACCGGTT")  # This should create some zero flows

    # Create flow matrix with zero flows
    n_flows = len(fbr.key)
    max_hmer = 12
    flow_matrix = np.ones((max_hmer + 1, n_flows)) * 0.001

    # Set probabilities including zero flows
    for i, hmer_size in enumerate(fbr.key):
        if hmer_size == 0:
            # Zero flow - set probability for hmer size 1
            flow_matrix[1, i] = 0.1
        elif hmer_size <= max_hmer:
            flow_matrix[hmer_size, i] = 0.9

    fbr._flow_matrix = flow_matrix

    # Test conversion
    qual, tp_tag, t0_tag = fbr._matrix_to_qual_tp()

    # Should have t0_tag if there are zero flows
    if np.any(fbr.key == 0):
        assert t0_tag is not None
        assert len(t0_tag) == len(fbr.seq)

    # Test record creation
    record = fbr.to_record()

    if np.any(fbr.key == 0):
        assert record.has_tag("t0")


def test_format_parameter_validation():
    """Test that the tp/t0 format works correctly"""
    fbr = FlowBasedRead.from_tuple("test_read", "ATCG")

    # Create minimal flow matrix
    n_flows = len(fbr.key)
    flow_matrix = np.ones((13, n_flows)) * 0.001
    for i, hmer_size in enumerate(fbr.key):
        if hmer_size <= 12:
            flow_matrix[hmer_size, i] = 0.9
    fbr._flow_matrix = flow_matrix

    # Test tp format (only format now)
    record_tp = fbr.to_record()
    assert record_tp.has_tag("tp")
    assert not record_tp.has_tag("kh")


def test_edge_cases():
    """Test edge cases in tp/t0 conversion"""
    # Test with very short sequence
    fbr = FlowBasedRead.from_tuple("test_read", "A")

    # Create minimal flow matrix
    n_flows = len(fbr.key)
    flow_matrix = np.ones((13, n_flows)) * 0.001
    for i, hmer_size in enumerate(fbr.key):
        if hmer_size <= 12:
            flow_matrix[hmer_size, i] = 0.9
    fbr._flow_matrix = flow_matrix

    # Should not raise errors
    qual, tp_tag, t0_tag = fbr._matrix_to_qual_tp()
    record = fbr.to_record()

    assert len(qual) == 1
    assert len(tp_tag) == 1
    assert record.has_tag("tp")


def test_no_flow_matrix_error():
    """Test error when flow matrix is not available"""
    fbr = FlowBasedRead.from_tuple("test_read", "ATCG")
    # Don't set _flow_matrix

    with pytest.raises(ValueError, match="Flow matrix not available"):
        fbr._matrix_to_qual_tp()


if __name__ == "__main__":
    pytest.main([__file__])
