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
    return Path(__file__).parent.parent / "resources"

def test_matrix_from_qual_tp(resources_dir):
    data = [x for x in pysam.AlignmentFile(pjoin(resources_dir, "chr9.sample.bam"))]
    expected = pickle.load(open(pjoin(resources_dir, "matrices.trim.pkl"), "rb"))
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(x, flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=12) for x in data
    ]
    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert np.allclose(expected[i], rec._flow_matrix)


def test_matrix_from_qual_tp_no_trim(resources_dir):
    data = [x for x in pysam.AlignmentFile(pjoin(resources_dir, "chr9.sample.bam"))]
    expected = pickle.load(open(pjoin(resources_dir, "matrices.pkl"), "rb"))
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(
            x, flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=12, spread_edge_probs=False
        )
        for x in data
    ]

    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert np.allclose(expected[i], rec._flow_matrix)


# test that spread probabilities on the first and last non-zero flow produces flat probabilities
# Since we read hmers 0-20 we expect P(0)=...P(20) = 1/21
def test_matrix_from_trimmed_read(resources_dir):
    data = [x for x in pysam.AlignmentFile(pjoin(resources_dir, "trimmed_read.bam"))]
    flow_based_read = fbr.FlowBasedRead.from_sam_record(
        data[0], flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=20, spread_edge_probs=True
    )

    np.testing.assert_array_almost_equal(flow_based_read._flow_matrix[:, 2], np.ones(21) / 21, 0.0001)
    np.testing.assert_array_almost_equal(flow_based_read._flow_matrix[:, -1], np.ones(21) / 21, 0.0001)
