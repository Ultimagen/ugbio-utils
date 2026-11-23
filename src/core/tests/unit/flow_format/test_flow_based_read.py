import pickle
import tempfile
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
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(x, flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=12) for x in data
    ]
    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert np.allclose(expected[i], rec._flow_matrix)


def test_matrix_from_qual_tp_no_trim(resources_dir):
    data = list(pysam.AlignmentFile(pjoin(resources_dir, "chr9.sample.bam")))
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
    data = list(pysam.AlignmentFile(pjoin(resources_dir, "trimmed_read.bam")))
    flow_based_read = fbr.FlowBasedRead.from_sam_record(
        data[0], flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=20, spread_edge_probs=True
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


def test_generate_key_from_sequence_terminated(flow_order):
    sequence = "AAGGTTCC"
    result = fbr.generate_key_from_sequence(sequence, flow_order, terminated=True)

    assert set(np.unique(result)).issubset({0, 1})
    assert result.sum() == len(sequence)

    expected = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    assert np.array_equal(result, expected)


def test_generate_key_from_sequence_terminated_and_truncate_raises(flow_order):
    sequence = "AAGGTTCC"
    with pytest.raises(ValueError):
        fbr.generate_key_from_sequence(sequence, flow_order, truncate=1, terminated=True)


def test_get_flow_matrix_column_for_base(resources_dir):
    data = list(pysam.AlignmentFile(pjoin(resources_dir, "chr9.sample.bam")))
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(x, flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=12) for x in data
    ]
    for rec in fbrs:
        for i in range(len(str(rec.record.query_sequence))):
            assert rec.get_flow_matrix_column_for_base(i)[0] == str(rec.record.query_sequence)[i]
            if i > 12 and i < len(str(rec.record.query_sequence)) - 12:
                assert (
                    rec.key[rec.base_to_flow_mapping[i]] == np.argmax(rec.get_flow_matrix_column_for_base(i)[1])
                ) or np.max(rec.get_flow_matrix_column_for_base(i)[1]) < 0.9


def _tp_record(qual: str, tp_tag: str | None, mq: str | None = "D", old_tp: list | None = None):
    rg = "@RG\tID:A" + (f"\tmq:{mq}" if mq else "")
    n = len(qual)
    fields = ["r1", "0", "chr1", "1", "60", f"{n}M", "*", "0", "0", "A" * n, qual, "RG:Z:A"]
    if tp_tag is not None:
        fields.append(f"tp:Z:{tp_tag}")
    if old_tp is not None:
        fields.append("tp:B:c," + ",".join(str(x) for x in old_tp))
    sam = f"@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000\n{rg}\n" + "\t".join(fields) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".sam") as f:
        f.write(sam)
        f.flush()
        return next(iter(pysam.AlignmentFile(f.name, "r")))


def test_decode_tp_old_bc_format():
    rec = _tp_record("DDD:", None, mq=None, old_tp=[0, 1, -1, 1])
    assert fbr.decode_tp_tag(rec).tolist() == [0, 1, -1, 1]


def test_decode_tp_sparse_z():
    rec = _tp_record("DDDD:::*:D:D::", "+--+++++")
    assert fbr.decode_tp_tag(rec).tolist() == [0, 0, 0, 0, 1, -1, -1, 1, 1, 0, 1, 0, 1, 1]


def test_decode_tp_absent_is_zeros():
    rec = _tp_record("DDDD", None)
    assert fbr.decode_tp_tag(rec).tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("tp_tag", ["+-", "+-+++", "+x+"])
def test_decode_tp_length_mismatch_or_bad_char(tp_tag):
    rec = _tp_record("DD:::", tp_tag)
    with pytest.raises(ValueError):
        fbr.decode_tp_tag(rec)


def test_decode_tp_no_mq_raises():
    rec = _tp_record("DD:::", "+-+", mq=None)
    with pytest.raises(ValueError):
        fbr.decode_tp_tag(rec)
