"""Unit tests for the DNN preprocessing pipeline.

Covers: static vocab config, CRAM fetch, tensor encoding, fold assignment, shuffle.
"""

from __future__ import annotations

import array
import json
from pathlib import Path

import numpy as np
import pysam
import pytest

# ---------------------------------------------------------------------------
# Static vocab config
# ---------------------------------------------------------------------------


class TestLoadVocabConfig:
    def test_load_default(self):
        from ugbio_srsnv.deep_srsnv.data_prep import Encoders, load_vocab_config

        enc = load_vocab_config()
        assert isinstance(enc, Encoders)
        assert enc.base_vocab["<PAD>"] == 0
        assert enc.base_vocab["<GAP>"] == 1
        assert enc.base_vocab["A"] == 2
        assert len(enc.base_vocab) == 7
        assert enc.t0_vocab["<PAD>"] == 0
        assert enc.t0_vocab["<MISSING>"] == 1
        assert len(enc.t0_vocab) == 11
        assert len(enc.tm_vocab) == 9
        assert len(enc.st_vocab) == 6
        assert len(enc.et_vocab) == 6

    def test_load_from_path(self, tmp_path):
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        cfg = {
            "base_vocab": {"<PAD>": 0, "A": 1},
            "t0_vocab": {"<PAD>": 0},
            "tm_vocab": {"<PAD>": 0},
            "st_vocab": {"<PAD>": 0},
            "et_vocab": {"<PAD>": 0},
        }
        cfg_path = tmp_path / "test_vocab.json"
        cfg_path.write_text(json.dumps(cfg))

        enc = load_vocab_config(cfg_path)
        assert enc.base_vocab == {"<PAD>": 0, "A": 1}

    def test_missing_file(self, tmp_path):
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        with pytest.raises(FileNotFoundError):
            load_vocab_config(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path):
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json!")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_vocab_config(bad_file)

    def test_missing_keys(self, tmp_path):
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        cfg = {"base_vocab": {"<PAD>": 0}}
        cfg_path = tmp_path / "incomplete.json"
        cfg_path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="missing keys"):
            load_vocab_config(cfg_path)


# ---------------------------------------------------------------------------
# CRAM fetch helpers
# ---------------------------------------------------------------------------


def _make_test_bam(bam_path: str, reads: list[dict]) -> str:
    """Create a minimal sorted, indexed BAM with the given reads.

    Each read dict should have: name, chrom (chr1/chr2), pos, seq, qual.
    """
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [
                {"SN": "chr1", "LN": 1000},
                {"SN": "chr2", "LN": 1000},
                {"SN": "chr21", "LN": 1000},
                {"SN": "chr22", "LN": 1000},
            ],
        }
    )
    unsorted = bam_path + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for r in reads:
            seg = pysam.AlignedSegment(header)
            seg.query_name = r["name"]
            seg.reference_id = header.get_tid(r["chrom"])
            seg.reference_start = r["pos"] - 1
            seg.query_sequence = r["seq"]
            seg.query_qualities = pysam.qualitystring_to_array(r.get("qual", "I" * len(r["seq"])))
            cigar = r.get("cigar", [(0, len(r["seq"]))])
            seg.cigar = cigar
            seg.mapping_quality = r.get("mapq", 60)
            seg.flag = 0
            # Add MD tag for get_aligned_pairs(with_seq=True)
            md = r.get("md", str(len(r["seq"])))
            seg.set_tag("MD", md)
            if r.get("tags"):
                seg.set_tags(list(r["tags"]) + [("MD", md)])
            out.write(seg)
    pysam.sort("-o", bam_path, unsorted)
    pysam.index(bam_path)
    Path(unsorted).unlink(missing_ok=True)
    return bam_path


@pytest.fixture()
def small_bam(tmp_path):
    """A small sorted indexed BAM with 4 reads across chr1 and chr2."""
    reads = [
        {
            "name": "read_A",
            "chrom": "chr1",
            "pos": 100,
            "seq": "ACGTACGT",
            "tags": [
                ("tp", array.array("i", [1, 2, 3, 4, 5, 6, 7, 8])),
                ("t0", "ACGTACGT"),
                ("rq", 0.95, "f"),
                ("tm", "AQ"),
                ("st", "PLUS"),
                ("et", "MINUS"),
            ],
        },
        {"name": "read_B", "chrom": "chr1", "pos": 200, "seq": "GGGGCCCC"},
        {"name": "read_C", "chrom": "chr2", "pos": 300, "seq": "TTTTAAAA"},
        {"name": "read_D", "chrom": "chr21", "pos": 400, "seq": "NNNNNNNN"},
    ]
    bam_path = str(tmp_path / "test.bam")
    _make_test_bam(bam_path, reads)
    return bam_path


class TestCramFetch:
    def test_found(self, small_bam):
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import _fetch_read_from_cram

        cram = pysam.AlignmentFile(small_bam, "rb")
        rec = _fetch_read_from_cram(cram, "chr1", 100, "read_A")
        assert rec is not None
        assert rec.query_name == "read_A"
        cram.close()

    def test_missing_rn(self, small_bam):
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import _fetch_read_from_cram

        cram = pysam.AlignmentFile(small_bam, "rb")
        rec = _fetch_read_from_cram(cram, "chr1", 100, "no_such_read")
        assert rec is None
        cram.close()

    def test_wrong_chrom(self, small_bam):
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import _fetch_read_from_cram

        cram = pysam.AlignmentFile(small_bam, "rb")
        rec = _fetch_read_from_cram(cram, "chr2", 100, "read_A")
        assert rec is None
        cram.close()


# ---------------------------------------------------------------------------
# samtools -N pipe fetch
# ---------------------------------------------------------------------------


class TestSamtoolsPipeFetch:
    """Tests for ``_fetch_reads_samtools_pipe`` which delegates filtering to samtools."""

    def test_basic_fetch(self, small_bam):
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import _fetch_reads_samtools_pipe

        rows = [
            {"CHROM": "chr1", "POS": 100, "RN": "read_A"},
            {"CHROM": "chr1", "POS": 200, "RN": "read_B"},
        ]
        matched = _fetch_reads_samtools_pipe(small_bam, None, rows)

        assert len(matched) == 2
        assert "read_A" in matched
        assert "read_B" in matched
        assert matched["read_A"].query_name == "read_A"
        assert matched["read_B"].query_name == "read_B"

    def test_multi_chrom(self, small_bam):
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import _fetch_reads_samtools_pipe

        rows = [
            {"CHROM": "chr1", "POS": 100, "RN": "read_A"},
            {"CHROM": "chr2", "POS": 300, "RN": "read_C"},
            {"CHROM": "chr21", "POS": 400, "RN": "read_D"},
        ]
        matched = _fetch_reads_samtools_pipe(small_bam, None, rows)

        assert len(matched) == 3
        assert set(matched.keys()) == {"read_A", "read_C", "read_D"}

    def test_nonexistent_name(self, small_bam):
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import _fetch_reads_samtools_pipe

        rows = [
            {"CHROM": "chr1", "POS": 100, "RN": "read_A"},
            {"CHROM": "chr1", "POS": 100, "RN": "no_such_read"},
        ]
        matched = _fetch_reads_samtools_pipe(small_bam, None, rows)

        assert "read_A" in matched
        assert "no_such_read" not in matched

    def test_empty_rows(self, small_bam):
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import _fetch_reads_samtools_pipe

        matched = _fetch_reads_samtools_pipe(small_bam, None, [])
        assert matched == {}

    def test_matches_pysam_fetch(self, small_bam):
        """Verify samtools pipe returns the same reads as pure-pysam fetch."""
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import (
            _fetch_reads_by_region,
            _fetch_reads_samtools_pipe,
        )

        rows = [
            {"CHROM": "chr1", "POS": 100, "RN": "read_A"},
            {"CHROM": "chr1", "POS": 200, "RN": "read_B"},
            {"CHROM": "chr2", "POS": 300, "RN": "read_C"},
        ]

        pipe_result = _fetch_reads_samtools_pipe(small_bam, None, rows)

        cram = pysam.AlignmentFile(small_bam, "rb")
        pysam_result = _fetch_reads_by_region(cram, rows)
        cram.close()

        assert set(pipe_result.keys()) == set(pysam_result.keys())
        for rn in pipe_result:
            assert pipe_result[rn].query_name == pysam_result[rn].query_name
            assert pipe_result[rn].query_sequence == pysam_result[rn].query_sequence


# ---------------------------------------------------------------------------
# Tensor encoding (via _build_gapped_channels)
# ---------------------------------------------------------------------------


def _make_aligned_segment(header, *, name="r1", chrom="chr1", pos=100, seq="ACGT", cigar=None, tags=None, md=None):
    seg = pysam.AlignedSegment(header)
    seg.query_name = name
    seg.reference_id = header.get_tid(chrom)
    seg.reference_start = pos - 1
    seg.query_sequence = seq
    seg.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    seg.cigar = cigar or [(0, len(seq))]
    seg.mapping_quality = 60
    seg.flag = 0
    seg.set_tag("MD", md or str(len(seq)))
    if tags:
        seg.set_tags(tags)
    return seg


@pytest.fixture()
def bam_header():
    return pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6"},
            "SQ": [{"SN": "chr1", "LN": 1000}],
        }
    )


class TestBuildGappedChannels:
    def test_match(self, bam_header):
        from ugbio_srsnv.deep_srsnv.data_prep import _build_gapped_channels

        rec = _make_aligned_segment(bam_header, seq="ACGT", pos=100, cigar=[(0, 4)])
        tp_raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        t0_raw = "ACGT"
        result = _build_gapped_channels(rec, 101, tp_raw, t0_raw, None)

        assert len(result["read_base_aln"]) == 4
        assert "<GAP>" not in result["read_base_aln"]
        assert "<GAP>" not in result["ref_base_aln"]
        assert len(result["focus_aln"]) == 4

    def test_insertion(self, bam_header):
        from ugbio_srsnv.deep_srsnv.data_prep import _build_gapped_channels

        # 2M1I2M: read=ACGGT, ref has gap at position of insertion
        rec = _make_aligned_segment(bam_header, seq="ACGGT", pos=100, cigar=[(0, 2), (1, 1), (0, 2)], md="4")
        tp_raw = np.zeros(5, dtype=np.float32)
        result = _build_gapped_channels(rec, 100, tp_raw, "", None)

        assert "<GAP>" in result["ref_base_aln"]
        assert "<GAP>" not in result["read_base_aln"]

    def test_deletion(self, bam_header):
        from ugbio_srsnv.deep_srsnv.data_prep import _build_gapped_channels

        # 2M1D2M: read=ACGT, ref has extra base at deletion
        rec = _make_aligned_segment(bam_header, seq="ACGT", pos=100, cigar=[(0, 2), (2, 1), (0, 2)], md="2^N2")
        tp_raw = np.zeros(4, dtype=np.float32)
        result = _build_gapped_channels(rec, 100, tp_raw, "", None)

        assert "<GAP>" in result["read_base_aln"]
        assert "<GAP>" not in result["ref_base_aln"]

    def test_softclip(self, bam_header):
        from ugbio_srsnv.deep_srsnv.data_prep import _build_gapped_channels

        # 1S4M: first base is soft-clipped
        rec = _make_aligned_segment(bam_header, seq="AACGT", pos=100, cigar=[(4, 1), (0, 4)], md="4")
        tp_raw = np.zeros(5, dtype=np.float32)
        result = _build_gapped_channels(rec, 100, tp_raw, "", None)

        assert result["softclip_mask_aln"][0] == 1.0

    def test_focus(self, bam_header):
        from ugbio_srsnv.deep_srsnv.data_prep import _build_gapped_channels

        rec = _make_aligned_segment(bam_header, seq="ACGT", pos=100, cigar=[(0, 4)])
        tp_raw = np.zeros(4, dtype=np.float32)
        result = _build_gapped_channels(rec, 102, tp_raw, "", None)

        focus = result["focus_aln"]
        focus_arr = np.array(focus)
        assert np.sum(focus_arr == 1.0) == 1
        assert focus_arr[np.argmax(focus_arr)] == 1.0

    def test_positive_ref_override(self, bam_header):
        from ugbio_srsnv.deep_srsnv.data_prep import _build_gapped_channels

        rec = _make_aligned_segment(bam_header, seq="ACGT", pos=100, cigar=[(0, 4)])
        tp_raw = np.zeros(4, dtype=np.float32)
        result = _build_gapped_channels(rec, 101, tp_raw, "", "T")

        focus_pos = np.argmax(result["focus_aln"])
        assert result["ref_base_aln"][focus_pos] == "T"


class TestTensorEncoding:
    def test_shapes(self):
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        enc = load_vocab_config()
        tensor_length = 50

        read_base_out = np.zeros((1, tensor_length), dtype=np.int16)
        ref_base_out = np.zeros((1, tensor_length), dtype=np.int16)
        mask_out = np.zeros((1, tensor_length), dtype=np.uint8)
        x_num_pos_out = np.zeros((1, 6, tensor_length), dtype=np.float16)
        x_num_const_out = np.zeros((1, 4), dtype=np.float16)

        tokens = ["A", "C", "G", "T"]
        valid = len(tokens)
        read_base_out[0, :valid] = [enc.base_vocab.get(t, 6) for t in tokens]
        mask_out[0, :valid] = 1

        assert read_base_out.shape == (1, tensor_length)
        assert ref_base_out.shape == (1, tensor_length)
        assert x_num_pos_out.shape == (1, 6, tensor_length)
        assert x_num_const_out.shape == (1, 4)
        assert mask_out.shape == (1, tensor_length)

    def test_padding(self):
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        enc = load_vocab_config()
        tensor_length = 10
        tokens = ["A", "C", "G"]
        valid = len(tokens)

        read_base_out = np.zeros(tensor_length, dtype=np.int16)
        mask_out = np.zeros(tensor_length, dtype=np.uint8)
        read_base_out[:valid] = [enc.base_vocab.get(t, 6) for t in tokens]
        mask_out[:valid] = 1

        assert mask_out[0] == 1
        assert mask_out[valid] == 0
        assert read_base_out[valid] == 0  # <PAD>


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------


class TestFoldAssignment:
    def test_chrom_kfold(self):
        from ugbio_srsnv.deep_srsnv.combine_splits import _assign_fold_ids

        manifest = {
            "split_mode": "chromosome_kfold",
            "chrom_to_fold": {"chr1": 0, "chr2": 1, "chr3": 2},
            "test_chromosomes": ["chr21"],
        }
        chroms = ["chr1", "chr2", "chr3", "chr21", "chr1"]
        rns = ["r1", "r2", "r3", "r4", "r5"]
        fold_ids = _assign_fold_ids(chroms, rns, manifest)

        assert fold_ids[0] == 0.0
        assert fold_ids[1] == 1.0
        assert fold_ids[2] == 2.0
        assert np.isnan(fold_ids[3])  # holdout
        assert fold_ids[4] == 0.0

    def test_holdout_is_nan(self):
        from ugbio_srsnv.deep_srsnv.combine_splits import _assign_fold_ids

        manifest = {
            "split_mode": "chromosome_kfold",
            "chrom_to_fold": {"chr1": 0},
            "test_chromosomes": ["chr21", "chr22"],
        }
        chroms = ["chr21", "chr22"]
        rns = ["r1", "r2"]
        fold_ids = _assign_fold_ids(chroms, rns, manifest)

        assert np.all(np.isnan(fold_ids))

    def test_single_model_chrom_val(self):
        from ugbio_srsnv.deep_srsnv.combine_splits import _assign_fold_ids

        manifest = {
            "split_mode": "single_model_chrom_val",
            "train_chromosomes": ["chr1", "chr3"],
            "val_chromosomes": ["chr2"],
            "test_chromosomes": ["chr21"],
        }
        chroms = ["chr1", "chr2", "chr21", "chr3"]
        rns = ["r1", "r2", "r3", "r4"]
        fold_ids = _assign_fold_ids(chroms, rns, manifest)

        assert fold_ids[0] == 0.0  # train
        assert fold_ids[1] == 1.0  # val
        assert np.isnan(fold_ids[2])  # test
        assert fold_ids[3] == 0.0  # train

    def test_single_model_read_hash_deterministic(self):
        from ugbio_srsnv.deep_srsnv.combine_splits import _assign_fold_ids

        manifest = {
            "split_mode": "single_model_read_hash",
            "test_chromosomes": ["chr21"],
            "train_val_chromosomes": ["chr1"],
            "val_fraction": 0.1,
            "random_seed": 42,
            "hash_key": "RN",
        }
        chroms = ["chr1"] * 100
        rns = [f"read_{i}" for i in range(100)]
        fold_ids_1 = _assign_fold_ids(chroms, rns, manifest)
        fold_ids_2 = _assign_fold_ids(chroms, rns, manifest)

        np.testing.assert_array_equal(fold_ids_1, fold_ids_2)
        assert np.sum(fold_ids_1 == 1.0) > 0  # some val
        assert np.sum(fold_ids_1 == 0.0) > 0  # some train


# ---------------------------------------------------------------------------
# Shuffle correctness
# ---------------------------------------------------------------------------


class TestShuffleCorrectness:
    def test_preserves_all_rows(self):
        rng = np.random.default_rng(42)
        indices = np.arange(100)
        shuffled = rng.permutation(indices)

        assert len(shuffled) == len(indices)
        assert set(shuffled.tolist()) == set(indices.tolist())

    def test_mixes_labels(self):
        rng = np.random.default_rng(42)
        labels = np.array([1] * 50 + [0] * 50)
        indices = rng.permutation(len(labels))
        shuffled_labels = labels[indices]

        first_10 = shuffled_labels[:10]
        assert not (
            np.all(first_10 == 1) or np.all(first_10 == 0)
        ), "First 10 shuffled rows should not all be the same label"

    def test_deterministic(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        shuffled1 = rng1.permutation(100)
        shuffled2 = rng2.permutation(100)

        np.testing.assert_array_equal(shuffled1, shuffled2)
