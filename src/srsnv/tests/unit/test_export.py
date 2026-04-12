"""Unit tests for ONNX export, inference engine wrappers, and DNNQualAnnotator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pysam
import pytest
import torch
from ugbio_srsnv.deep_srsnv.cnn_model import CNNReadClassifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_model():
    """A small CNNReadClassifier for fast tests."""
    return CNNReadClassifier(
        base_vocab_size=7,
        numeric_channels=10,
        tm_vocab_size=6,
        st_vocab_size=6,
        et_vocab_size=6,
        hidden_channels=32,
        n_blocks=2,
        dropout=0.0,
    ).eval()


@pytest.fixture()
def dummy_batch():
    """A single-element batch dict matching stream_cram_to_tensors output."""
    b, l_ = 4, 64
    return {
        "read_base_idx": torch.randint(0, 7, (b, l_), dtype=torch.int16),
        "ref_base_idx": torch.randint(0, 7, (b, l_), dtype=torch.int16),
        "tm_idx": torch.randint(0, 6, (b,), dtype=torch.int8),
        "st_idx": torch.randint(0, 6, (b,), dtype=torch.int8),
        "et_idx": torch.randint(0, 6, (b,), dtype=torch.int8),
        "x_num_pos": torch.randn(b, 6, l_, dtype=torch.float16),
        "x_num_const": torch.randn(b, 4, dtype=torch.float16),
        "mask": torch.ones(b, l_, dtype=torch.uint8),
    }


@pytest.fixture()
def metadata_json(tmp_path, small_model):
    """Write a minimal metadata JSON and a checkpoint, return (metadata_path, ckpt_path)."""
    from ugbio_srsnv.deep_srsnv.lightning_module import SRSNVLightningModule

    lit_model = SRSNVLightningModule(
        base_vocab_size=7,
        numeric_channels=10,
        tm_vocab_size=6,
        st_vocab_size=6,
        et_vocab_size=6,
        hidden_channels=32,
        n_blocks=2,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lr_scheduler="none",
    )
    ckpt_path = str(tmp_path / "model.ckpt")
    torch.save(
        {"state_dict": lit_model.state_dict(), "hyper_parameters": dict(lit_model.hparams)},
        ckpt_path,
    )
    metadata = {
        "model_type": "deep_srsnv_cnn_lightning",
        "prediction_model": "swa",
        "swa_checkpoint_paths": [ckpt_path],
        "best_checkpoint_paths": [],
        "onnx_path": None,
        "trt_engine_path": None,
        "quality_recalibration_table": [[0, 10, 20, 30], [0, 8, 18, 28]],
        "encoders": {
            "base_vocab": {"<PAD>": 0, "<GAP>": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6},
            "t0_vocab": {str(i): i for i in range(11)},
            "tm_vocab": {str(i): i for i in range(6)},
            "st_vocab": {str(i): i for i in range(6)},
            "et_vocab": {str(i): i for i in range(6)},
        },
        "channel_order": ["qual", "tp", "mask", "focus", "softclip_mask", "t0", "strand", "mapq", "rq", "mixed"],
        "training_parameters": {
            "hidden_channels": 32,
            "n_blocks": 2,
            "base_embed_dim": 16,
            "t0_embed_dim": 16,
            "cat_embed_dim": 4,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
    }
    meta_path = str(tmp_path / "metadata.json")
    Path(meta_path).write_text(json.dumps(metadata))
    return meta_path, ckpt_path


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------


class TestExportToOnnx:
    def test_export_creates_valid_onnx(self, small_model, tmp_path):
        from ugbio_srsnv.deep_srsnv.inference.export import export_to_onnx

        onnx_path = tmp_path / "model.onnx"
        result = export_to_onnx(small_model, onnx_path, tensor_length=64)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_dynamic_batch(self, small_model, tmp_path):
        """ONNX model should accept variable batch sizes."""
        from ugbio_srsnv.deep_srsnv.inference.export import export_to_onnx

        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(small_model, onnx_path, tensor_length=64)

        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        sess = ort.InferenceSession(str(onnx_path))
        for batch in [1, 4, 8]:
            feeds = {
                "read_base_idx": np.zeros((batch, 64), dtype=np.int64),
                "ref_base_idx": np.zeros((batch, 64), dtype=np.int64),
                "x_num": np.zeros((batch, 10, 64), dtype=np.float32),
                "mask": np.ones((batch, 64), dtype=np.float32),
                "tm_idx": np.zeros((batch,), dtype=np.int64),
                "st_idx": np.zeros((batch,), dtype=np.int64),
                "et_idx": np.zeros((batch,), dtype=np.int64),
            }
            outputs = sess.run(None, feeds)
            assert outputs[0].shape == (batch,)


class TestOnnxMatchesPytorch:
    def test_outputs_are_close(self, small_model, tmp_path):
        from ugbio_srsnv.deep_srsnv.inference.export import export_to_onnx

        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(small_model, onnx_path, tensor_length=64)

        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        batch = 4
        inputs_pt = {
            "read_base_idx": torch.randint(0, 7, (batch, 64), dtype=torch.long),
            "ref_base_idx": torch.randint(0, 7, (batch, 64), dtype=torch.long),
            "x_num": torch.randn(batch, 10, 64),
            "mask": torch.ones(batch, 64),
            "tm_idx": torch.randint(0, 6, (batch,)),
            "st_idx": torch.randint(0, 6, (batch,)),
            "et_idx": torch.randint(0, 6, (batch,)),
        }
        with torch.no_grad():
            pt_logits = small_model(**inputs_pt).numpy()

        sess = ort.InferenceSession(str(onnx_path))
        feeds = {k: v.numpy().astype(np.int64 if v.dtype == torch.long else np.float32) for k, v in inputs_pt.items()}
        ort_logits = sess.run(None, feeds)[0]

        np.testing.assert_allclose(pt_logits, ort_logits, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# trtexec command construction
# ---------------------------------------------------------------------------


class TestTrtexecCommand:
    def test_command_structure(self):
        from ugbio_srsnv.deep_srsnv.inference.export import build_trtexec_command

        cmd = build_trtexec_command("model.onnx", "model.engine", tensor_length=300)
        assert cmd[0] == "trtexec"
        assert "--onnx=model.onnx" in cmd
        assert "--saveEngine=model.engine" in cmd
        assert "--fp16" in cmd
        min_shapes = [c for c in cmd if c.startswith("--minShapes=")]
        assert len(min_shapes) == 1
        assert "1x300" in min_shapes[0]

    def test_no_fp16(self):
        from ugbio_srsnv.deep_srsnv.inference.export import build_trtexec_command

        cmd = build_trtexec_command("m.onnx", "m.engine", fp16=False)
        assert "--fp16" not in cmd


# ---------------------------------------------------------------------------
# PyTorchEngine
# ---------------------------------------------------------------------------


class TestPyTorchEngine:
    def test_predict_batch_returns_probabilities(self, small_model, dummy_batch):
        from ugbio_srsnv.deep_srsnv.inference.trt_engine import PyTorchEngine

        engine = PyTorchEngine(small_model, device="cpu")
        probs = engine.predict_batch(dummy_batch)
        assert probs.shape == (4,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_matches_direct_pytorch(self, small_model, dummy_batch):
        """PyTorchEngine output should match direct model forward pass."""
        from ugbio_srsnv.deep_srsnv.inference.trt_engine import PyTorchEngine, _compose_x_num

        engine = PyTorchEngine(small_model, device="cpu")
        probs_engine = engine.predict_batch(dummy_batch)

        x_num = torch.from_numpy(_compose_x_num(dummy_batch))
        with torch.no_grad():
            logits = small_model(
                read_base_idx=dummy_batch["read_base_idx"].long(),
                ref_base_idx=dummy_batch["ref_base_idx"].long(),
                x_num=x_num,
                mask=dummy_batch["mask"].float(),
                tm_idx=dummy_batch["tm_idx"].long(),
                st_idx=dummy_batch["st_idx"].long(),
                et_idx=dummy_batch["et_idx"].long(),
            )
            probs_direct = torch.sigmoid(logits).numpy()

        np.testing.assert_allclose(probs_engine, probs_direct, atol=1e-6)


# ---------------------------------------------------------------------------
# load_inference_engine factory
# ---------------------------------------------------------------------------


class TestLoadInferenceEngine:
    def test_pytorch_backend(self, metadata_json):
        from ugbio_srsnv.deep_srsnv.inference.trt_engine import PyTorchEngine, load_inference_engine

        meta_path, _ckpt = metadata_json
        engine = load_inference_engine(meta_path, backend="pytorch", device_id=0)
        assert isinstance(engine, PyTorchEngine)

    def test_trt_backend_missing_engine(self, metadata_json):
        from ugbio_srsnv.deep_srsnv.inference.trt_engine import load_inference_engine

        meta_path, _ = metadata_json
        with pytest.raises(FileNotFoundError, match="TRT engine not found"):
            load_inference_engine(meta_path, backend="trt")

    def test_invalid_backend(self, metadata_json):
        from ugbio_srsnv.deep_srsnv.inference.trt_engine import load_inference_engine

        meta_path, _ = metadata_json
        with pytest.raises(ValueError, match="Unknown backend"):
            load_inference_engine(meta_path, backend="invalid")


# ---------------------------------------------------------------------------
# Multi-GPU round robin
# ---------------------------------------------------------------------------


class TestMultiGPURoundRobin:
    def test_shards_distributed_across_engines(self):
        """Verify _predict_shard sends work to correct engine via round-robin."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import _predict_shard

        mock_engines = [MagicMock() for _ in range(3)]
        for eng in mock_engines:
            eng.predict_batch.return_value = np.array([0.5, 0.6])

        chunk = {
            "read_base_idx": torch.zeros(2, 10, dtype=torch.int16),
            "ref_base_idx": torch.zeros(2, 10, dtype=torch.int16),
            "tm_idx": torch.zeros(2, dtype=torch.int8),
            "st_idx": torch.zeros(2, dtype=torch.int8),
            "et_idx": torch.zeros(2, dtype=torch.int8),
            "x_num_pos": torch.zeros(2, 6, 10, dtype=torch.float16),
            "x_num_const": torch.zeros(2, 4, dtype=torch.float16),
            "mask": torch.ones(2, 10, dtype=torch.uint8),
            "chrom": np.array(["chr1", "chr1"], dtype=object),
            "pos": np.array([100, 200], dtype=np.int32),
            "rn": np.array(["read_A", "read_B"], dtype=object),
        }

        predictions = {}
        _predict_shard(chunk, mock_engines, gpu_idx=0, batch_size=256, predictions=predictions)
        assert mock_engines[0].predict_batch.called

        predictions2 = {}
        _predict_shard(chunk, mock_engines, gpu_idx=1, batch_size=256, predictions=predictions2)
        assert mock_engines[1].predict_batch.called

        predictions3 = {}
        _predict_shard(chunk, mock_engines, gpu_idx=4, batch_size=256, predictions=predictions3)
        assert mock_engines[1].predict_batch.called  # 4 % 3 == 1


# ---------------------------------------------------------------------------
# DNNQualAnnotator
# ---------------------------------------------------------------------------


class TestDNNQualAnnotator:
    def test_annotates_ml_qual_and_filter(self, tmp_path):
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import DNNQualAnnotator

        predictions = {
            ("chr1", 100, "read_A"): 0.99,
            ("chr1", 200, "read_B"): 0.1,
        }

        def quality_fn(mqual):
            return mqual * 0.9

        annotator = DNNQualAnnotator(
            predictions=predictions,
            quality_interpolation_fn=quality_fn,
            low_qual_threshold=10.0,
        )

        header = pysam.VariantHeader()
        header.add_sample("SAMPLE")
        header.add_line("##contig=<ID=chr1,length=1000>")
        header.add_line('##FORMAT=<ID=RN,Number=1,Type=String,Description="Read name">')
        header = annotator.edit_vcf_header(header)

        assert "ML_QUAL" in header.info
        assert "LowQual" in header.filters

    def test_quality_fn_applied(self):
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import _build_quality_fn

        metadata = {"quality_recalibration_table": [[0, 10, 20, 30], [0, 8, 18, 28]]}
        fn = _build_quality_fn(metadata)
        assert fn is not None
        assert fn(0) == 0
        assert fn(10) == pytest.approx(8.0)
        assert fn(15) == pytest.approx(13.0)

    def test_no_lut_returns_none(self):
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import _build_quality_fn

        assert _build_quality_fn({}) is None
        assert _build_quality_fn({"quality_recalibration_table": None}) is None


# ---------------------------------------------------------------------------
# _compose_x_num
# ---------------------------------------------------------------------------


class TestComposeXNum:
    def test_shape(self):
        from ugbio_srsnv.deep_srsnv.inference.trt_engine import _compose_x_num

        batch = {
            "x_num_pos": np.zeros((3, 6, 64), dtype=np.float16),
            "x_num_const": np.zeros((3, 4), dtype=np.float16),
        }
        result = _compose_x_num(batch)
        assert result.shape == (3, 10, 64)
        assert result.dtype == np.float32

    def test_torch_tensors(self):
        from ugbio_srsnv.deep_srsnv.inference.trt_engine import _compose_x_num

        batch = {
            "x_num_pos": torch.zeros(2, 6, 32, dtype=torch.float16),
            "x_num_const": torch.ones(2, 4, dtype=torch.float16),
        }
        result = _compose_x_num(batch)
        assert result.shape == (2, 10, 32)
        assert np.all(result[:, 6:, :] == 1.0)


# ---------------------------------------------------------------------------
# _process_shard_inference (reusing _process_shard without disk I/O)
# ---------------------------------------------------------------------------


class TestProcessShardInference:
    def test_returns_tensors_not_files(self, tmp_path):
        """Verify _process_shard returns tensors in memory (no disk writes)."""
        import array as _array

        from ugbio_srsnv.deep_srsnv.cram_to_tensors import _process_shard
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        header = pysam.AlignmentHeader.from_dict(
            {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10000}]}
        )
        bam_path = str(tmp_path / "test.bam")
        unsorted = str(tmp_path / "unsorted.bam")
        with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
            seg = pysam.AlignedSegment(header)
            seg.query_name = "read_0"
            seg.reference_id = 0
            seg.reference_start = 99
            seg.query_sequence = "A" * 20
            seg.query_qualities = pysam.qualitystring_to_array("I" * 20)
            seg.cigar = [(0, 20)]
            seg.mapping_quality = 60
            seg.flag = 0
            seg.set_tag("MD", "20")
            seg.set_tag("tp", _array.array("i", list(range(20))))
            seg.set_tag("t0", "A" * 20)
            seg.set_tag("rq", 0.9, "f")
            seg.set_tag("tm", "AQ")
            seg.set_tag("st", "PLUS")
            seg.set_tag("et", "MINUS")
            out.write(seg)
        pysam.sort("-o", bam_path, unsorted)
        pysam.index(bam_path)

        encoders = load_vocab_config()
        rows = [
            {
                "CHROM": "chr1",
                "POS": 100,
                "RN": "read_0",
                "REF": "A",
                "ALT": "T",
                "X_ALT": "T",
                "INDEX": 0,
                "REV": 0,
                "MAPQ": 60,
                "rq": 0.9,
                "tm": "AQ",
                "st": "PLUS",
                "et": "MINUS",
                "EDIST": 0,
            },
        ]
        sid, chunk, stats = _process_shard(
            shard_id=0,
            rows=rows,
            cram_path=bam_path,
            reference_path=None,
            encoders=encoders,
            tensor_length=64,
            label=False,
            max_edist=None,
            fetch_mode="pysam",
        )
        assert isinstance(chunk, dict)
        assert "read_base_idx" in chunk
        assert stats["output_rows"] == 1
        assert stats["missing_rows"] == 0
        assert chunk["read_base_idx"].shape == (1, 64)


# ---------------------------------------------------------------------------
# Ensemble manifest / fold-aware inference tests
# ---------------------------------------------------------------------------


class TestLoadEnsembleManifest:
    def test_valid_manifest(self, tmp_path):
        """Valid manifest loads without error."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import load_ensemble_manifest

        meta0 = tmp_path / "fold0_meta.json"
        meta1 = tmp_path / "fold1_meta.json"
        meta0.write_text(json.dumps({"model_type": "test"}))
        meta1.write_text(json.dumps({"model_type": "test"}))

        manifest = {
            "k_folds": 2,
            "chrom_to_fold": {"chr1": 0, "chr2": 1},
            "folds": [
                {"fold_idx": 0, "metadata_path": str(meta0)},
                {"fold_idx": 1, "metadata_path": str(meta1)},
            ],
        }
        manifest_path = tmp_path / "ensemble.json"
        manifest_path.write_text(json.dumps(manifest))

        result = load_ensemble_manifest(str(manifest_path))
        assert result["k_folds"] == 2
        assert len(result["folds"]) == 2

    def test_missing_key_raises(self, tmp_path):
        """Manifest missing required keys raises ValueError."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import load_ensemble_manifest

        manifest = {"k_folds": 2, "chrom_to_fold": {"chr1": 0}}
        manifest_path = tmp_path / "bad.json"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="missing required key.*folds"):
            load_ensemble_manifest(str(manifest_path))

    def test_fold_count_mismatch_raises(self, tmp_path):
        """k_folds != len(folds) raises ValueError."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import load_ensemble_manifest

        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"model_type": "test"}))

        manifest = {
            "k_folds": 3,
            "chrom_to_fold": {"chr1": 0},
            "folds": [{"fold_idx": 0, "metadata_path": str(meta)}],
        }
        manifest_path = tmp_path / "bad2.json"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="k_folds=3 but 1"):
            load_ensemble_manifest(str(manifest_path))

    def test_missing_metadata_file_raises(self, tmp_path):
        """Non-existent metadata_path raises FileNotFoundError."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import load_ensemble_manifest

        manifest = {
            "k_folds": 1,
            "chrom_to_fold": {"chr1": 0},
            "folds": [{"fold_idx": 0, "metadata_path": "/nonexistent/meta.json"}],
        }
        manifest_path = tmp_path / "bad3.json"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(FileNotFoundError):
            load_ensemble_manifest(str(manifest_path))


class TestPartitionRowsByFold:
    def test_basic_partitioning(self):
        """Rows are correctly assigned to folds by chromosome."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import _partition_rows_by_fold

        rows = [
            {"CHROM": "chr1", "POS": 100, "RN": "r1"},
            {"CHROM": "chr2", "POS": 200, "RN": "r2"},
            {"CHROM": "chr3", "POS": 300, "RN": "r3"},
            {"CHROM": "chr1", "POS": 150, "RN": "r4"},
        ]
        chrom_to_fold = {"chr1": 0, "chr2": 1, "chr3": 0}
        fold_rows, test_rows = _partition_rows_by_fold(rows, chrom_to_fold, k_folds=2)

        assert len(fold_rows[0]) == 3  # chr1 x2 + chr3 x1
        assert len(fold_rows[1]) == 1  # chr2
        assert len(test_rows) == 0

    def test_unmapped_chroms_go_to_test(self):
        """Chromosomes not in chrom_to_fold become test reads."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import _partition_rows_by_fold

        rows = [
            {"CHROM": "chr1", "POS": 100, "RN": "r1"},
            {"CHROM": "chrX", "POS": 200, "RN": "r2"},
            {"CHROM": "chrM", "POS": 300, "RN": "r3"},
        ]
        chrom_to_fold = {"chr1": 0}
        fold_rows, test_rows = _partition_rows_by_fold(rows, chrom_to_fold, k_folds=1)

        assert len(fold_rows[0]) == 1
        assert len(test_rows) == 2

    def test_empty_rows(self):
        """Empty input produces empty outputs."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import _partition_rows_by_fold

        fold_rows, test_rows = _partition_rows_by_fold([], {"chr1": 0}, k_folds=1)
        assert fold_rows == [[]]
        assert test_rows == []


class TestAggregateFoldProbabilities:
    def test_single_model_passthrough(self):
        """With K=1, aggregation returns the original probabilities."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import aggregate_fold_probabilities

        probs = np.array([[0.1, 0.5, 0.9]])
        result = aggregate_fold_probabilities(probs)
        np.testing.assert_allclose(result, [0.1, 0.5, 0.9], atol=1e-5)

    def test_symmetric_average(self):
        """Two models with symmetric probabilities around 0.5 average to ~0.5."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import aggregate_fold_probabilities

        probs = np.array([[0.2, 0.8], [0.8, 0.2]])
        result = aggregate_fold_probabilities(probs)
        np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-5)

    def test_logit_space_average(self):
        """Verify aggregation is done in logit space, not probability space."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import aggregate_fold_probabilities

        probs = np.array([[0.1, 0.9], [0.3, 0.7]])
        result = aggregate_fold_probabilities(probs)

        eps = 1e-7
        clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(clipped / (1 - clipped))
        expected = 1.0 / (1.0 + np.exp(-logits.mean(axis=0)))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_extreme_probabilities_clipped(self):
        """Probabilities at 0 and 1 are clipped to avoid inf logits."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import aggregate_fold_probabilities

        probs = np.array([[0.0, 1.0], [0.5, 0.5]])
        result = aggregate_fold_probabilities(probs)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0) and np.all(result <= 1)
