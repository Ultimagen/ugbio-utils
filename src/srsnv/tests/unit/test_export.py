"""Unit tests for ONNX export, inference engine wrappers, and DNNQualAnnotator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pysam
import pytest
import torch
from ugbio_srsnv.deep_srsnv.training.cnn_model import CNNReadClassifier

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
    from ugbio_srsnv.deep_srsnv.training.lightning_module import SRSNVLightningModule

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
    # Save as a Lightning-compatible checkpoint (includes pytorch-lightning_version key)
    torch.save(
        {
            "state_dict": lit_model.state_dict(),
            "hyper_parameters": dict(lit_model.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        ckpt_path,
    )
    metadata = {
        "model_type": "deep_srsnv_cnn_lightning",
        "prediction_model": "best_checkpoint",
        "best_checkpoint_paths": [ckpt_path],
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

        from ugbio_srsnv.deep_srsnv.preprocessing.cram_to_tensors import _process_shard
        from ugbio_srsnv.deep_srsnv.utils.vocab import load_vocab_config

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
