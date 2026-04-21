"""End-to-end smoke tests for the DNN VCF inference pipeline.

Uses small synthetic BAM + featuremap VCF + trained model checkpoint.
Tests the full flow: CRAM fetch -> tensorize -> engine predict -> VCF output.
"""

from __future__ import annotations

import array as _array
import json
from pathlib import Path

import numpy as np
import pysam
import pytest
import torch
from ugbio_srsnv.deep_srsnv.lightning_module import SRSNVLightningModule

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_READS = 10
SEQ_LEN = 30
SEQ = "A" * SEQ_LEN


def _make_test_bam(bam_path: str, reads: list[dict]) -> str:
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 10000}, {"SN": "chr2", "LN": 10000}],
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
            seg.query_qualities = pysam.qualitystring_to_array("I" * len(r["seq"]))
            seg.cigar = [(0, len(r["seq"]))]
            seg.mapping_quality = 60
            seg.flag = 0
            n = len(r["seq"])
            seg.set_tag("MD", str(n))
            seg.set_tag("tp", _array.array("i", list(range(n))))
            seg.set_tag("t0", "A" * n)
            seg.set_tag("rq", 0.9, "f")
            seg.set_tag("tm", "AQ")
            seg.set_tag("st", "PLUS")
            seg.set_tag("et", "MINUS")
            out.write(seg)
    pysam.sort("-o", bam_path, unsorted)
    pysam.index(bam_path)
    Path(unsorted).unlink(missing_ok=True)
    return bam_path


def _make_featuremap_vcf(vcf_path: str, reads: list[dict]) -> str:
    """Create a minimal featuremap VCF.gz with FORMAT/RN (bgzipped + indexed)."""
    if not vcf_path.endswith(".vcf.gz"):
        vcf_path = vcf_path + ".gz" if vcf_path.endswith(".vcf") else vcf_path + ".vcf.gz"

    header = pysam.VariantHeader()
    header.add_sample("SAMPLE")
    header.add_line("##contig=<ID=chr1,length=10000>")
    header.add_line("##contig=<ID=chr2,length=10000>")
    header.add_line('##FORMAT=<ID=RN,Number=1,Type=String,Description="Read name">')
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    header.add_line('##INFO=<ID=EDIST,Number=1,Type=Integer,Description="Edit distance">')

    with pysam.VariantFile(vcf_path, "wz", header=header) as out:
        for r in reads:
            rec = out.new_record()
            rec.contig = r["chrom"]
            rec.pos = r["pos"]
            rec.alleles = ("A", "T")
            rec.samples["SAMPLE"]["RN"] = r["name"]
            out.write(rec)
    pysam.tabix_index(vcf_path, preset="vcf", force=True)
    return vcf_path


def _make_parquet(path: str, reads: list[dict]) -> str:
    import polars as pl

    rows = []
    for r in reads:
        rows.append(
            {
                "CHROM": r["chrom"],
                "POS": r["pos"],
                "REF": "A",
                "ALT": "T",
                "X_ALT": "T",
                "RN": r["name"],
                "INDEX": 0,
                "REV": 0,
                "MAPQ": 60,
                "rq": 0.9,
                "tm": "AQ",
                "st": "PLUS",
                "et": "MINUS",
                "EDIST": 0,
            }
        )
    pl.DataFrame(rows).write_parquet(path)
    return path


@pytest.fixture()
def inference_data(tmp_path):
    """Create synthetic BAM, featuremap VCF, parquet, model checkpoint, and metadata."""
    reads = []
    for i in range(N_READS):
        chrom = "chr1" if i < 7 else "chr2"
        reads.append({"name": f"read_{i}", "chrom": chrom, "pos": 100 + i * 20, "seq": SEQ})

    bam_path = _make_test_bam(str(tmp_path / "source.bam"), reads)
    vcf_path = _make_featuremap_vcf(str(tmp_path / "featuremap.vcf"), reads)
    parquet_path = _make_parquet(str(tmp_path / "featuremap.parquet"), reads)

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
    ckpt_path = str(tmp_path / "model_swa.ckpt")
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
        "quality_recalibration_table": [
            list(range(101)),
            [max(0, x * 0.9) for x in range(101)],
        ],
        "encoders": {
            "base_vocab": {"<PAD>": 0, "<GAP>": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6},
            "t0_vocab": {
                "<PAD>": 0,
                "<MISSING>": 1,
                "<GAP>": 2,
                "-": 3,
                ":": 4,
                "A": 5,
                "C": 6,
                "D": 7,
                "G": 8,
                "N": 9,
                "T": 10,
            },
            "tm_vocab": {"<PAD>": 0, "<MISSING>": 1, "A": 2, "AQ": 3, "AQZ": 4, "AZ": 5},
            "st_vocab": {"<PAD>": 0, "<MISSING>": 1, "MINUS": 2, "MIXED": 3, "PLUS": 4, "UNDETERMINED": 5},
            "et_vocab": {"<PAD>": 0, "<MISSING>": 1, "MINUS": 2, "MIXED": 3, "PLUS": 4, "UNDETERMINED": 5},
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

    return {
        "bam_path": bam_path,
        "vcf_path": vcf_path,
        "parquet_path": parquet_path,
        "metadata_path": meta_path,
        "ckpt_path": ckpt_path,
        "tmp_path": tmp_path,
        "n_reads": N_READS,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDNNInferenceSmokeE2E:
    def test_pytorch_backend_e2e(self, inference_data):
        """Full pipeline with PyTorch backend produces annotated VCF."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import run_inference_pipeline

        output_vcf = str(inference_data["tmp_path"] / "output_pytorch.vcf")
        run_inference_pipeline(
            featuremap_vcf=inference_data["vcf_path"],
            cram_path=inference_data["bam_path"],
            metadata_path=inference_data["metadata_path"],
            output_vcf=output_vcf,
            backend="pytorch",
            checkpoint_path=inference_data["ckpt_path"],
            gpu_ids=[0] if torch.cuda.is_available() else None,
            num_cram_workers=1,
            shard_size=100,
            batch_size=64,
            tensor_length=64,
            parquet_path=inference_data["parquet_path"],
            fetch_mode="pysam",
        )

        assert Path(output_vcf).exists()
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
        assert len(records) > 0

        annotated = [r for r in records if "ML_QUAL" in r.info]
        assert len(annotated) > 0, "Expected at least some records with ML_QUAL annotation"

        for rec in annotated:
            ml_qual = rec.info["ML_QUAL"]
            assert ml_qual >= 0
            if rec.qual is not None:
                assert rec.qual >= 0

    def test_predictions_are_deterministic(self, inference_data):
        """Two runs with the same inputs produce identical predictions."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import run_inference_pipeline

        outputs = []
        for run_idx in range(2):
            out = str(inference_data["tmp_path"] / f"output_det_{run_idx}.vcf")
            run_inference_pipeline(
                featuremap_vcf=inference_data["vcf_path"],
                cram_path=inference_data["bam_path"],
                metadata_path=inference_data["metadata_path"],
                output_vcf=out,
                backend="pytorch",
                checkpoint_path=inference_data["ckpt_path"],
                gpu_ids=[0] if torch.cuda.is_available() else None,
                num_cram_workers=1,
                shard_size=100,
                batch_size=64,
                tensor_length=64,
                parquet_path=inference_data["parquet_path"],
                fetch_mode="pysam",
            )
            with pysam.VariantFile(out) as vcf:
                outputs.append(list(vcf))

        assert len(outputs[0]) == len(outputs[1])
        for r0, r1 in zip(outputs[0], outputs[1]):
            if "ML_QUAL" in r0.info and "ML_QUAL" in r1.info:
                np.testing.assert_allclose(r0.info["ML_QUAL"], r1.info["ML_QUAL"], atol=1e-4)

    def test_trt_vs_pytorch_equivalence(self, inference_data):
        """If TRT is available, compare outputs against PyTorch backend."""
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            pytest.skip("TensorRT not available")

        from ugbio_srsnv.deep_srsnv.inference.export import export_to_onnx, serialize_with_trtexec

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
        raw = torch.load(inference_data["ckpt_path"], weights_only=False)  # noqa: S301
        lit_model.load_state_dict(raw["state_dict"])

        onnx_path = str(inference_data["tmp_path"] / "model.onnx")
        export_to_onnx(lit_model.model, onnx_path, tensor_length=64)

        engine_path = str(inference_data["tmp_path"] / "model.engine")
        result = serialize_with_trtexec(onnx_path, engine_path, tensor_length=64)
        if result is None:
            pytest.skip("trtexec not available")

        with open(inference_data["metadata_path"]) as f:
            meta = json.load(f)
        meta["trt_engine_path"] = engine_path
        meta_trt = str(inference_data["tmp_path"] / "metadata_trt.json")
        Path(meta_trt).write_text(json.dumps(meta))

        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import run_inference_pipeline

        out_pt = str(inference_data["tmp_path"] / "out_pytorch.vcf")
        out_trt = str(inference_data["tmp_path"] / "out_trt.vcf")

        for out_path, backend, meta_path in [
            (out_pt, "pytorch", inference_data["metadata_path"]),
            (out_trt, "trt", meta_trt),
        ]:
            run_inference_pipeline(
                featuremap_vcf=inference_data["vcf_path"],
                cram_path=inference_data["bam_path"],
                metadata_path=meta_path,
                output_vcf=out_path,
                backend=backend,
                checkpoint_path=inference_data["ckpt_path"] if backend == "pytorch" else None,
                gpu_ids=[0],
                num_cram_workers=1,
                shard_size=100,
                batch_size=64,
                tensor_length=64,
                parquet_path=inference_data["parquet_path"],
                fetch_mode="pysam",
            )

        with pysam.VariantFile(out_pt) as vcf:
            pt_records = list(vcf)
        with pysam.VariantFile(out_trt) as vcf:
            trt_records = list(vcf)

        assert len(pt_records) == len(trt_records)
        for r_pt, r_trt in zip(pt_records, trt_records):
            if "ML_QUAL" in r_pt.info and "ML_QUAL" in r_trt.info:
                np.testing.assert_allclose(
                    r_pt.info["ML_QUAL"],
                    r_trt.info["ML_QUAL"],
                    atol=1e-2,
                    err_msg="TRT vs PyTorch ML_QUAL divergence exceeds FP16 tolerance",
                )


# ---------------------------------------------------------------------------
# Ensemble (k-fold) inference tests
# ---------------------------------------------------------------------------


def _make_fold_metadata(tmp_path, fold_idx, ckpt_path):
    """Create a per-fold metadata JSON pointing to the given checkpoint."""
    metadata = {
        "model_type": "deep_srsnv_cnn_lightning",
        "prediction_model": "swa",
        "swa_checkpoint_paths": [ckpt_path],
        "best_checkpoint_paths": [],
        "onnx_path": None,
        "trt_engine_path": None,
        "encoders": {
            "base_vocab": {"<PAD>": 0, "<GAP>": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6},
            "t0_vocab": {
                "<PAD>": 0,
                "<MISSING>": 1,
                "<GAP>": 2,
                "-": 3,
                ":": 4,
                "A": 5,
                "C": 6,
                "D": 7,
                "G": 8,
                "N": 9,
                "T": 10,
            },
            "tm_vocab": {"<PAD>": 0, "<MISSING>": 1, "A": 2, "AQ": 3, "AQZ": 4, "AZ": 5},
            "st_vocab": {"<PAD>": 0, "<MISSING>": 1, "MINUS": 2, "MIXED": 3, "PLUS": 4, "UNDETERMINED": 5},
            "et_vocab": {"<PAD>": 0, "<MISSING>": 1, "MINUS": 2, "MIXED": 3, "PLUS": 4, "UNDETERMINED": 5},
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
    meta_path = str(tmp_path / f"fold{fold_idx}_metadata.json")
    Path(meta_path).write_text(json.dumps(metadata))
    return meta_path


@pytest.fixture()
def ensemble_data(tmp_path):
    """Create data for k=2 fold ensemble: two separately-trained models."""
    reads = []
    for i in range(N_READS):
        chrom = "chr1" if i < 7 else "chr2"
        reads.append({"name": f"read_{i}", "chrom": chrom, "pos": 100 + i * 20, "seq": SEQ})

    bam_path = _make_test_bam(str(tmp_path / "source.bam"), reads)
    vcf_path = _make_featuremap_vcf(str(tmp_path / "featuremap.vcf"), reads)
    parquet_path = _make_parquet(str(tmp_path / "featuremap.parquet"), reads)

    model_kwargs = {
        "base_vocab_size": 7,
        "numeric_channels": 10,
        "tm_vocab_size": 6,
        "st_vocab_size": 6,
        "et_vocab_size": 6,
        "hidden_channels": 32,
        "n_blocks": 2,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "lr_scheduler": "none",
    }

    ckpt_paths = []
    meta_paths = []
    for fold_idx in range(2):
        lit_model = SRSNVLightningModule(**model_kwargs)
        ckpt = str(tmp_path / f"model_fold_{fold_idx}_swa.ckpt")
        torch.save(
            {"state_dict": lit_model.state_dict(), "hyper_parameters": dict(lit_model.hparams)},
            ckpt,
        )
        ckpt_paths.append(ckpt)
        meta_paths.append(_make_fold_metadata(tmp_path, fold_idx, ckpt))

    manifest = {
        "k_folds": 2,
        "chrom_to_fold": {"chr1": 0, "chr2": 1},
        "folds": [
            {"fold_idx": 0, "metadata_path": meta_paths[0]},
            {"fold_idx": 1, "metadata_path": meta_paths[1]},
        ],
        "quality_recalibration_table": [
            list(range(101)),
            [max(0, x * 0.9) for x in range(101)],
        ],
    }
    manifest_path = str(tmp_path / "ensemble_manifest.json")
    Path(manifest_path).write_text(json.dumps(manifest))

    return {
        "bam_path": bam_path,
        "vcf_path": vcf_path,
        "parquet_path": parquet_path,
        "manifest_path": manifest_path,
        "ckpt_paths": ckpt_paths,
        "meta_paths": meta_paths,
        "tmp_path": tmp_path,
        "n_reads": N_READS,
    }


class TestEnsembleInferenceSmokeE2E:
    def test_ensemble_produces_annotated_vcf(self, ensemble_data):
        """Ensemble inference with k=2 folds produces annotated VCF."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import run_inference_pipeline

        output_vcf = str(ensemble_data["tmp_path"] / "output_ensemble.vcf")
        run_inference_pipeline(
            featuremap_vcf=ensemble_data["vcf_path"],
            cram_path=ensemble_data["bam_path"],
            ensemble_manifest_path=ensemble_data["manifest_path"],
            output_vcf=output_vcf,
            backend="pytorch",
            gpu_ids=[0] if torch.cuda.is_available() else None,
            num_cram_workers=1,
            shard_size=100,
            batch_size=64,
            tensor_length=64,
            parquet_path=ensemble_data["parquet_path"],
            fetch_mode="pysam",
        )

        assert Path(output_vcf).exists()
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
        assert len(records) > 0

        annotated = [r for r in records if "ML_QUAL" in r.info]
        assert len(annotated) > 0, "Expected at least some records with ML_QUAL annotation"

        for rec in annotated:
            assert rec.info["ML_QUAL"] >= 0
            if rec.qual is not None:
                assert rec.qual >= 0

    def test_ensemble_predictions_are_deterministic(self, ensemble_data):
        """Two ensemble runs produce identical predictions."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import run_inference_pipeline

        outputs = []
        for run_idx in range(2):
            out = str(ensemble_data["tmp_path"] / f"output_ens_det_{run_idx}.vcf")
            run_inference_pipeline(
                featuremap_vcf=ensemble_data["vcf_path"],
                cram_path=ensemble_data["bam_path"],
                ensemble_manifest_path=ensemble_data["manifest_path"],
                output_vcf=out,
                backend="pytorch",
                gpu_ids=[0] if torch.cuda.is_available() else None,
                num_cram_workers=1,
                shard_size=100,
                batch_size=64,
                tensor_length=64,
                parquet_path=ensemble_data["parquet_path"],
                fetch_mode="pysam",
            )
            with pysam.VariantFile(out) as vcf:
                outputs.append(list(vcf))

        assert len(outputs[0]) == len(outputs[1])
        for r0, r1 in zip(outputs[0], outputs[1]):
            if "ML_QUAL" in r0.info and "ML_QUAL" in r1.info:
                np.testing.assert_allclose(r0.info["ML_QUAL"], r1.info["ML_QUAL"], atol=1e-4)

    def test_ensemble_matches_single_model_when_k1(self, inference_data):
        """With k=1 ensemble, results match single-model inference."""
        from ugbio_srsnv.deep_srsnv.inference.dnn_vcf_inference import run_inference_pipeline

        manifest = {
            "k_folds": 1,
            "chrom_to_fold": {"chr1": 0, "chr2": 0},
            "folds": [{"fold_idx": 0, "metadata_path": inference_data["metadata_path"]}],
            "quality_recalibration_table": [
                list(range(101)),
                [max(0, x * 0.9) for x in range(101)],
            ],
        }
        manifest_path = str(inference_data["tmp_path"] / "k1_manifest.json")
        Path(manifest_path).write_text(json.dumps(manifest))

        out_single = str(inference_data["tmp_path"] / "output_single.vcf")
        run_inference_pipeline(
            featuremap_vcf=inference_data["vcf_path"],
            cram_path=inference_data["bam_path"],
            metadata_path=inference_data["metadata_path"],
            output_vcf=out_single,
            backend="pytorch",
            checkpoint_path=inference_data["ckpt_path"],
            gpu_ids=[0] if torch.cuda.is_available() else None,
            num_cram_workers=1,
            shard_size=100,
            batch_size=64,
            tensor_length=64,
            parquet_path=inference_data["parquet_path"],
            fetch_mode="pysam",
        )

        out_ensemble = str(inference_data["tmp_path"] / "output_k1_ensemble.vcf")
        run_inference_pipeline(
            featuremap_vcf=inference_data["vcf_path"],
            cram_path=inference_data["bam_path"],
            ensemble_manifest_path=manifest_path,
            output_vcf=out_ensemble,
            backend="pytorch",
            gpu_ids=[0] if torch.cuda.is_available() else None,
            num_cram_workers=1,
            shard_size=100,
            batch_size=64,
            tensor_length=64,
            parquet_path=inference_data["parquet_path"],
            fetch_mode="pysam",
        )

        with pysam.VariantFile(out_single) as vcf:
            single_records = list(vcf)
        with pysam.VariantFile(out_ensemble) as vcf:
            ensemble_records = list(vcf)

        assert len(single_records) == len(ensemble_records)
        for rs, re in zip(single_records, ensemble_records):
            if "ML_QUAL" in rs.info and "ML_QUAL" in re.info:
                np.testing.assert_allclose(
                    rs.info["ML_QUAL"],
                    re.info["ML_QUAL"],
                    atol=1e-4,
                    err_msg="k=1 ensemble should match single model",
                )
