import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ugbio_srsnv import srsnv_dnn_bam_training as dnn_train


def _make_fake_tensor_cache(tmp_path: Path, n_rows: int = 80) -> str:
    """Create a minimal tensor cache on disk for testing."""
    import pickle

    length = 300
    chroms = ["chr1", "chr2", "chr21", "chr22"]
    split_ids = []
    for i in range(n_rows):
        chrom = chroms[i % len(chroms)]
        if chrom in {"chr21", "chr22"}:
            split_ids.append(-1)
        else:
            split_ids.append(i % 3)

    chunk = {
        "cache_format_version": 3,
        "read_base_idx": torch.zeros(n_rows, length, dtype=torch.int16),
        "ref_base_idx": torch.zeros(n_rows, length, dtype=torch.int16),
        "t0_idx": torch.zeros(n_rows, length, dtype=torch.int16),
        "x_num_pos": torch.randn(n_rows, 5, length).to(dtype=torch.float16),
        "x_num_const": torch.randn(n_rows, 7).to(dtype=torch.float16),
        "mask": torch.ones(n_rows, length, dtype=torch.uint8),
        "label": torch.tensor([int((i % 3) == 0) for i in range(n_rows)], dtype=torch.uint8),
        "split_id": torch.tensor(split_ids, dtype=torch.int8),
        "chrom": np.array([chroms[i % len(chroms)] for i in range(n_rows)], dtype=object),
        "pos": np.array([1000 + i for i in range(n_rows)], dtype=np.int32),
        "rn": np.array([f"read_{i}" for i in range(n_rows)], dtype=object),
    }

    cache_path = tmp_path / "tensor_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(cache_path)


@pytest.mark.integration
def test_dnn_lightning_training_pipeline(monkeypatch, tmp_path: Path) -> None:
    resources = Path(__file__).parent.parent / "resources"
    interval_list = resources / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"

    tensor_cache_path = _make_fake_tensor_cache(tmp_path, n_rows=80)

    def _fake_schema(*_args, **_kwargs):
        return {"schema_version": 1, "tag_counts": {"tp": 10, "t0": 10}}

    preprocess_index = {
        "cache_key": "test",
        "cache_hit": False,
        "total_shards": 1,
        "total_output_rows": 80,
        "tensor_cache_path": tensor_cache_path,
        "split_counts": {
            "0": {"rows": 14, "positives": 5, "negatives": 9},
            "1": {"rows": 13, "positives": 4, "negatives": 9},
            "2": {"rows": 13, "positives": 5, "negatives": 8},
            "-1": {"rows": 40, "positives": 14, "negatives": 26},
        },
        "chunk_split_stats": [],
    }

    def _fake_build_tensor_cache(**_kwargs):
        return preprocess_index

    monkeypatch.setattr(dnn_train, "discover_bam_schema", _fake_schema)
    monkeypatch.setattr(dnn_train, "build_tensor_cache", _fake_build_tensor_cache)

    fake_args = type(
        "Args",
        (),
        {
            "positive_bam": "ignored.bam",
            "negative_bam": "ignored.bam",
            "positive_parquet": "ignored.parquet",
            "negative_parquet": "ignored.parquet",
            "training_regions": str(interval_list),
            "output": str(tmp_path),
            "basename": "dnn_test",
            "k_folds": 3,
            "split_manifest_in": None,
            "split_manifest_out": str(tmp_path / "split.json"),
            "holdout_chromosomes": "chr21,chr22",
            "single_model_split": False,
            "val_fraction": 0.1,
            "split_hash_key": "RN",
            "max_rows_per_class": None,
            "epochs": 2,
            "patience": 5,
            "min_epochs": 1,
            "batch_size": 16,
            "eval_batch_size": None,
            "predict_batch_size": None,
            "learning_rate": 1e-3,
            "random_seed": 1,
            "length": 300,
            "lr_scheduler": "none",
            "lr_warmup_epochs": 1,
            "lr_min": 1e-6,
            "lr_step_size": 5,
            "lr_gamma": 0.5,
            "lr_patience": 3,
            "swa": False,
            "swa_lr": 1e-4,
            "swa_epoch_start": 0.7,
            "auto_lr_find": False,
            "auto_scale_batch_size": False,
            "use_amp": False,
            "use_tf32": False,
            "gradient_clip_val": None,
            "accumulate_grad_batches": 1,
            "devices": 1,
            "strategy": "auto",
            "preprocess_cache_dir": str(tmp_path / "cache"),
            "preprocess_num_workers": 1,
            "preprocess_max_ram_gb": 8.0,
            "preprocess_batch_rows": 1024,
            "preprocess_dry_run": False,
            "loader_num_workers": 0,
            "loader_prefetch_factor": 2,
            "loader_pin_memory": False,
            "verbose": False,
        },
    )()
    monkeypatch.setattr(dnn_train, "_cli", lambda: fake_args)

    dnn_train.main()

    assert (tmp_path / "dnn_test.featuremap_df.parquet").is_file()
    assert (tmp_path / "dnn_test.srsnv_dnn_metadata.json").is_file()
    meta = json.loads((tmp_path / "dnn_test.srsnv_dnn_metadata.json").read_text())
    assert meta["model_type"] == "deep_srsnv_cnn_lightning"
    assert "training_results" in meta
    assert "holdout_metrics" in meta
    assert "split_prevalence" in meta
    assert "model_architecture" in meta
    assert "split_manifest" in meta
    assert "st_vocab" in meta["encoders"]
    assert "et_vocab" in meta["encoders"]
    assert "focus" in meta["channel_order"]
    assert "softclip_mask" in meta["channel_order"]
    assert "best_checkpoint_paths" in meta
    assert len(meta["training_results"]) == 3
    assert "lr_scheduler" in meta["training_parameters"]


@pytest.mark.integration
def test_dnn_lightning_single_model_split(monkeypatch, tmp_path: Path) -> None:
    resources = Path(__file__).parent.parent / "resources"
    interval_list = resources / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"

    tensor_cache_path = _make_fake_tensor_cache(tmp_path, n_rows=60)

    preprocess_index = {
        "cache_key": "test_single",
        "cache_hit": False,
        "total_shards": 1,
        "total_output_rows": 60,
        "tensor_cache_path": tensor_cache_path,
        "split_counts": {
            "0": {"rows": 20, "positives": 7, "negatives": 13},
            "1": {"rows": 10, "positives": 3, "negatives": 7},
            "-1": {"rows": 30, "positives": 10, "negatives": 20},
        },
        "chunk_split_stats": [],
    }

    monkeypatch.setattr(dnn_train, "discover_bam_schema", lambda *a, **kw: {"schema_version": 1, "tag_counts": {}})
    monkeypatch.setattr(dnn_train, "build_tensor_cache", lambda **kw: preprocess_index)

    fake_args = type(
        "Args",
        (),
        {
            "positive_bam": "ignored.bam",
            "negative_bam": "ignored.bam",
            "positive_parquet": "ignored.parquet",
            "negative_parquet": "ignored.parquet",
            "training_regions": str(interval_list),
            "output": str(tmp_path),
            "basename": "single",
            "k_folds": 1,
            "split_manifest_in": None,
            "split_manifest_out": None,
            "holdout_chromosomes": "chr21,chr22",
            "single_model_split": True,
            "val_fraction": 0.15,
            "split_hash_key": "RN",
            "max_rows_per_class": None,
            "epochs": 2,
            "patience": 5,
            "min_epochs": 1,
            "batch_size": 16,
            "eval_batch_size": None,
            "predict_batch_size": None,
            "learning_rate": 1e-3,
            "random_seed": 42,
            "length": 300,
            "lr_scheduler": "cosine",
            "lr_warmup_epochs": 1,
            "lr_min": 1e-6,
            "lr_step_size": 5,
            "lr_gamma": 0.5,
            "lr_patience": 3,
            "swa": False,
            "swa_lr": 1e-4,
            "swa_epoch_start": 0.7,
            "auto_lr_find": False,
            "auto_scale_batch_size": False,
            "use_amp": False,
            "use_tf32": False,
            "gradient_clip_val": None,
            "accumulate_grad_batches": 1,
            "devices": 1,
            "strategy": "auto",
            "preprocess_cache_dir": str(tmp_path / "cache"),
            "preprocess_num_workers": 1,
            "preprocess_max_ram_gb": 8.0,
            "preprocess_batch_rows": 1024,
            "preprocess_dry_run": False,
            "loader_num_workers": 0,
            "loader_prefetch_factor": 2,
            "loader_pin_memory": False,
            "verbose": False,
        },
    )()
    monkeypatch.setattr(dnn_train, "_cli", lambda: fake_args)

    dnn_train.main()

    assert (tmp_path / "single.featuremap_df.parquet").is_file()
    meta = json.loads((tmp_path / "single.srsnv_dnn_metadata.json").read_text())
    assert len(meta["training_results"]) == 1
    assert meta["training_parameters"]["lr_scheduler"] == "cosine"
