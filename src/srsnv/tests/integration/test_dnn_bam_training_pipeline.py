import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ugbio_srsnv import srsnv_dnn_bam_training as dnn_train


def _make_fake_tensor_cache(tmp_path: Path, n_rows: int = 80) -> str:
    """Create a minimal tensor cache on disk for testing.

    Uses diverse chromosomes (chr1-chr9 + chr21/chr22) so that the
    DataModule can compute valid split_ids for any k-fold configuration.
    The cache does NOT contain split_id — the DataModule computes it
    from the chrom array and the split manifest.
    """
    import pickle  # noqa: PLC0415

    length = 300
    chroms = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr21", "chr22"]

    chunk = {
        "cache_format_version": 5,
        "read_base_idx": torch.zeros(n_rows, length, dtype=torch.int16),
        "ref_base_idx": torch.zeros(n_rows, length, dtype=torch.int16),
        "tm_idx": torch.ones(n_rows, dtype=torch.int8),
        "st_idx": torch.ones(n_rows, dtype=torch.int8),
        "et_idx": torch.ones(n_rows, dtype=torch.int8),
        "x_num_pos": torch.randn(n_rows, 6, length).to(dtype=torch.float16),
        "x_num_const": torch.randn(n_rows, 4).to(dtype=torch.float16),
        "mask": torch.ones(n_rows, length, dtype=torch.uint8),
        "label": torch.tensor([int((i % 3) == 0) for i in range(n_rows)], dtype=torch.uint8),
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
            "val_chromosomes": None,
            "pretrained_checkpoint": None,
            "weight_decay": 1e-4,
            "hidden_channels": 128,
            "n_blocks": 6,
            "base_embed_dim": 16,
            "cat_embed_dim": 4,
            "dropout": 0.3,
            "stats_positive": None,
            "stats_negative": None,
            "stats_featuremap": None,
            "mean_coverage": None,
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
            "val_chromosomes": None,
            "pretrained_checkpoint": None,
            "weight_decay": 1e-4,
            "hidden_channels": 128,
            "n_blocks": 6,
            "base_embed_dim": 16,
            "cat_embed_dim": 4,
            "dropout": 0.3,
            "stats_positive": None,
            "stats_negative": None,
            "stats_featuremap": None,
            "mean_coverage": None,
        },
    )()
    monkeypatch.setattr(dnn_train, "_cli", lambda: fake_args)

    dnn_train.main()

    assert (tmp_path / "single.featuremap_df.parquet").is_file()
    meta = json.loads((tmp_path / "single.srsnv_dnn_metadata.json").read_text())
    assert len(meta["training_results"]) == 1
    assert meta["training_parameters"]["lr_scheduler"] == "cosine"


@pytest.mark.integration
def test_dnn_pretrained_checkpoint_finetuning(monkeypatch, tmp_path: Path) -> None:
    """Train a model, then fine-tune from its checkpoint and verify pre-training eval runs."""
    resources = Path(__file__).parent.parent / "resources"
    interval_list = resources / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"

    tensor_cache_path = _make_fake_tensor_cache(tmp_path, n_rows=80)

    preprocess_index = {
        "cache_key": "test_pretrained",
        "cache_hit": False,
        "total_shards": 1,
        "total_output_rows": 80,
        "tensor_cache_path": tensor_cache_path,
    }

    monkeypatch.setattr(dnn_train, "discover_bam_schema", lambda *a, **kw: {"schema_version": 1, "tag_counts": {}})
    monkeypatch.setattr(dnn_train, "build_tensor_cache", lambda **kw: preprocess_index)

    base_attrs = {
        "positive_bam": "ignored.bam",
        "negative_bam": "ignored.bam",
        "positive_parquet": "ignored.parquet",
        "negative_parquet": "ignored.parquet",
        "training_regions": str(interval_list),
        "output": str(tmp_path / "phase1"),
        "basename": "base_model",
        "k_folds": 3,
        "split_manifest_in": None,
        "split_manifest_out": None,
        "holdout_chromosomes": "chr21,chr22",
        "val_chromosomes": None,
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
        "weight_decay": 1e-4,
        "random_seed": 1,
        "length": 300,
        "pretrained_checkpoint": None,
        "hidden_channels": 128,
        "n_blocks": 6,
        "base_embed_dim": 16,
        "cat_embed_dim": 4,
        "dropout": 0.3,
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
        "stats_positive": None,
        "stats_negative": None,
        "stats_featuremap": None,
        "mean_coverage": None,
    }

    # Phase 1: train a base model
    (tmp_path / "phase1").mkdir()
    base_args = type("Args", (), base_attrs)()
    monkeypatch.setattr(dnn_train, "_cli", lambda: base_args)  # noqa: PLW0108
    dnn_train.main()

    # Find the checkpoint produced by phase 1
    ckpt_files = list((tmp_path / "phase1").glob("*.ckpt"))
    assert ckpt_files, "Phase 1 should produce at least one checkpoint"
    ckpt_path = str(ckpt_files[0])

    # Phase 2: fine-tune from that checkpoint
    (tmp_path / "phase2").mkdir()
    ft_attrs = {
        **base_attrs,
        "output": str(tmp_path / "phase2"),
        "basename": "finetuned",
        "pretrained_checkpoint": ckpt_path,
    }
    ft_args = type("Args", (), ft_attrs)()
    monkeypatch.setattr(dnn_train, "_cli", lambda: ft_args)  # noqa: PLW0108
    dnn_train.main()

    assert (tmp_path / "phase2" / "finetuned.featuremap_df.parquet").is_file()
    meta = json.loads((tmp_path / "phase2" / "finetuned.srsnv_dnn_metadata.json").read_text())
    assert meta["training_parameters"]["pretrained_checkpoint"] == ckpt_path
