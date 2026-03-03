"""Tests for SWA fixes: EarlyStopping disabling, scheduler warnings,
SWA validation tracker, and SWA checkpoint loading in inference."""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock

import torch
from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging
from ugbio_srsnv.deep_srsnv.inference import (
    load_dnn_model_from_swa_checkpoint,
    load_dnn_models_from_metadata,
)
from ugbio_srsnv.deep_srsnv.lightning_module import SRSNVLightningModule
from ugbio_srsnv.deep_srsnv.swa_validation_tracker import SWAValidationTracker
from ugbio_srsnv.srsnv_dnn_bam_training import _build_callbacks


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "patience": 3,
        "swa": False,
        "swa_lr": 1e-4,
        "swa_epoch_start": 0.7,
        "epochs": 10,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _small_model_kwargs():
    return {
        "base_vocab_size": 8,
        "t0_vocab_size": 12,
        "numeric_channels": 9,
        "tm_vocab_size": 9,
        "st_vocab_size": 5,
        "et_vocab_size": 5,
        "hidden_channels": 16,
        "n_blocks": 2,
    }


def _make_batch(batch_size: int = 8, length: int = 50) -> dict:
    return {
        "read_base_idx": torch.randint(0, 7, (batch_size, length)),
        "ref_base_idx": torch.randint(0, 7, (batch_size, length)),
        "t0_idx": torch.randint(0, 10, (batch_size, length)),
        "tm_idx": torch.randint(0, 9, (batch_size,)),
        "st_idx": torch.randint(0, 5, (batch_size,)),
        "et_idx": torch.randint(0, 5, (batch_size,)),
        "x_num": torch.randn(batch_size, 9, length),
        "mask": torch.ones(batch_size, length),
        "label": torch.randint(0, 2, (batch_size,)).float(),
        "fold_id": torch.zeros(batch_size, dtype=torch.long),
    }


# ─── Phase 1: EarlyStopping + SWA ───────────────────────────────────────────


class TestEarlyStoppingDisabledWithSWA:
    def test_earlystopping_present_without_swa(self, tmp_path):
        args = _make_args(swa=False)
        callbacks, swa_cb = _build_callbacks(args, tmp_path, "test.", 0)
        assert any(isinstance(cb, EarlyStopping) for cb in callbacks)
        assert swa_cb is None

    def test_earlystopping_absent_with_swa(self, tmp_path):
        args = _make_args(swa=True, swa_epoch_start=5)
        callbacks, swa_cb = _build_callbacks(args, tmp_path, "test.", 0)
        assert not any(isinstance(cb, EarlyStopping) for cb in callbacks)
        assert swa_cb is not None
        assert isinstance(swa_cb, StochasticWeightAveraging)

    def test_swa_callback_returned_with_tracker(self, tmp_path):
        args = _make_args(swa=True, swa_epoch_start=3)
        callbacks, swa_cb = _build_callbacks(args, tmp_path, "test.", 0)
        assert any(isinstance(cb, SWAValidationTracker) for cb in callbacks)
        tracker = [cb for cb in callbacks if isinstance(cb, SWAValidationTracker)][0]
        assert tracker._swa_cb is swa_cb

    def test_swa_epoch_start_fraction_conversion(self, tmp_path):
        args = _make_args(swa=True, swa_epoch_start=0.5, epochs=20)
        callbacks, swa_cb = _build_callbacks(args, tmp_path, "test.", 0)
        assert swa_cb._swa_epoch_start == 10


# ─── Phase 3: SWA Validation Tracker ────────────────────────────────────────


class TestSWAValidationTracker:
    def test_is_swa_active_before_init(self):
        swa_cb = StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=3)
        tracker = SWAValidationTracker(swa_cb)
        trainer = MagicMock()
        trainer.current_epoch = 0
        assert not tracker._is_swa_active(trainer)

    def test_save_and_restore_state_preserves_weights(self):
        model = SRSNVLightningModule(**_small_model_kwargs(), lr_scheduler="none")
        original_params = [p.data.clone() for p in model.parameters()]

        state = SWAValidationTracker._save_state(model)
        for p in model.parameters():
            p.data.fill_(999.0)

        SWAValidationTracker._restore_state(model, state)
        for p_orig, p_restored in zip(original_params, model.parameters()):
            assert torch.equal(p_orig, p_restored.data)

    def test_save_and_restore_preserves_bn_stats(self):
        model = SRSNVLightningModule(**_small_model_kwargs(), lr_scheduler="none")
        model.train()
        batch = _make_batch()
        model._forward(batch)

        state = SWAValidationTracker._save_state(model)
        bn_modules = [m for m in model.modules() if hasattr(m, "running_mean") and m.running_mean is not None]
        original_means = [m.running_mean.clone() for m in bn_modules]

        for m in bn_modules:
            m.running_mean.fill_(0.0)
            m.running_var.fill_(1.0)

        SWAValidationTracker._restore_state(model, state)
        for m, orig_mean in zip(bn_modules, original_means):
            assert torch.equal(m.running_mean, orig_mean)

    def test_copy_averaged_weights(self):
        model_src = SRSNVLightningModule(**_small_model_kwargs(), lr_scheduler="none")
        model_dst = SRSNVLightningModule(**_small_model_kwargs(), lr_scheduler="none")
        for p in model_src.parameters():
            p.data.fill_(42.0)

        SWAValidationTracker._copy_averaged_weights(model_src, model_dst)
        for p_src, p_dst in zip(model_src.parameters(), model_dst.parameters()):
            assert torch.equal(p_src.data, p_dst.data)


# ─── Phase 5: SWA Checkpoint Loading ────────────────────────────────────────


class TestSWACheckpointLoading:
    def test_load_swa_checkpoint(self, tmp_path):
        model = SRSNVLightningModule(**_small_model_kwargs(), lr_scheduler="none")
        for p in model.parameters():
            p.data.fill_(7.0)

        ckpt_path = tmp_path / "swa.ckpt"
        torch.save(
            {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)},
            ckpt_path,
        )

        metadata = {
            "encoders": {
                "base_vocab": {str(i): i for i in range(8)},
                "t0_vocab": {str(i): i for i in range(12)},
                "tm_vocab": {str(i): i for i in range(9)},
                "st_vocab": {str(i): i for i in range(5)},
                "et_vocab": {str(i): i for i in range(5)},
            },
            "channel_order": ["qual", "tp", "mask", "focus", "softclip_mask", "strand", "mapq", "rq", "mixed"],
            "training_parameters": {
                "hidden_channels": 16,
                "n_blocks": 2,
                "base_embed_dim": 16,
                "t0_embed_dim": 16,
                "cat_embed_dim": 4,
                "dropout": 0.3,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
            },
        }

        loaded = load_dnn_model_from_swa_checkpoint(ckpt_path, metadata)
        for p in loaded.parameters():
            assert torch.allclose(p.data, torch.tensor(7.0))

    def test_load_models_prefers_swa_when_prediction_model_swa(self, tmp_path):
        model = SRSNVLightningModule(**_small_model_kwargs(), lr_scheduler="none")

        swa_path = str(tmp_path / "swa.ckpt")
        torch.save(
            {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)},
            swa_path,
        )

        metadata = {
            "prediction_model": "swa",
            "swa_checkpoint_paths": [swa_path],
            "best_checkpoint_paths": [],
            "encoders": {
                "base_vocab": {str(i): i for i in range(8)},
                "t0_vocab": {str(i): i for i in range(12)},
                "tm_vocab": {str(i): i for i in range(9)},
                "st_vocab": {str(i): i for i in range(5)},
                "et_vocab": {str(i): i for i in range(5)},
            },
            "channel_order": ["qual", "tp", "mask", "focus", "softclip_mask", "strand", "mapq", "rq", "mixed"],
            "training_parameters": {
                "hidden_channels": 16,
                "n_blocks": 2,
                "base_embed_dim": 16,
                "t0_embed_dim": 16,
                "cat_embed_dim": 4,
                "dropout": 0.3,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
            },
        }
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        models = load_dnn_models_from_metadata(meta_path)
        assert len(models) == 1
        assert isinstance(models[0], SRSNVLightningModule)

    def test_load_models_falls_back_without_swa_paths(self, tmp_path):
        metadata = {
            "prediction_model": "swa",
            "swa_checkpoint_paths": [],
            "best_checkpoint_paths": [],
            "training_results": [],
            "encoders": {
                "base_vocab": {str(i): i for i in range(8)},
                "t0_vocab": {str(i): i for i in range(12)},
                "tm_vocab": {},
                "st_vocab": {},
                "et_vocab": {},
            },
            "channel_order": [],
            "training_parameters": {},
        }
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        models = load_dnn_models_from_metadata(meta_path)
        assert len(models) == 0

    def test_prefer_swa_override(self, tmp_path):
        model = SRSNVLightningModule(**_small_model_kwargs(), lr_scheduler="none")

        swa_path = str(tmp_path / "swa.ckpt")
        torch.save(
            {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)},
            swa_path,
        )

        metadata = {
            "prediction_model": "best_checkpoint",
            "swa_checkpoint_paths": [swa_path],
            "best_checkpoint_paths": [],
            "encoders": {
                "base_vocab": {str(i): i for i in range(8)},
                "t0_vocab": {str(i): i for i in range(12)},
                "tm_vocab": {str(i): i for i in range(9)},
                "st_vocab": {str(i): i for i in range(5)},
                "et_vocab": {str(i): i for i in range(5)},
            },
            "channel_order": ["qual", "tp", "mask", "focus", "softclip_mask", "strand", "mapq", "rq", "mixed"],
            "training_parameters": {
                "hidden_channels": 16,
                "n_blocks": 2,
                "base_embed_dim": 16,
                "t0_embed_dim": 16,
                "cat_embed_dim": 4,
                "dropout": 0.3,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
            },
        }
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        # prefer_swa=True overrides prediction_model="best_checkpoint"
        models = load_dnn_models_from_metadata(meta_path, prefer_swa=True)
        assert len(models) == 1
