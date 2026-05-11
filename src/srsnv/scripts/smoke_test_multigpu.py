"""Smoke test for single/multi-GPU DNN training using a small pre-built cache.

Usage:
    uv run python src/srsnv/scripts/smoke_test_multigpu.py --devices 1
    uv run python src/srsnv/scripts/smoke_test_multigpu.py --devices 0,3
    uv run python src/srsnv/scripts/smoke_test_multigpu.py --devices auto
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import lightning
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import CSVLogger
from ugbio_core.logger import logger
from ugbio_srsnv.deep_srsnv.data_module import SRSNVDataModule
from ugbio_srsnv.deep_srsnv.data_prep import (
    _DEFAULT_SHM_DIR,
    load_cache_from_shm,
    load_full_tensor_cache,
    save_cache_to_shm,
)
from ugbio_srsnv.deep_srsnv.lightning_module import SRSNVLightningModule
from ugbio_srsnv.srsnv_dnn_bam_training import _parse_devices, _resolve_n_devices


def _setup_environment(args):
    """Configure logging, seeds, TF32, and resolve devices."""
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

    lightning.seed_everything(42, workers=True)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = _parse_devices(args.devices) if accelerator == "gpu" else 1
    n_devices = _resolve_n_devices(devices) if accelerator == "gpu" else 1
    logger.info("Devices: %s (n_devices=%d)", devices, n_devices)
    return out_dir, accelerator, devices, n_devices


def _load_cache(args, n_devices):
    """Load tensor cache, using shared memory for multi-GPU."""
    shm_cache_path = Path(_DEFAULT_SHM_DIR) / "deep_srsnv_shared_cache"
    if n_devices > 1:
        is_rank_zero = int(os.environ.get("LOCAL_RANK", "0")) == 0
        if is_rank_zero:
            full_cache = load_full_tensor_cache(args.cache)
            save_cache_to_shm(full_cache)
            del full_cache
        full_cache = load_cache_from_shm(shm_cache_path)
    else:
        full_cache = load_full_tensor_cache(args.cache)
    return full_cache, shm_cache_path


def _build_trainer(args, out_dir, accelerator, devices, n_devices):
    """Build Lightning Trainer with callbacks."""
    strategy = "ddp" if n_devices > 1 else "auto"
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=5),
        ModelCheckpoint(dirpath=out_dir, monitor="val_auc", mode="max", save_top_k=1),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=args.swa_epoch_start))
    csv_logger = CSVLogger(save_dir=out_dir, name="smoke_logs")

    trainer = lightning.Trainer(
        max_epochs=args.epochs,
        min_epochs=1,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision="32-true",
        callbacks=callbacks,
        logger=csv_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        default_root_dir=str(out_dir),
    )
    return trainer, strategy


def _post_training_rank0(args, out_dir, lit_model, dm, trainer, shm_cache_path, train_time, n_devices, effective_lr):
    """Rank-0 post-training: teardown DDP, predict, write summary."""
    logged = trainer.logged_metrics
    val_auc = float(logged.get("val_auc", 0))
    val_aupr = float(logged.get("val_aupr", 0))
    train_loss = float(logged.get("train_loss", 0))

    if dist.is_initialized():
        dist.destroy_process_group()

    if shm_cache_path.exists():
        shutil.rmtree(shm_cache_path, ignore_errors=True)
        logger.info("Cleaned up shared memory cache at %s", shm_cache_path)

    swa_ckpt = None
    if args.swa:
        swa_ckpt = str(out_dir / "smoke_swa_model.ckpt")
        torch.save(
            {"state_dict": lit_model.state_dict(), "hyper_parameters": dict(lit_model.hparams)},
            swa_ckpt,
        )
        logger.info("Saved SWA-averaged checkpoint: %s", swa_ckpt)

    predict_trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=True,
        default_root_dir=str(out_dir),
    )
    predictions = predict_trainer.predict(lit_model, datamodule=dm)
    n_preds = sum(b["probs"].shape[0] for b in predictions)
    logger.info("Predictions: %d samples (using %s model)", n_preds, "SWA" if args.swa else "best-ckpt")

    summary = {
        "devices": str(args.devices),
        "n_devices": n_devices,
        "world_size": n_devices,
        "epochs": args.epochs,
        "train_time_s": round(train_time, 1),
        "val_auc": val_auc,
        "val_aupr": val_aupr,
        "train_loss": train_loss,
        "n_predictions": n_preds,
        "effective_lr": effective_lr,
        "swa": args.swa,
        "swa_checkpoint": swa_ckpt,
    }
    summary_path = out_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    print(f"\nSMOKE TEST PASSED (devices={args.devices}, n_devices={n_devices}, val_auc={val_auc:.4f}, swa={args.swa})")


def main():
    ap = argparse.ArgumentParser(description="Multi-GPU smoke test")
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "small_cache/tensor_cache.pkl"))
    ap.add_argument("--devices", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--swa", action="store_true", help="Enable SWA")
    ap.add_argument("--swa-epoch-start", type=int, default=0, help="Epoch to start SWA")
    ap.add_argument("--output", default=os.path.join(tempfile.gettempdir(), "smoke_test_output"))
    args = ap.parse_args()

    out_dir, accelerator, devices, n_devices = _setup_environment(args)

    base_lr = 1e-3
    effective_lr = base_lr * math.sqrt(n_devices) if n_devices > 1 else base_lr
    if n_devices > 1:
        logger.info("LR scaling: %.6f -> %.6f (sqrt(%d))", base_lr, effective_lr, n_devices)

    full_cache, shm_cache_path = _load_cache(args, n_devices)

    from ugbio_srsnv.deep_srsnv.data_module import DataModuleConfig

    dm = SRSNVDataModule(
        full_cache=full_cache,
        train_split_ids={0},
        val_split_ids={1},
        test_split_ids={-1},
        loader_config=DataModuleConfig(train_batch_size=args.batch_size),
    )

    lit_model = SRSNVLightningModule(
        base_vocab_size=7,
        numeric_channels=10,
        tm_vocab_size=9,
        st_vocab_size=5,
        et_vocab_size=5,
        learning_rate=effective_lr,
        lr_scheduler="cosine",
    )

    n_params = sum(p.numel() for p in lit_model.model.parameters() if p.requires_grad)
    logger.info("Model params: %d", n_params)

    trainer, strategy = _build_trainer(args, out_dir, accelerator, devices, n_devices)

    logger.info("Starting training (strategy=%s)...", strategy)
    train_t0 = time.perf_counter()
    trainer.fit(lit_model, datamodule=dm)
    train_time = time.perf_counter() - train_t0
    logger.info("Training complete in %.1fs", train_time)

    if trainer.global_rank != 0:
        logger.info("Rank %d finished training. Cleaning up.", trainer.global_rank)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    _post_training_rank0(args, out_dir, lit_model, dm, trainer, shm_cache_path, train_time, n_devices, effective_lr)


if __name__ == "__main__":
    main()
