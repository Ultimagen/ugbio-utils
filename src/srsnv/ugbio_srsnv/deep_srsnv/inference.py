"""Utilities for loading trained DNN Lightning models and running inference."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from ugbio_core.logger import logger

from ugbio_srsnv.deep_srsnv.cnn_model import CNNReadClassifier
from ugbio_srsnv.deep_srsnv.lightning_module import SRSNVLightningModule


def load_dnn_model_from_checkpoint(
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> SRSNVLightningModule:
    """Load a trained SRSNVLightningModule from a ``.ckpt`` file.

    Parameters
    ----------
    ckpt_path
        Path to the Lightning checkpoint file.
    map_location
        Device mapping for ``torch.load`` (e.g. ``"cpu"``).

    Returns
    -------
    SRSNVLightningModule
        The loaded Lightning module in eval mode.
    """
    lit_model = SRSNVLightningModule.load_from_checkpoint(
        str(ckpt_path),
        map_location=map_location or "cpu",
    )
    lit_model.eval()
    logger.info("Loaded DNN Lightning model from checkpoint: %s", ckpt_path)
    return lit_model


def load_dnn_model_from_state_dict(
    state_dict_path: str | Path,
    metadata_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> CNNReadClassifier:
    """Load a CNNReadClassifier from a raw ``.pt`` state dict file.

    Parameters
    ----------
    state_dict_path
        Path to the ``.pt`` state dict file.
    metadata_path
        Path to the ``srsnv_dnn_metadata.json`` for encoder vocab sizes.
    map_location
        Device mapping for ``torch.load``.

    Returns
    -------
    CNNReadClassifier
        The loaded model in eval mode.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)
    encoders = metadata["encoders"]

    model = CNNReadClassifier(
        base_vocab_size=len(encoders["base_vocab"]),
        t0_vocab_size=len(encoders["t0_vocab"]),
        numeric_channels=12,
    )
    state = torch.load(str(state_dict_path), map_location=map_location or "cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded DNN model from state dict: %s", state_dict_path)
    return model


def load_dnn_models_from_metadata(
    metadata_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> list[SRSNVLightningModule | CNNReadClassifier]:
    """Load all fold models referenced in a metadata JSON.

    Supports both Lightning ``.ckpt`` checkpoints and raw ``.pt`` state dicts.

    Parameters
    ----------
    metadata_path
        Path to the ``srsnv_dnn_metadata.json``.
    map_location
        Device mapping for model loading.

    Returns
    -------
    list
        List of loaded models (one per fold), each in eval mode.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    ckpt_paths = metadata.get("best_checkpoint_paths", [])
    if ckpt_paths and all(ckpt_paths):
        models = []
        for ckpt_path in ckpt_paths:
            lit_model = load_dnn_model_from_checkpoint(ckpt_path, map_location=map_location)
            models.append(lit_model)
        return models

    training_results = metadata.get("training_results", [])
    model_dir = Path(metadata_path).parent
    encoders = metadata["encoders"]

    models = []
    for fold_result in training_results:
        fold_idx = fold_result["fold"]
        pt_path = model_dir / f"dnn_model_fold_{fold_idx}.pt"
        if pt_path.exists():
            model = CNNReadClassifier(
                base_vocab_size=len(encoders["base_vocab"]),
                t0_vocab_size=len(encoders["t0_vocab"]),
                numeric_channels=12,
            )
            state = torch.load(str(pt_path), map_location=map_location or "cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
            logger.info("Loaded DNN fold %d model from: %s", fold_idx, pt_path)
        else:
            logger.warning("No model file found for fold %d at %s", fold_idx, pt_path)
    return models
