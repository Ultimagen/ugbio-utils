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

    channel_order = metadata.get("channel_order", [])
    numeric_channels = len(channel_order) if channel_order else 10

    model = CNNReadClassifier(
        base_vocab_size=len(encoders["base_vocab"]),
        numeric_channels=numeric_channels,
        tm_vocab_size=len(encoders.get("tm_vocab", {})) or 1,
        st_vocab_size=len(encoders.get("st_vocab", {})) or 1,
        et_vocab_size=len(encoders.get("et_vocab", {})) or 1,
    )
    state = torch.load(str(state_dict_path), map_location=map_location or "cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded DNN model from state dict: %s", state_dict_path)
    return model


def _model_kwargs_from_metadata(metadata: dict) -> dict:
    """Extract CNNReadClassifier constructor kwargs from metadata."""
    encoders = metadata["encoders"]
    channel_order = metadata.get("channel_order", [])
    training_params = metadata.get("training_parameters", {})
    return {
        "base_vocab_size": len(encoders["base_vocab"]),
        "numeric_channels": len(channel_order) if channel_order else 10,
        "tm_vocab_size": len(encoders.get("tm_vocab", {})) or 1,
        "st_vocab_size": len(encoders.get("st_vocab", {})) or 1,
        "et_vocab_size": len(encoders.get("et_vocab", {})) or 1,
        "hidden_channels": training_params.get("hidden_channels", 128),
        "n_blocks": training_params.get("n_blocks", 6),
        "base_embed_dim": training_params.get("base_embed_dim", 16),
        "ref_embed_dim": training_params.get("base_embed_dim", 16),
        "cat_embed_dim": training_params.get("cat_embed_dim", 4),
        "dropout": training_params.get("dropout", 0.3),
    }


def load_dnn_model_from_swa_checkpoint(
    ckpt_path: str | Path,
    metadata: dict,
    *,
    map_location: str | torch.device | None = None,
) -> SRSNVLightningModule:
    """Load a model from an SWA checkpoint (raw state-dict format).

    SWA checkpoints are saved as ``{"state_dict": ..., "hyper_parameters": ...}``
    rather than a full Lightning checkpoint, so we reconstruct the module from
    metadata and load the state dict directly.

    Parameters
    ----------
    ckpt_path
        Path to the SWA ``.ckpt`` file.
    metadata
        Parsed metadata dict (for model constructor kwargs).
    map_location
        Device mapping for ``torch.load``.

    Returns
    -------
    SRSNVLightningModule
        The loaded Lightning module in eval mode.
    """
    model_kwargs = _model_kwargs_from_metadata(metadata)
    training_params = metadata.get("training_parameters", {})
    lit_model = SRSNVLightningModule(
        learning_rate=training_params.get("learning_rate", 1e-3),
        weight_decay=training_params.get("weight_decay", 1e-4),
        lr_scheduler="none",
        **model_kwargs,
    )
    raw = torch.load(str(ckpt_path), map_location=map_location or "cpu", weights_only=False)  # noqa: S301
    lit_model.load_state_dict(raw["state_dict"])
    lit_model.eval()
    logger.info("Loaded DNN model from SWA checkpoint: %s", ckpt_path)
    return lit_model


def load_dnn_models_from_metadata(
    metadata_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    prefer_swa: bool | None = None,
) -> list[SRSNVLightningModule | CNNReadClassifier]:
    """Load all fold models referenced in a metadata JSON.

    Supports Lightning ``.ckpt`` checkpoints, raw ``.pt`` state dicts, and
    SWA checkpoints.

    Parameters
    ----------
    metadata_path
        Path to the ``srsnv_dnn_metadata.json``.
    map_location
        Device mapping for model loading.
    prefer_swa
        If ``True``, prefer SWA checkpoints over best checkpoints.
        If ``None`` (default), auto-detect from ``prediction_model`` field
        in the metadata.

    Returns
    -------
    list
        List of loaded models (one per fold), each in eval mode.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    use_swa = prefer_swa if prefer_swa is not None else (metadata.get("prediction_model") == "swa")

    if use_swa:
        swa_paths = metadata.get("swa_checkpoint_paths") or []
        swa_paths = [p for p in swa_paths if p]
        if swa_paths:
            models = []
            for swa_path in swa_paths:
                lit_model = load_dnn_model_from_swa_checkpoint(swa_path, metadata, map_location=map_location)
                models.append(lit_model)
            return models
        logger.warning("SWA checkpoints requested but not found in metadata; falling back to best checkpoints")

    ckpt_paths = metadata.get("best_checkpoint_paths", [])
    if ckpt_paths and all(ckpt_paths):
        models = []
        for ckpt_path in ckpt_paths:
            lit_model = load_dnn_model_from_checkpoint(ckpt_path, map_location=map_location)
            models.append(lit_model)
        return models

    training_results = metadata.get("training_results", [])
    model_dir = Path(metadata_path).parent
    model_kwargs = _model_kwargs_from_metadata(metadata)

    models = []
    for fold_result in training_results:
        fold_idx = fold_result["fold"]
        pt_path = model_dir / f"dnn_model_fold_{fold_idx}.pt"
        if pt_path.exists():
            model = CNNReadClassifier(**model_kwargs)
            state = torch.load(str(pt_path), map_location=map_location or "cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
            logger.info("Loaded DNN fold %d model from: %s", fold_idx, pt_path)
        else:
            logger.warning("No model file found for fold %d at %s", fold_idx, pt_path)
    return models
