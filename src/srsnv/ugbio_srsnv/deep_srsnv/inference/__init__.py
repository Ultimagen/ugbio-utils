"""Inference package: model loading, engine wrappers, export, and VCF annotation.

Re-exports from ``model_loading`` for backward compatibility so that
``from ugbio_srsnv.deep_srsnv.inference import load_dnn_model_from_checkpoint``
continues to work.
"""

from ugbio_srsnv.deep_srsnv.inference.model_loading import (
    load_dnn_model_from_checkpoint,
    load_dnn_model_from_swa_checkpoint,
    load_dnn_models_from_metadata,
)

__all__ = [
    "load_dnn_model_from_checkpoint",
    "load_dnn_model_from_swa_checkpoint",
    "load_dnn_models_from_metadata",
]
