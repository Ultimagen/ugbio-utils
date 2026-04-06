"""Inference engine wrappers: TensorRT (production) and PyTorch (debug/fallback).

Both classes expose the same ``predict_batch(batch) -> np.ndarray`` interface so
the rest of the inference pipeline is backend-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from ugbio_core.logger import logger

if TYPE_CHECKING:
    from torch import nn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compose_x_num(batch: dict) -> np.ndarray:
    """Build the (B, 9, L) x_num array from split pos/const representations.

    Handles both numpy arrays and torch tensors in the input batch.
    """
    x_pos = batch["x_num_pos"]
    x_const = batch["x_num_const"]

    if isinstance(x_pos, torch.Tensor):
        x_pos = x_pos.numpy()
    if isinstance(x_const, torch.Tensor):
        x_const = x_const.numpy()

    x_pos = x_pos.astype(np.float32)
    x_const = x_const.astype(np.float32)

    b, _, seq_len = x_pos.shape
    x_const_expanded = np.broadcast_to(x_const[:, :, np.newaxis], (b, x_const.shape[1], seq_len))
    return np.concatenate([x_pos, x_const_expanded], axis=1)


def _to_numpy(tensor_or_array, dtype=np.float32) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.numpy().astype(dtype)
    return np.asarray(tensor_or_array, dtype=dtype)


def _to_numpy_long(tensor_or_array) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.numpy().astype(np.int64)
    return np.asarray(tensor_or_array, dtype=np.int64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# TensorRT Engine
# ---------------------------------------------------------------------------


class TRTEngine:
    """TensorRT inference engine bound to a specific GPU.

    Uses pre-allocated pinned host buffers and double-buffered CUDA streams
    for overlapping data transfer with computation.
    """

    def __init__(self, engine_path: str, device_id: int = 0, max_batch_size: int = 1024):
        try:
            import pycuda.driver as cuda  # noqa: PLC0415
            import tensorrt as trt  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "tensorrt and pycuda are required for TRTEngine. " "Install with: pip install tensorrt pycuda"
            ) from e

        cuda.init()
        self._cuda = cuda
        self._device_id = device_id
        self._cuda_ctx = cuda.Device(device_id).make_context()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._input_dtypes: dict[str, np.dtype] = {}
        self._input_shapes: dict[str, tuple] = {}
        self._output_shape: tuple = ()
        self._output_dtype: np.dtype = np.float32

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            shape = self._engine.get_tensor_profile_shape(name, 0) if mode == trt.TensorIOMode.INPUT else None
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                self._input_dtypes[name] = dtype
                self._input_shapes[name] = shape  # (min, opt, max) tuple
            else:
                self._output_names.append(name)
                self._output_dtype = dtype

        self._max_batch = max_batch_size

        self._streams = [cuda.Stream(), cuda.Stream()]
        self._active_stream = 0

        self._host_inputs: dict[str, np.ndarray] = {}
        self._device_inputs: dict[str, object] = {}
        self._host_output: np.ndarray | None = None
        self._device_output: object = None
        self._buffers_allocated = False

        logger.info(
            "TRTEngine initialized on GPU:%d from %s (inputs=%s, outputs=%s)",
            device_id,
            engine_path,
            self._input_names,
            self._output_names,
        )

    def _ensure_buffers(self, batch_size: int) -> None:
        """Allocate or reallocate pinned host + device buffers if needed."""
        if self._buffers_allocated and batch_size <= self._max_batch:
            return

        cuda = self._cuda
        alloc_batch = max(batch_size, self._max_batch)

        for name in self._input_names:
            opt_shape = self._input_shapes[name][1]
            shape = (alloc_batch,) + opt_shape[1:]
            nbytes = int(np.prod(shape)) * np.dtype(self._input_dtypes[name]).itemsize
            self._host_inputs[name] = cuda.pagelocked_empty(shape, self._input_dtypes[name])
            self._device_inputs[name] = cuda.mem_alloc(nbytes)

        out_nbytes = alloc_batch * np.dtype(self._output_dtype).itemsize
        self._host_output = cuda.pagelocked_empty((alloc_batch,), self._output_dtype)
        self._device_output = cuda.mem_alloc(out_nbytes)

        self._max_batch = alloc_batch
        self._buffers_allocated = True

    def predict_batch(self, batch: dict) -> np.ndarray:
        """Run inference on a batch dict and return sigmoid probabilities."""
        cuda = self._cuda

        x_num = _compose_x_num(batch)
        actual_batch = x_num.shape[0]
        self._ensure_buffers(actual_batch)

        inputs_np = {
            "read_base_idx": _to_numpy_long(batch["read_base_idx"]),
            "ref_base_idx": _to_numpy_long(batch["ref_base_idx"]),
            "t0_idx": _to_numpy_long(batch["t0_idx"]),
            "x_num": x_num,
            "mask": _to_numpy(batch["mask"]),
        }
        if "tm_idx" in batch:
            inputs_np["tm_idx"] = _to_numpy_long(batch["tm_idx"])
        if "st_idx" in batch:
            inputs_np["st_idx"] = _to_numpy_long(batch["st_idx"])
        if "et_idx" in batch:
            inputs_np["et_idx"] = _to_numpy_long(batch["et_idx"])

        stream = self._streams[self._active_stream]

        for name in self._input_names:
            if name not in inputs_np:
                continue
            arr = inputs_np[name]
            host_buf = self._host_inputs[name]
            host_buf[:actual_batch] = arr[:actual_batch]

            shape = (actual_batch,) + arr.shape[1:]
            self._context.set_input_shape(name, shape)
            cuda.memcpy_htod_async(self._device_inputs[name], host_buf[:actual_batch], stream)
            self._context.set_tensor_address(name, int(self._device_inputs[name]))

        for out_name in self._output_names:
            self._context.set_tensor_address(out_name, int(self._device_output))

        self._context.execute_async_v3(stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(self._host_output[:actual_batch], self._device_output, stream)
        stream.synchronize()

        logits = self._host_output[:actual_batch].copy()
        self._active_stream = 1 - self._active_stream

        return _sigmoid(logits)

    def predict_batch_prepared(self, batch: dict) -> np.ndarray:
        """Run inference on a pre-prepared batch (x_num already composed, dtypes already converted).

        Skips ``_compose_x_num`` and dtype conversions — the caller is responsible
        for providing arrays in the correct dtype (int64 for indices, float32 for x_num/mask).
        """
        cuda = self._cuda
        actual_batch = batch["x_num"].shape[0]
        self._ensure_buffers(actual_batch)

        stream = self._streams[self._active_stream]

        for name in self._input_names:
            if name not in batch:
                continue
            arr = batch[name]
            host_buf = self._host_inputs[name]
            host_buf[:actual_batch] = arr[:actual_batch]

            shape = (actual_batch,) + arr.shape[1:]
            self._context.set_input_shape(name, shape)
            cuda.memcpy_htod_async(self._device_inputs[name], host_buf[:actual_batch], stream)
            self._context.set_tensor_address(name, int(self._device_inputs[name]))

        for out_name in self._output_names:
            self._context.set_tensor_address(out_name, int(self._device_output))

        self._context.execute_async_v3(stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(self._host_output[:actual_batch], self._device_output, stream)
        stream.synchronize()

        logits = self._host_output[:actual_batch].copy()
        self._active_stream = 1 - self._active_stream

        return _sigmoid(logits)

    def push_context(self) -> None:
        """Push CUDA context onto the current thread's stack (for multi-threaded use)."""
        self._cuda_ctx.push()

    def pop_context(self) -> None:
        """Pop CUDA context from the current thread's stack."""
        self._cuda_ctx.pop()

    def close(self) -> None:
        """Release resources and detach the CUDA context."""
        self._host_inputs.clear()
        self._device_inputs.clear()
        self._host_output = None
        self._device_output = None
        self._context = None
        self._engine = None
        if self._cuda_ctx is not None:
            try:
                self._cuda_ctx.detach()
            except Exception:  # noqa: S110
                pass
            self._cuda_ctx = None
        logger.info("TRTEngine on GPU:%d closed", self._device_id)


# ---------------------------------------------------------------------------
# PyTorch Engine (debug / fallback)
# ---------------------------------------------------------------------------


class PyTorchEngine:
    """PyTorch-based inference engine with the same ``predict_batch`` API.

    Useful for validating TRT outputs and for environments without TensorRT.
    """

    def __init__(self, model: nn.Module, device: str = "cuda:0"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> np.ndarray:
        """Run inference and return sigmoid probabilities as numpy."""
        x_num_np = _compose_x_num(batch)
        x_num = torch.from_numpy(x_num_np).to(self.device)

        inputs = {
            "read_base_idx": torch.as_tensor(_to_numpy_long(batch["read_base_idx"])).to(self.device),
            "ref_base_idx": torch.as_tensor(_to_numpy_long(batch["ref_base_idx"])).to(self.device),
            "t0_idx": torch.as_tensor(_to_numpy_long(batch["t0_idx"])).to(self.device),
            "x_num": x_num,
            "mask": torch.from_numpy(_to_numpy(batch["mask"])).to(self.device),
        }
        if "tm_idx" in batch:
            inputs["tm_idx"] = torch.as_tensor(_to_numpy_long(batch["tm_idx"])).to(self.device)
        if "st_idx" in batch:
            inputs["st_idx"] = torch.as_tensor(_to_numpy_long(batch["st_idx"])).to(self.device)
        if "et_idx" in batch:
            inputs["et_idx"] = torch.as_tensor(_to_numpy_long(batch["et_idx"])).to(self.device)

        logits = self.model(**inputs)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def close(self) -> None:
        """No-op for symmetry with TRTEngine."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_inference_engine(
    metadata_path: str | Path,
    *,
    backend: str = "trt",
    device_id: int = 0,
    engine_path: str | None = None,
    checkpoint_path: str | None = None,
    max_batch_size: int = 1024,
) -> TRTEngine | PyTorchEngine:
    """Create an inference engine from model metadata.

    Parameters
    ----------
    metadata_path
        Path to ``srsnv_dnn_metadata.json``.
    backend
        ``"trt"`` for TensorRT or ``"pytorch"`` for PyTorch.
    device_id
        GPU device index.
    engine_path
        Override path to the ``.engine`` file (TRT backend).
    checkpoint_path
        Override path to a ``.ckpt`` file (PyTorch backend).
    max_batch_size
        Pre-allocated max batch size for TRT buffers.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    if backend == "trt":
        path = engine_path or metadata.get("trt_engine_path")
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"TRT engine not found at {path}. Run training with ONNX/TRT export, "
                f"or pass --engine-path explicitly."
            )
        return TRTEngine(str(path), device_id=device_id, max_batch_size=max_batch_size)

    if backend == "pytorch":
        from ugbio_srsnv.deep_srsnv.inference.model_loading import (  # noqa: PLC0415
            load_dnn_model_from_checkpoint,
            load_dnn_model_from_swa_checkpoint,
        )

        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        ckpt = checkpoint_path
        prefer_swa = metadata.get("prediction_model") == "swa"
        if not ckpt:
            if prefer_swa:
                swa_paths = metadata.get("swa_checkpoint_paths") or []
                ckpt = swa_paths[0] if swa_paths else None
            if not ckpt:
                best_paths = metadata.get("best_checkpoint_paths") or []
                ckpt = best_paths[0] if best_paths else None
        if not ckpt:
            raise FileNotFoundError("No checkpoint found in metadata for PyTorch backend")

        raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)  # noqa: S301
        is_swa_format = isinstance(raw, dict) and "state_dict" in raw and "pytorch-lightning_version" not in raw
        if is_swa_format:
            lit_model = load_dnn_model_from_swa_checkpoint(ckpt, metadata, map_location=device)
        else:
            lit_model = load_dnn_model_from_checkpoint(ckpt, map_location=device)

        return PyTorchEngine(lit_model.model, device=device)

    raise ValueError(f"Unknown backend: {backend!r}. Use 'trt' or 'pytorch'.")
