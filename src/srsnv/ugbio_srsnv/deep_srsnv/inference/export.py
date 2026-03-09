"""ONNX export and TensorRT serialization for the DNN read classifier."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch
from ugbio_core.logger import logger

from ugbio_srsnv.deep_srsnv.cnn_model import CNNReadClassifier


def export_to_onnx(
    model: CNNReadClassifier,
    output_path: str | Path,
    tensor_length: int = 300,
    batch_size: int = 1,
    opset_version: int = 17,
) -> Path:
    """Export a trained CNNReadClassifier to ONNX format.

    Parameters
    ----------
    model
        The trained model (should already be on CPU in eval mode).
    output_path
        Destination path for the ``.onnx`` file.
    tensor_length
        Fixed sequence length (L) used during training.
    batch_size
        Batch size for the dummy input (only affects the trace; the
        exported model will accept dynamic batch sizes).
    opset_version
        ONNX opset version.

    Returns
    -------
    Path
        The written ONNX file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu().eval()

    dummy_inputs = _build_dummy_inputs(model, batch_size, tensor_length)

    input_names = list(dummy_inputs.keys())
    output_names = ["logits"]

    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    args_tuple = tuple(dummy_inputs.values())

    with torch.no_grad():
        torch.onnx.export(
            model,
            args_tuple,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    try:
        import onnx  # noqa: PLC0415

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully: %s", output_path)
    except ImportError:
        logger.warning("onnx package not installed; skipping ONNX validation")
    except Exception:
        logger.exception("ONNX validation failed")
        raise

    logger.info("Exported ONNX model to %s", output_path)
    return output_path


def _build_dummy_inputs(
    model: CNNReadClassifier,
    batch_size: int,
    tensor_length: int,
) -> dict[str, torch.Tensor]:
    """Build dummy input tensors matching the model forward() signature."""
    inputs: dict[str, torch.Tensor] = {
        "read_base_idx": torch.zeros(batch_size, tensor_length, dtype=torch.long),
        "ref_base_idx": torch.zeros(batch_size, tensor_length, dtype=torch.long),
        "t0_idx": torch.zeros(batch_size, tensor_length, dtype=torch.long),
        "x_num": torch.zeros(batch_size, 9, tensor_length, dtype=torch.float32),
        "mask": torch.ones(batch_size, tensor_length, dtype=torch.float32),
    }
    if hasattr(model, "tm_emb"):
        inputs["tm_idx"] = torch.zeros(batch_size, dtype=torch.long)
    if hasattr(model, "st_emb"):
        inputs["st_idx"] = torch.zeros(batch_size, dtype=torch.long)
    if hasattr(model, "et_emb"):
        inputs["et_idx"] = torch.zeros(batch_size, dtype=torch.long)
    return inputs


def build_trtexec_command(
    onnx_path: str | Path,
    engine_path: str | Path,
    tensor_length: int = 300,
    min_batch: int = 1,
    opt_batch: int = 256,
    max_batch: int = 1024,
    fp16: bool = True,  # noqa: FBT001, FBT002
) -> list[str]:
    """Build the trtexec CLI command for serializing an ONNX model.

    Returns the command as a list of strings suitable for ``subprocess.run``.
    """
    seq_shapes = {
        "read_base_idx": f"{{B}}x{tensor_length}",
        "ref_base_idx": f"{{B}}x{tensor_length}",
        "t0_idx": f"{{B}}x{tensor_length}",
        "x_num": f"{{B}}x9x{tensor_length}",
        "mask": f"{{B}}x{tensor_length}",
        "tm_idx": "{B}",
        "st_idx": "{B}",
        "et_idx": "{B}",
    }

    def _shape_str(batch: int) -> str:
        parts = []
        for name, tmpl in seq_shapes.items():
            parts.append(f"{name}:{tmpl.format(B=batch)}")
        return ",".join(parts)

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={_shape_str(min_batch)}",
        f"--optShapes={_shape_str(opt_batch)}",
        f"--maxShapes={_shape_str(max_batch)}",
    ]
    if fp16:
        cmd.append("--fp16")
    return cmd


def serialize_with_trtexec(
    onnx_path: str | Path,
    engine_path: str | Path,
    tensor_length: int = 300,
    min_batch: int = 1,
    opt_batch: int = 256,
    max_batch: int = 1024,
    fp16: bool = True,  # noqa: FBT001, FBT002
) -> Path | None:
    """Run trtexec to convert an ONNX model to a TensorRT engine.

    Returns
    -------
    Path or None
        The engine path if successful, ``None`` if trtexec is not available.
    """
    if not shutil.which("trtexec"):
        logger.warning("trtexec not found on PATH; skipping TensorRT serialization")
        return None

    cmd = build_trtexec_command(
        onnx_path,
        engine_path,
        tensor_length,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        fp16=fp16,
    )
    logger.info("Running trtexec: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        logger.error("trtexec failed (rc=%d):\nstdout: %s\nstderr: %s", result.returncode, result.stdout, result.stderr)
        raise RuntimeError(f"trtexec failed with return code {result.returncode}")

    engine_path = Path(engine_path)
    logger.info("TensorRT engine saved to %s (%.1f MB)", engine_path, engine_path.stat().st_size / 1e6)
    return engine_path
