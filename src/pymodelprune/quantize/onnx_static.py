"""Static INT8 quantization through ONNX Runtime.

Static means activation ranges are measured once, on calibration data, instead of
on every batch. Unlike dynamic quantization it covers convolutions, and ONNX Runtime
ships hand tuned INT8 kernels for x86 and ARM, so this is where INT8 becomes *faster*
and not just smaller.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from pymodelprune.evaluate import reusable
from pymodelprune.export import require_onnx
from pymodelprune.fidelity import model_inputs

CALIBRATION_METHODS = {"minmax": "MinMax", "entropy": "Entropy", "percentile": "Percentile"}
"""Our names -> members of `onnxruntime.quantization.CalibrationMethod`."""


def check_calibration_method(method: str) -> None:
    if method not in CALIBRATION_METHODS:
        choices = " | ".join(CALIBRATION_METHODS)
        raise ValueError(f"calibration_method must be one of {choices}, got {method!r}")


def _calibration_reader(path: Path, batches: Iterable, max_batches: int):
    import onnxruntime
    from onnxruntime.quantization import CalibrationDataReader

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.iterator = self._feeds()

        def _feeds(self):
            for index, batch in enumerate(batches):
                if index >= max_batches:
                    return
                # multi-input models: a batch carries one tensor per graph input, labels after
                tensors = model_inputs(batch, len(input_names))
                yield {
                    name: tensor.detach().cpu().numpy()
                    for name, tensor in zip(input_names, tensors, strict=True)
                }

        def get_next(self):
            return next(self.iterator, None)

    return Reader()


def quantize_onnx_static(
    onnx_path: str | Path,
    output_path: str | Path,
    calibration: torch.Tensor | Iterable,
    max_batches: int = 16,
    per_channel: bool = True,
    exclude_nodes: Sequence[str] = (),
    calibration_method: str = "minmax",
) -> Path:
    """Quantize a float ONNX model to INT8 and return the new file's path.

    calibration: representative inputs. One tensor, or an iterable of batches
        (`(inputs, labels)` batches work, labels are ignored; a model with N inputs
        takes the first N items of every batch). A few hundred samples are enough;
        they only set the activation ranges.
    exclude_nodes: ONNX node names to keep in float (mixed precision) when a layer
        turns out to be too sensitive for INT8.
    calibration_method: how activation ranges are read off the calibration data.
        "minmax" takes the extremes, so one outlier stretches the range and costs
        resolution; "entropy" and "percentile" build histograms (slower, more memory)
        and clip outliers, which can recover accuracy on long-tailed activations.
    """
    check_calibration_method(calibration_method)
    require_onnx()
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    onnx_path, output_path = Path(onnx_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batches = [calibration] if isinstance(calibration, torch.Tensor) else reusable(calibration)

    with tempfile.TemporaryDirectory() as workdir:
        prepared = Path(workdir) / "prepared.onnx"
        quant_pre_process(str(onnx_path), str(prepared))
        quantize_static(
            str(prepared),
            str(output_path),
            _calibration_reader(prepared, batches, max_batches),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            nodes_to_exclude=list(exclude_nodes),
            calibrate_method=getattr(CalibrationMethod, CALIBRATION_METHODS[calibration_method]),
        )
    return output_path
