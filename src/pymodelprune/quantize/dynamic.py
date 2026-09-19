"""Dynamic INT8 quantization of Linear layers behind one interface, two backends.

`torch.ao.quantization` (the "legacy" backend) needs no extra install but is
deprecated. `torchao` is its successor and ships separately. Isolating both here
keeps that migration to a single file.

Only `nn.Linear` is quantized: dynamic quantization does not cover convolutions.
Whether INT8 is *faster* depends on the CPU kernels available; it is always smaller.
"""

from __future__ import annotations

import copy
import importlib.util
import warnings

import torch
from torch import nn

from pymodelprune.profile import weights_dtype

MODES = ("dynamic", "weight_only")
BACKENDS = ("auto", "torchao", "legacy")
_ENGINE_PREFERENCE = ("x86", "fbgemm", "qnnpack")


def torchao_available() -> bool:
    return importlib.util.find_spec("torchao") is not None


def select_quantized_engine() -> str:
    """Make sure a quantized kernel engine is active and return its name.

    On some platforms (Apple Silicon) PyTorch ships an engine but does not select
    it, and every quantized op fails with `NoQEngine` until one is chosen.
    """
    current = torch.backends.quantized.engine
    if current not in (None, "none"):
        return current
    supported = torch.backends.quantized.supported_engines
    for engine in _ENGINE_PREFERENCE:
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"this PyTorch build has no quantized engine (supported: {supported})")


def _quantize_legacy(model: nn.Module) -> nn.Module:
    select_quantized_engine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from torch.ao.quantization import quantize_dynamic as torch_quantize_dynamic
        except ImportError as error:
            raise RuntimeError(
                "torch.ao.quantization is gone from this PyTorch. "
                "Install the torchao backend: pip install 'pymodelprune[quant]'"
            ) from error
        return torch_quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def _quantize_torchao(model: nn.Module, mode: str) -> nn.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )

        config = (
            Int8DynamicActivationInt8WeightConfig() if mode == "dynamic" else Int8WeightOnlyConfig()
        )
        quantize_(model, config)
    # torchao decorates every converted Linear with an instance level `extra_repr`
    # functools.partial, which pickle rejects (no __name__), so torch.save(model)
    # would crash. It only affects print(model); drop it to keep the model savable.
    for module in model.modules():
        module.__dict__.pop("extra_repr", None)
    return model


def quantize_dynamic(model: nn.Module, mode: str = "dynamic", backend: str = "auto") -> nn.Module:
    """Return an INT8 copy of `model` (CPU). The input model is left untouched.

    mode="dynamic": INT8 weights, activations quantized on the fly per batch.
    mode="weight_only": INT8 weights, float activations (torchao only).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if weights_dtype(model) == "int8":
        raise ValueError("model is already quantized")
    if not any(isinstance(module, nn.Linear) for module in model.modules()):
        raise ValueError("model has no nn.Linear layers; dynamic quantization would do nothing")

    if backend == "auto":
        backend = "torchao" if torchao_available() else "legacy"
    if backend == "torchao" and not torchao_available():
        raise RuntimeError("torchao is not installed: pip install 'pymodelprune[quant]'")
    if backend == "legacy" and mode == "weight_only":
        raise ValueError("weight_only needs the torchao backend: pip install 'pymodelprune[quant]'")

    quantized = copy.deepcopy(model).cpu().eval()
    return (
        _quantize_legacy(quantized) if backend == "legacy" else _quantize_torchao(quantized, mode)
    )
