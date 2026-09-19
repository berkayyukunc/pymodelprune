"""Save optimized models and move them to ONNX Runtime, with verification."""

from __future__ import annotations

import contextlib
import copy
import gzip
import io
import os
import re
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Self

import torch
from torch import nn

from pymodelprune.evaluate import EvalFn
from pymodelprune.fidelity import as_inputs, compare_outputs, fidelity_batches
from pymodelprune.profile import ModelProfile, count_flops, count_parameters, measure_latency

_ANSI_CODES = re.compile(r"\x1b\[[0-9;]*m")


class OnnxMismatchError(RuntimeError):
    """The exported ONNX model does not reproduce the PyTorch outputs."""


def require_onnx() -> None:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "ONNX support is not installed: pip install 'pymodelprune[onnx]'"
        ) from error


def save_model(model: nn.Module, path: str | Path) -> Path:
    """Save the full model object (architecture + weights) with `torch.save`.

    Structurally pruned models have new layer shapes, so a bare state_dict would
    not load into the original architecture. Loading needs `weights_only=False`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pickling can fail half way (unpicklable attribute). Write next to the target and
    # rename, so a failure never leaves a truncated file, nor destroys an older good one.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(model, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class OrtModule(nn.Module):
    """An ONNX Runtime session that looks like a module, so the same profiling,
    accuracy and fidelity code runs on it unchanged."""

    def __init__(self, path: str | Path, threads: int | None = None) -> None:
        super().__init__()
        require_onnx()
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads or torch.get_num_threads()
        self.path = Path(path)
        self.session = onnxruntime.InferenceSession(
            str(self.path), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = [item.name for item in self.session.get_inputs()]

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.session is None:
            raise RuntimeError(f"the session for {self.path.name} is closed")
        feeds = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(self.input_names, inputs, strict=True)
        }
        return torch.from_numpy(self.session.run(None, feeds)[0])

    def close(self) -> None:
        """Release the session and its thread pool now.

        Left to the garbage collector, a session can still be alive when the interpreter
        starts tearing down, and racing ONNX Runtime's threads against that shutdown has
        crashed processes after all of their work was already done.
        """
        self.session = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _reason(error: BaseException) -> str:
    """First line of the root cause, without terminal colour codes. Exporter errors wrap
    the real problem in a screen of generic advice."""
    while error.__cause__ is not None:
        error = error.__cause__
    lines = _ANSI_CODES.sub("", str(error)).strip().splitlines()
    return f"{type(error).__name__}: {lines[0]}" if lines else type(error).__name__


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Both exporters print progress and warn about internals the caller cannot act on."""
    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        warnings.simplefilter("ignore")
        yield


def _export(model: nn.Module, inputs: tuple, path: Path, dynamic_batch: bool) -> None:
    # torch.export specializes dimensions of size 0/1, so a dynamic batch axis has to
    # be traced with a batch of at least 2.
    if dynamic_batch:
        inputs = tuple(torch.cat([item, item]) if item.shape[0] == 1 else item for item in inputs)
    failure = None
    with _quiet():
        try:
            shapes = (
                tuple({0: torch.export.Dim("batch")} for _ in inputs) if dynamic_batch else None
            )
            torch.onnx.export(
                model, inputs, str(path), dynamo=True, dynamic_shapes=shapes, external_data=False
            )
        except Exception as error:  # noqa: BLE001 - the dynamo exporter raises many unrelated types
            failure = error
    if failure is None:
        return

    # Outside `_quiet()` on purpose: the tracing exporter records only the path this
    # example took, so Python control flow is frozen. The caller has to hear about it.
    warnings.warn(
        f"the dynamo ONNX exporter failed ({_reason(failure)}); falling back to the tracing exporter, "
        "which freezes Python control flow (if/for on tensor values) into the graph",
        stacklevel=3,
    )
    names = [f"input_{index}" for index in range(len(inputs))]
    axes = {name: {0: "batch"} for name in names} if dynamic_batch else None
    with _quiet():
        torch.onnx.export(
            model, inputs, str(path), dynamo=False, input_names=names, dynamic_axes=axes
        )


def _cpu_copy(model: nn.Module) -> nn.Module:
    """ONNX Runtime answers on the CPU; an MPS/CUDA model is compared through a CPU copy."""
    return copy.deepcopy(model).cpu().eval()


def _cpu_inputs(example_input: torch.Tensor | tuple) -> tuple:
    return tuple(item.detach().cpu() for item in as_inputs(example_input))


def _verify(
    model: nn.Module, path: Path, inputs: tuple, atol: float, rtol: float, strict: bool = True
) -> float:
    """Compare a CPU eval model with the ONNX file on CPU inputs.

    `strict=False` is for probe inputs we made up ourselves: when PyTorch itself
    returns non-finite values there, the probe left the model's domain and proves nothing.
    """
    with torch.inference_mode():
        expected = model(*inputs)
    if not isinstance(expected, torch.Tensor):
        raise TypeError(
            f"only single-tensor outputs are supported, model returned {type(expected).__name__}"
        )
    if not strict and not bool(torch.isfinite(expected).all()):
        return 0.0
    try:
        with OrtModule(path) as session:
            actual = session(*inputs)
    except Exception as error:
        shapes = [tuple(item.shape) for item in inputs]
        raise OnnxMismatchError(
            f"ONNX Runtime failed on input shapes {shapes}: {_reason(error)}"
        ) from error
    if actual.shape != expected.shape:
        raise OnnxMismatchError(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    difference = float((actual - expected).abs().max()) if expected.numel() else 0.0
    tolerance = atol + rtol * (float(expected.abs().max()) if expected.numel() else 0.0)
    # `not <=` instead of `>`: a NaN difference compares False both ways and must fail
    if not difference <= tolerance:
        raise OnnxMismatchError(
            f"max abs difference {difference:.3g} exceeds atol={atol} + rtol={rtol} * max|output|"
        )
    return difference


def verify_onnx(
    model: nn.Module,
    path: str | Path,
    example_input: torch.Tensor | tuple,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> float:
    """Return the largest output difference between PyTorch and ONNX Runtime, or raise.

    Passes when `diff <= atol + rtol * max|expected|`, so models with large logits are
    not failed for float noise. The model and inputs may live on any device.
    """
    return _verify(_cpu_copy(model), Path(path), _cpu_inputs(example_input), atol, rtol)


def _probe_inputs(inputs: tuple, dynamic_batch: bool) -> list[tuple]:
    """Extra inputs that expose what a single example cannot.

    The negated example takes the other side of value dependent branches a tracer
    froze; a batch of another size catches a batch dimension baked into the graph.
    Integer tensors (token ids, indices) are not negated: that would leave their domain.
    """
    probes = [tuple(-item if item.is_floating_point() else item for item in inputs)]
    if dynamic_batch:
        size = 3 if inputs[0].shape[0] != 3 else 2
        probes.append(tuple(torch.cat([item] * 3)[:size] for item in inputs))
    return probes


def to_onnx(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    path: str | Path,
    verify: bool = True,
    atol: float = 1e-4,
    dynamic_batch: bool = True,
    rtol: float = 1e-4,
) -> Path:
    """Export to ONNX and, by default, prove the file reproduces the PyTorch outputs.

    Verification runs on the example, on its negation and (with `dynamic_batch`) on a
    batch of another size. A file that fails is deleted before the error is raised.
    """
    require_onnx()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = _cpu_inputs(example_input)
    _export(_cpu_copy(model), inputs, path, dynamic_batch)
    if verify:
        # a fresh copy: the comparison must not depend on what an exporter did to its model
        reference = _cpu_copy(model)
        try:
            _verify(reference, path, inputs, atol, rtol)
            for probe in _probe_inputs(inputs, dynamic_batch):
                _verify(reference, path, probe, atol, rtol, strict=False)
        except (OnnxMismatchError, TypeError):
            path.unlink(missing_ok=True)  # never leave a file behind that is known to be wrong
            raise
    return path


def onnx_is_quantized(path: str | Path) -> bool:
    import onnx

    quantized_ops = {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"}
    return any(node.op_type in quantized_ops for node in onnx.load(str(path)).graph.node)


def profile_onnx(
    path: str | Path,
    example_input: torch.Tensor | tuple,
    name: str = "onnx",
    source_model: nn.Module | None = None,
    warmup: int = 10,
    runs: int = 100,
    eval_fn: EvalFn | None = None,
    reference: nn.Module | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
    session: OrtModule | None = None,
) -> ModelProfile:
    """Profile an ONNX file with ONNX Runtime. Size columns describe the file on disk.

    Parameter counts (and FLOPs, for float models) are taken from `source_model`,
    the PyTorch model the file was exported from, since ONNX files do not expose them.
    `session` is an already open `OrtModule` for `path`; pass it to keep using the same
    session afterwards (e.g. for `retime_profiles`) instead of building a second one.
    """
    path = Path(path)
    own_session = session is None
    module = OrtModule(path) if own_session else session
    raw = path.read_bytes()
    quantized = onnx_is_quantized(path)
    total, nonzero = count_parameters(source_model) if source_model is not None else (0, 0)
    flops = None
    if source_model is not None and not quantized:
        flops = count_flops(source_model, example_input)

    fidelity = None
    if reference is not None:
        batches, num_inputs = fidelity_batches(example_input, fidelity_inputs)
        fidelity = compare_outputs(reference, module, batches, num_inputs)

    profile = ModelProfile(
        name=name,
        total_params=total,
        nonzero_params=nonzero,
        size_mb=len(raw) / 1e6,
        compressed_mb=len(gzip.compress(raw, compresslevel=6)) / 1e6,
        flops=flops,
        dtype="int8" if quantized else "fp32",
        latency=measure_latency(module, example_input, warmup=warmup, runs=runs),
        accuracy=float(eval_fn(module)) if eval_fn is not None else None,
        fidelity=fidelity,
    )
    if own_session:
        module.close()
    return profile
