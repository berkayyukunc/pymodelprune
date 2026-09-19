"""Measure a model: parameters, sparsity, size, FLOPs and inference latency."""

from __future__ import annotations

import contextlib
import dataclasses
import gzip
import io
import statistics
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from pymodelprune.evaluate import EvalFn
from pymodelprune.fidelity import Fidelity, as_inputs, compare_outputs, fidelity_batches


@dataclass(frozen=True)
class LatencyStats:
    median_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float
    runs: int


@dataclass(frozen=True)
class ModelProfile:
    name: str
    total_params: int
    nonzero_params: int
    size_mb: float
    latency: LatencyStats
    compressed_mb: float | None = None
    flops: int | None = None
    dtype: str = "fp32"
    accuracy: float | None = None
    fidelity: Fidelity | None = None

    @property
    def sparsity(self) -> float:
        """Fraction of parameters that are exactly zero (0.0 = dense)."""
        if self.total_params == 0:
            return 0.0
        return 1.0 - self.nonzero_params / self.total_params


def _is_packed_quantized(module: nn.Module) -> bool:
    """Legacy quantized layers hide their weight behind `weight()` and `_packed_params`."""
    return hasattr(module, "_packed_params") and callable(getattr(module, "weight", None))


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_quantized or type(tensor) not in (torch.Tensor, nn.Parameter)


def _dense(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.dequantize() if _is_quantized_tensor(tensor) else tensor


def effective_parameters(model: nn.Module) -> Iterator[torch.Tensor]:
    """Yield the weight tensors the forward pass really uses.

    `model.parameters()` is not enough: while `torch.nn.utils.prune` masks are
    attached it yields the unmasked `weight_orig`, and legacy quantized layers
    keep their weights outside the parameter list altogether.
    """
    seen: set[int] = set()
    for module in model.modules():
        if _is_packed_quantized(module):
            yield module.weight()
            bias = module.bias()
            if bias is not None:
                yield bias
            continue
        for name, param in module.named_parameters(recurse=False):
            if id(param) in seen:
                continue
            seen.add(id(param))
            masked_name = name.removesuffix("_orig")
            if masked_name != name and hasattr(module, f"{masked_name}_mask"):
                yield getattr(module, masked_name)
            else:
                yield param


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, nonzero) counts over the effective parameters."""
    total = 0
    nonzero = 0
    for tensor in effective_parameters(model):
        total += tensor.numel()
        nonzero += int(torch.count_nonzero(_dense(tensor.detach())))
    return total, nonzero


def weights_dtype(model: nn.Module) -> str:
    """'int8' if any weight is stored quantized, else the dtype of the first weight."""
    first = None
    for tensor in effective_parameters(model):
        if _is_quantized_tensor(tensor):
            return "int8"
        first = first if first is not None else tensor
    if first is None:
        return "-"
    return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(
        first.dtype, str(first.dtype).removeprefix("torch.")
    )


def _serialized_state(model: nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _sizes_mb(state: bytes) -> tuple[float, float]:
    """(raw, gzipped) size in MB of already serialized weights."""
    return len(state) / 1e6, len(gzip.compress(state, compresslevel=6)) / 1e6


def model_size_mb(model: nn.Module) -> float:
    """Size of the saved weights in MB, measured by serializing to memory."""
    return len(_serialized_state(model)) / 1e6


def compressed_size_mb(model: nn.Module) -> float:
    """Size of the saved weights after gzip. Zeros compress well, so this is where
    unstructured sparsity shows up even though the raw file does not shrink."""
    return _sizes_mb(_serialized_state(model))[1]


def _synchronizer(inputs: tuple) -> Callable[[], None]:
    """Accelerators run asynchronously: without waiting for the device, the timer
    would only measure how long it takes to *queue* the work."""
    device = next((item.device.type for item in inputs if isinstance(item, torch.Tensor)), "cpu")
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    return lambda: None


def count_flops(
    model: nn.Module, example_input: torch.Tensor | tuple[torch.Tensor, ...]
) -> int | None:
    """Floating point operations of one forward pass, or None if they cannot be counted.

    Deterministic and machine independent, unlike latency.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            model(*as_inputs(example_input))
        return counter.get_total_flops() or None
    except (RuntimeError, NotImplementedError, TypeError):
        return None
    finally:
        model.train(was_training)


def _latency_stats(timings_ms: list[float]) -> LatencyStats:
    timings_ms = sorted(timings_ms)
    p95_index = min(len(timings_ms) - 1, round(0.95 * (len(timings_ms) - 1)))
    return LatencyStats(
        median_ms=statistics.median(timings_ms),
        p95_ms=timings_ms[p95_index],
        mean_ms=statistics.fmean(timings_ms),
        std_ms=statistics.stdev(timings_ms),
        runs=len(timings_ms),
    )


@contextlib.contextmanager
def _eval_mode(models: Sequence[nn.Module]) -> Iterator[None]:
    modes = [(model, model.training) for model in models]
    for model in models:
        model.eval()
    try:
        yield
    finally:
        for model, was_training in modes:
            model.train(was_training)


def measure_latency_interleaved(
    models: Sequence[nn.Module],
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    warmup: int = 10,
    runs: int = 100,
    block: int = 20,
) -> list[LatencyStats]:
    """Time several models on the same input, in alternating blocks of `block` passes.

    Timing models one after the other compares the moments as much as the models: when
    the load on the machine drifts between two measurements, the unlucky model looks
    slow. Cycling through the candidates spreads that drift over all of them.

    The passes come in blocks rather than one at a time on purpose. A single pass each
    in turn means every model finds the caches filled by its neighbours, and the penalty
    grows with how many bytes the others moved. Measured that way a 44 MB model looked
    3x slower than it is, which inflated every speedup reported against it. Inside a
    block the weights stay hot, so only the first pass of each block pays that cost and
    the median ignores it. Warm-up and device synchronisation as in `measure_latency`.
    """
    if runs < 2:
        raise ValueError("runs must be at least 2")
    if block < 1:
        raise ValueError("block must be at least 1")

    inputs = as_inputs(example_input)
    synchronize = _synchronizer(inputs)
    timings_ms: list[list[float]] = [[] for _ in models]
    with _eval_mode(models), torch.inference_mode():
        for model in models:
            for _ in range(warmup):
                model(*inputs)
        done = 0
        while done < runs:
            passes = min(block, runs - done)
            for timings, model in zip(timings_ms, models, strict=True):
                for _ in range(passes):
                    synchronize()
                    start = time.perf_counter()
                    model(*inputs)
                    synchronize()
                    timings.append((time.perf_counter() - start) * 1000)
            done += passes
    return [_latency_stats(timings) for timings in timings_ms]


def measure_latency(
    model: nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    warmup: int = 10,
    runs: int = 100,
) -> LatencyStats:
    """Time single forward passes on the given input.

    The first `warmup` passes are discarded: they pay one-off costs (memory
    allocation, kernel selection) that do not reflect steady-state speed.
    CUDA and MPS inputs are synchronized around every pass, so the device work is timed.
    To compare models, prefer `measure_latency_interleaved`.
    """
    return measure_latency_interleaved([model], example_input, warmup=warmup, runs=runs)[0]


def retime_profiles(
    profiles: Sequence[ModelProfile],
    models: Sequence[nn.Module],
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    warmup: int = 10,
    runs: int = 100,
) -> list[ModelProfile]:
    """Copies of `profiles` whose latencies were measured together, interleaved.

    `models[i]` is the model behind `profiles[i]`. Profiles are usually taken minutes
    apart (a search or fine-tuning in between); re-timing them side by side right
    before rendering keeps the Speed column about the models, not about the moment.
    """
    if len(profiles) != len(models):
        raise ValueError("profiles and models must have the same length")
    latencies = measure_latency_interleaved(models, example_input, warmup=warmup, runs=runs)
    return [
        dataclasses.replace(profile, latency=latency)
        for profile, latency in zip(profiles, latencies, strict=True)
    ]


def profile_model(
    model: nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    name: str = "model",
    warmup: int = 10,
    runs: int = 100,
    eval_fn: EvalFn | None = None,
    reference: nn.Module | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
    flops: int | None = None,
) -> ModelProfile:
    """Measure everything we know how to measure about `model`.

    `eval_fn` adds a task score. `reference` (the original model) adds fidelity,
    computed on `fidelity_inputs` or, if omitted, on `example_input` (one sample:
    a `FewSamplesWarning` says so).
    `flops` is reported as given instead of being counted. It exists for INT8 models,
    whose kernels the counter cannot see: pass the count of the float model they were
    made from (same arithmetic, lower precision). Without it their FLOPs stay None.
    """
    fidelity = None
    if reference is not None:
        batches, num_inputs = fidelity_batches(example_input, fidelity_inputs)
        fidelity = compare_outputs(reference, model, batches, num_inputs)

    total, nonzero = count_parameters(model)
    dtype = weights_dtype(model)
    if flops is None and dtype != "int8":
        # quantized kernels are invisible to the FLOP counter; a partial count would mislead
        flops = count_flops(model, example_input)
    # serializing a large model is slow: do it once for both size columns
    size_mb, compressed_mb = _sizes_mb(_serialized_state(model))
    return ModelProfile(
        name=name,
        total_params=total,
        nonzero_params=nonzero,
        size_mb=size_mb,
        compressed_mb=compressed_mb,
        flops=flops,
        dtype=dtype,
        latency=measure_latency(model, example_input, warmup=warmup, runs=runs),
        # float(): numpy and 0-d tensor scores are common and do not survive json.dumps
        accuracy=float(eval_fn(model)) if eval_fn is not None else None,
        fidelity=fidelity,
    )
