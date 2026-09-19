"""One call: prune, quantize, measure every stage, report, save."""

from __future__ import annotations

import copy
import dataclasses
import platform
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from torch import nn

from pymodelprune._json import write_json
from pymodelprune.evaluate import EvalFn, reusable
from pymodelprune.export import OrtModule, profile_onnx, save_model, to_onnx
from pymodelprune.fidelity import FewSamplesWarning
from pymodelprune.profile import ModelProfile, profile_model, retime_profiles, weights_dtype
from pymodelprune.prune.structured import GradFn, prune_structured
from pymodelprune.prune.unstructured import make_permanent, prune_unstructured
from pymodelprune.quantize.dynamic import quantize_dynamic
from pymodelprune.quantize.onnx_static import check_calibration_method, quantize_onnx_static
from pymodelprune.report import render_profiles
from pymodelprune.search.budget import Budget, BudgetResult, FinetuneFn, search_budget

_IMPORTANCE = {
    "structured": ("l1", "l2", "bn", "taylor", "random"),
    # for single weights |w| and w**2 rank identically; bn/taylor score whole channels
    "unstructured": ("l1", "random"),
}


@dataclass(frozen=True)
class PruneConfig:
    """method="structured" removes whole channels (real speedup, needs a traceable
    model). method="unstructured" zeros single weights (works on anything, only the
    compressed size shrinks).

    importance: l1 | l2 | bn | taylor | random for structured, l1 | random for unstructured.
    scope: None picks the method's default, "uniform" for structured (same ratio in
        every group) and "global" for unstructured (one threshold over all weights).
    round_to / min_channels: structured only; keep channel counts a multiple of
        `round_to` and never below `min_channels`.
    budget: structured only; replaces `amount` and `scope` with an automatic search
        for per-group ratios.
    """

    method: str = "structured"
    amount: float = 0.5
    importance: str = "l1"
    scope: str | None = None
    round_to: int = 1
    budget: Budget | None = None
    min_channels: int = 1

    def __post_init__(self) -> None:
        if self.method not in _IMPORTANCE:
            raise ValueError("method must be 'structured' or 'unstructured'")
        if self.importance not in _IMPORTANCE[self.method]:
            choices = " | ".join(_IMPORTANCE[self.method])
            raise ValueError(
                f"importance {self.importance!r} is not available for {self.method} pruning "
                f"(choose from {choices})"
            )
        if not 0.0 <= self.amount < 1.0:
            raise ValueError("amount must be in [0, 1)")
        if self.round_to < 1 or self.min_channels < 1:
            raise ValueError("round_to and min_channels must be at least 1")
        if self.budget is not None and self.method != "structured":
            raise ValueError("a budget needs method='structured'")


@dataclass(frozen=True)
class QuantConfig:
    """mode="dynamic" / "weight_only": INT8 Linear layers, stays a PyTorch model.
    mode="onnx_static": full INT8 (convolutions too) as an ONNX file; needs
    calibration data and an `onnx_path` in `optimize`. `calibration_method`
    (minmax | entropy | percentile) picks how activation ranges are read off that data."""

    mode: str = "dynamic"
    backend: str = "auto"
    calibration: Any = None
    per_channel: bool = True
    exclude_nodes: tuple[str, ...] = ()
    calibration_method: str = "minmax"

    def __post_init__(self) -> None:
        if self.mode not in ("dynamic", "weight_only", "onnx_static"):
            raise ValueError("mode must be 'dynamic', 'weight_only' or 'onnx_static'")
        check_calibration_method(self.calibration_method)


@dataclass
class Stage:
    name: str
    profile: ModelProfile
    model: nn.Module | None = None
    onnx_path: Path | None = None


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "quantized_engine": torch.backends.quantized.engine,
    }


@dataclass
class OptimizationResult:
    stages: list[Stage]
    environment: dict[str, Any] = field(default_factory=environment_info)
    search: BudgetResult | None = None

    @property
    def profiles(self) -> list[ModelProfile]:
        return [stage.profile for stage in self.stages]

    @property
    def model(self) -> nn.Module:
        """The last stage that is still a PyTorch model."""
        return next(stage.model for stage in reversed(self.stages) if stage.model is not None)

    @property
    def onnx_path(self) -> Path | None:
        return self.stages[-1].onnx_path

    def report(self, console: Console | None = None) -> None:
        render_profiles(self.profiles, console=console)

    def save(self, path: str | Path) -> Path:
        """Write the final PyTorch model (`.pt`). ONNX stages are already on disk."""
        return save_model(self.model, path)

    def to_dict(self) -> dict[str, Any]:
        stages = []
        for stage in self.stages:
            entry = dataclasses.asdict(stage.profile)
            entry["sparsity"] = stage.profile.sparsity
            entry["onnx_path"] = str(stage.onnx_path) if stage.onnx_path else None
            stages.append(entry)
        result: dict[str, Any] = {"schema": 1, "environment": self.environment, "stages": stages}
        if self.search is not None:
            result["search"] = {
                "ratios": {str(gid): ratio for gid, ratio in self.search.ratios.items()},
                "baseline_score": self.search.baseline_score,
                "score": self.search.score,
                "flops_ratio": self.search.flops_ratio,
                "predicted_drop": self.search.predicted_drop,
                "refinements": self.search.refinements,
                "scale": self.search.scale,
                "history": [dataclasses.asdict(step) for step in self.search.history],
            }
        return result

    def to_json(self, path: str | Path) -> Path:
        return write_json(self.to_dict(), path)


def _to_cpu(example_input: torch.Tensor | tuple) -> torch.Tensor | tuple:
    if isinstance(example_input, tuple):
        return tuple(item.detach().cpu() for item in example_input)
    return example_input.detach().cpu()


def optimize(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    prune: PruneConfig | None = None,
    quantize: QuantConfig | None = None,
    eval_fn: EvalFn | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
    finetune_fn: FinetuneFn | None = None,
    grad_fn: GradFn | None = None,
    onnx_path: str | Path | None = None,
    warmup: int = 10,
    runs: int = 100,
    threads: int | None = None,
    onnx_atol: float = 1e-4,
    interleaved_timing: bool = True,
) -> OptimizationResult:
    """Prune, then quantize, profiling after every step. `model` is never modified.

    Everything runs on the CPU: the model is copied there and `example_input` is moved.
    eval_fn: model -> score (higher is better). Without it only fidelity is reported.
    fidelity_inputs: inputs for the fidelity column (default: `example_input`, a single
        sample, which earns a `FewSamplesWarning`). One-shot iterators are materialised.
    finetune_fn: trains the pruned model in place to recover accuracy. After
        unstructured pruning the masks stay attached while it runs, so sparsity survives.
    grad_fn: populates gradients, only needed for importance="taylor".
    onnx_path: where the INT8 ONNX file goes, required for quantize.mode="onnx_static".
        The float export is written next to it as `<stem>_fp32.onnx`.
    threads: CPU threads while measuring; the previous setting is restored afterwards.
    onnx_atol: absolute tolerance of the ONNX export verification.
    interleaved_timing: measure the latency of all stages together at the very end,
        one forward pass each in turn. Stages are produced minutes apart (a search or
        fine-tuning sits between them), so timing each one when it appears lets drifting
        machine load decide the Speed column: a model with half the FLOPs has shown up
        as "slower" that way. False times every stage on its own, as it is created.
    """
    if prune is None and quantize is None:
        raise ValueError("nothing to do: pass prune and/or quantize")
    if weights_dtype(model) == "int8":
        # INT8 tensors support neither channel slicing nor a second quantization pass
        raise ValueError("model is already quantized; optimize the float model instead")
    if quantize is not None and quantize.mode == "onnx_static":
        if onnx_path is None:
            raise ValueError("quantize.mode='onnx_static' needs onnx_path")
        if quantize.calibration is None and fidelity_inputs is None:
            raise ValueError("onnx_static needs calibration data (QuantConfig.calibration)")

    previous_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        # catch_warnings: `_run` mutes the few-samples reminder after its first stage
        with warnings.catch_warnings():
            return _run(
                model, _to_cpu(example_input), prune, quantize, eval_fn,
                reusable(fidelity_inputs), finetune_fn, grad_fn, onnx_path,
                warmup, runs, onnx_atol, interleaved_timing,
            )  # fmt: skip
    finally:
        if threads is not None:
            torch.set_num_threads(previous_threads)


def _prune(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    prune: PruneConfig,
    finetune_fn: FinetuneFn | None,
    grad_fn: GradFn | None,
) -> nn.Module:
    if prune.method == "unstructured":
        # Fine-tuning needs the masks still attached: once they are folded in, the
        # optimizer would update the zeros like any other weight and undo the sparsity.
        pruned = prune_unstructured(
            model, prune.amount, criterion=prune.importance, scope=prune.scope or "global",
            permanent=finetune_fn is None,
        )  # fmt: skip
        if finetune_fn is not None:
            finetune_fn(pruned)
            make_permanent(pruned)
        return pruned
    pruned = prune_structured(
        model, example_input, prune.amount, prune.importance, prune.scope or "uniform",
        prune.round_to, prune.min_channels, grad_fn=grad_fn,
    )  # fmt: skip
    if finetune_fn is not None:
        finetune_fn(pruned)
    return pruned


def _run(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    prune: PruneConfig | None,
    quantize: QuantConfig | None,
    eval_fn: EvalFn | None,
    fidelity_inputs: torch.Tensor | Iterable | None,
    finetune_fn: FinetuneFn | None,
    grad_fn: GradFn | None,
    onnx_path: str | Path | None,
    warmup: int,
    runs: int,
    onnx_atol: float,
    interleaved_timing: bool,
) -> OptimizationResult:
    original = copy.deepcopy(model).cpu().eval()
    measure = {"warmup": warmup, "runs": runs, "eval_fn": eval_fn}
    if interleaved_timing:
        # placeholder timing: every stage is timed again, together, once all of them exist
        measure.update(warmup=1, runs=2)
    compare = {"reference": original, "fidelity_inputs": fidelity_inputs}
    stages = [
        Stage("original", profile_model(original, example_input, "original", **measure), original)
    ]
    timed: list[nn.Module] = [original]  # what each stage runs on: its model or ONNX session
    current, search = original, None

    def add_stage(stage: Stage, runner: nn.Module) -> None:
        stages.append(stage)
        timed.append(runner)
        # every stage sees the same data: one few-samples reminder per call is enough
        warnings.simplefilter("ignore", FewSamplesWarning)

    if prune is not None:
        if prune.budget is not None:
            search = search_budget(
                current, example_input, prune.budget, eval_fn, fidelity_inputs,
                importance=prune.importance, grad_fn=grad_fn, finetune_fn=finetune_fn,
                round_to=prune.round_to, min_channels=prune.min_channels,
            )  # fmt: skip
            current = search.model
        else:
            current = _prune(current, example_input, prune, finetune_fn, grad_fn)
        current = current.cpu().eval()
        add_stage(
            Stage(
                "pruned",
                profile_model(current, example_input, "pruned", **measure, **compare),
                current,
            ),
            current,
        )

    if quantize is not None and quantize.mode == "onnx_static":
        int8_path = Path(onnx_path)
        # with_name, not with_suffix: "model.int8.onnx" must not overwrite or double a suffix
        float_path = int8_path.with_name(f"{int8_path.stem}_fp32.onnx")
        to_onnx(current, example_input, float_path, atol=onnx_atol)
        common = {"source_model": current, **measure, **compare}
        float_session = OrtModule(float_path)
        add_stage(
            Stage(
                "onnx fp32",
                profile_onnx(
                    float_path, example_input, "onnx fp32", session=float_session, **common
                ),
                None,
                float_path,
            ),
            float_session,
        )
        calibration = quantize.calibration if quantize.calibration is not None else fidelity_inputs
        quantize_onnx_static(
            float_path, int8_path, calibration, per_channel=quantize.per_channel,
            exclude_nodes=quantize.exclude_nodes,
            calibration_method=quantize.calibration_method,
        )  # fmt: skip
        int8_session = OrtModule(int8_path)
        add_stage(
            Stage(
                "onnx int8",
                profile_onnx(int8_path, example_input, "onnx int8", session=int8_session, **common),
                None,
                int8_path,
            ),
            int8_session,
        )
    elif quantize is not None:
        # same arithmetic, lower precision: the FLOP count of the float model still applies
        flops = stages[-1].profile.flops
        current = quantize_dynamic(current, mode=quantize.mode, backend=quantize.backend)
        profile = profile_model(current, example_input, "int8", flops=flops, **measure, **compare)
        add_stage(Stage("int8", profile, current), current)

    if interleaved_timing:
        profiles = retime_profiles(
            [stage.profile for stage in stages], timed, example_input, warmup=warmup, runs=runs
        )
        for stage, profile in zip(stages, profiles, strict=True):
            stage.profile = profile
    # The ONNX files stay on disk; only the live sessions go. Left to the garbage
    # collector they can outlive the caller and crash the process during shutdown.
    for measured in timed:
        if isinstance(measured, OrtModule):
            measured.close()
    return OptimizationResult(stages=stages, search=search)
