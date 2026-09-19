"""Command line interface."""

from __future__ import annotations

import contextlib
import importlib
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from rich.console import Console
from rich.table import Table
from torch import nn

from pymodelprune.demo_models import DEMO_INPUT_SHAPE, DEMO_MODELS
from pymodelprune.evaluate import reusable
from pymodelprune.fidelity import FewSamplesWarning
from pymodelprune.pipeline import PruneConfig, QuantConfig
from pymodelprune.pipeline import optimize as run_optimize
from pymodelprune.profile import profile_model
from pymodelprune.prune.graph import build_dependency_graph
from pymodelprune.quantize.dynamic import select_quantized_engine
from pymodelprune.report import render_profiles
from pymodelprune.search.budget import Budget
from pymodelprune.search.sensitivity import analyze_sensitivity

app = typer.Typer(
    help="Prune and quantize PyTorch models, then measure what you actually gained.",
    no_args_is_help=True,
)

_HOOK_HELP = "as 'package.module:function'. The current directory is importable."
ModelPath = Annotated[
    Path | None,
    typer.Argument(exists=True, dir_okay=False, help="Model saved with torch.save(model, path)."),
]
FactoryOption = Annotated[
    str | None,
    typer.Option(help=f"Function that builds the model, {_HOOK_HELP} Safer than unpickling."),
]
WeightsOption = Annotated[
    Path | None,
    typer.Option(exists=True, dir_okay=False, help="state_dict file for --factory."),
]
ShapeOption = Annotated[
    str, typer.Option("--input-shape", help="Example input shape, e.g. 1,3,224,224.")
]
EvalOption = Annotated[
    str | None, typer.Option("--eval", help=f"Scoring function model -> float, {_HOOK_HELP}")
]
DataOption = Annotated[
    str | None,
    typer.Option(help=f"Function returning an iterable of input batches, {_HOOK_HELP}"),
]
RunsOption = Annotated[int, typer.Option(min=2, help="Timed forward passes.")]

# `taylor` exists in the Python API only: it needs a grad_fn, which has no CLI form.
_CLI_IMPORTANCE = {"structured": ("l1", "l2", "bn", "random"), "unstructured": ("l1", "random")}


def _first_line(error: Exception) -> str:
    """torch errors can span a screen; the first line names the problem."""
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__


@contextlib.contextmanager
def _clean_errors() -> Iterator[None]:
    """Turn the errors a wrong argument, file or shape causes into one `error:` line,
    and the library's own warnings into one `warning:` line without source context."""
    try:
        with warnings.catch_warnings():  # also restores showwarning on exit
            default_show = warnings.showwarning

            def show(message, category, *args, **kwargs) -> None:
                if issubclass(category, FewSamplesWarning):
                    typer.echo(f"warning: {message}", err=True)
                else:
                    default_show(message, category, *args, **kwargs)

            warnings.showwarning = show
            yield
    except (typer.Exit, typer.Abort):  # click's Exit is a RuntimeError: let it through
        raise
    except (ValueError, RuntimeError, TypeError) as error:
        typer.echo(f"error: {_first_line(error)}", err=True)
        raise typer.Exit(code=1) from None


def _check_importance(importance: str, method: str = "structured") -> None:
    if method not in _CLI_IMPORTANCE:
        raise typer.BadParameter("--method must be 'structured' or 'unstructured'")
    if importance == "taylor":
        raise typer.BadParameter(
            "--importance taylor needs gradients (grad_fn) and is only available from "
            "the Python API"
        )
    if importance not in _CLI_IMPORTANCE[method]:
        choices = " | ".join(_CLI_IMPORTANCE[method])
        raise typer.BadParameter(
            f"--importance {importance!r} is not available for {method} pruning "
            f"(choose from {choices})"
        )


def _parse_shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise typer.BadParameter(f"expected comma separated integers, got {text!r}") from None
    if not shape or any(dim <= 0 for dim in shape):
        raise typer.BadParameter("every dimension must be a positive integer")
    return shape


def _load_hook(spec: str | None) -> Any:
    if spec is None:
        return None
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise typer.BadParameter(f"expected 'package.module:function', got {spec!r}")
    if "" not in sys.path and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        return getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise typer.BadParameter(f"cannot load {spec!r}: {error}") from None


def _load_data(spec: str | None) -> Any:
    """Call the --data hook; a generator is materialised because every stage re-reads it."""
    hook = _load_hook(spec)
    return reusable(hook()) if hook is not None else None


def _unpickle(path: Path) -> Any:
    # A model quantized with the legacy backend unpickles through quantized ops, which
    # fail with NoQEngine on platforms (Apple Silicon) where no engine is selected yet.
    select_quantized_engine()
    try:
        return torch.load(path, weights_only=False)
    except Exception as error:  # noqa: BLE001 - unpickling an arbitrary file can raise anything
        raise ValueError(f"cannot load {path.name}: {_first_line(error)}") from None


def _load_model(
    path: Path | None, factory: str | None, weights: Path | None, example: torch.Tensor
) -> nn.Module:
    if weights is not None and factory is None:
        raise typer.BadParameter("--weights needs --factory to build the architecture")
    if factory is not None:
        model = _load_hook(factory)()
        if weights is not None:
            model.load_state_dict(torch.load(weights, weights_only=True))
    elif path is not None:
        model = _unpickle(path)
    else:
        raise typer.BadParameter("give a model file or --factory")
    if not isinstance(model, nn.Module):
        raise TypeError(
            "file does not contain a full model. Save it with torch.save(model, path), "
            "or pass --factory together with --weights"
        )
    model.eval()
    # One plain forward pass up front: a wrong --input-shape is the most common mistake
    # and surfaces here with torch's own message instead of deep inside tracing.
    try:
        with torch.inference_mode():
            model(example)
    except (RuntimeError, ValueError, TypeError, IndexError) as error:
        shape = ",".join(str(dim) for dim in example.shape)
        raise ValueError(
            f"the model does not run on --input-shape {shape}: {_first_line(error)}"
        ) from None
    return model


@app.command()
def demo(
    model: Annotated[str, typer.Option(help=f"Built-in model: {', '.join(DEMO_MODELS)}.")] = "mlp",
    runs: RunsOption = 100,
) -> None:
    """Profile a small built-in model. Useful to check the installation."""
    if model not in DEMO_MODELS:
        raise typer.BadParameter(f"unknown model {model!r}, choose from {list(DEMO_MODELS)}")
    net = DEMO_MODELS[model]()
    example = torch.randn(DEMO_INPUT_SHAPE)
    render_profiles([profile_model(net, example, name=model, runs=runs)])


@app.command()
def profile(
    input_shape: ShapeOption,
    path: ModelPath = None,
    factory: FactoryOption = None,
    weights: WeightsOption = None,
    eval_hook: EvalOption = None,
    runs: RunsOption = 100,
) -> None:
    """Profile a model: parameters, size, FLOPs, latency and optionally accuracy.

    A model file is unpickled, which can run arbitrary code. Only load files you
    trust, or use --factory with --weights.
    """
    example = torch.randn(_parse_shape(input_shape))
    name = path.name if path else factory
    score = _load_hook(eval_hook)
    with _clean_errors():
        net = _load_model(path, factory, weights, example)
        profiles = [profile_model(net, example, name, runs=runs, eval_fn=score)]
    render_profiles(profiles)


@app.command()
def optimize(
    input_shape: ShapeOption,
    path: ModelPath = None,
    factory: FactoryOption = None,
    weights: WeightsOption = None,
    prune: Annotated[
        float | None, typer.Option(min=0.0, max=0.99, help="Fraction to prune, e.g. 0.5.")
    ] = None,
    method: Annotated[str, typer.Option(help="structured | unstructured")] = "structured",
    importance: Annotated[
        str, typer.Option(help="l1 | l2 | bn | random (unstructured: l1 | random)")
    ] = "l1",
    max_acc_drop: Annotated[
        float | None,
        typer.Option(
            help="Search pruning ratios automatically; tolerated loss in percentage points. "
            "Requires --eval or --data."
        ),
    ] = None,
    target_flops: Annotated[
        float | None,
        typer.Option(
            help="Search until FLOPs fall to this fraction, e.g. 0.5. Alone it does not limit "
            "the score loss (add --max-acc-drop). Requires --eval or --data."
        ),
    ] = None,
    quantize: Annotated[
        str, typer.Option(help="none | dynamic | weight_only | onnx_static")
    ] = "none",
    backend: Annotated[str, typer.Option(help="auto | torchao | legacy")] = "auto",
    eval_hook: EvalOption = None,
    data: DataOption = None,
    finetune: Annotated[
        str | None, typer.Option(help=f"Function training the pruned model in place, {_HOOK_HELP}")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output file: .pt, or .onnx for onnx_static.")
    ] = None,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Write the report as JSON.")
    ] = None,
    threads: Annotated[
        int | None, typer.Option(min=1, help="CPU threads used while timing.")
    ] = None,
    runs: RunsOption = 100,
) -> None:
    """Prune and/or quantize a model and report every stage against the original.

    --data supplies real inputs: they drive the fidelity column, the budget
    search and INT8 calibration. Without it fidelity is computed on the single
    random example input (and says so); a budget search then needs --eval,
    and onnx_static always needs --data.
    """
    searching = max_acc_drop is not None or target_flops is not None
    if prune is None and not searching and quantize == "none":
        raise typer.BadParameter("nothing to do: pass --prune, --max-acc-drop or --quantize")
    if prune is not None and searching:
        raise typer.BadParameter(
            "--prune cannot be combined with --max-acc-drop/--target-flops: "
            "the budget search decides the amount"
        )
    if prune is not None or searching:
        _check_importance(importance, method)
    if quantize == "onnx_static" and (out is None or out.suffix != ".onnx"):
        raise typer.BadParameter("--quantize onnx_static needs --out something.onnx")
    if quantize == "onnx_static" and data is None:
        raise typer.BadParameter("--quantize onnx_static needs --data for calibration")
    example = torch.randn(_parse_shape(input_shape))
    hooks = {
        "eval_fn": _load_hook(eval_hook),
        "fidelity_inputs": _load_data(data),
        "finetune_fn": _load_hook(finetune),
    }

    with _clean_errors():
        budget = None
        if searching:
            budget = Budget(
                max_drop=None if max_acc_drop is None else max_acc_drop / 100,
                target_flops_ratio=target_flops,
            )
        prune_config = None
        if prune is not None or budget is not None:
            prune_config = PruneConfig(method, prune or 0.0, importance, budget=budget)
        quant_config = None if quantize == "none" else QuantConfig(mode=quantize, backend=backend)
        result = run_optimize(
            _load_model(path, factory, weights, example),
            example,
            prune=prune_config,
            quantize=quant_config,
            onnx_path=out if quantize == "onnx_static" else None,
            runs=runs,
            threads=threads,
            **hooks,
        )

    result.report()
    if result.search is not None:
        typer.echo(
            f"search: predicted drop {result.search.predicted_drop:.2%}, "
            f"measured {result.search.measured_drop:.2%}, FLOPs x{result.search.flops_ratio:.2f}"
        )
        if max_acc_drop is None:
            # a FLOPs target alone prunes until it is reached, whatever that costs
            typer.echo(
                f"note: --target-flops without --max-acc-drop does not limit the score loss; "
                f"measured drop {result.search.measured_drop:.2%}"
            )
    if out is not None and quantize != "onnx_static":
        with _clean_errors():
            saved = result.save(out)
        typer.echo(f"saved {saved}")
    elif result.onnx_path is not None:
        typer.echo(f"saved {result.onnx_path}")
    if json_path is not None:
        typer.echo(f"report {result.to_json(json_path)}")


@app.command()
def sensitivity(
    input_shape: ShapeOption,
    path: ModelPath = None,
    factory: FactoryOption = None,
    weights: WeightsOption = None,
    eval_hook: EvalOption = None,
    data: DataOption = None,
    importance: Annotated[str, typer.Option(help="l1 | l2 | bn | random")] = "l1",
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Write the curves as JSON.")
    ] = None,
) -> None:
    """Show which layers tolerate structured pruning and which do not.

    Requires --eval (a task score) or --data (at least 32 samples, scored
    by agreement with the original model): a single random input cannot
    rank layers.
    """
    _check_importance(importance)
    if eval_hook is None and data is None:
        raise typer.BadParameter("sensitivity needs --eval or --data to score the pruned models")
    example = torch.randn(_parse_shape(input_shape))
    score, batches = _load_hook(eval_hook), _load_data(data)
    with _clean_errors():
        net = _load_model(path, factory, weights, example)
        graph = build_dependency_graph(net, example)
        report = analyze_sensitivity(
            net, example, eval_fn=score, fidelity_inputs=batches, importance=importance, graph=graph
        )
    blocked = [group for group in graph.groups if group.blockers and group.producers]
    if not report.curves:
        reasons = sorted({reason for group in graph.groups for reason in group.blockers})
        typer.echo(f"error: no prunable channel groups found. Blockers: {reasons}", err=True)
        raise typer.Exit(code=1)

    table = Table(box=None, pad_edge=False, padding=(0, 1), header_style="bold cyan")
    table.add_column("Group")
    table.add_column("Layers", no_wrap=True, max_width=36)
    ratios = [point.ratio for point in next(iter(report.curves.values()), [])]
    for ratio in ratios:
        table.add_column(f"-{ratio:.0%}", justify="right")
    for group_id, _ in report.ranking():
        drops = [f"{-report.drop(group_id, ratio):+.1%}" for ratio in ratios]
        table.add_row(str(group_id), ", ".join(report.layers[group_id]), *drops)
    console = Console()
    console.print(
        f"score change per pruned share ({report.metric}, baseline "
        f"{report.baseline_score:.1%}), most sensitive first"
    )
    console.print(table)
    for group in blocked:
        console.print(
            f"[dim]not prunable: {', '.join(group.producers)} ({group.blockers[0]})[/dim]"
        )
    if json_path is not None:
        report.to_json(json_path)
        typer.echo(f"report {json_path}")
