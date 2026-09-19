"""Structured pruning: physically remove channels and neurons.

Unlike unstructured pruning this shrinks the tensors, so parameters, FLOPs, file
size and latency all really go down. The price is bookkeeping: every tensor in a
`ChannelGroup` has to be sliced with the same indices (see `graph.py`).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping

import torch
from torch import nn

from pymodelprune.prune.graph import ChannelGroup, DependencyGraph, build_dependency_graph
from pymodelprune.prune.importance import CRITERIA, group_importance

GradFn = Callable[[nn.Module], None]
"""Runs backward passes on the model so that every parameter has a `.grad`."""


def _slice(module: nn.Module, name: str, index: torch.Tensor, dim: int) -> None:
    tensor = getattr(module, name, None)
    if tensor is None:
        return
    sliced = tensor.detach().index_select(dim, index.to(tensor.device)).clone()
    if isinstance(tensor, nn.Parameter):
        setattr(module, name, nn.Parameter(sliced, requires_grad=tensor.requires_grad))
    else:
        setattr(module, name, sliced)


def _apply_group(model: nn.Module, group: ChannelGroup, keep: torch.Tensor) -> None:
    kept = len(keep)
    for name in group.producers:
        module = model.get_submodule(name)
        _slice(module, "weight", keep, 0)
        _slice(module, "bias", keep, 0)
        if isinstance(module, nn.Linear):
            module.out_features = kept
        else:
            module.out_channels = kept
    for name in group.norms:
        module = model.get_submodule(name)
        for tensor_name in ("weight", "bias", "running_mean", "running_var"):
            _slice(module, tensor_name, keep, 0)
        module.num_features = kept
    for name in group.depthwise:
        module = model.get_submodule(name)
        _slice(module, "weight", keep, 0)
        _slice(module, "bias", keep, 0)
        module.in_channels = module.out_channels = module.groups = kept
    for name, fan in group.consumers:
        module = model.get_submodule(name)
        columns = (keep[:, None] * fan + torch.arange(fan)[None, :]).flatten()
        _slice(module, "weight", columns, 1)
        if isinstance(module, nn.Linear):
            module.in_features = len(columns)
        else:
            module.in_channels = kept


def _zero_group(model: nn.Module, group: ChannelGroup, keep: torch.Tensor) -> None:
    drop = torch.ones(group.channels, dtype=torch.bool)
    drop[keep] = False
    with torch.no_grad():
        for name in group.producers + group.norms + group.depthwise:
            module = model.get_submodule(name)
            for tensor_name in ("weight", "bias"):
                tensor = getattr(module, tensor_name, None)
                if tensor is not None:
                    tensor[drop.to(tensor.device)] = 0


def _round_half_up(value: float) -> int:
    """Python's round() sends halves to the even neighbour (4.5 -> 4 but 5.5 -> 6), which
    makes the kept channel count jump unevenly along a sweep of ratios."""
    return math.floor(value + 0.5)


def select_channels(
    scores: torch.Tensor, ratio: float, round_to: int = 1, min_channels: int = 1
) -> torch.Tensor:
    """Indices (sorted) of the channels to keep after removing `ratio` of them."""
    if round_to < 1 or min_channels < 1:
        raise ValueError("round_to and min_channels must be at least 1")
    channels = len(scores)
    kept = _round_half_up(channels * (1.0 - ratio))
    if round_to > 1 and channels >= round_to:
        kept = max(round_to, _round_half_up(kept / round_to) * round_to)
    kept = max(min(min_channels, channels), min(kept, channels))
    return torch.topk(scores, kept).indices.sort().values


def _global_ratios(
    scores: Mapping[int, torch.Tensor], amount: float, max_ratio: float
) -> dict[int, float]:
    """One threshold over all groups, so unimportant groups lose more channels."""
    threshold = torch.quantile(torch.cat(list(scores.values())), amount)
    return {
        group_id: min(float((score < threshold).float().mean()), max_ratio)
        for group_id, score in scores.items()
    }


def plan_pruning(
    model: nn.Module,
    graph: DependencyGraph,
    amount: float | Mapping[int, float],
    importance: str = "l1",
    scope: str = "uniform",
    round_to: int = 1,
    min_channels: int = 1,
    max_ratio: float = 0.9,
) -> dict[int, torch.Tensor]:
    """Decide which channels survive. Returns {group id: indices to keep}.

    `amount` is one ratio for every group, or an explicit {group id: ratio} mapping.
    scope="global" turns a single amount into per-group ratios using one shared
    importance threshold. For importance="taylor" gradients must already be populated.
    """
    if importance not in CRITERIA:
        raise ValueError(f"importance must be one of {sorted(CRITERIA)}")
    if scope not in ("uniform", "global"):
        raise ValueError("scope must be 'uniform' or 'global'")

    prunable = {group.id: group for group in graph.prunable_groups}
    for group in prunable.values():
        first = model.get_submodule(group.producers[0])
        if first.weight.shape[0] != group.channels:
            raise ValueError(
                "the dependency graph does not match the model (was it already pruned?). "
                "Build a new graph with build_dependency_graph."
            )
    if isinstance(amount, Mapping):
        unknown = set(amount) - set(prunable)
        if unknown:
            raise ValueError(f"not prunable group ids: {sorted(unknown)}")
        ratios = dict(amount)
    else:
        ratios = {group_id: float(amount) for group_id in prunable}
    if any(not 0.0 <= ratio < 1.0 for ratio in ratios.values()):
        raise ValueError("pruning ratios must be in [0, 1)")

    scores = {gid: group_importance(model, prunable[gid], importance) for gid in ratios}
    if scope == "global" and not isinstance(amount, Mapping) and scores:
        ratios = _global_ratios(scores, float(amount), max_ratio)
    return {
        gid: select_channels(scores[gid], ratio, round_to, min_channels)
        for gid, ratio in ratios.items()
        if ratio > 0.0
    }


def apply_plan(
    model: nn.Module, graph: DependencyGraph, plan: Mapping[int, torch.Tensor]
) -> nn.Module:
    """Remove channels in place according to `plan` and return the model."""
    for group_id, keep in plan.items():
        _apply_group(model, graph.group(group_id), keep)
    return model


def mask_plan(
    model: nn.Module, graph: DependencyGraph, plan: Mapping[int, torch.Tensor]
) -> nn.Module:
    """Zero (instead of remove) the channels a plan would drop, in place.

    A masked model and a pruned model must produce identical outputs. That is the
    correctness check for the whole dependency analysis, and the test suite runs it.
    """
    for group_id, keep in plan.items():
        _zero_group(model, graph.group(group_id), keep)
    return model


def prune_structured(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    amount: float | Mapping[int, float] = 0.5,
    importance: str = "l1",
    scope: str = "uniform",
    round_to: int = 1,
    min_channels: int = 1,
    grad_fn: GradFn | None = None,
    graph: DependencyGraph | None = None,
) -> nn.Module:
    """Return a physically smaller copy of `model`. The input model is left untouched.

    `amount` is the fraction of channels removed from every prunable group (or a
    {group id: ratio} mapping). The layers feeding the model output are never pruned.
    """
    graph = graph or build_dependency_graph(model, example_input)
    if not graph.prunable_groups:
        reasons = sorted({reason for group in graph.groups for reason in group.blockers})
        raise ValueError(f"no prunable channel groups found. Blockers: {reasons}")

    pruned = copy.deepcopy(model)
    if importance == "taylor":
        if grad_fn is None:
            raise ValueError("importance='taylor' needs grad_fn to populate gradients")
        pruned.zero_grad(set_to_none=True)
        grad_fn(pruned)
    plan = plan_pruning(pruned, graph, amount, importance, scope, round_to, min_channels)
    apply_plan(pruned, graph, plan)
    pruned.zero_grad(set_to_none=True)
    return pruned
