"""Budget driven structured pruning: give a limit, get per-group pruning ratios.

Greedy search over the sensitivity curves: every step moves the group that
buys the most FLOPs for the least predicted score loss. Single-group losses do not
simply add up when several groups are pruned together, so the prediction is only
used to steer. The final model is always re-measured, and scaled back if it
breaks the limit.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from torch import nn

from pymodelprune.evaluate import EvalFn, reusable
from pymodelprune.profile import count_flops
from pymodelprune.prune.graph import DependencyGraph, build_dependency_graph
from pymodelprune.prune.structured import GradFn, apply_plan, plan_pruning
from pymodelprune.search.sensitivity import (
    SensitivityReport,
    analyze_sensitivity,
    make_score_fn,
)

FinetuneFn = Callable[[nn.Module], None]
"""Trains the pruned model in place for a short while to recover accuracy."""


@dataclass(frozen=True)
class Budget:
    """At least one limit is required.

    max_drop: largest tolerated score loss, in the units of the metric
        (accuracy in [0, 1] -> 0.01 means one percentage point).
    target_flops_ratio: stop once FLOPs fall to this fraction of the original.
    With both set, the search stops at whichever limit is hit first.

    `target_flops_ratio` alone puts NO limit on the score: the search keeps pruning
    until the target is reached, whatever that costs, and nothing is scaled back
    afterwards. The ratios come from a grid, so the target is overshot to the first
    grid point at or below it. Read `BudgetResult.measured_drop`, or set `max_drop` too.
    """

    max_drop: float | None = None
    target_flops_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.max_drop is None and self.target_flops_ratio is None:
            raise ValueError("set max_drop, target_flops_ratio, or both")
        if self.max_drop is not None and self.max_drop < 0:
            raise ValueError("max_drop must be >= 0")
        if self.target_flops_ratio is not None and not 0.0 < self.target_flops_ratio < 1.0:
            raise ValueError("target_flops_ratio must be in (0, 1)")


@dataclass(frozen=True)
class SearchStep:
    flops_ratio: float
    predicted_drop: float


@dataclass
class BudgetResult:
    """`model`, `ratios`, `score` and `flops_ratio` describe the returned model.

    `predicted_drop` and `history` describe the greedy selection BEFORE verification.
    When the measured drop broke the limit, every selected ratio was multiplied by
    `scale`, so they no longer describe `model`; `scale == 1.0` means they still do.
    """

    model: nn.Module
    ratios: dict[int, float]
    baseline_score: float
    score: float
    flops_ratio: float
    predicted_drop: float
    refinements: int
    history: list[SearchStep] = field(default_factory=list)
    """Greedy path: an estimate of the accuracy/FLOPs trade-off curve."""
    scale: float = 1.0
    """Bisect factor applied to the greedy ratios: 1.0 when the selection passed
    verification as it was, 0.0 when no scale fit the limit (nothing pruned)."""

    @property
    def measured_drop(self) -> float:
        return self.baseline_score - self.score


def _check_report(
    report: SensitivityReport,
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    graph: DependencyGraph,
) -> None:
    """A report from another model has curves for the wrong groups and would silently
    steer the search with numbers that mean nothing here."""
    layers = {group.id: list(group.producers) for group in graph.prunable_groups}
    if report.layers != layers:
        raise ValueError("the sensitivity report does not match this model's prunable groups")
    if report.baseline_flops != count_flops(model, example_input):
        raise ValueError("the sensitivity report was measured on a model with different FLOPs")


def search_budget(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    budget: Budget,
    eval_fn: EvalFn | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
    report: SensitivityReport | None = None,
    importance: str = "l1",
    grad_fn: GradFn | None = None,
    finetune_fn: FinetuneFn | None = None,
    graph: DependencyGraph | None = None,
    max_refinements: int = 5,
    round_to: int = 1,
    min_channels: int = 1,
) -> BudgetResult:
    """Find per-group pruning ratios that respect `budget`, and return the pruned model.

    Scores come from `eval_fn`, or from agreement with the original on `fidelity_inputs`
    (at least 32 samples). A `report` from `analyze_sensitivity` on this very model skips
    the expensive analysis. When nothing can be pruned within the budget, the result
    holds an untouched copy of the model and `finetune_fn` is not called.
    """
    graph = graph or build_dependency_graph(model, example_input)
    if not graph.prunable_groups:
        reasons = sorted({reason for group in graph.groups for reason in group.blockers})
        raise ValueError(f"no prunable channel groups found. Blockers: {reasons}")
    fidelity_inputs = reusable(fidelity_inputs)
    if report is None:
        report = analyze_sensitivity(
            model, example_input, eval_fn, fidelity_inputs, importance=importance,
            grad_fn=grad_fn, graph=graph, round_to=round_to, min_channels=min_channels,
        )  # fmt: skip
    else:
        _check_report(report, model, example_input, graph)
    _, score_fn = make_score_fn(model, example_input, eval_fn, fidelity_inputs)

    scored = copy.deepcopy(model)
    if importance == "taylor":
        if grad_fn is None:
            raise ValueError("importance='taylor' needs grad_fn to populate gradients")
        grad_fn(scored)

    def build(ratios: dict[int, float]) -> nn.Module:
        active = {gid: ratio for gid, ratio in ratios.items() if ratio > 0.0}
        plan = plan_pruning(
            scored, graph, active, importance, round_to=round_to, min_channels=min_channels
        )
        return apply_plan(copy.deepcopy(model), graph, plan)

    def flops_ratio(ratios: dict[int, float]) -> float:
        return count_flops(build(ratios), example_input) / report.baseline_flops

    grid = {gid: [0.0] + [point.ratio for point in curve] for gid, curve in report.curves.items()}
    ratios = dict.fromkeys(grid, 0.0)
    target = budget.target_flops_ratio

    def predicted_drop(candidate_ratios: dict[int, float]) -> float:
        # Summed over the current ratios, not accumulated move by move: on a curve that
        # is not monotone a running sum of clipped increments drifts away from the table.
        return sum(max(report.drop(gid, ratio), 0.0) for gid, ratio in candidate_ratios.items())

    predicted = 0.0
    history = [SearchStep(1.0, 0.0)]
    while target is None or history[-1].flops_ratio > target:
        best = None
        for gid, steps in grid.items():
            # Every ratio above the current one is a candidate, not only the next grid
            # point: a bump early in a curve must not hide the cheap ratios behind it.
            for following in (step for step in steps if step > ratios[gid]):
                gain = report.saved_flops(gid, following) - report.saved_flops(gid, ratios[gid])
                if gain <= 0:
                    continue
                after = predicted_drop({**ratios, gid: following})
                if budget.max_drop is None or after <= budget.max_drop:
                    cost = (max(after - predicted, 0.0) + 1e-6) / gain
                    if best is None or cost < best[0]:
                        best = (cost, gid, following, after)
                reached = history[-1].flops_ratio - gain / report.baseline_flops
                if target is not None and reached <= target:
                    break  # longer jumps in this group would only overshoot the target further
        if best is None:
            break
        _, gid, following, predicted = best
        ratios[gid] = following
        history.append(SearchStep(flops_ratio(ratios), predicted))

    def verify(candidate_ratios: dict[int, float]) -> tuple[nn.Module, float]:
        if not any(ratio > 0.0 for ratio in candidate_ratios.values()):
            # Nothing fits the budget. Fine-tuning here would hand back a model that
            # differs from the original although the result says nothing was pruned.
            return copy.deepcopy(model), report.baseline_score
        candidate = build(candidate_ratios)
        if finetune_fn is not None:
            finetune_fn(candidate)
        return candidate, score_fn(candidate)

    candidate, score = verify(ratios)
    refinements, accepted_scale = 0, 1.0
    if budget.max_drop is not None and report.baseline_score - score > budget.max_drop:
        # Predictions were too optimistic: bisect a common scale factor on all ratios.
        low, high, accepted = 0.0, 1.0, None
        for _ in range(max_refinements):
            refinements += 1
            scale = (low + high) / 2
            scaled = {gid: ratio * scale for gid, ratio in ratios.items()}
            trial, trial_score = verify(scaled)
            if report.baseline_score - trial_score <= budget.max_drop:
                low, accepted = scale, (scaled, trial, trial_score)
            else:
                high = scale
        accepted_scale = low
        if accepted is None:
            ratios = dict.fromkeys(ratios, 0.0)
            candidate, score = copy.deepcopy(model), report.baseline_score
        else:
            ratios, candidate, score = accepted

    return BudgetResult(
        model=candidate,
        ratios={gid: ratio for gid, ratio in ratios.items() if ratio > 0.0},
        baseline_score=report.baseline_score,
        score=score,
        flops_ratio=count_flops(candidate, example_input) / report.baseline_flops,
        predicted_drop=predicted,
        refinements=refinements,
        history=history,
        scale=accepted_scale,
    )
