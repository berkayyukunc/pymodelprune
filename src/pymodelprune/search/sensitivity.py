"""Per-group sensitivity analysis: prune one group at a time and watch the score.

Layers differ wildly in how much pruning they tolerate. The curves measured here
are what lets the budget search remove more from robust groups and less from
fragile ones, instead of one ratio everywhere.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from pymodelprune._json import write_json
from pymodelprune.evaluate import EvalFn, reusable
from pymodelprune.fidelity import MIN_SAMPLES, FewSamplesWarning, compare_outputs, fidelity_batches
from pymodelprune.profile import count_flops
from pymodelprune.prune.graph import DependencyGraph, build_dependency_graph
from pymodelprune.prune.structured import GradFn, apply_plan, plan_pruning

DEFAULT_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class SensitivityPoint:
    ratio: float
    score: float
    flops: int


@dataclass
class SensitivityReport:
    metric: str
    baseline_score: float
    baseline_flops: int
    curves: dict[int, list[SensitivityPoint]] = field(default_factory=dict)
    layers: dict[int, list[str]] = field(default_factory=dict)

    def drop(self, group_id: int, ratio: float) -> float:
        """Score lost by pruning only `group_id` at a measured `ratio` (0 for ratio 0)."""
        if ratio == 0.0:
            return 0.0
        point = next(p for p in self.curves[group_id] if p.ratio == ratio)
        return self.baseline_score - point.score

    def saved_flops(self, group_id: int, ratio: float) -> int:
        if ratio == 0.0:
            return 0
        point = next(p for p in self.curves[group_id] if p.ratio == ratio)
        return self.baseline_flops - point.flops

    def ranking(self, ratio: float | None = None) -> list[tuple[int, float]]:
        """Groups ordered from most to least sensitive at `ratio` (default: the middle one)."""
        if not self.curves:
            return []
        ratios = [p.ratio for p in next(iter(self.curves.values()))]
        ratio = ratio if ratio is not None else ratios[len(ratios) // 2]
        drops = [(gid, self.drop(gid, ratio)) for gid in self.curves]
        return sorted(drops, key=lambda item: item[1], reverse=True)

    def to_json(self, path: str | Path) -> None:
        """Strict JSON: a NaN score (an eval_fn that broke down) is written as null."""
        write_json(asdict(self), path)

    @classmethod
    def from_json(cls, path: str | Path) -> SensitivityReport:
        raw = json.loads(Path(path).read_text())

        def score(value: float | None) -> float:
            return float("nan") if value is None else value

        return cls(
            metric=raw["metric"],
            baseline_score=score(raw["baseline_score"]),
            baseline_flops=raw["baseline_flops"],
            curves={
                int(gid): [
                    SensitivityPoint(point["ratio"], score(point["score"]), point["flops"])
                    for point in points
                ]
                for gid, points in raw["curves"].items()
            },
            layers={int(gid): names for gid, names in raw["layers"].items()},
        )


def make_score_fn(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    eval_fn: EvalFn | None,
    fidelity_inputs: torch.Tensor | Iterable | None,
) -> tuple[str, EvalFn]:
    """The user's metric if there is one, else agreement with the original model.

    The fallback refuses to run on fewer than 32 samples: agreement on a handful of
    inputs is all-or-nothing, and a search steered or verified by it would be a lie.
    """
    if eval_fn is not None:
        # float(): numpy and 0-d tensor scores would not survive the JSON reports
        return "eval_fn", lambda candidate: float(eval_fn(candidate))
    batches, num_inputs = fidelity_batches(example_input, reusable(fidelity_inputs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FewSamplesWarning)  # about to become an error
        samples = compare_outputs(model, model, batches, num_inputs).samples
    if samples < MIN_SAMPLES:
        raise ValueError(
            f"the search needs a score: pass eval_fn, or fidelity_inputs with at least "
            f"{MIN_SAMPLES} samples (got {samples}). CLI: --eval or --data"
        )
    return (
        "fidelity",
        lambda candidate: compare_outputs(model, candidate, batches, num_inputs).agreement,
    )


def analyze_sensitivity(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    eval_fn: EvalFn | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    importance: str = "l1",
    grad_fn: GradFn | None = None,
    graph: DependencyGraph | None = None,
    progress: Callable[[int, int], None] | None = None,
    round_to: int = 1,
    min_channels: int = 1,
) -> SensitivityReport:
    """Measure score and FLOPs for every (prunable group, ratio) pair, one group at a time.

    Costs `groups x ratios` evaluations, so pass an `eval_fn` over a small validation
    subset. Without `eval_fn`, agreement with the original model on `fidelity_inputs`
    is used instead; that needs at least 32 samples (ValueError otherwise).
    `round_to` / `min_channels` must match the pruning the curves will be used for.
    """
    ratios = sorted(set(ratios))
    if not ratios or not all(0.0 < ratio < 1.0 for ratio in ratios):
        raise ValueError("ratios must be in (0, 1)")
    graph = graph or build_dependency_graph(model, example_input)
    baseline_flops = count_flops(model, example_input)
    if baseline_flops is None:
        raise ValueError("cannot count FLOPs for this model")

    fidelity_inputs = reusable(fidelity_inputs)
    metric, score_fn = make_score_fn(model, example_input, eval_fn, fidelity_inputs)
    scored = copy.deepcopy(model)
    if importance == "taylor":
        if grad_fn is None:
            raise ValueError("importance='taylor' needs grad_fn to populate gradients")
        grad_fn(scored)

    report = SensitivityReport(
        metric=metric, baseline_score=score_fn(model), baseline_flops=baseline_flops
    )
    total, done = len(graph.prunable_groups) * len(ratios), 0
    for group in graph.prunable_groups:
        report.layers[group.id] = list(group.producers)
        report.curves[group.id] = []
        for ratio in ratios:
            plan = plan_pruning(
                scored, graph, {group.id: ratio}, importance,
                round_to=round_to, min_channels=min_channels,
            )  # fmt: skip
            candidate = apply_plan(copy.deepcopy(model), graph, plan)
            report.curves[group.id].append(
                SensitivityPoint(ratio, score_fn(candidate), count_flops(candidate, example_input))
            )
            done += 1
            if progress is not None:
                progress(done, total)
    return report
