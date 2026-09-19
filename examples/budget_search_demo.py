"""Let the tool decide how much to prune: "lose at most one accuracy point".

uv run python examples/train_fashion_mnist.py   # once
uv run python examples/budget_search_demo.py --max-drop 0.01
"""

from __future__ import annotations

import torch
import typer
from fashion_data import get_loaders, model_path

from pymodelprune import Budget, PruneConfig, build_dependency_graph, make_accuracy_fn, optimize
from pymodelprune.demo_models import DEMO_INPUT_SHAPE


def main(model: str = "cnn", max_drop: float = 0.01) -> None:
    original = torch.load(model_path(model), weights_only=False)
    _, test_loader = get_loaders()
    example = torch.randn(DEMO_INPUT_SHAPE)

    result = optimize(
        original,
        example,
        prune=PruneConfig(budget=Budget(max_drop=max_drop)),
        eval_fn=make_accuracy_fn(test_loader),
        fidelity_inputs=test_loader,
    )
    result.report()

    graph = build_dependency_graph(original, example)
    print("\nchosen pruning ratio per group:")
    for group in graph.prunable_groups:
        ratio = result.search.ratios.get(group.id, 0.0)
        print(f"  {', '.join(group.producers):<24} {ratio:.0%}")
    print(
        f"predicted drop {result.search.predicted_drop:.2%}, measured {result.search.measured_drop:.2%}, "
        f"refinements {result.search.refinements}"
    )
    print("greedy path (FLOPs ratio -> predicted drop):")
    print(
        "  "
        + "  ".join(f"{s.flops_ratio:.2f}->{s.predicted_drop:.1%}" for s in result.search.history)
    )


if __name__ == "__main__":
    typer.run(main)
