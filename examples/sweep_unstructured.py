"""Unstructured pruning sweep: what really changes as more weights become zero?

uv run python examples/train_fashion_mnist.py   # once
uv run python examples/sweep_unstructured.py --model cnn
"""

from __future__ import annotations

import torch
import typer
from fashion_data import get_loaders, model_path

from pymodelprune import make_accuracy_fn, profile_model, render_profiles, retime_profiles
from pymodelprune.demo_models import DEMO_INPUT_SHAPE
from pymodelprune.prune import prune_unstructured


def main(model: str = "mlp", scope: str = "global") -> None:
    original = torch.load(model_path(model), weights_only=False)
    _, test_loader = get_loaders()
    eval_fn = make_accuracy_fn(test_loader)
    example = torch.randn(DEMO_INPUT_SHAPE)

    models = [original]
    profiles = [profile_model(original, example, name="original", eval_fn=eval_fn)]
    for amount in (0.3, 0.5, 0.7, 0.9, 0.95):
        pruned = prune_unstructured(original, amount, scope=scope)
        models.append(pruned)
        profiles.append(
            profile_model(
                pruned,
                example,
                name=f"pruned {amount:.0%}",
                eval_fn=eval_fn,
                reference=original,
                fidelity_inputs=test_loader,
            )
        )
    # an evaluation sits between the rows: time them side by side so load drift cancels out
    render_profiles(retime_profiles(profiles, models, example))


if __name__ == "__main__":
    typer.run(main)
