"""Structured pruning sweep: channels are removed for real, so everything shrinks.

uv run python examples/train_fashion_mnist.py   # once
uv run python examples/sweep_structured.py --model cnn --finetune
"""

from __future__ import annotations

import torch
import typer
from fashion_data import get_loaders, make_finetune_fn, model_path

from pymodelprune import (
    make_accuracy_fn,
    make_classification_grad_fn,
    profile_model,
    prune_structured,
    render_profiles,
    retime_profiles,
)
from pymodelprune.demo_models import DEMO_INPUT_SHAPE


def main(model: str = "cnn", importance: str = "l1", finetune: bool = False) -> None:
    original = torch.load(model_path(model), weights_only=False)
    train_loader, test_loader = get_loaders()
    eval_fn = make_accuracy_fn(test_loader)
    recover = make_finetune_fn(train_loader)
    example = torch.randn(DEMO_INPUT_SHAPE)
    # taylor ranks channels by weight x gradient, so it needs gradients from real data
    grad_fn = make_classification_grad_fn(train_loader, 8) if importance == "taylor" else None

    models = [original]
    profiles = [profile_model(original, example, name="original", eval_fn=eval_fn)]
    for amount in (0.3, 0.5, 0.7, 0.9):
        pruned = prune_structured(original, example, amount, importance=importance, grad_fn=grad_fn)
        if finetune:
            recover(pruned)
        models.append(pruned)
        profiles.append(
            profile_model(
                pruned,
                example,
                name=f"-{amount:.0%} ch" + (" +ft" if finetune else ""),
                eval_fn=eval_fn,
                reference=original,
                fidelity_inputs=test_loader,
            )
        )
    # fine-tuning sits between the rows: time them side by side so load drift cancels out
    render_profiles(retime_profiles(profiles, models, example))


if __name__ == "__main__":
    typer.run(main)
