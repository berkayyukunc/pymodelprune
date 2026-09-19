"""Damage a trained model on purpose and watch accuracy and fidelity react.

    uv run python examples/train_fashion_mnist.py   # once
    uv run python examples/fidelity_demo.py

Gaussian noise is added to every weight, scaled by that tensor's own standard
deviation. This stands in for the damage pruning and quantization will do later.
"""

from __future__ import annotations

import copy

import torch
import typer
from fashion_data import get_loaders, model_path
from torch import nn

from pymodelprune import make_accuracy_fn, profile_model, render_profiles, retime_profiles
from pymodelprune.demo_models import DEMO_INPUT_SHAPE


def add_weight_noise(model: nn.Module, relative_std: float) -> nn.Module:
    noisy = copy.deepcopy(model)
    with torch.no_grad():
        for param in noisy.parameters():
            if param.numel() > 1:
                param.add_(torch.randn_like(param) * param.std() * relative_std)
    return noisy


def main(model: str = "mlp", seed: int = 0) -> None:
    path = model_path(model)
    if not path.exists():
        raise typer.BadParameter(f"{path} not found, run train_fashion_mnist.py first")
    original = torch.load(path, weights_only=False)
    _, test_loader = get_loaders()
    eval_fn = make_accuracy_fn(test_loader)
    example = torch.randn(DEMO_INPUT_SHAPE)

    torch.manual_seed(seed)
    models = [original]
    profiles = [profile_model(original, example, name="original", eval_fn=eval_fn)]
    for relative_std in (0.1, 0.5, 1.0, 2.0):
        models.append(add_weight_noise(original, relative_std))
        profiles.append(
            profile_model(
                models[-1],
                example,
                name=f"noise x{relative_std}",
                eval_fn=eval_fn,
                reference=original,
                fidelity_inputs=test_loader,
            )
        )
    # same architecture in every row: timed side by side, the latencies should agree
    render_profiles(retime_profiles(profiles, models, example))


if __name__ == "__main__":
    typer.run(main)
