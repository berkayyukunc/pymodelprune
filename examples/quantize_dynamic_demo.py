"""Dynamic INT8 quantization: every backend and mode, on both demo models.

uv run python examples/train_fashion_mnist.py   # once
uv run python examples/quantize_dynamic_demo.py
"""

from __future__ import annotations

import torch
import typer
from fashion_data import get_loaders, model_path

from pymodelprune import make_accuracy_fn, profile_model, render_profiles, retime_profiles
from pymodelprune.demo_models import DEMO_INPUT_SHAPE
from pymodelprune.quantize import quantize_dynamic
from pymodelprune.quantize.dynamic import torchao_available


def main() -> None:
    _, test_loader = get_loaders()
    eval_fn = make_accuracy_fn(test_loader)
    example = torch.randn(DEMO_INPUT_SHAPE)
    variants = [("legacy", "dynamic")]
    if torchao_available():
        variants += [("torchao", "dynamic"), ("torchao", "weight_only")]

    for name in ("mlp", "cnn"):
        original = torch.load(model_path(name), weights_only=False)
        models = [original]
        profiles = [profile_model(original, example, name=name, eval_fn=eval_fn)]
        for backend, mode in variants:
            quantized = quantize_dynamic(original, mode=mode, backend=backend)
            models.append(quantized)
            label = f"{backend[:3]} {mode[:6]}"
            profiles.append(
                profile_model(
                    quantized,
                    example,
                    name=label,
                    eval_fn=eval_fn,
                    reference=original,
                    fidelity_inputs=test_loader,
                    # same arithmetic, lower precision: the float count still applies
                    flops=profiles[0].flops,
                )
            )
        # an evaluation sits between the rows: time them side by side so load drift cancels out
        render_profiles(retime_profiles(profiles, models, example))
        print()


if __name__ == "__main__":
    typer.run(main)
