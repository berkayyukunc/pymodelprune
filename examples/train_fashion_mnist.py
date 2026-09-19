"""Train the two demo models on FashionMNIST and profile them.

    uv run python examples/train_fashion_mnist.py

Training uses the Mac GPU (MPS) or CUDA when available. Profiling always runs on CPU.
Saved models land in data/models/ and are reused by the later examples.
"""

from __future__ import annotations

import time

import torch
import typer
from fashion_data import MODEL_DIR, get_loaders, model_path
from torch import nn
from torch.utils.data import DataLoader

from pymodelprune import accuracy, make_accuracy_fn, profile_model, render_profiles
from pymodelprune.demo_models import DEMO_INPUT_SHAPE, DEMO_MODELS


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(model: nn.Module, loader: DataLoader, epochs: int, device: torch.device) -> None:
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = loss_fn(model(inputs), labels)  # how wrong are we?
            optimizer.zero_grad()
            loss.backward()  # which way should every weight move?
            optimizer.step()  # move them a little
            running_loss += loss.item()
        print(f"  epoch {epoch}/{epochs}  loss {running_loss / len(loader):.4f}")

    model.cpu().eval()


def main(epochs: int = 3, seed: int = 0) -> None:
    device = pick_device()
    train_loader, test_loader = get_loaders()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"training on {device}")

    profiles = []
    for name, factory in DEMO_MODELS.items():
        torch.manual_seed(seed)
        model = factory()
        print(f"{name}: accuracy before training {accuracy(model, test_loader):.2%}")
        start = time.perf_counter()
        train(model, train_loader, epochs, device)
        print(f"{name}: trained in {time.perf_counter() - start:.0f}s")

        torch.save(model, model_path(name))
        profiles.append(
            profile_model(
                model,
                torch.randn(DEMO_INPUT_SHAPE),
                name=name,
                eval_fn=make_accuracy_fn(test_loader),
            )
        )

    render_profiles(profiles, compare=False)
    print(f"models saved to {MODEL_DIR}")


if __name__ == "__main__":
    typer.run(main)
