"""Task metrics. The library never owns a dataset: callers hand it an `EvalFn`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

import torch
from torch import nn

EvalFn = Callable[[nn.Module], float]
"""Takes a model, returns a score where higher is better (e.g. accuracy in [0, 1])."""

Batches = TypeVar("Batches")


def reusable(batches: Batches) -> Batches | list:
    """Return `batches` in a form that can be iterated more than once.

    Every stage (search, profiling, calibration) walks the data again. A generator
    is empty after the first walk, so one-shot iterators are materialised as a list;
    anything re-iterable (tensor, list, DataLoader) is returned unchanged.
    """
    if batches is None or isinstance(batches, torch.Tensor):
        return batches
    return list(batches) if iter(batches) is batches else batches


def model_device(model: nn.Module) -> torch.device:
    """Device of the first parameter; CPU for modules without any (an `OrtModule`)."""
    first = next(model.parameters(), None)
    return first.device if first is not None else torch.device("cpu")


def accuracy(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    max_batches: int | None = None,
) -> float:
    """Top-1 classification accuracy in [0, 1] over `(inputs, labels)` batches."""
    device = model_device(model)
    was_training = model.training
    model.eval()

    correct = 0
    seen = 0
    try:
        with torch.inference_mode():
            for index, (inputs, labels) in enumerate(batches):
                if max_batches is not None and index >= max_batches:
                    break
                predictions = model(inputs.to(device)).argmax(dim=-1)
                correct += int((predictions == labels.to(device)).sum())
                seen += labels.numel()
    finally:
        model.train(was_training)

    if seen == 0:
        raise ValueError("no samples to evaluate")
    return correct / seen


def make_accuracy_fn(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    max_batches: int | None = None,
) -> EvalFn:
    """Bind a dataset to `accuracy`. `batches` must be re-iterable (a DataLoader or a list)."""

    def eval_fn(model: nn.Module) -> float:
        return accuracy(model, batches, max_batches=max_batches)

    return eval_fn
