"""How closely does an optimized model reproduce the original's outputs?

Needs no labels, so it is available even when the caller has no `EvalFn`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from pymodelprune.evaluate import model_device

MIN_SAMPLES = 32
"""Below this many samples an agreement percentage is mostly noise."""


class FewSamplesWarning(UserWarning):
    """Fidelity was computed on too few samples to mean much."""


@dataclass(frozen=True)
class Fidelity:
    agreement: float
    """Fraction of samples where both models pick the same class (argmax of the last dim)."""
    max_abs_diff: float
    """Largest output difference; NaN as soon as either model produced a NaN."""
    samples: int


def as_inputs(example_input: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple:
    return example_input if isinstance(example_input, tuple) else (example_input,)


def model_inputs(batch: torch.Tensor | tuple | list, num_inputs: int = 1) -> tuple:
    """The positional model inputs inside one batch; trailing items (labels) are dropped."""
    if isinstance(batch, (tuple, list)):
        if len(batch) < num_inputs:
            raise ValueError(f"the model takes {num_inputs} inputs, the batch has {len(batch)}")
        return tuple(batch[:num_inputs])
    if num_inputs != 1:
        raise ValueError(f"the model takes {num_inputs} inputs, the batch is a single tensor")
    return (batch,)


def fidelity_batches(
    example_input: torch.Tensor | tuple, fidelity_inputs: torch.Tensor | Iterable | None
) -> tuple[Iterable, int]:
    """(batches, num_inputs) for `compare_outputs`; falls back to the example input."""
    inputs = as_inputs(example_input)
    batches = fidelity_inputs if fidelity_inputs is not None else [inputs]
    return batches, len(inputs)


def _as_batches(inputs: torch.Tensor | Iterable) -> Iterable:
    return [inputs] if isinstance(inputs, torch.Tensor) else inputs


def _forward(model: nn.Module, inputs: tuple) -> torch.Tensor:
    output = model(*inputs)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"fidelity needs a tensor output, model returned {type(output).__name__}")
    return output


def compare_outputs(
    reference: nn.Module,
    candidate: nn.Module,
    inputs: torch.Tensor | Iterable,
    num_inputs: int = 1,
) -> Fidelity:
    """Run both models on the same inputs and compare their outputs.

    `inputs` is one batch tensor, or an iterable of batches. A tuple/list batch
    contributes its first `num_inputs` items as the model inputs and the rest
    (labels) is ignored, so a DataLoader works: `(x, label)` with num_inputs=1,
    `(x1, x2)` or `(x1, x2, label)` with num_inputs=2.
    Both models must live on the same device; every batch is moved there (the device
    of `reference`'s first parameter, CPU for an `OrtModule`), so data that was left
    on an accelerator still compares against models that are profiled on the CPU.
    Warns (`FewSamplesWarning`) below 32 samples: the agreement is then mostly noise.
    """
    modes = [(model, model.training) for model in (reference, candidate)]
    for model, _ in modes:
        model.eval()

    device = model_device(reference)
    same = 0
    samples = 0
    max_abs_diff = 0.0
    try:
        with torch.inference_mode():
            for batch in _as_batches(inputs):
                # lazily, batch by batch: a whole dataset need not fit on the device twice
                arguments = tuple(item.to(device) for item in model_inputs(batch, num_inputs))
                expected = _forward(reference, arguments)
                actual = _forward(candidate, arguments)
                if expected.shape != actual.shape:
                    raise ValueError(
                        f"output shapes differ: {tuple(expected.shape)} vs {tuple(actual.shape)}"
                    )
                difference = float((expected - actual).abs().max())
                # max() drops NaN (every comparison with it is False), which would
                # report broken outputs as a perfect 0.0. Once NaN, stay NaN.
                if math.isnan(difference) or difference > max_abs_diff:
                    max_abs_diff = difference
                same += int((expected.argmax(dim=-1) == actual.argmax(dim=-1)).sum())
                samples += expected.argmax(dim=-1).numel()
    finally:
        for model, was_training in modes:
            model.train(was_training)

    if samples == 0:
        raise ValueError("no samples to compare")
    if samples < MIN_SAMPLES:
        warnings.warn(
            f"fidelity was computed on only {samples} sample(s), which says little about "
            "real data; pass fidelity_inputs (CLI: --data) with representative batches",
            FewSamplesWarning,
            stacklevel=2,
        )
    return Fidelity(agreement=same / samples, max_abs_diff=max_abs_diff, samples=samples)
