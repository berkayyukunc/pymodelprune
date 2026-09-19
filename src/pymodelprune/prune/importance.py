"""Channel importance criteria for structured pruning.

Every criterion returns one score per channel of a group, higher = keep. Scores of
the group's layers are divided by their mean and then averaged. Without the first
step the layer with the largest weights would decide alone; without the second,
groups with many layers (residual streams) would look more important than the rest
under a global threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from pymodelprune.prune.graph import ChannelGroup


def _normalized(score: torch.Tensor) -> torch.Tensor:
    return score / score.mean().clamp_min(1e-12)


def _weight_norm(model: nn.Module, group: ChannelGroup, p: int) -> torch.Tensor:
    total = torch.zeros(group.channels)
    for name in group.producers:
        weight = model.get_submodule(name).weight.detach().float().cpu()
        total += _normalized(weight.flatten(1).norm(p=p, dim=1))
    return total / len(group.producers)


def l1(model: nn.Module, group: ChannelGroup) -> torch.Tensor:
    return _weight_norm(model, group, 1)


def l2(model: nn.Module, group: ChannelGroup) -> torch.Tensor:
    return _weight_norm(model, group, 2)


def bn_scale(model: nn.Module, group: ChannelGroup) -> torch.Tensor:
    """|gamma| of the group's BatchNorm layers (Network Slimming). Falls back to L1."""
    scales = [model.get_submodule(name).weight for name in group.norms]
    scales = [scale.detach().abs().float().cpu() for scale in scales if scale is not None]
    if not scales:
        return l1(model, group)
    return sum(_normalized(scale) for scale in scales) / len(scales)


def taylor(model: nn.Module, group: ChannelGroup) -> torch.Tensor:
    """First order estimate of the loss change when a channel is removed: |sum(w * dL/dw)|."""
    total = torch.zeros(group.channels)
    for name in group.producers:
        module = model.get_submodule(name)
        if module.weight.grad is None:
            raise ValueError(f"'{name}' has no gradient; run backward passes before taylor scoring")
        score = (module.weight.detach() * module.weight.grad).flatten(1).sum(dim=1)
        if module.bias is not None and module.bias.grad is not None:
            score = score + module.bias.detach() * module.bias.grad
        total += _normalized(score.abs().float().cpu())
    return total / len(group.producers)


def random(model: nn.Module, group: ChannelGroup) -> torch.Tensor:
    return torch.rand(group.channels)


CRITERIA: dict[str, Callable[[nn.Module, ChannelGroup], torch.Tensor]] = {
    "l1": l1,
    "l2": l2,
    "bn": bn_scale,
    "taylor": taylor,
    "random": random,
}


def group_importance(model: nn.Module, group: ChannelGroup, criterion: str) -> torch.Tensor:
    return CRITERIA[criterion](model, group)


def make_classification_grad_fn(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]], max_batches: int = 8
) -> Callable[[nn.Module], None]:
    """Gradient provider for `importance="taylor"`: cross entropy over a few batches."""

    def grad_fn(model: nn.Module) -> None:
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()  # keep BatchNorm statistics frozen while collecting gradients
        loss_fn = nn.CrossEntropyLoss()
        try:
            for index, (inputs, labels) in enumerate(batches):
                if index >= max_batches:
                    break
                loss_fn(model(inputs.to(device)), labels.to(device)).backward()
        finally:
            model.train(was_training)

    return grad_fn
