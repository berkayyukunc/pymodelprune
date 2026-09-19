"""Unstructured (element-wise) magnitude pruning on top of `torch.nn.utils.prune`.

Zeros individual weights. The tensors keep their shape, so dense kernels do the
same amount of work afterwards: expect a smaller *compressed* file, not a faster model.
"""

from __future__ import annotations

import copy

import torch.nn.utils.prune as torch_prune
from torch import nn

DEFAULT_LAYERS: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv1d, nn.Conv2d)

# For single weights |w| and w**2 rank identically, so there is no separate "l2"
# option here. The L1/L2 distinction only matters for structured pruning.
_METHODS = {"l1": torch_prune.L1Unstructured, "random": torch_prune.RandomUnstructured}


def make_permanent(model: nn.Module) -> nn.Module:
    """Fold pruning masks into the weights and drop the `_orig`/`_mask` tensors, in place."""
    for module in model.modules():
        for name in [n.removesuffix("_mask") for n, _ in module.named_buffers(recurse=False)]:
            if hasattr(module, f"{name}_orig") and hasattr(module, f"{name}_mask"):
                torch_prune.remove(module, name)
    return model


def prune_unstructured(
    model: nn.Module,
    amount: float,
    criterion: str = "l1",
    scope: str = "global",
    layers: tuple[type[nn.Module], ...] = DEFAULT_LAYERS,
    permanent: bool = True,
) -> nn.Module:
    """Return a copy of `model` with a fraction `amount` of its weights set to zero.

    scope="global" ranks all weights together, so layers end up with different
    sparsity. scope="layer" removes the same fraction from every layer.
    Biases and normalization layers are never pruned.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("amount must be in [0, 1)")
    if criterion not in _METHODS:
        raise ValueError(f"criterion must be one of {sorted(_METHODS)}")
    if scope not in ("global", "layer"):
        raise ValueError("scope must be 'global' or 'layer'")

    pruned = copy.deepcopy(model)
    targets = [(module, "weight") for module in pruned.modules() if isinstance(module, layers)]
    if not targets:
        raise ValueError(f"model has no layers of type {[t.__name__ for t in layers]}")
    if amount == 0.0:
        return pruned

    method = _METHODS[criterion]
    if scope == "global":
        torch_prune.global_unstructured(targets, pruning_method=method, amount=amount)
    else:
        for module, name in targets:
            method.apply(module, name, amount=amount)

    return make_permanent(pruned) if permanent else pruned
