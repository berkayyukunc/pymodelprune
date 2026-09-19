"""Pruning methods."""

from pymodelprune.prune.graph import (
    ChannelGroup,
    DependencyGraph,
    UntraceableModelError,
    build_dependency_graph,
)
from pymodelprune.prune.importance import make_classification_grad_fn
from pymodelprune.prune.structured import apply_plan, mask_plan, plan_pruning, prune_structured
from pymodelprune.prune.unstructured import make_permanent, prune_unstructured

__all__ = [
    "ChannelGroup",
    "DependencyGraph",
    "UntraceableModelError",
    "apply_plan",
    "build_dependency_graph",
    "make_classification_grad_fn",
    "make_permanent",
    "mask_plan",
    "plan_pruning",
    "prune_structured",
    "prune_unstructured",
]
