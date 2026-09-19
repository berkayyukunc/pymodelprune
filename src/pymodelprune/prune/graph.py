"""Dependency analysis for structured pruning.

Removing output channel `c` of a layer forces matching removals elsewhere: the
BatchNorm that follows, the input side of every layer that reads the tensor, and
the producers of every other branch it is added to. A `ChannelGroup` is one such set
of tensors that must always be sliced with the same channel indices.

The model is traced with `torch.fx`, shapes are recorded with one forward pass,
and the graph is walked once in topological order. Anything not understood marks
the affected groups as not prunable: skipping a layer is safe, guessing is not.
"""

from __future__ import annotations

import copy
import math
import operator
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.utils import parametrize


class UntraceableModelError(RuntimeError):
    """The model cannot be traced by torch.fx (usually data dependent control flow)."""


@dataclass
class ChannelGroup:
    id: int
    channels: int
    producers: list[str] = field(default_factory=list)
    """Conv/Linear modules whose *output* channels belong to the group."""
    norms: list[str] = field(default_factory=list)
    depthwise: list[str] = field(default_factory=list)
    """Depthwise convolutions: input and output channels are tied together."""
    consumers: list[tuple[str, int]] = field(default_factory=list)
    """(module, fan): modules whose *input* reads the group. After a flatten each
    channel spans `fan` consecutive input features."""
    blockers: list[str] = field(default_factory=list)

    @property
    def prunable(self) -> bool:
        return not self.blockers and bool(self.producers)


@dataclass
class DependencyGraph:
    groups: list[ChannelGroup]

    @property
    def prunable_groups(self) -> list[ChannelGroup]:
        return [group for group in self.groups if group.prunable]

    def group(self, group_id: int) -> ChannelGroup:
        for group in self.groups:
            if group.id == group_id:
                return group
        raise KeyError(group_id)


# Shape preserving ops with f(0) == 0. That property is what makes "zero a channel"
# and "remove a channel" equivalent, so sigmoid-like functions are deliberately absent.
_PASSTHROUGH_MODULES = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.GELU,
    nn.SiLU,
    nn.Tanh,
    nn.ELU,
    nn.Hardswish,
    nn.Hardtanh,
    nn.Mish,
    nn.Identity,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
)
_PASSTHROUGH_FUNCTIONS = {
    F.relu,
    F.relu6,
    F.leaky_relu,
    F.gelu,
    F.silu,
    F.tanh,
    F.elu,
    F.hardswish,
    F.hardtanh,
    F.mish,
    F.dropout,
    F.max_pool1d,
    F.max_pool2d,
    F.avg_pool1d,
    F.avg_pool2d,
    F.adaptive_avg_pool1d,
    F.adaptive_avg_pool2d,
    F.adaptive_max_pool1d,
    F.adaptive_max_pool2d,
    torch.relu,
    torch.tanh,
}
_PASSTHROUGH_METHODS = {"relu", "relu_", "tanh", "contiguous", "clone", "detach"}
_JOIN_FUNCTIONS = {operator.add, operator.iadd, torch.add, operator.sub, torch.sub}
_SCALE_FUNCTIONS = {operator.mul, torch.mul, operator.truediv, torch.div}
_DIVISIONS = {operator.truediv, torch.div, "div", "div_"}
_SHAPE_QUERIES = {"size", "dim", "numel", "ndimension"}
_JOIN_METHODS = {"add", "add_", "sub", "sub_"}
_SCALE_METHODS = {"mul", "mul_", "div", "div_"}
_RESHAPE_FUNCTIONS = {torch.flatten, torch.reshape}
_RESHAPE_METHODS = {"flatten", "view", "reshape"}
_FLATTENS = {torch.flatten, "flatten"}


class _Groups:
    """Union-find over channel groups."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.data: list[ChannelGroup] = []

    def new(self, channels: int) -> int:
        group_id = len(self.parent)
        self.parent.append(group_id)
        self.data.append(ChannelGroup(id=group_id, channels=channels))
        return group_id

    def find(self, group_id: int) -> int:
        while self.parent[group_id] != group_id:
            self.parent[group_id] = self.parent[self.parent[group_id]]
            group_id = self.parent[group_id]
        return group_id

    def get(self, group_id: int) -> ChannelGroup:
        return self.data[self.find(group_id)]

    def block(self, group_id: int | None, reason: str) -> None:
        if group_id is not None and reason not in self.get(group_id).blockers:
            self.get(group_id).blockers.append(reason)

    def union(self, first: int, second: int) -> int:
        a, b = self.find(first), self.find(second)
        if a == b:
            return a
        keep, drop = self.data[a], self.data[b]
        if keep.channels != drop.channels:
            keep.blockers.append("joined branches have different channel counts")
        keep.producers += drop.producers
        keep.norms += drop.norms
        keep.depthwise += drop.depthwise
        keep.consumers += drop.consumers
        keep.blockers += [reason for reason in drop.blockers if reason not in keep.blockers]
        self.parent[b] = a
        return a


def _shape(node: fx.Node) -> tuple[int, ...] | None:
    meta = node.meta.get("tensor_meta")
    return tuple(meta.shape) if hasattr(meta, "shape") else None


def _symbolic_trace(model: nn.Module) -> fx.GraphModule:
    try:
        return fx.symbolic_trace(model)
    except Exception as error:
        raise UntraceableModelError(
            f"torch.fx could not trace {type(model).__name__}: {error}. Structured pruning "
            "needs a static graph; unstructured pruning and quantization still work."
        ) from error


def trace(model: nn.Module, example_input: torch.Tensor | tuple) -> fx.GraphModule:
    """Trace a private copy of `model` and annotate every node with its output shape.

    A model whose graph differs between train and eval mode (auxiliary heads, ...) is
    analysed in train mode, so layers that only run during training are sliced too.
    """
    if isinstance(example_input, list):
        example_input = tuple(example_input)
    inputs = example_input if isinstance(example_input, tuple) else (example_input,)
    try:
        private = copy.deepcopy(model)
    except RuntimeError as error:
        raise ValueError(
            f"cannot copy the model ({error}). If pruning masks from torch.nn.utils.prune "
            "are attached, fold them in first with pymodelprune.prune.make_permanent(model)."
        ) from error

    traced = _symbolic_trace(private.eval())
    train_traced = _symbolic_trace(copy.deepcopy(private).train())
    if train_traced.code != traced.code:
        eval_modules = {n.target for n in traced.graph.nodes if n.op == "call_module"}
        train_modules = {n.target for n in train_traced.graph.nodes if n.op == "call_module"}
        if not eval_modules <= train_modules:
            raise UntraceableModelError(
                "the model runs layers in eval mode that it skips in train mode "
                f"({sorted(eval_modules - train_modules)}); structured pruning cannot cover both"
            )
        traced = train_traced
        # BatchNorm refuses a single sample in train mode
        inputs = tuple(torch.cat([item, item]) if item.shape[0] == 1 else item for item in inputs)

    try:
        with torch.no_grad():
            ShapeProp(traced).propagate(*inputs)
    except Exception as error:
        raise ValueError(
            f"the example input does not run through the model: {str(error).splitlines()[0]}. "
            "Check its shape, dtype and device (it must match the model's device)."
        ) from error
    return traced


def _literal_sizes(node: fx.Node) -> list[int]:
    """Hard coded sizes in view/reshape calls, e.g. the 400 in `x.view(-1, 400)`."""
    sizes = []
    for arg in node.args[1:]:
        items = arg if isinstance(arg, (tuple, list)) else (arg,)
        sizes += [item for item in items if isinstance(item, int) and item != -1]
    return sizes


def build_dependency_graph(
    model: nn.Module, example_input: torch.Tensor | tuple
) -> DependencyGraph:
    traced = trace(model, example_input)
    modules = dict(traced.named_modules())
    calls = Counter(node.target for node in traced.graph.nodes if node.op == "call_module")
    groups = _Groups()
    # node -> (group id, fan). Missing/None means "channels not tracked here".
    flow: dict[fx.Node, tuple[int, int] | None] = {}
    # A module called more than once has ONE set of output channels: later calls join
    # the group of the first, so the producer is listed in exactly one (blocked) group.
    first_call: dict[str, int] = {}

    def block_inputs(node: fx.Node, reason: str) -> None:
        for source in node.all_input_nodes:
            if flow.get(source) is not None:
                groups.block(flow[source][0], reason)

    def passthrough(node: fx.Node) -> tuple[int, int] | None:
        source = node.all_input_nodes[0]
        before, after = _shape(source), _shape(node)
        if flow.get(source) is None:
            return None
        if before is None or after is None or len(before) != len(after) or before[1] != after[1]:
            groups.block(flow[source][0], f"'{node.name}' changes the channel dimension")
            return None
        return flow[source]

    def reshape(node: fx.Node) -> tuple[int, int] | None:
        source = node.all_input_nodes[0]
        tracked, before, after = flow.get(source), _shape(source), _shape(node)
        if tracked is None:
            return None
        flattens = (
            before is not None
            and after is not None
            and len(after) == 2
            and after[0] == before[0]
            and after[1] == math.prod(before[1:])
        )
        if flattens and node.op != "call_module" and node.target not in _FLATTENS:
            flattens = not _literal_sizes(node)  # a hard coded size breaks once channels go
        if not flattens:
            groups.block(tracked[0], f"unsupported reshape '{node.name}'")
            return None
        return tracked[0], tracked[1] * math.prod(before[2:])

    def join(node: fx.Node, scaling: bool) -> tuple[int, int] | None:
        operands = [*node.args, *(node.kwargs[key] for key in ("other",) if key in node.kwargs)]
        tensors = [item for item in operands if isinstance(item, fx.Node) and _shape(item)]
        tracked = [flow.get(item) for item in tensors]
        divides = node.target in _DIVISIONS
        if len(tensors) == 1:  # tensor (op) scalar
            if tracked[0] is None:
                return None
            if not scaling:
                groups.block(tracked[0][0], f"constant added at '{node.name}'")
            elif divides and operands[0] is not tensors[0]:
                groups.block(tracked[0][0], f"division by a tensor at '{node.name}'")
            return tracked[0]
        if divides:  # 0 / 0 for removed channels: zeroing and removing would disagree
            block_inputs(node, f"division by a tensor at '{node.name}'")
            return None
        if any(item is None for item in tracked) or len({item[1] for item in tracked}) != 1:
            block_inputs(node, f"cannot align branches at '{node.name}'")
            return None
        shapes = [_shape(item) for item in tensors]
        if len({len(shape) for shape in shapes}) != 1 or len({shape[1] for shape in shapes}) != 1:
            block_inputs(node, f"broadcast over channels at '{node.name}'")
            return None
        root = tracked[0][0]
        for other in tracked[1:]:
            root = groups.union(root, other[0])
        return root, tracked[0][1]

    def module_problem(module: nn.Module, name: str) -> str | None:
        if parametrize.is_parametrized(module) or hasattr(module, "weight_g"):
            return f"'{name}' has parametrized weights"
        return None

    def call_module(node: fx.Node) -> tuple[int, int] | None:
        module = modules[node.target]
        source = flow.get(node.all_input_nodes[0]) if node.all_input_nodes else None
        shared = calls[node.target] > 1

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            is_linear = isinstance(module, nn.Linear)
            out_channels = module.out_features if is_linear else module.out_channels
            in_channels = module.in_features if is_linear else module.in_channels
            conv_groups = 1 if is_linear else module.groups

            if conv_groups > 1 and conv_groups == in_channels == out_channels:
                if source is None:
                    return None
                problem = module_problem(module, node.target)
                if shared or source[1] != 1:
                    problem = f"depthwise conv '{node.target}' cannot be sliced"
                if problem:
                    groups.block(source[0], problem)
                else:
                    groups.get(source[0]).depthwise.append(node.target)
                return source

            produced = first_call.get(node.target)
            if produced is None:
                produced = first_call[node.target] = groups.new(out_channels)
                groups.get(produced).producers.append(node.target)
            problem = module_problem(module, node.target)
            if conv_groups > 1:
                problem = f"grouped convolution '{node.target}'"
            elif shared:
                problem = f"module '{node.target}' is called more than once"
            elif is_linear and len(_shape(node) or ()) != 2:
                problem = f"linear layer '{node.target}' runs on a non 2-D input"
            if problem:
                groups.block(produced, problem)
                if source is not None:
                    groups.block(source[0], problem)
            elif source is not None:
                if in_channels != groups.get(source[0]).channels * source[1]:
                    groups.block(source[0], f"input of '{node.target}' does not match its source")
                else:
                    groups.get(source[0]).consumers.append((node.target, source[1]))
            return produced, 1

        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if source is None:
                return None
            if shared or source[1] != 1 or not module.affine:
                groups.block(source[0], f"batch norm '{node.target}' cannot be sliced")
            else:
                groups.get(source[0]).norms.append(node.target)
            return source

        if isinstance(module, nn.Flatten):
            return reshape(node)
        if isinstance(module, nn.Hardtanh) and not module.min_val <= 0.0 <= module.max_val:
            block_inputs(node, f"'{node.target}' does not map 0 to 0")
            return None
        if isinstance(module, _PASSTHROUGH_MODULES):
            return passthrough(node)
        block_inputs(node, f"unsupported module '{node.target}' ({type(module).__name__})")
        return None

    used_parameters: set[str] = set()
    for node in traced.graph.nodes:
        if node.op == "call_module":
            flow[node] = call_module(node)
        elif node.op == "output":
            block_inputs(node, "model output")
        elif node.op in ("call_function", "call_method"):
            function = node.op == "call_function"
            passes = _PASSTHROUGH_FUNCTIONS if function else _PASSTHROUGH_METHODS
            joins = _JOIN_FUNCTIONS if function else _JOIN_METHODS
            scales = _SCALE_FUNCTIONS if function else _SCALE_METHODS
            reshapes = _RESHAPE_FUNCTIONS if function else _RESHAPE_METHODS
            name = getattr(node.target, "__name__", str(node.target))
            shape_query = node.target in _SHAPE_QUERIES or (
                node.target is getattr and node.args[1:] == ("shape",)
            )
            flow[node] = None
            if shape_query:
                pass  # x.size(0), x.shape: reads the shape, does not touch the channels
            elif _shape(node) is None:  # split, sort, max(dim=...): tuples we cannot follow
                block_inputs(node, f"unsupported operation '{name}'")
            elif node.target is F.hardtanh and (len(node.args) > 1 or node.kwargs):
                block_inputs(node, "hardtanh with custom limits may not map 0 to 0")
            elif node.target in passes:
                flow[node] = passthrough(node)
            elif node.target in joins or node.target in scales:
                flow[node] = join(node, scaling=node.target in scales)
            elif node.target in reshapes:
                flow[node] = reshape(node)
            else:
                block_inputs(node, f"unsupported operation '{name}'")
        else:  # placeholder, get_attr
            flow[node] = None
            if node.op == "get_attr":
                used_parameters.add(str(node.target).rpartition(".")[0])

    # Parameters reached outside their module (F.conv2d(x, self.a.weight)) or tied between
    # modules would silently go out of sync once one copy is sliced.
    owners = Counter(
        id(param) for module in modules.values() for param in module.parameters(recurse=False)
    )
    for name, module in modules.items():
        if any(owners[id(param)] > 1 for param in module.parameters(recurse=False)):
            used_parameters.add(name)
    roots = sorted({groups.find(index) for index in range(len(groups.parent))})
    result = [groups.data[root] for root in roots]
    for group in result:
        members = [
            *group.producers,
            *group.norms,
            *group.depthwise,
            *(c for c, _ in group.consumers),
        ]
        touched = sorted(used_parameters.intersection(members))
        if touched:
            group.blockers.append(f"parameters of {touched} are also used outside their module")
    return DependencyGraph(groups=result)
