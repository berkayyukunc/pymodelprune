# How it works

## Why unstructured pruning does not make a model faster

`torch.nn.utils.prune` multiplies each weight tensor by a 0/1 mask. The tensor keeps its
shape, dense kernels do the same number of multiplications, and `torch.save` writes every
zero as a full 32-bit float. So the raw file size, the FLOPs and the latency stay put.
Only a general purpose compressor notices the zeros, which is why the profile has a
`gz MB` column.

While the masks are attached, PyTorch keeps both `weight_orig` and `weight_mask`, and the
saved file roughly doubles. `prune_unstructured` therefore folds the masks into the weights
by default (`permanent=True`). The profiler also counts the *effective* weights, because
`model.parameters()` would yield the unmasked `weight_orig` and report 0% sparsity.

Getting a real speedup means changing tensor shapes. That is structured pruning.

## Structured pruning

### The problem

```text
Conv A (out: 64) -> BN -> ReLU -> Conv B (in: 64)
                             \
                              (+) <- other branch, Conv C (out: 64)
```

Removing output channel 5 of `A` also requires removing element 5 of the BatchNorm
(`weight`, `bias`, `running_mean`, `running_var`), input channel 5 of `B`, and, because of
the addition, output channel 5 of `C`. Miss one and the model either crashes or, worse,
runs and silently computes something else.

### Building the dependency graph

`build_dependency_graph(model, example_input)`:

1. Deep-copies the model, puts it in `eval()` mode and traces it with
   `torch.fx.symbolic_trace`. A failure raises `UntraceableModelError`. If the graph differs
   between train and eval mode (auxiliary heads that only run during training), the train
   graph is analysed instead, so those layers are sliced too.
2. Runs `ShapeProp` once with the example input, so every node knows its output shape. An
   example input that does not run through the model raises a `ValueError`.
3. Walks the nodes once, in topological order, maintaining `flow[node] = (group, fan)`:
   which channel group the tensor produced by this node belongs to, and how many
   consecutive features each channel spans (`fan`, 1 until a flatten). `None` means
   "channels are not tracked here", which is the state of the model input.

A `ChannelGroup` collects everything that must be sliced with the same indices:

| Field | Meaning | Sliced along |
|---|---|---|
| `producers` | Conv/Linear modules whose **output** channels belong to the group | dim 0 of `weight`, `bias` |
| `norms` | BatchNorm layers on the group's tensors | `weight`, `bias`, `running_mean`, `running_var` |
| `depthwise` | Depthwise convolutions running on the group | dim 0 of `weight` (shape `[C, 1, k, k]`), `bias` |
| `consumers` | `(module, fan)`: modules whose **input** reads the group | dim 1 of `weight` |
| `blockers` | Reasons the group must not be pruned | - |

A group is `prunable` when it has at least one producer and no blockers.

### Rules per node type

**Conv1d / Conv2d / Linear.** The layer closes the incoming group and opens a new one. If
its input is tracked and `in_channels == group.channels * fan`, it is registered as a
consumer `(name, fan)` of the incoming group; a mismatch blocks the incoming group. Its
output starts a fresh group with the layer as the first producer and `fan = 1`. Three cases
block both the new and the incoming group: a grouped convolution that is not depthwise, a
module that is called more than once in the graph (weight sharing), and a `Linear` whose
output is not 2-D (applied over a sequence or a feature map).

**BatchNorm1d / BatchNorm2d.** Added to `norms`; the tensor stays in the same group. Blocked
if the module is shared, has `affine=False`, or comes after a flatten (`fan != 1`).

**Pass-through operations.** ReLU, ReLU6, LeakyReLU, GELU, SiLU, Tanh, ELU, Hardswish,
Hardtanh, Mish, Identity, dropout, max/average/adaptive pooling, and `contiguous`, `clone`,
`detach`. The tensor stays in its group. This list is deliberately restricted to functions
that keep the channel dimension and satisfy `f(0) = 0`; the reason is the equivalence test
below. The rule is checked, not assumed: a `Hardtanh` whose limits exclude 0 is refused,
and so is any listed operation whose recorded output shape changes the channel dimension.

**Flatten.** `nn.Flatten`, `torch.flatten`, `x.flatten`, `x.view`, `x.reshape` are accepted
only if the recorded shapes show a flatten to 2-D that keeps the batch dimension:
`(N, C, H, W) -> (N, C*H*W)`. The group is kept and the fan is multiplied by `H*W`. Any
other reshape blocks the group, and so does a `view`/`reshape` with a hard coded feature
count such as `x.view(-1, 400)`: it would flatten correctly today and break the moment
channels are removed. When the consumer is sliced, kept channel `c` expands to
the feature columns

```text
columns = [c*fan, c*fan + 1, ..., c*fan + fan - 1]   for every kept c
```

For `TinyCNN` the last convolution has 64 channels on a 7x7 map, so the classifier is
registered as `("classifier", 49)` and keeping 32 channels keeps `32 * 49` input features.

**add / sub: union-find merge.** A residual addition forces all of its operands to keep the
same channels. Groups live in a union-find structure; at an `add` the groups of all operand
tensors are merged into one (producers, norms, depthwise layers, consumers and blockers are
concatenated). Because an identity shortcut carries the group of the previous block
forward, a whole residual stage collapses into a single group. In torchvision's ResNet-18
the stem convolution, `layer1.0.conv2` and `layer1.1.conv2` are producers of one group, and
`layer2.0.conv2`, `layer2.1.conv2` and `layer2.0.downsample.0` of the next. The inner
convolution of every block (`conv1`) keeps a group of its own. ResNet-18 ends up with 12
prunable groups.

The merge is refused, and all operands are blocked, if one operand is not tracked (for
example a learned parameter added to the tensor), if the operands have different fans, or
if the addition broadcasts over the channel dimension. Adding a Python scalar blocks the
group too: a removed channel would lose that constant, a zeroed one would not.

**mul / div.** Two tracked tensors that are multiplied are merged exactly like `add`.
Multiplying or dividing by a Python scalar passes through, since `0 * k = 0`. Dividing *by*
a tensor is refused: a zeroed channel would compute `0 / 0`, a removed one nothing, and the
two would disagree.

**Depthwise convolutions.** A convolution with `groups == in_channels == out_channels`
does not mix channels, so its output stays in the group of its input and the module is
added to `depthwise`. Slicing it removes rows of its `[C, 1, k, k]` weight and sets
`in_channels = out_channels = groups = kept`. This is what makes MobileNetV2 prunable.

**Model output.** Every group that reaches the output node is blocked with the reason
`model output`: the classifier must keep its classes.

**Everything else.** An unknown module, function or method blocks the groups of all of its
tracked inputs, and the reason names the node: `unsupported operation 'cat'`,
`unsupported module 'norm' (LayerNorm)`. The same holds for operations that return tuples
(`split`, `sort`), for layers with parametrized weights (`weight_norm`,
`torch.nn.utils.parametrize`), and for parameters that are used outside their own module
or tied between modules, since one copy would go out of sync once the other is sliced.
Reading a shape (`x.size(0)`, `x.shape`) is harmless and ignored.

Squeeze-and-excitation blocks fall out of these rules naturally: the gate passes through a sigmoid, which is unknown, so the gate tensor is
untracked, and the `mul` that applies it cannot align its branches and blocks the gated
channels as well.

### The correctness guard: zeroing equals removing

`mask_plan` zeroes what `apply_plan` would remove: for every dropped channel the producer
weights and biases, the BatchNorm scale and shift, and the depthwise filters are set to 0.
The channel's activation is then exactly 0 everywhere downstream, as long as every
operation on the way maps 0 to 0. A consumer multiplies that channel by its weights and
adds 0 to its sum. Removing the channel and the matching weight columns must therefore
produce **the same output**, up to floating point noise.

```python
import copy

import torch

from pymodelprune.demo_models import TinyCNN
from pymodelprune.prune import apply_plan, build_dependency_graph, mask_plan, plan_pruning

model, example = TinyCNN().eval(), torch.randn(4, 1, 28, 28)
graph = build_dependency_graph(model, example)
plan = plan_pruning(model, graph, 0.5, "l1")
masked = mask_plan(copy.deepcopy(model), graph, plan)
pruned = apply_plan(copy.deepcopy(model), graph, plan)
assert torch.allclose(masked(example), pruned(example), atol=1e-5)
```

If a dependency is missed (a BatchNorm not sliced, a wrong flatten mapping, a residual
branch forgotten) the two outputs differ, or the pruned model fails with a shape error. The
test suite runs this check with randomized BatchNorm statistics on `TinyMLP`, `TinyCNN`, a
residual network with a 1x1 downsample branch, a depthwise network, 12 randomly generated
CNNs, and torchvision's `resnet18`, `resnet50`, `mobilenet_v2` and `vgg11_bn`, for the
`l1`, `l2`, `bn` and `random` criteria.

The same argument explains the pass-through list. `sigmoid(0) = 0.5`: a zeroed channel
would keep feeding 0.5 into the next layer while a removed channel feeds nothing, so the
equivalence, and with it the safety argument, breaks. Such operations are refused.

### Choosing which channels to keep

A criterion returns one score per channel, higher meaning keep. Within a group, the score
of every layer is divided by its mean before the layers are summed; otherwise the layer
with the largest weights would decide alone.

| `importance` | Score of channel `c` |
|---|---|
| `l1` | L1 norm of row `c` of each producer's weight |
| `l2` | L2 norm of the same rows |
| `bn` | `abs(gamma_c)` of the group's BatchNorm layers (Network Slimming); falls back to `l1` if the group has none |
| `taylor` | `abs(sum(w * dL/dw))` over row `c` of each producer (plus `b * dL/db`): a first order estimate of the loss change if the channel were removed |
| `random` | Uniform random; the baseline any criterion has to beat |

`taylor` needs gradients. `prune_structured` clears them, calls your `grad_fn(model)`, then
scores. `make_classification_grad_fn(batches, max_batches=8)` builds one that accumulates
cross-entropy gradients over a few batches with the model in `eval()` mode, so BatchNorm
statistics are not disturbed.

How many channels a group keeps:

- `kept = round(channels * (1 - ratio))`, rounded to a multiple of `round_to` if given,
  never below `min_channels`. The top `kept` channels by score survive, in their original
  order.
- `scope="uniform"` applies the same ratio to every prunable group.
- `scope="global"` computes one threshold, the `amount` quantile of all normalized scores of
  all groups, and prunes each group by the share of its channels under the threshold
  (capped at 90% per group), so unimportant layers lose more.
- `amount` may also be an explicit `{group id: ratio}` mapping. Group ids are the ones in
  `graph.prunable_groups`; they are stable for a given model but not contiguous.

After slicing, module attributes are updated (`out_channels`, `in_channels`, `groups`,
`out_features`, `in_features`, `num_features`), so the pruned model is an ordinary
`nn.Module`: it trains, pickles and exports to ONNX. Because layer shapes changed, it is
saved as a full model (`save_model`), not as a `state_dict` for the original class.

## Sensitivity analysis and budget search

Layers differ widely in how much pruning they tolerate, so one ratio for all of them is
rarely the best trade.

### Sensitivity

`analyze_sensitivity` prunes **one group at a time** at each ratio of a grid (default 0.1,
0.3, 0.5, 0.7, 0.9), without fine-tuning, and records the score and the FLOPs of the
resulting model. The score is your `eval_fn`; without one it is fidelity, the agreement
with the original model's predictions on `fidelity_inputs`. The cost is
`groups x ratios` evaluations, so use a small validation subset. The report can be saved
(`to_json`) and passed back to `search_budget(report=...)` to try several budgets without
measuring again.

### Greedy selection

Every group starts at ratio 0 on its grid. At each step the search looks at moving each
group one grid position forward and computes

```text
extra_drop = max(drop(group, next) - drop(group, current), 0)
gain       = saved_flops(group, next) - saved_flops(group, current)
cost       = (extra_drop + 1e-6) / gain
```

Moves with no FLOPs gain, or that would push the predicted total drop over
`Budget.max_drop`, are discarded; the cheapest remaining move is taken. The search stops
when no move is left, or when the FLOPs of the combined model, which is actually built and
counted at every step, reach `Budget.target_flops_ratio`. The sequence of
`(flops_ratio, predicted_drop)` points is kept as `BudgetResult.history`, an estimate of
the accuracy/FLOPs trade-off curve.

### Verification and bisection

The predicted drop assumes that single-group losses add up. They do not, exactly: pruning
two layers together can hurt more, or less, than the sum. So the prediction is only used to
steer, and the chosen configuration is then **built, fine-tuned if a `finetune_fn` was
given, and measured**.

If the measured drop exceeds `max_drop`, all ratios are multiplied by a common scale factor
found by bisection on `[0, 1]` (at most `max_refinements=5` rounds, each one a real prune,
fine-tune and evaluation). The largest scale that satisfies the limit wins. If none does,
the unpruned model is returned: the search never hands back a model that violates the
accuracy limit it was given. `BudgetResult` reports `predicted_drop`, `measured_drop` and
`refinements` so the quality of the prediction is visible. In the FashionMNIST example the
prediction was 0.65%, the measurement -0.03% (the pruned model was marginally better), and
no refinement was needed.

## Quantization

### Dynamic INT8

`quantize_dynamic` stores `nn.Linear` weights as INT8; activations are quantized on the fly
(`mode="dynamic"`) or left in float (`mode="weight_only"`, torchao only). Convolutions are
not covered, so a CNN barely changes. Two backends sit behind one function:

- `torchao`, the maintained successor, installed with the `[quant]` extra;
- `legacy`, `torch.ao.quantization.quantize_dynamic`, which needs no extra package but is
  deprecated upstream.

`backend="auto"` prefers torchao when it is installed. Keeping both behind one interface
confines the migration to a single file. On Apple Silicon PyTorch ships a quantized engine
(`qnnpack`) but does not select it, and every quantized op fails with `NoQEngine`;
`select_quantized_engine` picks one automatically.

INT8 is always smaller. Whether it is faster depends on which kernels exist for the CPU:
on the development machine PyTorch's FP32 matrix multiplication is much better optimized
than its INT8 path, and no dynamic INT8 variant was faster than FP32; on the Linear-only
model all of them were several times slower.

### Why static INT8 goes through ONNX Runtime

Static quantization measures activation ranges once, on calibration data, instead of on
every batch. It covers convolutions and removes the per-batch overhead, so it is the variant
that can actually be faster. PyTorch's current static path (PT2E) depends on backend
specific quantizers, and there was no ready one for ARM CPUs such as Apple Silicon when this
was written. ONNX Runtime's static quantization runs on any CPU and ships hand tuned INT8
kernels for x86 and ARM. The price is that the result is an `.onnx` file rather than a
PyTorch model, which is acceptable: a deployment that cares about CPU latency usually moves
to a dedicated runtime anyway.

The path in `optimize(..., quantize=QuantConfig(mode="onnx_static"))`:

1. `to_onnx` exports the (pruned) float model with a dynamic batch axis. The new
   `torch.export` based exporter is tried first, the TorchScript exporter is the fallback.
   The fallback traces, which freezes Python control flow on tensor values into the graph,
   so it is announced with a warning that names why the first exporter failed.
   Since `torch.export` specializes dimensions of size 1, a batch of 1 is duplicated for
   tracing.
2. The file is run in ONNX Runtime and compared with the PyTorch output on three inputs:
   the example, its negation (the other side of a value dependent branch) and a batch of
   another size (a frozen batch dimension). A difference above
   `atol + rtol * max|output|` (both `1e-4`), or a NaN, raises `OnnxMismatchError` and the
   file is deleted.
3. `quantize_onnx_static` runs ONNX Runtime's pre-processing, then `quantize_static` in QDQ
   format with INT8 weights and activations, per-channel weights by default, calibrated on
   up to 16 batches. `exclude_nodes` keeps sensitive nodes in float.
4. `profile_onnx` measures both files through `OrtModule`, an `nn.Module` wrapper around an
   ONNX Runtime session, so latency, accuracy and fidelity come from the same code as for
   the PyTorch stages. The INT8 file is not checked against a tolerance; its quality shows
   up in the `Acc` and `Fid` columns.

## Measurement

- **Latency**: `warmup=10` discarded passes, then `runs=100` timed passes of a single
  forward call under `torch.inference_mode()`; median, p95, mean and standard deviation are
  recorded, the table shows the median. The pipeline always measures on CPU;
  `measure_latency` on a CUDA or MPS input synchronizes the device around every pass.
  `optimize(threads=...)` pins the thread count for the duration of the call.
- **Size**: the `state_dict` serialized to memory with `torch.save`; `gz MB` is the same
  bytes through gzip level 6 (shown, like `Sparse`, only when some stage is at least 5%
  sparse). For ONNX stages, the file on disk.
- **FLOPs**: `torch.utils.flop_counter.FlopCounterMode` on one forward pass. Deterministic
  and machine independent. Quantized kernels are invisible to the counter, so INT8 models
  report no FLOPs of their own; in `optimize` the dynamic INT8 stage carries over the count
  of the float model it was made from (same arithmetic, lower precision).
- **Parameters and sparsity**: counted over the weights the forward pass really uses,
  including masked weights and the packed parameters of legacy quantized layers.
- **Accuracy**: whatever your `eval_fn` returns. **Fidelity**: share of inputs with the same
  `argmax` as the original model, plus the largest absolute output difference.
- **Environment**: Python, PyTorch, platform, thread count and quantized engine are stored
  in the JSON report, because latency without that context is not reproducible.
