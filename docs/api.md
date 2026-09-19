# API reference

Everything below is importable from the top-level package (`from pymodelprune import ...`)
unless a module path is given. Signatures are copied from the source.

Conventions shared by all functions:

- `example_input` is one tensor or a tuple of tensors (for models with several inputs).
- Functions that optimize a model return a **new** model; the argument is left untouched.
- `fidelity_inputs` and `calibration` accept a single tensor or an iterable of batches.
  `(inputs, labels)` batches are accepted wherever inputs are needed, and the labels are
  ignored. For a model with N inputs (an `example_input` tuple of N tensors) a batch is a
  tuple whose first N items are the inputs: `(x1, x2)` or `(x1, x2, labels)`.
- One-shot iterators (generators) are materialised as a list by `optimize`,
  `analyze_sensitivity`, `search_budget` and `quantize_onnx_static`. `accuracy` and
  `make_accuracy_fn` still need something re-iterable (a list or `DataLoader`).
- Fidelity computed on fewer than 32 samples emits a `FewSamplesWarning`
  (a `UserWarning`). The searches go further and raise, see [Search](#search).

## Pipeline

### `optimize`

```python
def optimize(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    prune: PruneConfig | None = None,
    quantize: QuantConfig | None = None,
    eval_fn: EvalFn | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
    finetune_fn: FinetuneFn | None = None,
    grad_fn: GradFn | None = None,
    onnx_path: str | Path | None = None,
    warmup: int = 10,
    runs: int = 100,
    threads: int | None = None,
    onnx_atol: float = 1e-4,
) -> OptimizationResult: ...
```

Prunes, then quantizes, and profiles after every step. The model is copied, moved to CPU
and put in `eval()` mode first; `example_input` is moved to the CPU as well.

- `eval_fn`: `model -> float`, higher is better. Without it only fidelity is reported.
- `fidelity_inputs`: inputs for the `Fid` column; also the fallback calibration data for
  `onnx_static`. The default is `example_input`, a single sample: the call then warns once
  (`FewSamplesWarning`) because such a `Fid` value says little.
- `finetune_fn`: `model -> None`, trains the pruned model in place. With a budget it is run
  on every candidate the search verifies. After unstructured pruning the masks stay
  attached while it runs and are folded in afterwards, so the sparsity survives training.
- `grad_fn`: only needed for `importance="taylor"`.
- `onnx_path`: where the INT8 ONNX file is written; required for `mode="onnx_static"`. The
  float export lands next to it as `<stem>_fp32.onnx` (`model.onnx` -> `model_fp32.onnx`).
- `threads`: calls `torch.set_num_threads` for the duration of the call; the previous
  value is restored afterwards, also on failure.
- `onnx_atol`: absolute tolerance of the ONNX export verification (see `to_onnx`).

Stages, in order: `original`; `pruned` (if `prune`); then either `int8` (dynamic or
weight-only) or `onnx fp32` and `onnx int8` (static).

Raises `ValueError` if neither `prune` nor `quantize` is given, if the model is already
quantized, or if `onnx_static` lacks `onnx_path` or calibration data.

### `PruneConfig`

```python
@dataclass(frozen=True)
class PruneConfig:
    method: str = "structured"  # "structured" | "unstructured"
    amount: float = 0.5
    importance: str = "l1"
    scope: str | None = None
    round_to: int = 1
    budget: Budget | None = None
    min_channels: int = 1
```

- `method="structured"` removes whole channels and needs an fx-traceable model;
  `"unstructured"` zeroes single weights and works on anything.
- `importance`: `"l1"`, `"l2"`, `"bn"`, `"taylor"`, `"random"` for structured pruning;
  only `"l1"` and `"random"` for unstructured pruning.
- `scope`: `None` picks the method's default, `"uniform"` for structured (other option
  `"global"`) and `"global"` for unstructured (other option `"layer"`).
- `round_to`: keep channel counts at multiples of this number (structured only).
- `min_channels`: never leave a group with fewer channels than this (structured only).
- `budget`: replaces `amount` and `scope` with an automatic search (structured only);
  `round_to` and `min_channels` still apply.

Invalid combinations raise `ValueError` at construction: an unknown `method`, an
`importance` the method does not offer, `amount` outside `[0, 1)`, `round_to` or
`min_channels` below 1, a `budget` with `method="unstructured"`.

### `QuantConfig`

```python
@dataclass(frozen=True)
class QuantConfig:
    mode: str = "dynamic"  # "dynamic" | "weight_only" | "onnx_static"
    backend: str = "auto"  # "auto" | "torchao" | "legacy"
    calibration: Any = None
    per_channel: bool = True
    exclude_nodes: tuple[str, ...] = ()
```

`backend` applies to `dynamic` and `weight_only`; `calibration`, `per_channel` and
`exclude_nodes` apply to `onnx_static`.

### `OptimizationResult`

```python
@dataclass
class OptimizationResult:
    stages: list[Stage]  # Stage(name, profile, model, onnx_path)
    environment: dict[str, Any]
    search: BudgetResult | None = None
```

| Member | Description |
|---|---|
| `profiles` | `list[ModelProfile]`, one per stage. |
| `model` | The last stage that is still a PyTorch model (for `onnx_static`: the pruned float model). |
| `onnx_path` | Path of the final ONNX file, or `None`. |
| `report(console=None)` | Prints the comparison table. |
| `save(path) -> Path` | Saves `model` as a full model with `torch.save`. |
| `to_dict()` / `to_json(path) -> Path` | Profiles, environment and, after a search, ratios and history. `"schema": 1`. The file is strict JSON: non-finite numbers (a NaN score) are written as `null`. |

## Profiling

### `profile_model`

```python
def profile_model(
    model: nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    name: str = "model",
    warmup: int = 10,
    runs: int = 100,
    eval_fn: EvalFn | None = None,
    reference: nn.Module | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
) -> ModelProfile: ...
```

`eval_fn` adds accuracy (stored as a plain `float`, so numpy and 0-d tensor scores are
fine). `reference` (the original model) adds fidelity, computed on `fidelity_inputs` or, if
omitted, on `example_input`. Latency is measured on the device the model and input live on
(CUDA and MPS are synchronized around every pass); the library's own numbers are always CPU.

### `ModelProfile` and `LatencyStats`

```python
@dataclass(frozen=True)
class ModelProfile:
    name: str
    total_params: int
    nonzero_params: int
    size_mb: float
    latency: LatencyStats
    compressed_mb: float | None = None
    flops: int | None = None
    dtype: str = "fp32"
    accuracy: float | None = None
    fidelity: Fidelity | None = None

    @property
    def sparsity(self) -> float: ...  # share of parameters that are exactly zero


@dataclass(frozen=True)
class LatencyStats:
    median_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float
    runs: int
```

Sizes are in MB (10^6 bytes). `flops` is `None` for INT8 models.

### Single measurements

```python
def count_parameters(model: nn.Module) -> tuple[int, int]: ...  # (total, nonzero)
def count_flops(
    model: nn.Module, example_input: torch.Tensor | tuple[torch.Tensor, ...]
) -> int | None: ...
def model_size_mb(model: nn.Module) -> float: ...
def measure_latency(
    model: nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    warmup: int = 10,
    runs: int = 100,
) -> LatencyStats: ...
```

`count_parameters` counts the weights the forward pass really uses: masked weights while
`torch.nn.utils.prune` masks are attached, dequantized packed weights for legacy quantized
layers. `count_flops` returns `None` when the forward pass cannot be counted.
`model_size_mb` serializes the `state_dict` to memory. `measure_latency` needs `runs >= 2`;
when the input lives on a CUDA or MPS device it waits for the device before and after every
timed pass.

### Comparing several models

```python
def measure_latency_interleaved(
    models: Sequence[nn.Module],
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    warmup: int = 10,
    runs: int = 100,
    block: int = 20,
) -> list[LatencyStats]: ...


def retime_profiles(
    profiles: Sequence[ModelProfile],
    models: Sequence[nn.Module],
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    warmup: int = 10,
    runs: int = 100,
) -> list[ModelProfile]: ...
```

Timing models one after the other compares the moments as much as the models: a drift in
machine load between two measurements shows up as a speedup. `measure_latency_interleaved`
cycles through the candidates in blocks of `block` passes so the drift reaches all of them.

The passes come in blocks rather than one at a time because a strict round robin lets each
model evict its neighbours' weights from the CPU caches, and the penalty grows with how
many bytes the others moved. Measured that way a 44 MB model looked three times slower than
it is, which inflated every speedup reported against it. Inside a block the weights stay
hot, so only the first pass of each block pays that cost and the median ignores it.

`retime_profiles` returns copies of `profiles` with their `latency` replaced by such a
measurement; `models[i]` is the model behind `profiles[i]`, and an `OrtModule` counts as a
model. `optimize` calls it on all of its stages before returning, because they are otherwise
profiled minutes apart.

### `render_profiles`

```python
def render_profiles(
    profiles: Sequence[ModelProfile],
    console: Console | None = None,
    compare: bool = True,
) -> None: ...
```

Prints a table. With `compare`, the first profile is the baseline and the others get `Size`
(below 1 is smaller) and `Speed` (above 1 is faster) ratios. Optional columns (`Type`,
`FLOPs`, `Acc`, `Fid`) appear only when some profile has a value for them; `Sparse` and
`gz MB` appear only when some profile is at least 5% sparse. Without the two sparsity
columns the widest table fits an 80-column terminal.

## Accuracy and fidelity

```python
EvalFn = Callable[[nn.Module], float]  # higher is better


def accuracy(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    max_batches: int | None = None,
) -> float: ...


def make_accuracy_fn(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    max_batches: int | None = None,
) -> EvalFn: ...
```

`accuracy` is top-1 classification accuracy in `[0, 1]`; inputs are moved to the model's
device. `make_accuracy_fn` binds a dataset to it. Both keep the `(inputs, labels)` batch
contract and need re-iterable data.

```python
def reusable(batches: Batches) -> Batches | list: ...
```

Returns `batches` unchanged unless it is a one-shot iterator (a generator), which is
materialised as a list so that several stages can walk it.

```python
def compare_outputs(
    reference: nn.Module,
    candidate: nn.Module,
    inputs: torch.Tensor | Iterable,
    num_inputs: int = 1,
) -> Fidelity: ...


@dataclass(frozen=True)
class Fidelity:
    agreement: float  # share of samples with the same argmax over the last dimension
    max_abs_diff: float
    samples: int
```

Needs no labels. Both models must be on the same device as the inputs, and must return a
single tensor. A tuple/list batch contributes its first `num_inputs` items as the model
inputs; the rest (labels) is ignored. `max_abs_diff` is NaN as soon as either model
produced a NaN. Emits `FewSamplesWarning` below 32 samples.

## Pruning

### `prune_unstructured`

```python
def prune_unstructured(
    model: nn.Module,
    amount: float,
    criterion: str = "l1",  # "l1" | "random"
    scope: str = "global",  # "global" | "layer"
    layers: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv1d, nn.Conv2d),
    permanent: bool = True,
) -> nn.Module: ...
```

Zeroes a fraction `amount` in `[0, 1)` of the weights. `scope="global"` ranks all weights
together; `"layer"` removes the same fraction from every layer. Biases and normalization
layers are never pruned. With `permanent=False` the `torch.nn.utils.prune` masks stay
attached; `pymodelprune.prune.make_permanent(model)` folds them in later.

### `build_dependency_graph`

```python
def build_dependency_graph(
    model: nn.Module, example_input: torch.Tensor | tuple
) -> DependencyGraph: ...
```

Returns a `DependencyGraph` with `groups`, `prunable_groups` and `group(group_id)`. Each
`ChannelGroup` has `id`, `channels`, `producers`, `norms`, `depthwise`, `consumers`
(`(module name, fan)` pairs), `blockers` and the property `prunable`. Raises
`pymodelprune.prune.UntraceableModelError` if `torch.fx` cannot trace the model.
See [How it works](how-it-works.md).

### `prune_structured`

```python
def prune_structured(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    amount: float | Mapping[int, float] = 0.5,
    importance: str = "l1",  # "l1" | "l2" | "bn" | "taylor" | "random"
    scope: str = "uniform",  # "uniform" | "global"
    round_to: int = 1,
    min_channels: int = 1,
    grad_fn: GradFn | None = None,
    graph: DependencyGraph | None = None,
) -> nn.Module: ...
```

Returns a physically smaller copy. `amount` is the fraction of channels removed from every
prunable group, in `[0, 1)`, or a `{group id: ratio}` mapping. Pass a prebuilt `graph` to
skip tracing. Raises `ValueError` when no group is prunable (the message lists the
blockers), when a mapping names a group that is not prunable, or when
`importance="taylor"` is used without `grad_fn`.

The lower-level steps are available from `pymodelprune.prune`:

```python
def plan_pruning(
    model,
    graph,
    amount,
    importance="l1",
    scope="uniform",
    round_to=1,
    min_channels=1,
    max_ratio=0.9,
) -> dict[int, torch.Tensor]: ...  # {group id: indices to keep}
def apply_plan(model, graph, plan) -> nn.Module: ...  # removes channels, in place
def mask_plan(model, graph, plan) -> nn.Module: ...  # zeroes them instead, in place
```

### `make_classification_grad_fn`

```python
def make_classification_grad_fn(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]], max_batches: int = 8
) -> Callable[[nn.Module], None]: ...
```

Gradient provider for `importance="taylor"`: accumulates cross-entropy gradients over a
few batches, with the model in `eval()` mode so BatchNorm statistics stay frozen. Any
callable that leaves `.grad` populated on the model's parameters works as a `grad_fn`.

## Search

### `analyze_sensitivity`

```python
def analyze_sensitivity(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    eval_fn: EvalFn | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
    ratios: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    importance: str = "l1",
    grad_fn: GradFn | None = None,
    graph: DependencyGraph | None = None,
    progress: Callable[[int, int], None] | None = None,
    round_to: int = 1,
    min_channels: int = 1,
) -> SensitivityReport: ...
```

Measures score and FLOPs for every (prunable group, ratio) pair, one group at a time, which
costs `groups x ratios` evaluations. Without `eval_fn`, agreement with the original model
on `fidelity_inputs` is the score; that fallback raises `ValueError` on fewer than 32
samples (so also when neither argument is given), because agreement on a handful of inputs
cannot rank layers. `round_to` and `min_channels` should match the pruning the curves are
meant for. `progress(done, total)` is called after every evaluation.

`SensitivityReport` (from `pymodelprune.search`): `metric`, `baseline_score`,
`baseline_flops`, `curves`, `layers`; methods `drop(group_id, ratio)`,
`saved_flops(group_id, ratio)`, `ranking(ratio=None)`, `to_json(path)` and the class method
`from_json(path)`. `drop` and `saved_flops` only accept ratios that were measured.

### `Budget` and `search_budget`

```python
@dataclass(frozen=True)
class Budget:
    max_drop: float | None = None
    target_flops_ratio: float | None = None  # in (0, 1)
```

At least one limit is required. `max_drop` is in the units of the metric: with accuracy in
`[0, 1]`, `0.01` is one percentage point. With both limits set the search stops at
whichever is hit first.

```python
def search_budget(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    budget: Budget,
    eval_fn: EvalFn | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
    report: SensitivityReport | None = None,
    importance: str = "l1",
    grad_fn: GradFn | None = None,
    finetune_fn: FinetuneFn | None = None,
    graph: DependencyGraph | None = None,
    max_refinements: int = 5,
    round_to: int = 1,
    min_channels: int = 1,
) -> BudgetResult: ...
```

Pass a saved `report` to skip the sensitivity measurements; it must come from this very
model and example shape (same prunable groups, same FLOPs), otherwise `ValueError`. The
same 32-sample rule as in `analyze_sensitivity` applies when there is no `eval_fn`. Raises
`ValueError` when the model has no prunable group (the message lists the blockers). When
no group can be pruned within the budget, `model` is an untouched copy, `ratios` is empty
and `finetune_fn` is not called. Returns a `BudgetResult`
(`pymodelprune.search`): `model`, `ratios` (`{group id: ratio}`, pruned groups only),
`baseline_score`, `score`, `flops_ratio`, `predicted_drop`, `refinements`, `history`
(list of `SearchStep(flops_ratio, predicted_drop)`) and the property `measured_drop`.

## Quantization

### `quantize_dynamic`

```python
def quantize_dynamic(
    model: nn.Module, mode: str = "dynamic", backend: str = "auto"
) -> nn.Module: ...
```

INT8 copy of the model, on CPU. Only `nn.Linear` layers are quantized.

- `mode="dynamic"`: INT8 weights, activations quantized per batch.
  `mode="weight_only"`: INT8 weights, float activations (torchao only).
- `backend="auto"` uses torchao when installed, else `legacy` (`torch.ao.quantization`).

Raises `ValueError` for a model that is already quantized, a model without `nn.Linear`
layers or `weight_only` with the legacy backend, and `RuntimeError` when the requested
backend is not available. Models from both backends can be saved with `save_model`.

### `quantize_onnx_static`

```python
def quantize_onnx_static(
    onnx_path: str | Path,
    output_path: str | Path,
    calibration: torch.Tensor | Iterable,
    max_batches: int = 16,
    per_channel: bool = True,
    exclude_nodes: Sequence[str] = (),
) -> Path: ...
```

Quantizes a float ONNX model to INT8 (QDQ format, INT8 weights and activations) with ONNX
Runtime and returns the new file's path. A few hundred representative samples are enough
for calibration. `exclude_nodes` lists ONNX node names to keep in float. Needs the `[onnx]`
extra. For a graph with N inputs every calibration batch must be a tuple whose first N
items are the inputs, in graph order.

## Saving and ONNX

```python
def save_model(model: nn.Module, path: str | Path) -> Path: ...
```

Saves the full model object with `torch.save`. Structurally pruned models have new layer
shapes, so a bare `state_dict` would not load into the original architecture. Load with
`torch.load(path, weights_only=False)`. The file is written to a temporary name and renamed
into place, so a failed save never leaves a truncated file or destroys an older one.

```python
def to_onnx(
    model: nn.Module,
    example_input: torch.Tensor | tuple,
    path: str | Path,
    verify: bool = True,
    atol: float = 1e-4,
    dynamic_batch: bool = True,
    rtol: float = 1e-4,
) -> Path: ...


def verify_onnx(
    model: nn.Module,
    path: str | Path,
    example_input: torch.Tensor | tuple,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> float: ...
```

`verify_onnx` returns the largest absolute output difference between PyTorch and ONNX
Runtime, or raises `pymodelprune.export.OnnxMismatchError` when it exceeds
`atol + rtol * max|expected|` (or is NaN). Model and inputs may live on any device; the
comparison runs on CPU copies. Models that return anything but a single tensor raise
`TypeError`.

`to_onnx` exports and, by default, verifies three times: on the example, on the negated
example (floating point inputs only) and, with `dynamic_batch`, on a batch of another size
built by repeating the example. The extra probes catch value-dependent Python control flow
and batch sizes that were frozen into the graph; a probe on which PyTorch itself returns
non-finite values is skipped. A file that fails verification is deleted. When the dynamo
exporter fails and the tracing exporter takes over, a `UserWarning` names the reason.

```python
class OrtModule(nn.Module):
    def __init__(self, path: str | Path, threads: int | None = None) -> None: ...
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor: ...
```

An ONNX Runtime CPU session that behaves like a module (first output only), so
`accuracy`, `compare_outputs` and `measure_latency` run on it unchanged.

```python
def profile_onnx(
    path: str | Path,
    example_input: torch.Tensor | tuple,
    name: str = "onnx",
    source_model: nn.Module | None = None,
    warmup: int = 10,
    runs: int = 100,
    eval_fn: EvalFn | None = None,
    reference: nn.Module | None = None,
    fidelity_inputs: torch.Tensor | Iterable | None = None,
) -> ModelProfile: ...
```

Profiles an ONNX file with ONNX Runtime. Size columns describe the file on disk. Parameter
counts, and FLOPs for float files, are taken from `source_model`, the PyTorch model the file
was exported from, because ONNX files do not expose them; without it they are 0.

## Demo models

`pymodelprune.demo_models.TinyMLP` and `TinyCNN`: small classifiers for `1x28x28` inputs,
used by `pymodelprune demo`, the examples and the tests. `DEMO_INPUT_SHAPE = (1, 1, 28, 28)`.
