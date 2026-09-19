# pymodelprune

Prune and quantize PyTorch models, then measure what you actually gained.

`pymodelprune` applies unstructured pruning, structured (channel) pruning, dynamic INT8 and
ONNX Runtime static INT8 to a model, and after every step reports parameters, sparsity, file
size, gzip size, FLOPs, CPU latency, accuracy and fidelity next to the original.

## Why this exists

Most compression tutorials stop at "50% of the weights are now zero". That number says
nothing about the file you ship or the latency you pay. On the machine this project was
developed on:

- PyTorch's built-in pruning left the file size and the latency exactly where they were.
- Dynamic INT8 made a model four times smaller and several times *slower*.
- Removing channels for real made every metric move, and one epoch of fine-tuning brought
  the accuracy back.

The library is built around that observation: every optimization returns a new model, and
every model is profiled against the original with the same code, so the report shows what
changed and what did not. Numbers in this README are copied from the output of the scripts
in [`examples/`](examples/), not estimated.

## The headline result

ResNet-18 on CIFAR-10, pruned to half its channels, fine-tuned for 3 epochs, then exported
to ONNX Runtime and quantized to INT8. Measured on an Apple Silicon CPU, batch size 1:

| Model     | Params | Type |    MB |  FLOPs | p50 ms |   Acc |  Size | Speed |
|-----------|-------:|------|------:|-------:|-------:|------:|------:|------:|
| original  |  11.2M | fp32 | 44.77 |   1.1B |  4.444 | 90.7% | 1.00x | 1.00x |
| optimized |   2.8M | int8 |  2.92 | 278.6M |  0.617 | 89.1% | 0.07x | 7.20x |

15x smaller and 7.2x faster for 1.6 accuracy points, reproducible with
`examples/resnet18_cifar10.py`. [Full stage-by-stage breakdown below.](#5-resnet-18-on-cifar-10-end-to-end-with-onnx-runtime-static-int8)

## Install

```bash
pip install pymodelprune            # core: torch, numpy, typer, rich
pip install "pymodelprune[quant]"   # + torchao (current dynamic INT8 backend)
pip install "pymodelprune[onnx]"    # + onnx, onnxruntime, onnxscript (export, static INT8)
pip install "pymodelprune[all]"     # everything above
```

Python 3.10+ and PyTorch 2.2+. Optional dependencies are really optional: a missing extra
raises an error that names the `pip install` line to run.

Development setup with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/berkayyukunc/pymodelprune.git && cd pymodelprune
uv sync                              # core + all extras + dev tools + torchvision for examples
uv run pymodelprune demo --model cnn
```

## Quickstart

### See it work in one minute

```bash
uv run python examples/train_fashion_mnist.py   # trains two small models, under a minute
uv run python examples/sweep_structured.py --model cnn --finetune
```

The second script removes channels at four ratios, fine-tunes for one epoch and prints:

```text
Model        Params    MB   FLOPs  p50 ms    Acc    Fid   Size  Speed
original      50.4K  0.21    7.7M   0.114  89.2%      -  1.00x  1.00x
-30% ch +ft   31.4K  0.13    3.8M   0.101  90.8%  91.2%  0.63x  1.13x
-50% ch +ft   20.6K  0.09    2.1M   0.091  89.0%  87.6%  0.42x  1.25x
-70% ch +ft   11.2K  0.05  830.1K   0.086  89.4%  90.5%  0.24x  1.33x
-90% ch +ft    3.2K  0.02  111.7K   0.074  83.9%  86.7%  0.09x  1.55x
```

Seven out of ten channels can go and, after one epoch of fine-tuning, the model is as
accurate as it started: a quarter of the size, a ninth of the arithmetic.

### Python

```python
import torch

from pymodelprune import PruneConfig, QuantConfig, make_accuracy_fn, optimize
from pymodelprune.demo_models import TinyCNN

model = TinyCNN()  # stand-in: use your own trained nn.Module
example = torch.randn(1, 1, 28, 28)

# stand-in data: any re-iterable of (inputs, labels) batches, e.g. a DataLoader
val_batches = [(torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))) for _ in range(4)]

result = optimize(
    model,
    example,
    prune=PruneConfig(method="structured", amount=0.5, importance="l1"),
    quantize=QuantConfig(mode="dynamic"),
    eval_fn=make_accuracy_fn(val_batches),  # or any callable: model -> float
    fidelity_inputs=val_batches,
)
result.report()  # one table: original, pruned, int8
result.to_json("report.json")  # same numbers plus environment info
result.save("model_int8.pt")  # full model; load with torch.load(path, weights_only=False)
```

`model` is never modified. `result.model` is the final PyTorch model and
`result.stages` holds every intermediate model with its profile.

Instead of choosing a ratio, state a limit and let the search pick a ratio per layer:

```python
from pymodelprune import Budget


def finetune(pruned):  # optional: trains the pruned model in place
    optimizer = torch.optim.SGD(pruned.parameters(), lr=0.01)
    pruned.train()
    for inputs, labels in val_batches:  # use your training loader here
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(pruned(inputs), labels).backward()
        optimizer.step()
    pruned.eval()


result = optimize(
    model,
    example,
    prune=PruneConfig(budget=Budget(max_drop=0.01)),  # lose at most 1 accuracy point
    eval_fn=make_accuracy_fn(val_batches),
    fidelity_inputs=val_batches,
    finetune_fn=finetune,
)
result.report()
result.save("model_pruned.pt")
print(result.search.ratios, result.search.measured_drop)
```

### Command line

```bash
pymodelprune demo --model cnn                              # check the installation

pymodelprune profile model.pt --input-shape 1,3,224,224

pymodelprune optimize model.pt --input-shape 1,3,32,32 \
    --prune 0.5 --quantize dynamic \
    --eval my_project.hooks:evaluate --data my_project.hooks:batches \
    --out model_opt.pt --json report.json

pymodelprune optimize model.pt --input-shape 1,3,32,32 \
    --max-acc-drop 1.0 --eval my_project.hooks:evaluate     # budget search

pymodelprune sensitivity model.pt --input-shape 1,3,32,32 --eval my_project.hooks:evaluate
```

`model.pt` is a file written by `torch.save(model, path)`. Loading it unpickles arbitrary
code, so for files you do not trust use `--factory package.module:build_model --weights state_dict.pt`.

## What actually happens

Measured on an Apple Silicon Mac, CPU only, PyTorch 2.14, batch size 1, median of 100
timed runs. The models are the two demo networks (`TinyMLP`, 669.7K parameters;
`TinyCNN`, 50.4K parameters) trained on FashionMNIST by
`examples/train_fashion_mnist.py`. `Fid` is the share of test inputs on which the
optimized model predicts the same class as the original. `Size` and `Speed` are ratios
against the first row. Latency is machine specific; rerun the scripts for your hardware.

### 1. Unstructured pruning: only the gzip size moves

`examples/sweep_unstructured.py --model mlp` (global magnitude pruning via `torch.nn.utils.prune`)

| Model      | Sparse |   MB | gz MB | FLOPs | p50 ms |   Acc |  Size | Speed |
|------------|-------:|-----:|------:|------:|-------:|------:|------:|------:|
| original   |   0.0% | 2.68 |  2.49 |  1.3M |  0.020 | 86.9% | 1.00x | 1.00x |
| pruned 30% |  30.0% | 2.68 |  1.95 |  1.3M |  0.020 | 87.0% | 1.00x | 1.00x |
| pruned 50% |  49.9% | 2.68 |  1.51 |  1.3M |  0.020 | 86.9% | 1.00x | 1.00x |
| pruned 70% |  69.9% | 2.68 |  1.03 |  1.3M |  0.020 | 85.5% | 1.00x | 1.00x |
| pruned 90% |  89.9% | 2.68 |  0.51 |  1.3M |  0.020 | 62.4% | 1.00x | 1.00x |
| pruned 95% |  94.9% | 2.68 |  0.36 |  1.3M |  0.020 | 43.8% | 1.00x | 1.00x |

Zeros are still stored and still multiplied: the file, the FLOPs and the latency do not
change, only the compressed size does. While the pruning masks are still attached
(`permanent=False`) the saved `TinyMLP` even doubles, from 2.68 MB to 5.36 MB, because
PyTorch stores the original weights and the masks side by side.

### 2. Dynamic INT8: smaller, not faster

`examples/quantize_dynamic_demo.py`

| Model               | Type |   MB | p50 ms |   Acc |    Fid |  Size | Speed |
|---------------------|------|-----:|-------:|------:|-------:|------:|------:|
| TinyMLP             | fp32 | 2.68 |  0.020 | 86.9% |      - | 1.00x | 1.00x |
| legacy, dynamic     | int8 | 0.68 |  0.119 | 87.0% |  99.9% | 0.25x | 0.17x |
| torchao, dynamic    | int8 | 0.68 |  0.209 | 87.0% |  99.9% | 0.25x | 0.10x |
| torchao, weight only| int8 | 0.68 |  0.083 | 86.9% | 100.0% | 0.25x | 0.24x |
| TinyCNN             | fp32 | 0.21 |  0.193 | 89.2% |      - | 1.00x | 1.00x |
| legacy, dynamic     | int8 | 0.11 |  0.222 | 89.3% |  99.9% | 0.55x | 0.87x |
| torchao, dynamic    | int8 | 0.11 |  0.268 | 89.3% |  99.9% | 0.55x | 0.72x |
| torchao, weight only| int8 | 0.11 |  0.198 | 89.2% |  99.9% | 0.55x | 0.98x |

The Linear-only model shrinks 4x at no accuracy cost, but runs at 0.10x to 0.24x the speed
of FP32: this CPU's FP32 matrix kernels are better tuned than its INT8 ones. The CNN only
shrinks to 0.55x and gains no speed, because dynamic quantization covers `nn.Linear` only
and most of its work is in convolutions.

### 3. Structured pruning: everything shrinks, fine-tuning recovers accuracy

`examples/sweep_structured.py --model cnn` with and without `--finetune`. The last two
columns and the latency come from the fine-tuned run; "Acc, no ft" is the same pruning
without any training afterwards.

| Model     | Params |   MB |  FLOPs | p50 ms | Acc, no ft | Acc, 1 epoch ft |  Size | Speed |
|-----------|-------:|-----:|-------:|-------:|-----------:|----------------:|------:|------:|
| original  |  50.4K | 0.21 |   7.7M |  0.114 |      89.2% |           89.2% | 1.00x | 1.00x |
| -30% ch   |  31.4K | 0.13 |   3.8M |  0.101 |      83.9% |           90.8% | 0.63x | 1.13x |
| -50% ch   |  20.6K | 0.09 |   2.1M |  0.091 |      72.2% |           89.0% | 0.42x | 1.25x |
| -70% ch   |  11.2K | 0.05 | 830.1K |  0.086 |      59.2% |           89.4% | 0.24x | 1.33x |
| -90% ch   |   3.2K | 0.02 | 111.7K |  0.074 |      16.0% |           83.9% | 0.09x | 1.55x |

Channels are physically removed, so parameters, file size, FLOPs and latency all fall
together. Pruning alone is destructive, but one epoch of fine-tuning brings the network
with 70% of its channels removed (about a quarter of the size, a ninth of the FLOPs) back
to the original accuracy.

### 4. Budget search: "lose at most one accuracy point"

`examples/budget_search_demo.py --max-drop 0.01` (no fine-tuning)

| Model    | Params |   MB | FLOPs | p50 ms |   Acc |   Fid |  Size | Speed |
|----------|-------:|-----:|------:|-------:|------:|------:|------:|------:|
| original |  50.4K | 0.21 |  7.7M |  0.164 | 89.2% |     - | 1.00x | 1.00x |
| pruned   |  25.4K | 0.11 |  4.1M |  0.153 | 88.6% | 90.1% | 0.52x | 1.08x |

```text
chosen pruning ratio per group:
  features.0               0%
  features.4              50%
predicted drop 0.61%, measured 0.61%, refinements 0
```

The search measured each layer's sensitivity first: removing half of `features.0` costs
14.8 accuracy points, removing half of `features.4` costs 0.3. So it left the fragile first
convolution alone and took half of the second one, for the same FLOPs saving. The
prediction only steers the search; the returned model is always re-measured against the
limit, and its ratios are scaled back if it misses.

### 5. ResNet-18 on CIFAR-10, end to end, with ONNX Runtime static INT8

`examples/resnet18_cifar10.py` trains ResNet-18, compares the importance criteria, removes
50% of the channels with Taylor importance, fine-tunes, exports to ONNX and quantizes to
INT8 with ONNX Runtime. Static quantization covers convolutions and ONNX Runtime ships
tuned INT8 kernels for x86 and ARM, which is where INT8 becomes a latency win and not only
a size win.

Trained for 10 epochs, then 50% of the channels removed and 3 epochs of fine-tuning.
Every stage is timed together at the end, in alternating blocks, so the Speed column
compares the models rather than the moments they were measured.

| Model     | Params | Type |    MB |  FLOPs | p50 ms |   Acc |   Fid |  Size | Speed |
|-----------|-------:|------|------:|-------:|-------:|------:|------:|------:|------:|
| original  |  11.2M | fp32 | 44.77 |   1.1B |  4.444 | 90.7% |     - | 1.00x | 1.00x |
| pruned    |   2.8M | fp32 | 11.25 | 278.6M |  1.731 | 89.1% | 93.4% | 0.25x | 2.57x |
| onnx fp32 |   2.8M | fp32 | 11.27 | 278.6M |  1.317 | 89.1% | 93.4% | 0.25x | 3.37x |
| onnx int8 |   2.8M | int8 |  2.92 |      - |  0.617 | 89.1% | 93.0% | 0.07x | 7.20x |

Each row changes one thing. Pruning removes half the channels, so parameters, file size and
FLOPs all fall about 4x. Moving to ONNX Runtime changes the runtime, not the model: same
weights, same accuracy, 1.31x less time. Static INT8 changes the data type, not the runtime:
4x smaller again and 2.13x faster. Together: **90.7% to 89.1% accuracy for a 15x smaller
model that answers 7.2x faster.**

The script also compares the importance criteria at ratios where the model still works:

| Ratio pruned | random |    l1 |    l2 |    bn | taylor |
|--------------|-------:|------:|------:|------:|-------:|
| 5%           |  83.5% | 89.7% | 89.2% | 86.1% |  84.2% |
| 10%          |  71.7% | 81.7% | 81.6% | 67.2% |  78.1% |
| 20%          |  29.5% | 34.9% | 31.4% | 29.2% |  44.5% |

No criterion dominates, and the ranking changes with the ratio. It matters even less after
fine-tuning: pruning 50% of the channels with `l1`, `taylor` and `random` and giving each
the same 3 epochs gives 89.18%, 89.55% and 89.02%. Random channel selection lands within
half a point of the informed criteria, which says the surviving *width* matters more than
which channels survived. What does make a measurable difference is the per-layer ratio: on
the demo CNN, removing half of the first convolution costs 14.8 accuracy points and half of
the second costs 0.3, for the same FLOPs saving. That is what the sensitivity analysis and
the budget search are for.

Reproduce with `uv run python examples/resnet18_cifar10.py` (about 30 minutes on an Apple
Silicon laptop, most of it the initial training). Absolute latencies are machine specific.

## How structured pruning works

Removing output channel `c` of one layer forces matching removals elsewhere. The model is
traced with `torch.fx`, shapes are recorded with one forward pass, and the graph is walked
once to build **dependency groups**: sets of tensors that must always be sliced with the
same channel indices.

| Node reached by a tracked tensor | Rule |
|---|---|
| `Conv1d`, `Conv2d`, `Linear` | Its *input* side joins the group as a consumer; its output starts a new group. |
| `BatchNorm1d/2d` | `weight`, `bias`, `running_mean`, `running_var` are sliced with the group. |
| ReLU, GELU, SiLU, Tanh, pooling, dropout, ... | Passes through. Only shape-preserving ops with `f(0) = 0` are on this list. |
| `flatten` / `view` / `reshape` to `(N, C*H*W)` | Channel `c` becomes the `H*W` consecutive features `c*H*W ... (c+1)*H*W - 1` of the next `Linear`. |
| `add`, `sub` (residual connections) | The groups of all branches are merged (union-find): they must keep the same channels. |
| `mul` of two tracked tensors | Merged like `add`. Multiplying or dividing by a Python scalar passes through; dividing by a tensor is refused. |
| Depthwise convolution | Input and output channels are tied, so it is sliced with its source group and `groups` is updated. |
| Anything else | The groups involved are marked **not prunable**, with the reason recorded. |

A residual stage ends up as one group. For the first stage of torchvision's ResNet-18:

```text
           +------------------ identity ------------------+
           |                                              v
 conv1 -> bn1 -> relu -+-> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+) -> relu -> ...
 [stem]                    [block 0, inner]        [block 0, outer]

 group A (the residual stream), one shared set of kept channels:
   producers  stem conv1, layer1.0.conv2, layer1.1.conv2      (output channels sliced)
   norms      stem bn1,   layer1.0.bn2,   layer1.1.bn2
   consumers  layer1.0.conv1, layer1.1.conv1,
              layer2.0.conv1, layer2.0.downsample.0           (input channels sliced)

 group B (inside block 0), pruned independently of A:
   producers  layer1.0.conv1    norms  layer1.0.bn1    consumers  layer1.0.conv2
```

**The guard: zeroing must equal removing.** For any pruning plan, the library can either
zero the dropped channels (producer weights and biases, BatchNorm scale and shift) or
physically remove them. If the dependency analysis is right, both models produce the same
output. The test suite asserts this for the demo models, a residual net, a depthwise net,
12 randomly generated CNNs and torchvision's ResNet-18, ResNet-50, MobileNetV2 and
VGG-11-BN, for every importance criterion. This equivalence is also why `sigmoid` is not a
pass-through op: `sigmoid(0) = 0.5`, so a zeroed channel would still carry signal that a
removed channel cannot.

**What is refused, and why.** Skipping a layer is safe; guessing is not. A group is left
untouched, and the reason reported, when it:

- feeds the model output (the number of classes must not change),
- passes through an unknown module or function (`torch.cat`, `LayerNorm`, `sigmoid`,
  attention, ...),
- is reshaped by anything other than a flatten to 2-D (a `view` with a hard coded feature
  count included),
- has a constant added to it, is divided by a tensor, or is joined with a branch that is
  not tracked or broadcasts over channels,
- belongs to a module that is called more than once, a grouped (non depthwise)
  convolution, a `Linear` applied to an input with more than two dimensions, or a layer
  whose weights are parametrized, tied, or used outside the module.

If nothing is prunable, `prune_structured` raises a `ValueError` listing the blockers. If
the model cannot be traced at all (data dependent control flow), it raises
`UntraceableModelError`. More detail in [docs/how-it-works.md](docs/how-it-works.md).

## Features

- **Profiling**: parameters, sparsity, serialized size, gzip size, FLOPs, CPU latency
  (warm-up, median, p95), accuracy through a user supplied `eval_fn`, label-free fidelity.
  Counts stay correct with pruning masks attached and with packed quantized weights.
- **Unstructured pruning**: global or per-layer magnitude pruning on top of
  `torch.nn.utils.prune`, made permanent by default.
- **Structured pruning**: `torch.fx` dependency graph, BatchNorm, flatten, residual and
  depthwise handling; importance by L1, L2, BatchNorm scale, first-order Taylor, or random;
  uniform or global ratios, explicit per-group ratios, `round_to` for hardware friendly
  channel counts; fine-tune hook.
- **Sensitivity analysis and budget search**: per-layer ratios from "lose at most X" and/or
  "reach this fraction of the FLOPs", verified on the real pruned model.
- **Dynamic INT8**: one interface over `torchao` and the legacy `torch.ao.quantization`
  backend, with automatic quantized-engine selection (needed on Apple Silicon).
- **ONNX**: export with a dynamic batch axis, verified against the PyTorch outputs;
  ONNX Runtime static INT8 with calibration data; ONNX files profiled with the same table.
- **One call and a CLI**: `optimize()` runs prune, fine-tune, quantize and profiles every
  stage; reports as a terminal table or JSON with environment info.
- **Input models are never modified.** Every function works on a copy.

## CLI reference

| Command | Purpose |
|---|---|
| `pymodelprune demo [--model mlp\|cnn]` | Profile a built-in model; checks the installation. |
| `pymodelprune profile [MODEL] --input-shape ...` | Profile one model, optionally with `--eval`. |
| `pymodelprune optimize [MODEL] --input-shape ...` | Prune and/or quantize, report every stage. |
| `pymodelprune sensitivity [MODEL] --input-shape ...` | Per-layer tolerance to structured pruning. |

Main `optimize` options: `--prune 0.5`, `--method structured|unstructured`,
`--importance l1|l2|bn|random`, `--max-acc-drop 1.0` (percentage points),
`--target-flops 0.5`, `--quantize none|dynamic|weight_only|onnx_static`,
`--backend auto|torchao|legacy`, `--eval`, `--data`, `--finetune`
(hooks given as `package.module:function`), `--out`, `--json`, `--threads`, `--runs`.
Full reference: [docs/cli.md](docs/cli.md).

## Limitations

- Structured pruning needs a model that `torch.fx` can trace: no data dependent control
  flow. Unstructured pruning and quantization work on anything.
- No `torch.cat`: DenseNet, U-Net and Inception-style models are reported as not prunable.
- No structured pruning of Hugging Face transformers (attention, `LayerNorm` and `Linear`
  layers on 3-D inputs are all refused).
- Sigmoid-gated blocks (squeeze-and-excitation) are skipped, together with the channels
  they gate. `Linear` layers on inputs with more than two dimensions are skipped.
- The layers that produce the model output are never pruned.
- Latency numbers are specific to the machine, thread count and batch size they were
  measured on. FLOPs, parameters and sizes are deterministic.
- Dynamic INT8 may be slower than FP32, as it was on the machine above. Measure before
  you ship.
- The `legacy` quantization backend wraps `torch.ao.quantization`, which is deprecated
  upstream; `torchao` replaces it but is tied to specific PyTorch versions.
- Without `fidelity_inputs` / `--data` the `Fid` column is computed on the single example
  input (a warning says so), and a budget search or sensitivity analysis without `eval_fn`
  refuses to run on fewer than 32 samples.
- ONNX export supports single-tensor outputs. It is verified on three inputs, which catches
  most frozen control flow, not all of it.
- Accuracy helpers assume classification (`argmax` over the last dimension). Any other
  task works through a custom `eval_fn`.

See [docs/limitations.md](docs/limitations.md).

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The test suite makes no claims about speed; it checks exact things (shapes, FLOPs, sizes,
output equivalence). Tests that need `torchao`, `onnx`/`onnxruntime` or `torchvision` skip
when those are missing, and CI runs one job with the core dependencies only.

Reproduce the tables:

```bash
uv run python examples/train_fashion_mnist.py      # trains both demo models, under a minute
uv run python examples/sweep_unstructured.py --model mlp
uv run python examples/quantize_dynamic_demo.py
uv run python examples/sweep_structured.py --model cnn --finetune
uv run python examples/budget_search_demo.py --max-drop 0.01
uv run python examples/resnet18_cifar10.py          # downloads CIFAR-10, trains ResNet-18
```

Documentation is built with MkDocs: `pip install mkdocs-material`, then `mkdocs serve`.

## License

MIT, see [LICENSE](LICENSE).
