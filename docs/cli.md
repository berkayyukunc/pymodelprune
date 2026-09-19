# CLI reference

```text
pymodelprune [OPTIONS] COMMAND [ARGS]...

Commands:
  demo         Profile a small built-in model. Useful to check the installation.
  profile      Profile a model: parameters, size, FLOPs, latency and optionally accuracy.
  optimize     Prune and/or quantize a model and report every stage against the original.
  sensitivity  Show which layers tolerate structured pruning and which do not.
```

## Loading a model

`profile`, `optimize` and `sensitivity` accept the model in one of two ways.

| Form | Example | Notes |
|---|---|---|
| Full model file | `pymodelprune profile model.pt ...` | Written by `torch.save(model, "model.pt")`. Loaded with `weights_only=False`, which unpickles and can therefore **run arbitrary code**. Only load files you trust. The model's class must be importable. |
| Factory + weights | `--factory my_pkg.models:build --weights state_dict.pt` | The factory is called without arguments and must return an `nn.Module`; the `state_dict` is loaded with `weights_only=True`. `--weights` is optional. |

A file that holds a bare `state_dict` is rejected with a message that points to
`--factory`/`--weights`.

## Hooks

Data, metrics and training are Python functions referenced as `package.module:function`.
The current working directory is importable, so a `hooks.py` next to your model is enough.

| Option | Signature | Purpose |
|---|---|---|
| `--factory` | `() -> nn.Module` | Builds the model. |
| `--eval` | `(model) -> float` | Score, higher is better. Fills the `Acc` column; metric of the search. |
| `--data` | `() -> iterable of batches` | Real inputs for fidelity, the budget search and INT8 calibration. Batches may be tensors or `(inputs, labels)`. A generator is fine: it is materialised as a list, because the data is read more than once. |
| `--finetune` | `(model) -> None` | Trains the pruned model in place. |

Without `--data` the single random example input stands in, so the `Fid` column is
computed on one random sample and says very little; a `warning:` line says so whenever
fidelity is computed on fewer than 32 samples. The budget search and `sensitivity` refuse
to run on that: they need `--eval`, or `--data` with at least 32 samples.
`--quantize onnx_static` always needs `--data` for calibration.

## `demo`

```text
pymodelprune demo [--model mlp|cnn] [--runs N]
```

| Option | Default | Description |
|---|---|---|
| `--model` | `mlp` | Built-in model: `mlp` or `cnn` (input shape `1,1,28,28`). |
| `--runs` | `100` | Timed forward passes (at least 2). |

## `profile`

```text
pymodelprune profile [MODEL] --input-shape 1,3,224,224 [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--input-shape` | required | Example input shape, comma separated. The first number is the batch size used for timing. |
| `--factory`, `--weights` | - | See [Loading a model](#loading-a-model). |
| `--eval` | - | Adds the `Acc` column. |
| `--runs` | `100` | Timed forward passes. |

## `optimize`

```text
pymodelprune optimize [MODEL] --input-shape 1,3,32,32 [OPTIONS]
```

At least one of `--prune`, `--max-acc-drop`, `--target-flops` or `--quantize` is required.

| Option | Default | Description |
|---|---|---|
| `--input-shape` | required | Example input shape. |
| `--factory`, `--weights` | - | See [Loading a model](#loading-a-model). |
| `--prune` | - | Fraction to prune, `0.0` to `0.99`. |
| `--method` | `structured` | `structured` removes channels; `unstructured` zeroes single weights. |
| `--importance` | `l1` | `l1`, `l2`, `bn` or `random`. With `--method unstructured` only `l1` and `random` are valid. `taylor` needs a gradient function and is available from Python only. |
| `--max-acc-drop` | - | Budget search: tolerated score loss **in percentage points** (`1.0` means one point, i.e. `Budget(max_drop=0.01)`). Requires `--eval` or `--data`. |
| `--target-flops` | - | Budget search: stop once FLOPs fall to this fraction, e.g. `0.5`. Requires `--eval` or `--data`. |
| `--quantize` | `none` | `none`, `dynamic`, `weight_only` or `onnx_static`. |
| `--backend` | `auto` | Dynamic quantization backend: `auto`, `torchao` or `legacy`. |
| `--eval`, `--data`, `--finetune` | - | See [Hooks](#hooks). |
| `--out` | - | Output file: `.pt` (full model), or `.onnx`, which is required with `--quantize onnx_static`. |
| `--json` | - | Write the report, the environment and the search history as JSON. |
| `--threads` | PyTorch default | CPU threads used while timing. |
| `--runs` | `100` | Timed forward passes. |

Either budget option switches on the search, which is structured only. Combining it with
`--prune` is an error: the search decides the amount. After a search the CLI prints a summary line:

```text
search: predicted drop <predicted>%, measured <measured>%, FLOPs x<remaining fraction>
```

With `--quantize onnx_static --out model.onnx`, the float export is kept next to the INT8
file as `model_fp32.onnx`.

Examples:

```bash
# remove half of the channels, fine-tune, save
pymodelprune optimize model.pt --input-shape 1,3,32,32 --prune 0.5 \
    --eval hooks:evaluate --data hooks:batches --finetune hooks:finetune --out pruned.pt

# search: lose at most one accuracy point
pymodelprune optimize model.pt --input-shape 1,3,32,32 --max-acc-drop 1.0 \
    --eval hooks:evaluate --data hooks:batches --json report.json

# prune, then static INT8 through ONNX Runtime
pymodelprune optimize model.pt --input-shape 1,3,32,32 --prune 0.3 \
    --quantize onnx_static --data hooks:batches --out model_int8.onnx
```

Errors raised by the library or by loading the model (nothing prunable, untraceable model,
missing extra, a file that is not a model, an `--input-shape` the model does not accept,
an already quantized model, ...) are printed as one `error: ...` line and the exit code
is 1. Invalid option combinations are reported by the option parser with exit code 2.

## `sensitivity`

```text
pymodelprune sensitivity [MODEL] --input-shape 1,3,32,32 [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--input-shape` | required | Example input shape. |
| `--factory`, `--weights` | - | See [Loading a model](#loading-a-model). |
| `--eval` | - | Metric. Without it, fidelity to the original model is used. One of `--eval` and `--data` is required. |
| `--data` | - | Inputs for the fidelity metric, at least 32 samples. |
| `--importance` | `l1` | `l1`, `l2`, `bn` or `random`. |
| `--json` | - | Write the curves as JSON (readable by `SensitivityReport.from_json`). |

Output for the FashionMNIST `TinyCNN`, with an `--eval` hook scoring accuracy over
2000 test images (`examples/train_fashion_mnist.py` produces the model):

```text
score change per pruned share (eval_fn, baseline 89.6%), most sensitive first
Group  Layers       -10%   -30%    -50%    -70%    -90%
0      features.0  -2.7%  -5.1%  -14.8%  -24.5%  -65.0%
1      features.4  -1.1%  +0.9%   -0.3%   -5.4%  -28.0%
not prunable: classifier (model output)
```

Each cell is the score change when only that group is pruned by that share, without
fine-tuning. Groups are ordered by their drop at the middle ratio. Layers that were refused
are listed at the bottom with the first reason. Here the first convolution is the fragile
one and the second tolerates half its channels going away, which is exactly the imbalance
`optimize` exploits when you hand it a `Budget` instead of a fixed ratio.

Curves are only as meaningful as the inputs behind them: pass real data with `--data`, and
prefer `--eval` when you have labels.
