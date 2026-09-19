# Quickstart

## Install

```bash
pip install pymodelprune            # core
pip install "pymodelprune[quant]"   # + torchao
pip install "pymodelprune[onnx]"    # + onnx, onnxruntime, onnxscript
pip install "pymodelprune[all]"     # everything
```

Check the installation:

```bash
pymodelprune demo --model cnn
```

## What you need to bring

| Input | Required | Used for |
|---|---|---|
| `model` | yes | Any `nn.Module`. It is deep-copied, moved to CPU and put in `eval()` mode. |
| `example_input` | yes | One input tensor (or a tuple of tensors). Drives tracing, FLOPs and latency, so its batch size is the batch size the latency refers to. |
| `eval_fn` | no | `model -> float`, higher is better. Fills the `Acc` column and steers the budget search. |
| `fidelity_inputs` | no | A tensor or an iterable of batches (`(inputs, labels)` batches are fine, labels are ignored; generators are materialised). Fills the `Fid` column; also the fallback calibration set for static INT8. Defaults to `example_input`, a single sample, which earns a `FewSamplesWarning`: pass at least 32 real samples. |
| `finetune_fn` | no | `model -> None`, trains the pruned model in place. |
| `grad_fn` | only for `importance="taylor"` | `model -> None`, runs backward passes so parameters have `.grad`. |

`make_accuracy_fn(batches)` builds an `eval_fn` for classification from any re-iterable of
`(inputs, labels)` batches (a `DataLoader` or a list, not a generator).

## Prune, quantize, report

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
    eval_fn=make_accuracy_fn(val_batches),
    fidelity_inputs=val_batches,
)
result.report()
result.to_json("report.json")
```

The report has one row per stage: `original`, `pruned`, `int8`. With an untrained model
and random labels the accuracy column is meaningless, of course; the snippet only shows
the mechanics.

Useful attributes of the result:

```python
result.stages  # list of Stage(name, profile, model, onnx_path)
result.profiles  # the ModelProfile of every stage
result.model  # the last stage that is still a PyTorch model
result.environment  # python, torch, platform, threads, quantized engine
```

## Let the search choose the ratios

```python
from pymodelprune import Budget


def finetune(pruned):
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
    prune=PruneConfig(budget=Budget(max_drop=0.01)),  # at most one accuracy point
    eval_fn=make_accuracy_fn(val_batches),
    fidelity_inputs=val_batches,
    finetune_fn=finetune,
)
result.report()
result.save("model_pruned.pt")

print(result.search.ratios)  # {group id: ratio}
print(result.search.predicted_drop, result.search.measured_drop, result.search.refinements)
```

`Budget.max_drop` is expressed in the units of your metric: with accuracy in `[0, 1]`,
`0.01` is one percentage point. `Budget(target_flops_ratio=0.5)` stops once the FLOPs have
halved; with both limits set the search stops at whichever is reached first.

The sensitivity analysis behind the search costs `groups x ratios` evaluations, so give it
an `eval_fn` over a small validation subset (`make_accuracy_fn(loader, max_batches=4)`).

## Real INT8 speed: ONNX Runtime

```python
result = optimize(
    model,
    example,
    prune=PruneConfig(amount=0.5),
    quantize=QuantConfig(mode="onnx_static", calibration=val_batches),
    eval_fn=make_accuracy_fn(val_batches),
    fidelity_inputs=val_batches,
    onnx_path="out/model.int8.onnx",
)
result.report()  # original, pruned, onnx fp32, onnx int8
print(result.onnx_path)
```

The float export is written next to the INT8 file as `<stem>_fp32.onnx` (here
`out/model.int8_fp32.onnx`). The ONNX
stages are measured through ONNX Runtime by the same profiling code, so their latency,
accuracy and fidelity are directly comparable with the PyTorch rows.

## Using the building blocks directly

```python
from pymodelprune import (
    build_dependency_graph,
    profile_model,
    prune_structured,
    render_profiles,
)

graph = build_dependency_graph(model, example)
for group in graph.groups:
    status = "prunable" if group.prunable else f"blocked: {group.blockers}"
    print(group.id, group.channels, group.producers, status)

eval_fn = make_accuracy_fn(val_batches)
before = profile_model(model, example, name="original", eval_fn=eval_fn)
smaller = prune_structured(model, example, amount=0.5)
after = profile_model(
    smaller,
    example,
    name="pruned",
    eval_fn=eval_fn,
    reference=model,
    fidelity_inputs=val_batches,
)
render_profiles([before, after])
```

## Command line

The CLI takes a model file written by `torch.save(model, path)`, or a factory function plus
a `state_dict`. Data, metrics and training are supplied as hooks,
`package.module:function`; the current directory is importable.

```python
# hooks.py
import torch

from pymodelprune import accuracy
from pymodelprune.demo_models import TinyCNN

_BATCHES = [(torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))) for _ in range(4)]


def build_model():  # --factory
    return TinyCNN()


def evaluate(model):  # --eval: model -> float, higher is better
    return accuracy(model, _BATCHES)


def batches():  # --data: returns an iterable of input batches (at least 32 samples)
    return _BATCHES
```

```bash
pymodelprune profile --factory hooks:build_model --input-shape 1,1,28,28 --eval hooks:evaluate

pymodelprune optimize --factory hooks:build_model --input-shape 1,1,28,28 \
    --prune 0.5 --eval hooks:evaluate --data hooks:batches \
    --out model_pruned.pt --json report.json

pymodelprune sensitivity --factory hooks:build_model --input-shape 1,1,28,28 \
    --eval hooks:evaluate
```

See the [CLI reference](cli.md) for every option.
