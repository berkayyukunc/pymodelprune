# pymodelprune

Prune and quantize PyTorch models, then measure what you actually gained.

`pymodelprune` is a small library and CLI for CPU-oriented model compression. It applies an
optimization, returns a new model, and profiles that model against the original with the
same measurement code every time: parameters, sparsity, file size, gzip size, FLOPs,
latency, accuracy and fidelity.

## The headline result

ResNet-18 on CIFAR-10, pruned to half its channels, fine-tuned for 3 epochs, exported to
ONNX Runtime and quantized to INT8. Apple Silicon CPU, batch size 1, every stage timed
together at the end:

| Model     | Params | Type |    MB |  FLOPs | p50 ms |   Acc |  Size | Speed |
|-----------|-------:|------|------:|-------:|-------:|------:|------:|------:|
| original  |  11.2M | fp32 | 44.77 |   1.1B |  4.444 | 90.7% | 1.00x | 1.00x |
| pruned    |   2.8M | fp32 | 11.25 | 278.6M |  1.731 | 89.1% | 0.25x | 2.57x |
| onnx fp32 |   2.8M | fp32 | 11.27 | 278.6M |  1.317 | 89.1% | 0.25x | 3.37x |
| onnx int8 |   2.8M | int8 |  2.92 |      - |  0.617 | 89.1% | 0.07x | 7.20x |

15x smaller, 7.2x faster, 1.6 accuracy points. Reproduce with
`uv run python examples/resnet18_cifar10.py`.

## The point

Compression techniques are often reported with the one metric that improved. This project
reports all of them, including the ones that did not move:

| Technique | What it changes | What it does not |
|---|---|---|
| Unstructured pruning (`torch.nn.utils.prune`) | gzip size | raw file size, FLOPs, latency |
| Dynamic INT8 (`nn.Linear` only) | file size of Linear-heavy models (about 4x) | convolutions; and it may be slower than FP32 |
| Structured pruning | parameters, FLOPs, file size, latency | the layers that produce the model output |
| ONNX Runtime static INT8 | file size and latency, convolutions included | the output is an `.onnx` file, not a PyTorch model |

The measured tables behind this summary are in the project README, and every one of them
can be regenerated with a script from `examples/`.

## Design rules

- **The input model is never modified.** Every function deep-copies and returns a new model,
  which is what makes before/after comparisons trustworthy.
- **The library never owns a dataset.** Accuracy comes from your `eval_fn(model) -> float`.
  Fidelity (agreement with the original model's predictions) needs no labels at all.
- **Skip, do not guess.** Structured pruning only touches layers whose dependencies it
  fully understands; everything else is left alone and the reason is reported.
- **Correctness is tested by equivalence.** Zeroing channels and removing channels must
  give identical outputs. See [How it works](how-it-works.md).
- **Heavy dependencies are optional.** Core: `torch`, `numpy`, `typer`, `rich`. Extras:
  `[quant]` (torchao), `[onnx]` (onnx, onnxruntime, onnxscript).
- **Speed is measured on CPU**, with warm-up runs discarded and the median reported.
  Candidates are timed in alternating blocks rather than one after the other, so a drift in
  machine load cannot masquerade as a speedup. Tests never assert on latency.

## Where to go next

- [Quickstart](quickstart.md): install, first `optimize` call, the CLI.
- [How it works](how-it-works.md): the dependency graph, importance criteria, budget search,
  and why static INT8 goes through ONNX Runtime.
- [CLI](cli.md) and [API reference](api.md).
- [Limitations](limitations.md): what is out of scope and what is refused.
