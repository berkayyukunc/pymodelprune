# Limitations

This page lists what the library does not do, and what it refuses to do on purpose.

## Structured pruning

**The model must be traceable by `torch.fx`.** Data dependent control flow
(`if x.sum() > 0:`), and Python constructs that symbolic tracing cannot follow, raise
`UntraceableModelError`. Unstructured pruning and quantization still work on such models.

**Unsupported patterns are skipped, not guessed.** A channel group is left untouched, and
the reason is recorded in `ChannelGroup.blockers`, when it meets:

| Pattern | Consequence |
|---|---|
| `torch.cat` / concatenation | DenseNet, U-Net and Inception-style models are not prunable. |
| Attention, `LayerNorm`, `GroupNorm`, embeddings | No structured pruning of transformers, including Hugging Face models. |
| `Linear` applied to an input with more than two dimensions | Skipped, together with the group feeding it. |
| Sigmoid, hard-sigmoid and softmax gates | Squeeze-and-excitation blocks are skipped, and so are the channels they gate. |
| Reshapes other than a flatten to `(N, features)` | `view`/`reshape`/`permute`/`transpose` tricks (channel shuffle, pixel shuffle) block the group. |
| Grouped convolutions that are not depthwise | Skipped. |
| A module called more than once, tied weights, weights used functionally outside their module | Skipped. |
| Parametrized weights (`weight_norm`, `torch.nn.utils.parametrize`), `BatchNorm` with `affine=False` | Skipped. |
| Division by a tensor; operations returning tuples (`split`, `sort`, `max(dim=...)`) | Skipped. |
| A constant or a learned parameter added to the tensor | Skipped. |
| Operations that are simply not on the list (`x.mean((2, 3))`, `ConvTranspose2d`, `Conv3d`, ...) | Skipped. |

**The layers producing the model output are never pruned**, so the output shape is always
preserved. If no group at all is prunable, `prune_structured` raises a `ValueError` that
lists the blockers; `pymodelprune sensitivity` prints them per layer.

**Coverage that is tested**: plain CNNs and MLPs, VGG-11-BN, ResNet-18/50, MobileNetV2
(depthwise convolutions), and randomly generated CNNs. Anything else may work, may be
partly skipped, and is covered by the equivalence check only if you run it yourself
(see [How it works](how-it-works.md#the-correctness-guard-zeroing-equals-removing)).

**Pruning without fine-tuning is destructive.** Expect to pass a `finetune_fn`. The library
calls it but does not implement training.

**A pruned model is saved as a full pickled module**, because its layer shapes no longer
match the original class. Loading it needs `torch.load(path, weights_only=False)` and the
original class to be importable; only load files you trust.

## Budget search

- The greedy search assumes single-layer losses add up. They do not exactly, which is why
  the result is re-measured and scaled back if needed; it is a good configuration, not a
  proven optimum.
- The sensitivity curves are measured without fine-tuning, so they are pessimistic for
  workflows that fine-tune afterwards.
- A `target_flops_ratio` is a stopping rule, not a guarantee: the grid ends at 90% per group,
  and a simultaneous `max_drop` takes priority.
- Without an `eval_fn` the search optimizes fidelity on the inputs you provide, and refuses
  to run (`ValueError`) on fewer than 32 samples: agreement on a handful of inputs is
  all-or-nothing. Fidelity only measures closeness to the original, not task quality.
- A saved `SensitivityReport` is only accepted for the model and example shape it was
  measured on (same prunable groups, same FLOPs).

## Quantization

- **Dynamic INT8 covers `nn.Linear` only.** Convolutional models barely change.
- **Dynamic INT8 may be slower than FP32.** On the Apple Silicon development machine no
  dynamic variant was faster than FP32, and on a Linear-only model all were several times
  slower. INT8 speed depends on the kernels available for your CPU.
- **The `legacy` backend is deprecated upstream.** It wraps `torch.ao.quantization`, which
  PyTorch has announced for removal. The `torchao` backend replaces it but is a separate
  package tied to specific PyTorch versions. `weight_only` mode needs torchao.
- **An already quantized model cannot be optimized again.** `quantize_dynamic` and
  `optimize` raise `ValueError("model is already quantized")`; start from the float model.
- **Static INT8 produces an ONNX file**, not a PyTorch model, and needs the `[onnx]` extra
  plus representative calibration data. INT8 accuracy is reported, not enforced: check the
  `Acc`/`Fid` columns and use `exclude_nodes` for sensitive layers.
- **ONNX export supports single-tensor outputs only.** Models returning tuples or dicts
  raise `TypeError`. Verification runs on the example, its negation and a second batch
  size, which catches most frozen control flow but cannot prove the absence of it: when the
  exporter warns that it fell back to tracing, check the model for `if`/`for` on tensor values.
- FLOPs are not reported for INT8 models; the counter cannot see quantized kernels.

## Measurement

- **Latency is machine specific.** It depends on the CPU, the thread count, the batch size
  of `example_input`, the PyTorch/ONNX Runtime build, and whatever else the machine is doing.
  Parameters, sizes and FLOPs are deterministic; latency ratios are not portable. Small
  differences between two rows can be noise, since no confidence interval is computed.
- The pipeline measures latency on CPU by design: `optimize` moves the model and the
  example input there. `measure_latency` and `profile_model` called directly do time CUDA
  and MPS models correctly (the device is synchronized around every pass).
- **Fidelity needs data.** Without `fidelity_inputs` / `--data` the `Fid` column is computed
  on the single example input and a `FewSamplesWarning` says so; below 32 samples the
  percentage is mostly noise.
- `accuracy`, `make_accuracy_fn` and fidelity assume classification: a single output tensor
  and `argmax` over its last dimension. Other tasks need a custom `eval_fn`, and fidelity's
  `agreement` may not be meaningful for them (`max_abs_diff` still is).
- FLOPs are whatever `torch.utils.flop_counter` counts (matrix multiplications,
  convolutions and attention kernels), not a full operation count.

## Scope

Not included, deliberately: training loops, datasets, knowledge distillation, quantization
aware training, N:M or block sparsity, sparse inference kernels, and GPU deployment
(TensorRT and the like).
