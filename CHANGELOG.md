# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions 0.1.0 to 0.4.0 were developed together and share one date.

## [0.4.0] - 2026-09-19

### Added

- `to_onnx`: ONNX export with a dynamic batch axis. The exported file is verified against
  the PyTorch outputs by default and `OnnxMismatchError` is raised above the tolerance.
- `verify_onnx`: largest output difference between a PyTorch model and an ONNX file.
- `OrtModule`: an ONNX Runtime session wrapped as an `nn.Module`, so profiling, accuracy and
  fidelity run on ONNX files unchanged.
- `profile_onnx`: profile an ONNX file (file size, gzip size, latency, accuracy, fidelity)
  into the same `ModelProfile` used for PyTorch models.
- `quantize_onnx_static`: static INT8 quantization through ONNX Runtime with calibration
  data, per-channel weights, `calibration_method` (`minmax`, `entropy`, `percentile`) and
  `exclude_nodes` for mixed precision.
- `QuantConfig(mode="onnx_static")` and the `onnx_path` argument of `optimize`, which add
  `onnx fp32` and `onnx int8` stages to the report.
- CLI: `optimize --quantize onnx_static --out model.onnx`.
- `save_model` and `OptimizationResult.save` for full-model checkpoints of structurally
  pruned models.
- `measure_latency_interleaved` and `retime_profiles`: time several models in alternating
  blocks of passes, so a drift in machine load hits every candidate equally. `optimize`
  re-times all of its stages together at the end, because they are otherwise measured
  minutes apart. The passes come in blocks (20 by default) rather than one at a time: a
  strict round robin lets each model evict its neighbours' weights from the caches, which
  made a 44 MB model look 3x slower than it is and inflated every speedup measured against
  it.
- `examples/resnet18_cifar10.py`: end-to-end run on ResNet-18 / CIFAR-10, plus an importance
  criterion comparison at several pruning ratios.
- Optional dependency group `[onnx]`.
- `compare_outputs(num_inputs=...)` and multi-input support (an `example_input` tuple) in
  `profile_model`, `profile_onnx`, `optimize`, the searches and ONNX calibration.
- `reusable`: materialises one-shot iterators; applied to `fidelity_inputs`, calibration
  data and the CLI `--data` hook, so generators work everywhere.
- `FewSamplesWarning` when fidelity is computed on fewer than 32 samples (once per
  `optimize` call).
- `PruneConfig.min_channels`; `round_to` / `min_channels` arguments on `search_budget` and
  `analyze_sensitivity`; `optimize(onnx_atol=...)`; `rtol` on `to_onnx` / `verify_onnx`.
- `BudgetResult.scale`: the factor a bisection had to apply to the greedy selection, so the
  report says when `predicted_drop` and `history` no longer describe the returned model.
- `profile_model(flops=...)`: carry the float model's FLOP count onto an INT8 profile, where
  the counter cannot see quantized kernels.

### Changed

- `analyze_sensitivity` and `search_budget` without an `eval_fn` raise `ValueError` on fewer
  than 32 fidelity samples instead of ranking layers on one random tensor. The CLI
  `sensitivity` command requires `--eval` or `--data`.
- `to_onnx` verifies on the example, its negation and a second batch size, warns when it
  falls back to the tracing exporter, and deletes a file that fails verification.
- `verify_onnx` passes when `diff <= atol + rtol * max|expected|` and fails on NaN.
- The float ONNX side file of `optimize` is named `<stem>_fp32.onnx` (was `<stem>.fp32.onnx`).
- The report table shows `Sparse` and `gz MB` only when some stage is at least 5% sparse,
  so the full structured + INT8 comparison fits 80 columns.
- `optimize(threads=...)` restores the previous thread count afterwards.
- `PruneConfig` validates `importance` against the method, `amount`, `round_to` and
  `min_channels`. CLI: `--prune` together with a budget option is an error.
- `search_budget` validates that a given `report` belongs to the model, raises when nothing
  is prunable, and returns an untouched copy (without calling `finetune_fn`) when no ratio
  fits the budget.
- `search_budget` predicts the total score drop from the current ratios instead of
  accumulating clipped per-step increments, and considers every ratio above the current one
  as a candidate move. A bump early in a sensitivity curve no longer hides the cheaper
  ratios behind it, which used to make small budgets prune nothing at all.
- `Budget(target_flops_ratio=...)` alone documents that it bounds FLOPs only: the score is
  not verified and the target is met at the first grid point at or below it.
- A module called more than once now joins one blocked group instead of opening a new group
  per call, so every producer appears exactly once in `DependencyGraph.groups`.
- `select_channels` rounds halves up, so a sweep of ratios does not jump unevenly.
- `OrtModule.close()` (and the context manager form) releases a session and its thread pool
  on the spot. `optimize`, `verify_onnx` and `profile_onnx` close the sessions they open:
  left to the garbage collector they could outlive the caller and crash the process during
  interpreter shutdown, after all of the work was already done.
- Removed the unused `[plot]` extra.

### Fixed

- Models quantized by the torchao backend can be saved (`save_model`,
  `OptimizationResult.save`, `--out`); `save_model` writes atomically.
- `finetune_fn` after unstructured pruning no longer destroys the sparsity.
- `measure_latency` synchronizes CUDA and MPS devices around every timed pass.
- NaN outputs are no longer reported as `max_abs_diff == 0.0` or accepted by `verify_onnx`.
- `to_json` writes strict JSON (non-finite numbers become `null`) and scores returned as
  numpy or tensor scalars are stored as floats.
- `optimize`, `to_onnx` and `verify_onnx` accept models and inputs on MPS/CUDA.
- Quantizing an already quantized model raises `ValueError("model is already quantized")`.
- CLI: user errors (bad option values, corrupt files, wrong `--input-shape`, missing
  `--data` for `onnx_static`, ...) end as a one-line message without a traceback; `--json`
  creates parent directories; legacy INT8 `.pt` files load on Apple Silicon.
- `profile_model` serializes the weights once for both size columns.

## [0.3.0] - 2026-09-19

### Added

- `analyze_sensitivity`: prunes one dependency group at a time over a grid of ratios and
  records score and FLOPs. `SensitivityReport` with `drop`, `saved_flops`, `ranking` and a
  JSON round trip.
- `search_budget` and `Budget(max_drop=..., target_flops_ratio=...)`: greedy per-group ratio
  search over the sensitivity curves, followed by a real measurement of the chosen model
  and a bisection on a common scale factor if the limit is exceeded.
- `PruneConfig(budget=...)` in `optimize`; `OptimizationResult.search` and the search
  history in the JSON report.
- Fidelity (agreement with the original model) as the search metric when no `eval_fn` is
  given.
- CLI: `sensitivity` command; `optimize --max-acc-drop` and `--target-flops`.
- `examples/budget_search_demo.py`.

## [0.2.0] - 2026-09-19

### Added

- `prune_structured`: physically removes channels and neurons, so parameters, FLOPs, file
  size and latency all decrease.
- `build_dependency_graph`: `torch.fx` based dependency analysis producing `ChannelGroup`s.
  Handles BatchNorm, flatten into `Linear`, residual `add`/`sub` (union-find merge),
  element-wise `mul`, and depthwise convolutions. Unsupported patterns are marked not
  prunable with a recorded reason instead of being guessed; untraceable models raise
  `UntraceableModelError`.
- `plan_pruning`, `apply_plan` and `mask_plan`; the zero-versus-remove equivalence test that
  guards the dependency analysis, run on hand-written, randomly generated and torchvision
  architectures.
- Importance criteria: `l1`, `l2`, `bn` (BatchNorm scale), `taylor` (first order, through
  `grad_fn` / `make_classification_grad_fn`) and `random`.
- `scope="uniform" | "global"`, explicit `{group id: ratio}` amounts, `round_to` and
  `min_channels`.
- `finetune_fn` hook in `optimize` to recover accuracy after pruning.
- `count_flops` and a FLOPs column in the report.
- `PruneConfig(method="structured")` became the default pruning method of `optimize`.
- `examples/sweep_structured.py`.

## [0.1.0] - 2026-09-19

### Added

- `profile_model` and `ModelProfile`: parameters, sparsity, serialized size, gzip size, CPU
  latency (`measure_latency`: warm-up, median, p95, mean, standard deviation), weight dtype.
  Parameter counting that stays correct with pruning masks attached and with packed
  quantized weights.
- `accuracy`, `make_accuracy_fn` and the `EvalFn` contract (`model -> float`); the library
  never owns a dataset.
- `compare_outputs` and `Fidelity`: label-free agreement between an optimized model and the
  original.
- `prune_unstructured`: global or per-layer magnitude pruning on top of
  `torch.nn.utils.prune`, permanent by default.
- `quantize_dynamic`: dynamic and weight-only INT8 for `nn.Linear` behind one interface with
  two backends (`torchao`, and the deprecated `torch.ao.quantization` as `legacy`), with
  automatic quantized-engine selection.
- `optimize`, `PruneConfig`, `QuantConfig` and `OptimizationResult`: prune, quantize and
  profile every stage in one call; terminal report and JSON report with environment info.
- `render_profiles`: comparison table with size and speed ratios against the first row.
- CLI: `demo`, `profile` and `optimize`, with `package.module:function` hooks for `--eval`,
  `--data` and `--factory`.
- Demo models `TinyMLP` and `TinyCNN`; FashionMNIST example scripts.
- Optional dependency groups `[quant]`, `[plot]` (removed again later) and `[all]`.
