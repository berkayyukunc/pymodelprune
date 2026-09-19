"""Prune and quantize PyTorch models, then measure what you actually gained."""

from pymodelprune.evaluate import EvalFn, accuracy, make_accuracy_fn, reusable
from pymodelprune.export import OrtModule, profile_onnx, save_model, to_onnx, verify_onnx
from pymodelprune.fidelity import FewSamplesWarning, Fidelity, compare_outputs
from pymodelprune.pipeline import OptimizationResult, PruneConfig, QuantConfig, optimize
from pymodelprune.profile import (
    LatencyStats,
    ModelProfile,
    count_flops,
    count_parameters,
    measure_latency,
    measure_latency_interleaved,
    model_size_mb,
    profile_model,
    retime_profiles,
)
from pymodelprune.prune import (
    build_dependency_graph,
    make_classification_grad_fn,
    prune_structured,
    prune_unstructured,
)
from pymodelprune.quantize import quantize_dynamic
from pymodelprune.quantize.onnx_static import quantize_onnx_static
from pymodelprune.report import render_profiles
from pymodelprune.search import Budget, analyze_sensitivity, search_budget

__version__ = "0.4.0"

__all__ = [
    "Budget",
    "EvalFn",
    "FewSamplesWarning",
    "Fidelity",
    "LatencyStats",
    "ModelProfile",
    "OptimizationResult",
    "OrtModule",
    "PruneConfig",
    "QuantConfig",
    "accuracy",
    "analyze_sensitivity",
    "build_dependency_graph",
    "compare_outputs",
    "count_flops",
    "count_parameters",
    "make_accuracy_fn",
    "make_classification_grad_fn",
    "measure_latency",
    "measure_latency_interleaved",
    "model_size_mb",
    "optimize",
    "profile_model",
    "profile_onnx",
    "prune_structured",
    "prune_unstructured",
    "quantize_dynamic",
    "quantize_onnx_static",
    "render_profiles",
    "retime_profiles",
    "reusable",
    "save_model",
    "search_budget",
    "to_onnx",
    "verify_onnx",
]
