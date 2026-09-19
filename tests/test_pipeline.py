import copy
import json
import warnings

import numpy as np
import pytest
import torch
from torch import nn

from pymodelprune import Budget, PruneConfig, QuantConfig, make_accuracy_fn, optimize
from pymodelprune.demo_models import TinyCNN, TinyMLP
from pymodelprune.fidelity import FewSamplesWarning

EXAMPLE = torch.randn(1, 1, 28, 28)
FAST = {"warmup": 1, "runs": 3}


def test_prune_then_quantize():
    model = TinyMLP()
    before = copy.deepcopy(model.state_dict())
    result = optimize(
        model, EXAMPLE, PruneConfig(amount=0.5), QuantConfig(backend="legacy"), **FAST
    )
    assert [stage.name for stage in result.stages] == ["original", "pruned", "int8"]
    original, pruned, int8 = result.profiles
    assert pruned.total_params < 0.6 * original.total_params and pruned.flops < original.flops
    assert (
        int8.size_mb < 0.4 * pruned.size_mb and int8.dtype == "int8" and int8.flops == pruned.flops
    )
    assert original.fidelity is None and pruned.fidelity is not None
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert result.model(EXAMPLE).shape == (1, 10)


def test_unstructured_method_and_single_steps():
    result = optimize(TinyCNN(), EXAMPLE, PruneConfig("unstructured", 0.7), **FAST)
    assert result.profiles[1].sparsity > 0.6
    assert result.profiles[1].total_params == result.profiles[0].total_params
    only_quant = optimize(TinyMLP(), EXAMPLE, quantize=QuantConfig(backend="legacy"), **FAST)
    assert [stage.name for stage in only_quant.stages] == ["original", "int8"]


def test_budget_and_finetune(trained_blobnet, blob_batches):
    calls = []
    result = optimize(
        trained_blobnet, torch.randn(1, 1, 8, 8), PruneConfig(budget=Budget(max_drop=0.02)),
        eval_fn=make_accuracy_fn(blob_batches), finetune_fn=calls.append, **FAST,
    )  # fmt: skip
    assert result.search is not None and calls
    assert result.profiles[0].accuracy - result.profiles[1].accuracy <= 0.02 + 1e-9
    assert "search" in result.to_dict()


def test_json_report_and_save(tmp_path):
    result = optimize(TinyMLP(), EXAMPLE, PruneConfig(amount=0.25), **FAST)
    data = json.loads(result.to_json(tmp_path / "out" / "report.json").read_text())
    assert data["schema"] == 1 and {"torch", "threads", "machine"} <= set(data["environment"])
    assert [stage["name"] for stage in data["stages"]] == ["original", "pruned"]
    assert {"total_params", "size_mb", "flops", "latency", "sparsity", "fidelity"} <= set(
        data["stages"][1]
    )
    restored = torch.load(result.save(tmp_path / "m.pt"), weights_only=False)
    assert torch.equal(restored(EXAMPLE), result.model(EXAMPLE))


def test_onnx_static(tmp_path, trained_blobnet, blob_batches):
    pytest.importorskip("onnxruntime")
    result = optimize(
        trained_blobnet, torch.randn(1, 1, 8, 8), PruneConfig(amount=0.25),
        QuantConfig("onnx_static", calibration=blob_batches), eval_fn=make_accuracy_fn(blob_batches),
        onnx_path=tmp_path / "blob.onnx", **FAST,
    )  # fmt: skip
    assert [stage.name for stage in result.stages] == [
        "original",
        "pruned",
        "onnx fp32",
        "onnx int8",
    ]
    assert result.onnx_path == tmp_path / "blob.onnx" and result.onnx_path.exists()
    assert result.profiles[-1].dtype == "int8" and result.profiles[-1].accuracy > 0.85
    assert result.model is result.stages[1].model


def test_argument_validation():
    with pytest.raises(ValueError):
        optimize(TinyMLP(), EXAMPLE)
    with pytest.raises(ValueError):
        optimize(TinyMLP(), EXAMPLE, quantize=QuantConfig("onnx_static"))
    with pytest.raises(ValueError):
        PruneConfig(method="magic")
    with pytest.raises(ValueError):
        PruneConfig(method="unstructured", budget=Budget(max_drop=0.1))
    with pytest.raises(ValueError):
        QuantConfig(mode="fp4")
    with pytest.raises(ValueError, match="l1 | random"):
        PruneConfig(method="unstructured", importance="l2")
    with pytest.raises(ValueError, match="taylor"):
        PruneConfig(importance="magnitude")
    for bad in ({"amount": 1.0}, {"amount": -0.1}, {"round_to": 0}, {"min_channels": 0}):
        with pytest.raises(ValueError):
            PruneConfig(**bad)


def sgd_steps(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.randn(16, 1, 28, 28)).square().mean().backward()
        optimizer.step()


def test_finetuning_keeps_unstructured_sparsity():
    result = optimize(
        TinyMLP(), EXAMPLE, PruneConfig("unstructured", 0.7), finetune_fn=sgd_steps, **FAST
    )
    assert result.profiles[1].sparsity == pytest.approx(0.7, abs=0.02)
    assert not any(name.endswith("_orig") for name, _ in result.model.named_parameters())


class TwoInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.left, self.right = nn.Linear(8, 32), nn.Linear(4, 32)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, a, b):
        return self.head(self.left(a) + self.right(b))


def test_two_input_model():
    example = (torch.randn(1, 8), torch.randn(1, 4))
    batches = [(torch.randn(16, 8), torch.randn(16, 4)) for _ in range(2)]
    config = PruneConfig(amount=0.5), QuantConfig(backend="legacy")
    result = optimize(TwoInputs(), example, *config, fidelity_inputs=batches, **FAST)
    assert [profile.fidelity.samples for profile in result.profiles[1:]] == [32, 32]
    assert result.profiles[1].total_params < result.profiles[0].total_params
    with pytest.warns(FewSamplesWarning):
        fallback = optimize(TwoInputs(), example, *config, **FAST)
    assert fallback.profiles[-1].fidelity.samples == 1


def test_one_shot_iterator_feeds_every_stage():
    batches = (torch.randn(16, 1, 28, 28) for _ in range(2))
    result = optimize(
        TinyMLP(), EXAMPLE, PruneConfig(amount=0.5), QuantConfig(backend="legacy"),
        fidelity_inputs=batches, **FAST,
    )  # fmt: skip
    assert [profile.fidelity.samples for profile in result.profiles[1:]] == [32, 32]


def test_few_samples_warning_is_raised_once_per_call():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimize(TinyMLP(), EXAMPLE, PruneConfig(amount=0.5), QuantConfig(backend="legacy"), **FAST)
    assert sum(issubclass(item.category, FewSamplesWarning) for item in caught) == 1


def test_json_is_valid_with_numpy_and_nan_scores(tmp_path):
    big = torch.randn(32, 1, 28, 28)
    result = optimize(
        TinyMLP(), EXAMPLE, PruneConfig(amount=0.25), eval_fn=lambda _: np.float32(0.5),
        fidelity_inputs=big, **FAST,
    )  # fmt: skip
    assert type(result.profiles[0].accuracy) is float
    nan_result = optimize(
        TinyMLP(), EXAMPLE, PruneConfig(amount=0.25), eval_fn=lambda _: float("nan"),
        fidelity_inputs=big, **FAST,
    )  # fmt: skip

    def reject(constant):
        raise AssertionError(f"bare {constant} is not JSON")

    text = nan_result.to_json(tmp_path / "nan.json").read_text()
    assert json.loads(text, parse_constant=reject)["stages"][0]["accuracy"] is None
    json.loads(result.to_json(tmp_path / "numpy.json").read_text(), parse_constant=reject)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_mps_model_and_input_are_moved_to_the_cpu():
    model = TinyMLP().to("mps")
    result = optimize(model, EXAMPLE.to("mps"), PruneConfig(amount=0.5), **FAST)
    assert next(result.model.parameters()).device.type == "cpu"
    assert next(model.parameters()).device.type == "mps"


def test_threads_are_restored_even_on_failure():
    before = torch.get_num_threads()
    other = 1 if before > 1 else 2
    optimize(TinyMLP(), EXAMPLE, PruneConfig(amount=0.5), threads=other, **FAST)
    assert torch.get_num_threads() == before
    with pytest.raises(ValueError):
        optimize(
            nn.Sequential(nn.Flatten(), nn.Linear(784, 10)),
            EXAMPLE,
            PruneConfig(),
            threads=other,
            **FAST,
        )
    assert torch.get_num_threads() == before


def test_round_to_and_min_channels_reach_the_pruner(trained_blobnet, blob_batches):
    example = torch.randn(1, 1, 8, 8)
    uniform = optimize(trained_blobnet, example, PruneConfig(amount=0.9, min_channels=6), **FAST)
    assert uniform.model.features[0].out_channels == 6
    searched = optimize(
        trained_blobnet, example, PruneConfig(round_to=4, budget=Budget(target_flops_ratio=0.5)),
        eval_fn=make_accuracy_fn(blob_batches), **FAST,
    )  # fmt: skip
    widths = [m.out_channels for m in searched.model.modules() if isinstance(m, nn.Conv2d)]
    assert searched.search.ratios and all(width % 4 == 0 for width in widths)


def test_onnx_side_file_name_and_tolerance(tmp_path, trained_blobnet, blob_batches):
    pytest.importorskip("onnxruntime")
    from pymodelprune.export import OnnxMismatchError

    arguments = (trained_blobnet, torch.randn(1, 1, 8, 8))
    config = {"quantize": QuantConfig("onnx_static", calibration=blob_batches), **FAST}
    result = optimize(*arguments, onnx_path=tmp_path / "model.int8.onnx", **config)
    assert result.stages[1].onnx_path == tmp_path / "model.int8_fp32.onnx"
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "model.int8.onnx",
        "model.int8_fp32.onnx",
    ]
    with pytest.raises(OnnxMismatchError):
        optimize(*arguments, onnx_path=tmp_path / "strict.onnx", onnx_atol=-1.0, **config)
