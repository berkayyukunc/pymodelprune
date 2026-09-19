import copy
import pickle

import pytest
import torch
from torch import nn

from pymodelprune import (
    compare_outputs,
    count_parameters,
    model_size_mb,
    profile_model,
    save_model,
)
from pymodelprune.demo_models import TinyCNN, TinyMLP
from pymodelprune.profile import weights_dtype
from pymodelprune.prune import prune_unstructured
from pymodelprune.quantize import quantize_dynamic
from pymodelprune.quantize.dynamic import torchao_available

needs_torchao = pytest.mark.skipif(not torchao_available(), reason="torchao not installed")
BACKENDS = ["legacy", pytest.param("torchao", marks=needs_torchao)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_mlp_shrinks_about_four_times(backend):
    model = TinyMLP()
    quantized = quantize_dynamic(model, backend=backend)
    assert model_size_mb(model) / model_size_mb(quantized) > 3.5
    assert weights_dtype(quantized) == "int8"
    assert weights_dtype(model) == "fp32"


@pytest.mark.parametrize("backend", BACKENDS)
def test_outputs_stay_close(backend):
    torch.manual_seed(0)
    model = TinyMLP()
    result = compare_outputs(
        model, quantize_dynamic(model, backend=backend), torch.randn(64, 1, 28, 28)
    )
    assert result.agreement > 0.9
    assert result.max_abs_diff < 0.05


@pytest.mark.parametrize("backend", BACKENDS)
def test_parameter_count_survives_quantization(backend):
    model = TinyMLP()
    total, _ = count_parameters(quantize_dynamic(model, backend=backend))
    assert total == count_parameters(model)[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sparsity_survives_quantization(backend):
    pruned = prune_unstructured(TinyMLP(), 0.6)
    total, nonzero = count_parameters(quantize_dynamic(pruned, backend=backend))
    assert 1 - nonzero / total == pytest.approx(0.6, abs=0.02)


def test_input_model_is_untouched():
    model = TinyMLP()
    before = copy.deepcopy(model.state_dict())
    quantize_dynamic(model, backend="legacy")
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert weights_dtype(model) == "fp32"


@pytest.mark.parametrize("backend", BACKENDS)
def test_quantized_model_survives_save_and_load(backend, tmp_path):
    quantized = quantize_dynamic(TinyMLP(), backend=backend)
    restored = torch.load(save_model(quantized, tmp_path / "q.pt"), weights_only=False)
    example = torch.randn(4, 1, 28, 28)
    assert torch.equal(restored(example), quantized(example))
    assert [item.name for item in tmp_path.iterdir()] == ["q.pt"]


def test_failed_save_keeps_the_old_file_and_leaves_no_debris(tmp_path):
    model = nn.Linear(2, 2)
    model.callback = lambda: None  # lambdas cannot be pickled
    (tmp_path / "m.pt").write_bytes(b"previous good file")
    with pytest.raises((AttributeError, pickle.PicklingError)):
        save_model(model, tmp_path / "m.pt")
    assert (tmp_path / "m.pt").read_bytes() == b"previous good file"
    assert [item.name for item in tmp_path.iterdir()] == ["m.pt"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_quantizing_twice_is_a_clear_error(backend):
    quantized = quantize_dynamic(TinyMLP(), backend=backend)
    with pytest.raises(ValueError, match="already quantized"):
        quantize_dynamic(quantized, backend=backend)


def test_convolutions_are_left_alone():
    quantized = quantize_dynamic(TinyCNN(), backend="legacy")
    convs = [m for m in quantized.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 2 and all(c.weight.dtype == torch.float32 for c in convs)


def test_profile_reports_int8_and_skips_flops():
    profile = profile_model(
        quantize_dynamic(TinyMLP(), backend="legacy"), torch.randn(1, 1, 28, 28), warmup=1, runs=3
    )
    assert profile.dtype == "int8" and profile.flops is None


@needs_torchao
def test_weight_only_mode():
    model = TinyMLP()
    quantized = quantize_dynamic(model, mode="weight_only", backend="torchao")
    assert model_size_mb(model) / model_size_mb(quantized) > 3.5


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        quantize_dynamic(TinyMLP(), mode="static")
    with pytest.raises(ValueError):
        quantize_dynamic(TinyMLP(), backend="tensorrt")
    with pytest.raises(ValueError):
        quantize_dynamic(TinyMLP(), mode="weight_only", backend="legacy")
    with pytest.raises(ValueError):
        quantize_dynamic(nn.Sequential(nn.Conv2d(1, 1, 3)))
