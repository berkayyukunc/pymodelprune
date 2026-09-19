import copy

import pytest
import torch
from torch import nn

from pymodelprune import count_parameters, model_size_mb, profile_model
from pymodelprune.demo_models import TinyCNN, TinyMLP
from pymodelprune.profile import compressed_size_mb
from pymodelprune.prune import make_permanent, prune_unstructured


def weight_sparsity(model: nn.Module) -> float:
    weights = [m.weight for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    zeros = sum(int((w == 0).sum()) for w in weights)
    return zeros / sum(w.numel() for w in weights)


@pytest.mark.parametrize("scope", ["global", "layer"])
@pytest.mark.parametrize("amount", [0.3, 0.7])
def test_amount_is_respected(scope, amount):
    pruned = prune_unstructured(TinyCNN(), amount, scope=scope)
    assert weight_sparsity(pruned) == pytest.approx(amount, abs=0.01)


def test_global_scope_prunes_layers_unevenly():
    pruned = prune_unstructured(TinyMLP(), 0.5, scope="global")
    per_layer = [
        float((m.weight == 0).float().mean()) for m in pruned.modules() if isinstance(m, nn.Linear)
    ]
    assert max(per_layer) - min(per_layer) > 0.01


def test_layer_scope_prunes_layers_evenly():
    pruned = prune_unstructured(TinyMLP(), 0.5, scope="layer")
    for module in pruned.modules():
        if isinstance(module, nn.Linear):
            assert float((module.weight == 0).float().mean()) == pytest.approx(0.5, abs=0.01)


def test_l1_removes_the_smallest_weights():
    layer = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.1, -5.0, 0.2, 3.0]]))
    pruned = prune_unstructured(layer, 0.5)
    assert pruned.weight.tolist() == [[0.0, -5.0, 0.0, 3.0]]


def test_input_model_is_untouched():
    model = TinyMLP()
    before = copy.deepcopy(model.state_dict())
    prune_unstructured(model, 0.9)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_permanent_result_has_no_mask_tensors():
    pruned = prune_unstructured(TinyMLP(), 0.5)
    assert not [k for k in pruned.state_dict() if k.endswith(("_orig", "_mask"))]


def test_masked_model_reports_true_sparsity():
    # regression: count_parameters used to read weight_orig and report 0% sparsity
    masked = prune_unstructured(TinyMLP(), 0.5, permanent=False)
    assert [k for k in masked.state_dict() if k.endswith("_mask")]
    total, nonzero = count_parameters(masked)
    assert total == count_parameters(TinyMLP())[0]
    assert 1 - nonzero / total == pytest.approx(0.5, abs=0.01)
    assert count_parameters(make_permanent(masked)) == (total, nonzero)


def test_pruning_shrinks_only_the_compressed_size():
    model = TinyMLP()
    pruned = prune_unstructured(model, 0.9)
    assert model_size_mb(pruned) == pytest.approx(model_size_mb(model), rel=0.01)
    assert compressed_size_mb(pruned) < 0.5 * compressed_size_mb(model)


def test_masks_double_the_file():
    model = TinyMLP()
    masked = prune_unstructured(model, 0.5, permanent=False)
    assert model_size_mb(masked) > 1.8 * model_size_mb(model)


def test_output_shape_and_flops_unchanged():
    example = torch.randn(2, 1, 28, 28)
    base = profile_model(TinyCNN(), example, warmup=1, runs=3)
    pruned = profile_model(prune_unstructured(TinyCNN(), 0.8), example, warmup=1, runs=3)
    assert pruned.flops == base.flops
    assert pruned.sparsity > 0.7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amount": 1.0},
        {"amount": -0.1},
        {"amount": 0.5, "criterion": "l2"},
        {"amount": 0.5, "scope": "x"},
    ],
)
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        prune_unstructured(TinyMLP(), **kwargs)


def test_rejects_model_without_prunable_layers():
    with pytest.raises(ValueError):
        prune_unstructured(nn.Sequential(nn.ReLU()), 0.5)
