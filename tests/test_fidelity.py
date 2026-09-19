import copy
import math
import warnings

import pytest
import torch
from rich.console import Console
from torch import nn

from pymodelprune import (
    PruneConfig,
    QuantConfig,
    compare_outputs,
    optimize,
    profile_model,
    render_profiles,
)
from pymodelprune.demo_models import TinyCNN, TinyMLP
from pymodelprune.fidelity import FewSamplesWarning


def test_model_agrees_with_its_copy():
    model = TinyMLP()
    result = compare_outputs(model, copy.deepcopy(model), torch.randn(32, 1, 28, 28))
    assert result.agreement == 1.0
    assert result.max_abs_diff == 0.0
    assert result.samples == 32


def test_unrelated_models_disagree():
    torch.manual_seed(0)
    result = compare_outputs(TinyMLP(), TinyMLP(), torch.randn(256, 1, 28, 28))
    assert result.agreement < 0.9
    assert result.max_abs_diff > 0


def test_accepts_labelled_batches():
    model = TinyMLP()
    batches = [(torch.randn(8, 1, 28, 28), torch.zeros(8)) for _ in range(3)]
    assert compare_outputs(model, model, batches).samples == 24


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare_outputs(nn.Linear(4, 3), nn.Linear(4, 5), torch.randn(2, 4))


def test_rejects_non_tensor_output():
    class TupleOut(nn.Module):
        def forward(self, x):
            return x, x

    with pytest.raises(TypeError):
        compare_outputs(TupleOut(), TupleOut(), torch.randn(2, 4))


def test_rejects_empty_inputs():
    with pytest.raises(ValueError):
        compare_outputs(nn.Linear(4, 3), nn.Linear(4, 3), [])


def test_profile_carries_accuracy_and_fidelity():
    model = TinyMLP()
    example = torch.randn(4, 1, 28, 28)
    result = profile_model(
        model, example, warmup=1, runs=3, eval_fn=lambda _: 0.5, reference=copy.deepcopy(model)
    )
    assert result.accuracy == 0.5
    assert result.fidelity is not None and result.fidelity.agreement == 1.0


def test_table_shows_optional_columns_only_when_present():
    example = torch.randn(4, 1, 28, 28)
    plain = profile_model(TinyMLP(), example, warmup=1, runs=3)
    scored = profile_model(
        TinyMLP(), example, name="scored", warmup=1, runs=3, eval_fn=lambda _: 0.5
    )

    console = Console(record=True, width=200)
    render_profiles([plain], console=console)
    assert "Acc" not in console.export_text()

    render_profiles([plain, scored], console=console)
    text = console.export_text()
    assert "Acc" in text and "50.0%" in text


def test_nan_outputs_are_not_reported_as_identical():
    broken = nn.Linear(4, 3)
    with torch.no_grad():
        broken.weight.fill_(float("nan"))
    batches = [torch.randn(16, 4), torch.randn(16, 4)]
    assert math.isnan(compare_outputs(nn.Linear(4, 3), broken, batches).max_abs_diff)
    nan_first = [torch.full((16, 4), float("nan")), torch.randn(16, 4)]
    assert math.isnan(compare_outputs(nn.Linear(4, 3), nn.Linear(4, 3), nan_first).max_abs_diff)


class TwoInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.left, self.right = nn.Linear(8, 16), nn.Linear(4, 16)

    def forward(self, a, b):
        return self.left(a) + self.right(b)


def test_multi_input_batches():
    model = TwoInputs()
    pair = (torch.randn(32, 8), torch.randn(32, 4))
    assert compare_outputs(model, model, [pair], num_inputs=2).samples == 32
    assert compare_outputs(model, model, [[*pair, torch.zeros(32)]], num_inputs=2).samples == 32
    with pytest.raises(ValueError):
        compare_outputs(model, model, pair[0], num_inputs=2)
    example = (torch.randn(1, 8), torch.randn(1, 4))
    with pytest.warns(FewSamplesWarning):
        assert (
            profile_model(model, example, warmup=1, runs=3, reference=model).fidelity.samples == 1
        )


def test_few_samples_warn():
    model = TinyMLP()
    with pytest.warns(FewSamplesWarning, match="fidelity_inputs"):
        compare_outputs(model, model, torch.randn(31, 1, 28, 28))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare_outputs(model, model, torch.randn(32, 1, 28, 28))


def test_full_structured_int8_table_fits_80_columns():
    torch.manual_seed(0)
    result = optimize(
        TinyCNN(), torch.randn(1, 1, 28, 28), PruneConfig(amount=0.5), QuantConfig(backend="legacy"),
        eval_fn=lambda _: 0.987, fidelity_inputs=torch.randn(32, 1, 28, 28), warmup=1, runs=3,
    )  # fmt: skip
    console = Console(record=True, width=80)
    result.report(console)
    text = console.export_text()
    assert all(header in text for header in ("Type", "FLOPs", "Acc", "Fid", "Size", "Speed"))
    assert "Sparse" not in text and "gz MB" not in text
    assert "…" not in text and all(len(line) <= 80 for line in text.splitlines())


def test_sparsity_columns_appear_for_sparse_models():
    result = optimize(
        TinyCNN(), torch.randn(1, 1, 28, 28), PruneConfig("unstructured", 0.5), warmup=1, runs=3,
        fidelity_inputs=torch.randn(32, 1, 28, 28),
    )  # fmt: skip
    console = Console(record=True, width=200)
    result.report(console)
    text = console.export_text()  # export_text() clears the record: read it once
    assert "Sparse" in text and "gz MB" in text
