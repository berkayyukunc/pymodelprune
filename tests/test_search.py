import copy

import pytest
import torch
from torch import nn

from pymodelprune import Budget, accuracy, analyze_sensitivity, make_accuracy_fn, search_budget
from pymodelprune.prune import build_dependency_graph
from pymodelprune.search import SensitivityReport

EXAMPLE = torch.randn(1, 1, 8, 8)


@pytest.fixture(scope="module")
def report(trained_blobnet, blob_batches):
    return analyze_sensitivity(
        trained_blobnet, EXAMPLE, make_accuracy_fn(blob_batches), ratios=(0.25, 0.5, 0.75)
    )


def test_fixture_model_is_actually_trained(trained_blobnet, blob_batches):
    assert accuracy(trained_blobnet, blob_batches) > 0.95


def test_report_covers_every_prunable_group(report, trained_blobnet):
    graph = build_dependency_graph(trained_blobnet, EXAMPLE)
    assert set(report.curves) == {group.id for group in graph.prunable_groups}
    assert report.metric == "eval_fn" and report.baseline_score > 0.95
    for curve in report.curves.values():
        assert [point.ratio for point in curve] == [0.25, 0.5, 0.75]
        flops = [point.flops for point in curve]
        assert flops == sorted(flops, reverse=True) and flops[0] < report.baseline_flops


def test_report_helpers_and_json_roundtrip(report, tmp_path):
    group_id = next(iter(report.curves))
    assert report.drop(group_id, 0.0) == 0.0 and report.saved_flops(group_id, 0.0) == 0
    assert report.saved_flops(group_id, 0.5) > 0
    assert [gid for gid, _ in report.ranking()] and len(report.ranking()) == len(report.curves)
    report.to_json(tmp_path / "s.json")
    assert SensitivityReport.from_json(tmp_path / "s.json") == report


def test_progress_callback_and_fidelity_fallback(trained_blobnet, blob_batches):
    seen = []
    result = analyze_sensitivity(
        trained_blobnet,
        EXAMPLE,
        fidelity_inputs=blob_batches,
        ratios=(0.5,),
        progress=lambda d, t: seen.append((d, t)),
    )
    assert result.metric == "fidelity" and result.baseline_score == 1.0
    assert seen[-1][0] == seen[-1][1] == len(result.curves)


def test_rejects_bad_ratios(trained_blobnet):
    with pytest.raises(ValueError):
        analyze_sensitivity(trained_blobnet, EXAMPLE, ratios=(0.0, 0.5))


def test_budget_respects_the_score_limit(trained_blobnet, blob_batches, report):
    eval_fn = make_accuracy_fn(blob_batches)
    result = search_budget(trained_blobnet, EXAMPLE, Budget(max_drop=0.02), eval_fn, report=report)
    assert result.measured_drop <= 0.02 + 1e-9
    assert result.flops_ratio < 0.9, "a trained toy model should tolerate some pruning"
    assert accuracy(result.model, blob_batches) == pytest.approx(result.score)
    ratios = [step.flops_ratio for step in result.history]
    assert ratios == sorted(ratios, reverse=True)


def test_budget_reaches_a_flops_target(trained_blobnet, blob_batches, report):
    eval_fn = make_accuracy_fn(blob_batches)
    result = search_budget(
        trained_blobnet, EXAMPLE, Budget(target_flops_ratio=0.5), eval_fn, report=report
    )
    assert result.flops_ratio <= 0.5
    assert result.model(EXAMPLE).shape == (1, 4)


def test_zero_tolerance_never_returns_a_worse_model(trained_blobnet, blob_batches, report):
    eval_fn = make_accuracy_fn(blob_batches)
    result = search_budget(trained_blobnet, EXAMPLE, Budget(max_drop=0.0), eval_fn, report=report)
    assert result.measured_drop <= 1e-9


def test_finetune_hook_is_called(trained_blobnet, blob_batches, report):
    calls = []
    search_budget(
        trained_blobnet, EXAMPLE, Budget(target_flops_ratio=0.7), make_accuracy_fn(blob_batches),
        report=report, finetune_fn=lambda model: calls.append(model),
    )  # fmt: skip
    assert len(calls) >= 1


@pytest.mark.parametrize(
    "kwargs", [{}, {"max_drop": -1}, {"target_flops_ratio": 1.5}, {"target_flops_ratio": 0.0}]
)
def test_budget_validation(kwargs):
    with pytest.raises(ValueError):
        Budget(**kwargs)


def test_fidelity_fallback_refuses_a_handful_of_samples(trained_blobnet, blob_batches):
    few = blob_batches[0][0][:31]
    with pytest.raises(ValueError, match="at least 32 samples"):
        analyze_sensitivity(trained_blobnet, EXAMPLE, ratios=(0.5,))
    with pytest.raises(ValueError, match="at least 32 samples"):
        search_budget(trained_blobnet, EXAMPLE, Budget(target_flops_ratio=0.5), fidelity_inputs=few)
    one_shot = (batch for batch in blob_batches)
    result = search_budget(
        trained_blobnet, EXAMPLE, Budget(target_flops_ratio=0.7), fidelity_inputs=one_shot
    )
    assert result.flops_ratio <= 0.7


def test_report_from_another_model_is_rejected(trained_blobnet, blob_batches, report):
    eval_fn = make_accuracy_fn(blob_batches)
    wider = nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(288, 4))
    with pytest.raises(ValueError, match="does not match"):
        search_budget(wider, EXAMPLE, Budget(max_drop=0.1), eval_fn, report=report)
    with pytest.raises(ValueError, match="different FLOPs"):
        search_budget(
            trained_blobnet, torch.randn(2, 1, 8, 8), Budget(max_drop=0.1), eval_fn, report=report
        )


def test_round_to_and_min_channels_shape_the_result(trained_blobnet, blob_batches):
    eval_fn = make_accuracy_fn(blob_batches)
    result = search_budget(
        trained_blobnet,
        EXAMPLE,
        Budget(target_flops_ratio=0.4),
        eval_fn,
        round_to=4,
        min_channels=4,
    )
    widths = [m.out_channels for m in result.model.modules() if isinstance(m, nn.Conv2d)]
    assert result.ratios and all(width % 4 == 0 and width >= 4 for width in widths)


def test_unprunable_model_raises_like_prune_structured(blob_batches):
    single = nn.Sequential(nn.Flatten(), nn.Linear(64, 4))
    with pytest.raises(ValueError, match="no prunable channel groups"):
        search_budget(
            single, EXAMPLE, Budget(target_flops_ratio=0.5), make_accuracy_fn(blob_batches)
        )


def test_empty_search_returns_the_model_untouched(trained_blobnet, blob_batches, report):
    calls = []
    # every pruning step is predicted to cost something, so a worse-than-nothing budget stays empty
    fragile = copy.deepcopy(report)
    fragile.baseline_score = 2.0
    result = search_budget(
        trained_blobnet, EXAMPLE, Budget(max_drop=0.0), make_accuracy_fn(blob_batches),
        report=fragile, finetune_fn=calls.append,
    )  # fmt: skip
    assert result.ratios == {} and not calls and result.flops_ratio == 1.0
    expected = trained_blobnet.state_dict()
    assert all(
        torch.equal(value, expected[key]) for key, value in result.model.state_dict().items()
    )
    assert result.model is not trained_blobnet


def test_two_input_model_with_the_fidelity_fallback():
    class TwoInputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.left, self.right = nn.Linear(8, 32), nn.Linear(4, 32)
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))

        def forward(self, a, b):
            return self.head(self.left(a) + self.right(b))

    example = (torch.randn(1, 8), torch.randn(1, 4))
    batches = [(torch.randn(32, 8), torch.randn(32, 4), torch.zeros(32))]
    result = analyze_sensitivity(TwoInputs(), example, fidelity_inputs=batches, ratios=(0.5,))
    assert result.metric == "fidelity" and result.baseline_score == 1.0 and result.curves
    with pytest.raises(ValueError, match="at least 32 samples"):
        analyze_sensitivity(TwoInputs(), example, ratios=(0.5,))
