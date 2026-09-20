import itertools

import numpy as np
import pytest
import torch
from torch import nn

import pymodelprune.profile as profile_module
from pymodelprune import count_parameters, measure_latency, model_size_mb, profile_model


def test_count_parameters_linear():
    # 10*5 weights + 5 biases
    total, nonzero = count_parameters(nn.Linear(10, 5))
    assert total == 55
    assert nonzero <= total


def test_count_parameters_sees_zeros():
    layer = nn.Linear(10, 5)
    with torch.no_grad():
        layer.weight.zero_()
    total, nonzero = count_parameters(layer)
    assert total == 55
    assert nonzero <= 5  # only the biases can be nonzero


def test_model_size_matches_float32_storage():
    layer = nn.Linear(1000, 1000, bias=False)
    expected_mb = 1000 * 1000 * 4 / 1e6  # float32 = 4 bytes
    assert model_size_mb(layer) == pytest.approx(expected_mb, rel=0.01)


def test_measure_latency_returns_sane_stats():
    stats = measure_latency(nn.Linear(8, 8), torch.randn(1, 8), warmup=2, runs=20)
    assert stats.runs == 20
    assert 0 < stats.median_ms <= stats.p95_ms


def test_measure_latency_restores_training_mode():
    model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    model.train()
    measure_latency(model, torch.randn(1, 8), warmup=1, runs=2)
    assert model.training


def test_measure_latency_rejects_single_run():
    with pytest.raises(ValueError):
        measure_latency(nn.Linear(8, 8), torch.randn(1, 8), runs=1)


def test_profile_model_sparsity():
    layer = nn.Linear(10, 10, bias=False)
    with torch.no_grad():
        layer.weight[:5].zero_()
    result = profile_model(layer, torch.randn(1, 10), warmup=1, runs=5)
    assert result.sparsity == pytest.approx(0.5)


@pytest.mark.parametrize(("device", "attribute"), [("cuda", "cuda"), ("mps", "mps"), ("cpu", None)])
def test_latency_waits_for_the_accelerator(monkeypatch, device, attribute):
    """Accelerators queue work, so a timer that does not wait measures the queueing.

    Checked by counting the waits rather than by comparing durations: the mechanism is
    what matters, and this project tells its users not to trust a latency number as proof.
    """
    waits = []
    if attribute is not None:
        monkeypatch.setattr(
            getattr(torch, attribute), "synchronize", lambda *a, **k: waits.append(device)
        )
    example = torch.zeros(1, 4)
    monkeypatch.setattr(type(example), "device", property(lambda self: torch.device(device)))
    profile_module.measure_latency_interleaved([nn.Linear(4, 4)], example, warmup=1, runs=5)
    # one wait before the timer starts and one after the pass, for each timed run
    assert len(waits) == (10 if attribute else 0)


def test_profile_serializes_the_weights_once(monkeypatch):
    calls = []
    original = profile_module._serialized_state
    monkeypatch.setattr(
        profile_module, "_serialized_state", lambda model: calls.append(1) or original(model)
    )
    layer = nn.Linear(100, 100)
    result = profile_model(layer, torch.randn(1, 100), warmup=1, runs=3)
    assert len(calls) == 1
    assert result.size_mb == pytest.approx(model_size_mb(layer))
    assert 0 < result.compressed_mb == pytest.approx(profile_module.compressed_size_mb(layer))


def test_scores_are_stored_as_plain_floats():
    for score in (np.float32(0.5), torch.tensor(0.5)):
        result = profile_model(
            nn.Linear(4, 4), torch.randn(1, 4), warmup=1, runs=3, eval_fn=lambda _, s=score: s
        )
        assert type(result.accuracy) is float and result.accuracy == 0.5


class Recorder(nn.Module):
    """A model that writes its own name into a shared log on every forward pass."""

    def __init__(self, name: str, log: list[str], width: int = 64):
        super().__init__()
        self.name, self.log = name, log
        self.layer = nn.Linear(width, width)

    def forward(self, x):
        self.log.append(self.name)
        return self.layer(x)


def call_blocks(log: list[str]) -> list[int]:
    """Lengths of the runs of consecutive calls to the same model."""
    blocks = []
    for name in log:
        if blocks and log[sum(blocks) - 1] == name:
            blocks[-1] += 1
        else:
            blocks.append(1)
    return blocks


def test_interleaved_timing_alternates_in_blocks():
    log: list[str] = []
    models = [Recorder("a", log), Recorder("b", log)]
    example = torch.randn(1, 64)
    profile_module.measure_latency_interleaved(models, example, warmup=0, runs=10, block=5)
    # 5 passes of a, 5 of b, twice: the cycling is what spreads machine load drift, the
    # blocks are what keeps each model's weights in cache while it is being timed.
    assert call_blocks(log) == [5, 5, 5, 5]
    assert log.count("a") == log.count("b") == 10


def test_interleaved_timing_handles_a_ragged_last_cycle():
    log: list[str] = []
    models = [Recorder("a", log), Recorder("b", log)]
    stats = profile_module.measure_latency_interleaved(
        models, torch.randn(1, 64), warmup=0, runs=7, block=5
    )
    assert call_blocks(log) == [5, 5, 2, 2]
    assert [item.runs for item in stats] == [7, 7]


def test_interleaved_warmup_runs_before_any_timing():
    log: list[str] = []
    models = [Recorder("a", log), Recorder("b", log)]
    profile_module.measure_latency_interleaved(
        models, torch.randn(1, 64), warmup=3, runs=4, block=2
    )
    assert log[:6] == ["a", "a", "a", "b", "b", "b"]  # every model is warm before the first timing


def test_every_model_is_measured_the_same_way():
    """The candidates must get identical treatment: same passes, same cycles, same turns.

    An earlier version of this test compared the measured medians of identical models
    instead. It was flaky on shared CI machines (1.7x apart on one run), and asserting on
    latency is exactly what this project tells its users not to do.
    """
    log: list[str] = []
    models = [Recorder(name, log) for name in "abc"]
    stats = profile_module.measure_latency_interleaved(
        models, torch.randn(1, 64), warmup=2, runs=12, block=4
    )
    assert [item.runs for item in stats] == [12, 12, 12]
    assert call_blocks(log[6:]) == [4] * 9  # 3 cycles x 3 models, after the warm-up
    turns = [name for name, _ in itertools.groupby(log[6:])]
    assert turns == list("abc") * 3


def test_interleaved_rejects_bad_arguments():
    for kwargs in ({"runs": 1}, {"block": 0}):
        with pytest.raises(ValueError):
            profile_module.measure_latency_interleaved(
                [nn.Linear(4, 4)], torch.randn(1, 4), **kwargs
            )


def test_retime_profiles_replaces_only_the_latency():
    example = torch.randn(1, 64)
    models = [Recorder("a", []), Recorder("b", [])]
    profiles = [profile_model(m, example, name=m.name, warmup=1, runs=3) for m in models]
    retimed = profile_module.retime_profiles(profiles, models, example, warmup=1, runs=8)
    assert [item.name for item in retimed] == ["a", "b"]
    assert all(item.latency.runs == 8 for item in retimed)
    assert [item.total_params for item in retimed] == [item.total_params for item in profiles]
    with pytest.raises(ValueError):
        profile_module.retime_profiles(profiles, models[:1], example)
