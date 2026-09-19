import pytest
import torch
from torch import nn

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from pymodelprune import (
    OrtModule,
    accuracy,
    make_accuracy_fn,
    profile_onnx,
    prune_structured,
    quantize_onnx_static,
    to_onnx,
    verify_onnx,
)
from pymodelprune.demo_models import TinyCNN, TinyMLP
from pymodelprune.export import OnnxMismatchError, onnx_is_quantized
from pymodelprune.fidelity import FewSamplesWarning

EXAMPLE = torch.randn(1, 1, 28, 28)


def test_export_is_verified_and_batch_is_dynamic(tmp_path):
    model = TinyCNN().eval()
    path = to_onnx(model, EXAMPLE, tmp_path / "cnn.onnx")
    assert verify_onnx(model, path, EXAMPLE) < 1e-4
    batch = torch.randn(5, 1, 28, 28)
    assert torch.allclose(OrtModule(path)(batch), model(batch), atol=1e-4)


def test_export_does_not_touch_the_model(tmp_path):
    model = TinyCNN().train()
    to_onnx(model, EXAMPLE, tmp_path / "cnn.onnx")
    assert model.training


def test_mismatch_is_detected(tmp_path):
    path = to_onnx(TinyMLP().eval(), EXAMPLE, tmp_path / "mlp.onnx")
    with pytest.raises(OnnxMismatchError):
        verify_onnx(TinyMLP().eval(), path, EXAMPLE)


def test_structurally_pruned_model_exports(tmp_path):
    pruned = prune_structured(TinyCNN().eval(), EXAMPLE, 0.5)
    to_onnx(pruned, EXAMPLE, tmp_path / "pruned.onnx")


def test_profile_onnx(tmp_path):
    model = TinyCNN().eval()
    path = to_onnx(model, EXAMPLE, tmp_path / "cnn.onnx")
    profile = profile_onnx(path, EXAMPLE, source_model=model, reference=model, warmup=1, runs=3)
    assert profile.dtype == "fp32" and profile.total_params == 50378 and profile.flops
    assert profile.fidelity.agreement == 1.0
    assert profile.size_mb == pytest.approx(path.stat().st_size / 1e6)


def test_static_int8(tmp_path, trained_blobnet, blob_batches):
    example = torch.randn(1, 1, 8, 8)
    float_path = to_onnx(trained_blobnet, example, tmp_path / "blob.onnx")
    int8_path = quantize_onnx_static(float_path, tmp_path / "blob.int8.onnx", blob_batches)
    assert onnx_is_quantized(int8_path) and not onnx_is_quantized(float_path)
    baseline = accuracy(trained_blobnet, blob_batches)
    assert accuracy(OrtModule(int8_path), blob_batches) > baseline - 0.05
    profile = profile_onnx(
        int8_path, example, eval_fn=make_accuracy_fn(blob_batches), warmup=1, runs=3
    )
    assert profile.dtype == "int8" and profile.flops is None and profile.accuracy > 0.9


def test_static_int8_accepts_a_single_tensor(tmp_path, trained_blobnet, blob_batches):
    float_path = to_onnx(trained_blobnet, torch.randn(1, 1, 8, 8), tmp_path / "blob.onnx")
    out = quantize_onnx_static(
        float_path, tmp_path / "q.onnx", blob_batches[0][0], exclude_nodes=()
    )
    assert out.exists()


class Branching(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 4)

    def forward(self, x):
        return self.layer(x) if float(x.sum()) > 0 else self.layer(x) * 3


def test_frozen_control_flow_never_passes_silently(tmp_path):
    path = tmp_path / "branching.onnx"
    with (
        pytest.warns(UserWarning, match="falling back to the tracing exporter"),
        pytest.raises(OnnxMismatchError),
    ):
        to_onnx(Branching().eval(), torch.ones(1, 8), path)
    assert not path.exists(), "a file that failed verification must not be left behind"


def test_nan_outputs_fail_verification(tmp_path):
    model = nn.Linear(8, 4).eval()
    path = to_onnx(model, torch.ones(1, 8), tmp_path / "linear.onnx")
    with pytest.raises(OnnxMismatchError):
        verify_onnx(model, path, torch.full((1, 8), float("nan")))


def test_tolerance_scales_with_the_outputs(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).eval()
    with torch.no_grad():
        model[2].weight.mul_(1e6)
    example = torch.randn(4, 64)
    path = to_onnx(model, example, tmp_path / "big.onnx")
    assert verify_onnx(model, path, example) > 1e-4, "float noise on 1e6 logits exceeds atol alone"
    with pytest.raises(OnnxMismatchError):
        verify_onnx(model, path, example, rtol=0.0)


def test_tuple_outputs_are_rejected_clearly(tmp_path):
    class TupleOut(nn.Module):
        def forward(self, x):
            return x + 1, x

    with pytest.raises(TypeError, match="only single-tensor outputs"):
        to_onnx(TupleOut(), torch.ones(1, 8), tmp_path / "tuple.onnx")
    assert not (tmp_path / "tuple.onnx").exists()


class TwoInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.left, self.right = nn.Linear(8, 16), nn.Linear(4, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, a, b):
        return self.head(torch.relu(self.left(a) + self.right(b)))


def test_two_input_model_exports_calibrates_and_profiles(tmp_path):
    torch.manual_seed(0)
    model, example = TwoInputs().eval(), (torch.randn(1, 8), torch.randn(1, 4))
    batches = [(torch.randn(16, 8), torch.randn(16, 4), torch.zeros(16)) for _ in range(3)]
    float_path = to_onnx(model, example, tmp_path / "two.onnx")
    int8_path = quantize_onnx_static(float_path, tmp_path / "two.int8.onnx", batches)
    profile = profile_onnx(
        int8_path, example, reference=model, fidelity_inputs=batches, warmup=1, runs=3
    )
    assert profile.dtype == "int8" and profile.fidelity.samples == 48
    with pytest.warns(FewSamplesWarning):
        assert (
            profile_onnx(float_path, example, reference=model, warmup=1, runs=3).fidelity.samples
            == 1
        )


def test_one_shot_calibration_iterator(tmp_path, trained_blobnet, blob_batches):
    float_path = to_onnx(trained_blobnet, torch.randn(1, 1, 8, 8), tmp_path / "blob.onnx")
    out = quantize_onnx_static(float_path, tmp_path / "q.onnx", (batch for batch in blob_batches))
    assert onnx_is_quantized(out)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_mps_model_exports_and_verifies(tmp_path):
    model, example = TinyCNN().eval().to("mps"), EXAMPLE.to("mps")
    path = to_onnx(model, example, tmp_path / "cnn.onnx")
    assert verify_onnx(model, path, example) < 1e-3
    assert next(model.parameters()).device.type == "mps"


def test_session_can_be_closed_and_refuses_use_afterwards(tmp_path):
    path = to_onnx(TinyCNN().eval(), EXAMPLE, tmp_path / "cnn.onnx")
    with OrtModule(path) as session:
        assert session(EXAMPLE).shape == (1, 10)
    assert session.session is None
    with pytest.raises(RuntimeError, match="closed"):
        session(EXAMPLE)
    assert path.exists()  # closing releases the session, not the file


def test_optimize_leaves_no_open_onnx_session(tmp_path, trained_blobnet, blob_batches):
    import gc

    import onnxruntime

    from pymodelprune import PruneConfig, QuantConfig, optimize

    optimize(
        trained_blobnet, torch.randn(1, 1, 8, 8), PruneConfig(amount=0.25),
        QuantConfig("onnx_static", calibration=blob_batches), fidelity_inputs=blob_batches,
        onnx_path=tmp_path / "blob.onnx", warmup=1, runs=3,
    )  # fmt: skip
    gc.collect()
    live = [o for o in gc.get_objects() if isinstance(o, onnxruntime.InferenceSession)]
    assert not live, f"{len(live)} ONNX Runtime sessions still open after optimize"
