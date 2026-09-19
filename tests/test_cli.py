import contextlib
import re

import pytest
import torch
from typer.testing import CliRunner

from pymodelprune.cli import app
from pymodelprune.demo_models import DEMO_MODELS, TinyCNN

runner = CliRunner()

# Rich formats usage errors, and the formatting lands in the middle of the words these
# tests look for: it wraps the message in a box at the terminal width, and it highlights
# option names, which puts colour codes inside "--data". CI forces colour on, so these
# assertions were green locally and red there. Strip both before matching.
_BOX_DRAWING = str.maketrans("", "", "\u2502\u256d\u256e\u2570\u256f\u2500")
_ANSI_CODES = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")


def test_demo_runs():
    result = runner.invoke(app, ["demo", "--model", "cnn", "--runs", "5"])
    assert result.exit_code == 0
    assert "cnn" in result.output


def test_demo_rejects_unknown_model():
    result = runner.invoke(app, ["demo", "--model", "nope"])
    assert result.exit_code != 0


def test_profile_saved_model(tmp_path):
    path = tmp_path / "cnn.pt"
    torch.save(TinyCNN(), path)
    result = runner.invoke(app, ["profile", str(path), "--input-shape", "1,1,28,28", "--runs", "5"])
    assert result.exit_code == 0
    assert "cnn.pt" in result.output


def test_profile_rejects_state_dict(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save(TinyCNN().state_dict(), path)
    result = runner.invoke(app, ["profile", str(path), "--input-shape", "1,1,28,28"])
    assert result.exit_code == 1


HOOKS = """
import torch
from pymodelprune.demo_models import DEMO_MODELS, TinyCNN

def build():
    return TinyCNN()

def batches():
    torch.manual_seed(0)
    return [torch.randn(16, 1, 28, 28) for _ in range(2)]

def one_shot():
    return (batch for batch in batches())

def score(model):
    return 0.5
"""


def test_optimize_writes_model_and_report(tmp_path):
    path = tmp_path / "mlp.pt"
    torch.save(DEMO_MODELS["mlp"](), path)
    args = [
        "optimize",
        str(path),
        "--input-shape",
        "1,1,28,28",
        "--prune",
        "0.5",
        "--quantize",
        "dynamic",
    ]
    args += [
        "--backend",
        "legacy",
        "--out",
        str(tmp_path / "opt.pt"),
        "--json",
        str(tmp_path / "r.json"),
        "--runs",
        "3",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "pruned" in result.output and "int8" in result.output
    assert (tmp_path / "opt.pt").exists() and (tmp_path / "r.json").exists()


def test_optimize_with_factory_and_hooks(tmp_path, monkeypatch):
    (tmp_path / "my_hooks.py").write_text(HOOKS)
    torch.save(TinyCNN().state_dict(), tmp_path / "w.pt")
    monkeypatch.chdir(tmp_path)
    args = [
        "optimize",
        "--factory",
        "my_hooks:build",
        "--weights",
        "w.pt",
        "--input-shape",
        "1,1,28,28",
    ]
    args += [
        "--target-flops",
        "0.6",
        "--eval",
        "my_hooks:score",
        "--data",
        "my_hooks:batches",
        "--runs",
        "3",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "search:" in result.output and "50.0%" in result.output


def test_optimize_error_paths(tmp_path):
    path = tmp_path / "mlp.pt"
    torch.save(DEMO_MODELS["mlp"](), path)
    base = ["optimize", str(path), "--input-shape", "1,1,28,28"]
    assert runner.invoke(app, base).exit_code != 0  # nothing to do
    assert runner.invoke(app, [*base, "--quantize", "onnx_static"]).exit_code != 0  # no --out
    assert runner.invoke(app, [*base, "--prune", "0.5", "--eval", "nope:missing"]).exit_code != 0
    assert (
        runner.invoke(
            app, ["optimize", str(path), "--input-shape", "a,b", "--prune", "0.5"]
        ).exit_code
        != 0
    )
    assert (
        runner.invoke(app, ["optimize", "--input-shape", "1,1,28,28", "--prune", "0.5"]).exit_code
        != 0
    )


def test_sensitivity_command(tmp_path, monkeypatch):
    (tmp_path / "my_hooks.py").write_text(HOOKS)
    torch.save(TinyCNN(), tmp_path / "cnn.pt")
    monkeypatch.chdir(tmp_path)
    args = ["sensitivity", "cnn.pt", "--input-shape", "1,1,28,28", "--data", "my_hooks:one_shot"]
    result = runner.invoke(app, [*args, "--json", "new_dir/s.json"])
    assert result.exit_code == 0, result.output
    assert "features.0" in result.output and (tmp_path / "new_dir" / "s.json").exists()


def cli_text(result) -> str:
    """Everything the command printed, as one line without Rich's box drawing."""
    streams = [result.output]
    with contextlib.suppress(ValueError):  # click only separates them when asked to
        streams.append(result.stderr)
    plain = _ANSI_CODES.sub("", " ".join(streams)).translate(_BOX_DRAWING)
    return " ".join(plain.split())


def fails_cleanly(args: list[str], expected: str) -> bool:
    """A user error is one readable message and a non-zero exit code, never a traceback."""
    result = runner.invoke(app, args)
    text = cli_text(result)
    assert result.exit_code != 0, f"expected a failure for {args}, got:\n{text}"
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"expected a clean exit for {args}, got {result.exception!r}"
    )
    assert "Traceback" not in text, f"traceback leaked for {args}:\n{text}"
    assert expected in text, f"expected {expected!r} in the output of {args}, got:\n{text}"
    return True


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (["--prune", "0.5", "--method", "foo"], "structured"),
        (["--quantize", "foo"], "error: mode must be"),
        (["--max-acc-drop", "-1"], "error: max_drop"),
        (["--target-flops", "1.5"], "error: target_flops_ratio"),
        (["--method", "unstructured", "--target-flops", "0.5"], "error: a budget needs"),
        (["--target-flops", "0.5"], "error: the search needs a score"),
        (["--prune", "0.5", "--max-acc-drop", "1"], "budget search decides the amount"),
        (["--prune", "0.5", "--importance", "foo"], "choose from l1 | l2 | bn | random"),
        (["--prune", "0.5", "--importance", "taylor"], "Python API"),
        (["--prune", "0.5", "--method", "unstructured", "--importance", "l2"], "l1 | random"),
        (["--quantize", "onnx_static", "--out", "q.onnx"], "needs --data for calibration"),
    ],
)
def test_optimize_user_errors_are_clean(tmp_path, extra, expected):
    torch.save(TinyCNN(), tmp_path / "cnn.pt")
    base = ["optimize", str(tmp_path / "cnn.pt"), "--input-shape", "1,1,28,28", "--runs", "3"]
    assert fails_cleanly([*base, *extra], expected)


def test_bad_files_and_shapes_are_clean_errors(tmp_path, monkeypatch):
    (tmp_path / "my_hooks.py").write_text(HOOKS)
    monkeypatch.chdir(tmp_path)
    torch.save(TinyCNN(), tmp_path / "cnn.pt")
    torch.save(TinyCNN().state_dict(), tmp_path / "weights.pt")
    (tmp_path / "garbage.pt").write_bytes(b"not a model")
    cnn, garbage, weights = (
        str(tmp_path / name) for name in ("cnn.pt", "garbage.pt", "weights.pt")
    )
    shape = ["--input-shape", "1,1,28,28"]
    for command in (["profile"], ["optimize", "--prune", "0.5"]):
        assert fails_cleanly([*command, cnn, "--input-shape", "1,3,28,28", "--runs", "3"], "error:")
        assert fails_cleanly([*command, garbage, *shape], "error: cannot load garbage.pt")
    wrong_shape = ["sensitivity", cnn, "--input-shape", "1,3,28,28", "--eval", "my_hooks:score"]
    assert fails_cleanly(wrong_shape, "error:")
    assert fails_cleanly(["sensitivity", cnn, *shape], "needs --eval or --data")
    assert fails_cleanly(
        ["profile", cnn, "--weights", weights, *shape], "--weights needs --factory"
    )


def test_saved_legacy_int8_model_can_be_profiled_again(tmp_path):
    torch.save(DEMO_MODELS["mlp"](), tmp_path / "mlp.pt")
    shape = ["--input-shape", "1,1,28,28", "--runs", "3"]
    args = [
        "optimize",
        str(tmp_path / "mlp.pt"),
        *shape,
        "--quantize",
        "dynamic",
        "--backend",
        "legacy",
    ]
    assert runner.invoke(app, [*args, "--out", str(tmp_path / "q.pt")]).exit_code == 0
    result = runner.invoke(app, ["profile", str(tmp_path / "q.pt"), *shape])
    assert result.exit_code == 0 and "int8" in result.output
    assert fails_cleanly(
        ["optimize", str(tmp_path / "q.pt"), *shape, "--quantize", "dynamic"], "already quantized"
    )
