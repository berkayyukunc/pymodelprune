import pytest
import torch
from torch import nn

from pymodelprune import accuracy, make_accuracy_fn


def identity_classifier() -> nn.Linear:
    """Predicts the index of the largest input value."""
    layer = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(3))
    return layer


def test_accuracy_known_result():
    inputs = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    labels = torch.tensor([0, 1, 2, 2])  # last one is wrong on purpose
    assert accuracy(identity_classifier(), [(inputs, labels)]) == pytest.approx(0.75)


def test_accuracy_spans_batches():
    batch = (torch.eye(3), torch.tensor([0, 1, 2]))
    wrong = (torch.eye(3), torch.tensor([1, 2, 0]))
    assert accuracy(identity_classifier(), [batch, wrong]) == pytest.approx(0.5)


def test_accuracy_max_batches():
    batch = (torch.eye(3), torch.tensor([0, 1, 2]))
    wrong = (torch.eye(3), torch.tensor([1, 2, 0]))
    assert accuracy(identity_classifier(), [batch, wrong], max_batches=1) == pytest.approx(1.0)


def test_accuracy_rejects_empty_data():
    with pytest.raises(ValueError):
        accuracy(identity_classifier(), [])


def test_accuracy_restores_training_mode():
    model = identity_classifier()
    model.train()
    accuracy(model, [(torch.eye(3), torch.tensor([0, 1, 2]))])
    assert model.training


def test_make_accuracy_fn_is_reusable():
    eval_fn = make_accuracy_fn([(torch.eye(3), torch.tensor([0, 1, 2]))])
    model = identity_classifier()
    assert eval_fn(model) == eval_fn(model) == pytest.approx(1.0)
