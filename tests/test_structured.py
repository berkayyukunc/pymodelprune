import copy
import random

import pytest
import torch
from torch import nn

from pymodelprune import count_parameters
from pymodelprune.demo_models import TinyCNN, TinyMLP
from pymodelprune.profile import count_flops
from pymodelprune.prune import (
    UntraceableModelError,
    apply_plan,
    build_dependency_graph,
    make_classification_grad_fn,
    mask_plan,
    plan_pruning,
    prune_structured,
)


class ResidualNet(nn.Module):
    """Stem + two residual blocks (one with a 1x1 downsample) + classifier."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.a1 = nn.Conv2d(16, 24, 3, padding=1)
        self.a1_bn = nn.BatchNorm2d(24)
        self.a2 = nn.Conv2d(24, 16, 3, padding=1)
        self.a2_bn = nn.BatchNorm2d(16)
        self.b1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.b1_bn = nn.BatchNorm2d(32)
        self.b2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.down = nn.Sequential(nn.Conv2d(16, 32, 1, stride=2), nn.BatchNorm2d(32))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(x + self.a2_bn(self.a2(torch.relu(self.a1_bn(self.a1(x))))))
        x = torch.relu(self.down(x) + self.b2(torch.relu(self.b1_bn(self.b1(x)))))
        return self.fc(torch.flatten(self.pool(x), 1))


class DepthwiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.Conv2d(16, 8, 1),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.net(x)


def randomize_norms(model: nn.Module) -> nn.Module:
    """Fresh BatchNorm layers are identities; give them real statistics."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.uniform_(module.weight, 0.5, 1.5)
            nn.init.normal_(module.bias, std=0.5)
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 2.0)
    return model.eval()


def assert_zeroing_equals_removing(model, example, amount=0.5, importance="l1", atol=1e-5):
    model = randomize_norms(model)
    graph = build_dependency_graph(model, example)
    assert graph.prunable_groups, [g.blockers for g in graph.groups]
    plan = plan_pruning(model, graph, amount, importance)
    masked = mask_plan(copy.deepcopy(model), graph, plan)
    pruned = apply_plan(copy.deepcopy(model), graph, plan)
    with torch.no_grad():
        expected, actual = masked(example), pruned(example)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=atol), float((actual - expected).abs().max())
    assert count_parameters(pruned)[0] < count_parameters(model)[0]
    return graph, pruned


CASES = [
    (TinyMLP, (4, 1, 28, 28)),
    (TinyCNN, (4, 1, 28, 28)),
    (ResidualNet, (2, 3, 16, 16)),
    (DepthwiseNet, (2, 3, 8, 8)),
]


@pytest.mark.parametrize("factory,shape", CASES)
@pytest.mark.parametrize("importance", ["l1", "l2", "bn", "random"])
def test_zeroing_equals_removing(factory, shape, importance):
    torch.manual_seed(0)
    assert_zeroing_equals_removing(factory(), torch.randn(shape), importance=importance)


def test_residual_branches_share_one_group():
    graph = build_dependency_graph(ResidualNet(), torch.randn(1, 3, 16, 16))
    by_producer = {name: group for group in graph.groups for name in group.producers}
    assert by_producer["stem.0"] is by_producer["a2"]  # identity path ties them
    assert by_producer["down.0"] is by_producer["b2"]
    assert by_producer["a1"] is not by_producer["a2"]
    assert by_producer["a1"].prunable


def test_model_output_is_never_pruned():
    for factory, shape in CASES:
        model, example = factory(), torch.randn(shape)
        pruned = prune_structured(model, example, 0.8)
        assert pruned(example).shape == model(example).shape


def test_flatten_maps_channels_to_feature_blocks():
    graph = build_dependency_graph(TinyCNN(), torch.randn(1, 1, 28, 28))
    last_conv = next(g for g in graph.groups if "features.4" in g.producers)
    assert last_conv.consumers == [("classifier", 49)]  # 7x7 features per channel
    assert last_conv.norms == ["features.5"]


def test_depthwise_conv_is_tied_to_its_input():
    graph = build_dependency_graph(DepthwiseNet(), torch.randn(1, 3, 8, 8))
    group = next(g for g in graph.groups if "net.0" in g.producers)
    assert group.depthwise == ["net.3"] and group.norms == ["net.1", "net.4"]
    pruned = prune_structured(DepthwiseNet(), torch.randn(1, 3, 8, 8), 0.5)
    assert pruned.net[3].groups == pruned.net[3].in_channels == 8


def test_structured_pruning_really_shrinks_the_model():
    model, example = TinyCNN(), torch.randn(1, 1, 28, 28)
    pruned = prune_structured(model, example, 0.5)
    assert count_parameters(pruned)[0] < 0.6 * count_parameters(model)[0]
    assert count_flops(pruned, example) < 0.6 * count_flops(model, example)
    assert pruned.features[0].out_channels == 16 and pruned.features[4].in_channels == 16


def test_input_model_is_untouched():
    model = TinyCNN()
    before = copy.deepcopy(model.state_dict())
    prune_structured(model, torch.randn(1, 1, 28, 28), 0.5)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_pruned_model_is_trainable_and_picklable(tmp_path):
    example = torch.randn(4, 1, 28, 28)
    pruned = prune_structured(TinyCNN(), example, 0.5).train()
    pruned(example).sum().backward()
    assert all(p.grad is not None for p in pruned.parameters())
    torch.save(pruned, tmp_path / "pruned.pt")
    restored = torch.load(tmp_path / "pruned.pt", weights_only=False).eval()
    assert torch.allclose(restored(example), pruned.eval()(example))


def test_l1_drops_the_weakest_neurons():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.diag(torch.tensor([5.0, 0.1, 3.0, 0.2])))
    pruned = prune_structured(model, torch.randn(1, 4), 0.5)
    assert pruned[0].weight.abs().sum(dim=1).tolist() == [5.0, 3.0]
    assert pruned[1 + 1].in_features == 2


def test_explicit_ratios_and_global_scope():
    model, example = TinyCNN(), torch.randn(1, 1, 28, 28)
    graph = build_dependency_graph(model, example)
    first, _second = (g.id for g in graph.prunable_groups)
    pruned = prune_structured(model, example, {first: 0.75}, graph=graph)
    assert pruned.features[0].out_channels == 8 and pruned.features[4].out_channels == 64
    with pytest.raises(ValueError):
        prune_structured(model, example, {999: 0.5}, graph=graph)
    global_pruned = prune_structured(model, example, 0.5, scope="global")
    assert count_parameters(global_pruned)[0] < count_parameters(model)[0]


def test_round_to_and_min_channels():
    pruned = prune_structured(TinyCNN(), torch.randn(1, 1, 28, 28), 0.45, round_to=8)
    assert pruned.features[0].out_channels % 8 == 0 and pruned.features[4].out_channels % 8 == 0


def test_taylor_importance():
    torch.manual_seed(0)
    example = torch.randn(8, 1, 28, 28)
    grad_fn = make_classification_grad_fn([(example, torch.randint(0, 10, (8,)))])
    model = TinyCNN()
    pruned = prune_structured(model, example, 0.5, importance="taylor", grad_fn=grad_fn)
    assert pruned(example).shape == (8, 10)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError):
        prune_structured(model, example, 0.5, importance="taylor")


class CatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Conv2d(3, 4, 1), nn.Conv2d(3, 4, 1)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.head(torch.cat([self.a(x), self.b(x)], dim=1))


class SigmoidGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Conv2d(3, 4, 1), nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.b(torch.sigmoid(self.a(x)))


class SharedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 2)

    def forward(self, x):
        return self.b(torch.relu(self.a(torch.relu(self.a(x)))))


class AddsConstant(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 2)

    def forward(self, x):
        return self.b(self.a(x) + 1.0)


@pytest.mark.parametrize(
    "factory,shape",
    [
        (CatNet, (1, 3, 4, 4)),
        (SigmoidGate, (1, 3, 4, 4)),
        (SharedLayer, (1, 4)),
        (AddsConstant, (1, 4)),
    ],
)
def test_unsupported_patterns_are_refused_not_guessed(factory, shape):
    graph = build_dependency_graph(factory(), torch.randn(shape))
    assert not graph.prunable_groups
    assert any(group.blockers for group in graph.groups)
    with pytest.raises(ValueError, match="Blockers"):
        prune_structured(factory(), torch.randn(shape), 0.5)


def test_layer_norm_blocks_pruning():
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 2))
    assert not build_dependency_graph(model, torch.randn(2, 8)).prunable_groups


def test_data_dependent_control_flow_gives_a_clear_error():
    class Branchy(nn.Module):
        def forward(self, x):
            return x if x.sum() > 0 else -x

    with pytest.raises(UntraceableModelError):
        build_dependency_graph(Branchy(), torch.randn(2, 2))


def random_cnn(rng: random.Random) -> nn.Module:
    layers, channels = [], 3
    for _ in range(rng.randint(2, 5)):
        out = rng.choice([4, 6, 8, 12])
        layers.append(
            nn.Conv2d(channels, out, rng.choice([1, 3]), padding=1, bias=rng.random() < 0.5)
        )
        if rng.random() < 0.6:
            layers.append(nn.BatchNorm2d(out))
        layers.append(rng.choice([nn.ReLU, nn.GELU, nn.SiLU, nn.Identity])())
        if rng.random() < 0.3:
            layers.append(nn.MaxPool2d(2, ceil_mode=True))
        channels = out
    pool = rng.choice([1, 2])
    layers += [
        nn.AdaptiveAvgPool2d(pool),
        nn.Flatten(),
        nn.Linear(channels * pool * pool, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    ]
    return nn.Sequential(*layers)


@pytest.mark.parametrize("seed", range(12))
def test_random_architectures(seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    assert_zeroing_equals_removing(
        random_cnn(rng), torch.randn(2, 3, 16, 16), amount=rng.choice([0.25, 0.5, 0.75])
    )


torchvision_models = pytest.importorskip("torchvision.models")


@pytest.mark.parametrize("name", ["resnet18", "resnet50", "mobilenet_v2", "vgg11_bn"])
def test_torchvision_models(name):
    torch.manual_seed(0)
    model = getattr(torchvision_models, name)(weights=None)
    graph, _pruned = assert_zeroing_equals_removing(
        model, torch.randn(1, 3, 64, 64), amount=0.4, atol=1e-4
    )
    assert len(graph.prunable_groups) >= 5


# Regressions found by adversarial testing: every pattern below used to produce a wrong
# or crashing model. Each must now either be refused or keep zeroing == removing.
class _Branches(nn.Module):
    def __init__(self, head_in=8):
        super().__init__()
        self.a, self.b = nn.Conv2d(3, 8, 1), nn.Conv2d(3, 8, 1)
        self.h, self.k, self.fc = nn.Conv2d(head_in, 2, 1), nn.Conv2d(8, 2, 1), nn.Linear(128, 2)


def _net(forward, head_in=8):
    return type("Net", (_Branches,), {"forward": forward})(head_in)


ADVERSARIAL = {
    "keyword_add": lambda s, x: s.h(torch.add(s.a(x), other=s.b(x))),
    "method_keyword_add": lambda s, x: s.h(s.a(x).add(other=s.b(x))),
    "keyword_mul": lambda s, x: s.h(torch.mul(s.a(x), other=s.b(x))),
    "sort": lambda s, x: s.h(s.a(x).sort(dim=1)[0]),
    "max_dim": lambda s, x: s.k(s.a(x) + torch.max(s.b(x), dim=1, keepdim=True)[0]),
    "literal_view": lambda s, x: s.fc(s.a(x).view(-1, 128)),
    "literal_reshape": lambda s, x: s.fc(torch.reshape(s.a(x), (-1, 128))),
    "weight_reuse": lambda s, x: s.h(s.a(x)) + s.k(nn.functional.conv2d(x, s.a.weight)),
    "reciprocal": lambda s, x: s.h(1.0 / s.a(x)),
    "tensor_division": lambda s, x: s.h(s.a(x) / s.b(x)),
    "hardtanh_offset": lambda s, x: s.h(nn.functional.hardtanh(s.a(x), 0.5, 2.0)),
    "broadcast_rank": lambda s, x: s.h(s.a(x) + s.fc(s.b(x).flatten(1))[:, :1, None, None]),
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_patterns_never_yield_a_wrong_model(name):
    torch.manual_seed(0)
    model, example = _net(ADVERSARIAL[name]).eval(), torch.randn(2, 3, 4, 4)
    graph = build_dependency_graph(model, example)
    plan = plan_pruning(model, graph, 0.5)
    masked = mask_plan(copy.deepcopy(model), graph, plan)(example)
    pruned = apply_plan(copy.deepcopy(model), graph, plan)(example)
    assert torch.allclose(masked, pruned, atol=1e-5)


def test_split_is_refused():
    model = _net(lambda s, x: s.h(s.a(x).split(4, dim=1)[0]), head_in=4)
    assert not [
        g
        for g in build_dependency_graph(model, torch.randn(2, 3, 4, 4)).prunable_groups
        if "a" in g.producers
    ]


def test_scalar_division_and_scaling_stay_prunable():
    model = _net(lambda s, x: s.h(s.a(x) / 2.0 * 0.5))
    assert_zeroing_equals_removing(model, torch.randn(2, 3, 4, 4))


def test_unusual_modules_are_refused():
    offset = nn.Sequential(nn.Conv2d(3, 8, 1), nn.Hardtanh(0.5, 2.0), nn.Conv2d(8, 2, 1))
    plain_bn = nn.Sequential(
        nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8, affine=False), nn.Conv2d(8, 2, 1)
    )
    normed = nn.Sequential(
        nn.utils.parametrizations.weight_norm(nn.Conv2d(3, 8, 1)), nn.Conv2d(8, 2, 1)
    )
    tied = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    tied[2].weight = tied[0].weight
    for model, shape in [(offset, (1, 3, 4, 4)), (plain_bn, (1, 3, 4, 4)), (normed, (1, 3, 4, 4))]:
        assert not build_dependency_graph(model, torch.randn(shape)).prunable_groups
    tied_groups = build_dependency_graph(tied, torch.randn(2, 8)).prunable_groups
    assert not [g for g in tied_groups if {"0", "2"} & set(g.producers)]


def test_train_only_branches_are_pruned_consistently():
    def forward(s, x):
        features = s.a(x)
        return (s.h(features), s.fc(features.flatten(1))) if s.training else s.h(features)

    model = _net(forward).eval()
    example = torch.randn(1, 3, 4, 4)
    graph = build_dependency_graph(model, example)
    if graph.prunable_groups:
        pruned = prune_structured(model, example, 0.5, graph=graph).train()
        pruned(torch.randn(2, 3, 4, 4))


def test_select_channels_rounding_and_validation():
    from pymodelprune.prune.structured import select_channels

    assert len(select_channels(torch.rand(16), 0.8, round_to=8)) == 8
    assert len(select_channels(torch.rand(4), 0.5, round_to=8)) == 2  # fewer channels than round_to
    assert len(select_channels(torch.rand(16), 0.99)) == 1
    for bad in ({"round_to": 0}, {"min_channels": 0}):
        with pytest.raises(ValueError):
            select_channels(torch.rand(16), 0.5, **bad)


def test_global_scope_edge_cases():
    model, example = TinyCNN(), torch.randn(1, 1, 28, 28)
    untouched = prune_structured(model, example, 0.0, scope="global")
    assert count_parameters(untouched) == count_parameters(model)
    resnet = torchvision_models.resnet18(weights=None)
    pruned = prune_structured(resnet, torch.randn(1, 3, 64, 64), 0.5, scope="global")
    assert pruned.layer1[0].conv2.out_channels < 64, "residual streams must not be immune"


def test_clear_errors_for_bad_inputs():
    from pymodelprune.prune import prune_unstructured

    with pytest.raises(ValueError, match="example input"):
        build_dependency_graph(TinyCNN(), torch.randn(1, 3, 28, 28))
    masked = prune_unstructured(TinyCNN(), 0.5, permanent=False)
    with pytest.raises(ValueError, match="make_permanent"):
        build_dependency_graph(masked, torch.randn(1, 1, 28, 28))
    model, example = TinyCNN(), torch.randn(1, 1, 28, 28)
    graph = build_dependency_graph(model, example)
    with pytest.raises(ValueError, match="already pruned"):
        prune_structured(prune_structured(model, example, 0.5), example, 0.5, graph=graph)
