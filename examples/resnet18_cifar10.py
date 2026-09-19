"""End to end on a real network: ResNet-18 on CIFAR-10.

    uv run python examples/resnet18_cifar10.py

1. trains ResNet-18 (cached in data/models/ after the first run)
2. compares channel importance criteria at several pruning ratios
3. prunes 50% of the channels, fine-tunes, exports to ONNX and quantizes to INT8
4. prints one table with every stage, measured on CPU

Training uses MPS/CUDA when available. Downloads CIFAR-10 (~170 MB) on first use.
"""

from __future__ import annotations

import time

import torch
import typer
from fashion_data import DATA_DIR, MODEL_DIR, pick_device
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from pymodelprune import (
    PruneConfig,
    QuantConfig,
    accuracy,
    make_accuracy_fn,
    make_classification_grad_fn,
    optimize,
    prune_structured,
)

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CRITERIA = ("random", "l1", "l2", "bn", "taylor")


def cifar_resnet18() -> nn.Module:
    """torchvision ResNet-18 with the ImageNet stem swapped for one that suits 32x32 images."""
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    normalize = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train = datasets.CIFAR10(DATA_DIR, True, transforms.Compose(augment + normalize), download=True)
    test = datasets.CIFAR10(DATA_DIR, False, transforms.Compose(normalize), download=True)
    generator = torch.Generator().manual_seed(0)
    return (
        DataLoader(train, batch_size, shuffle=True, generator=generator, drop_last=True),
        DataLoader(test, 500),
    )


def train(model: nn.Module, loader: DataLoader, epochs: int, max_lr: float) -> None:
    device = pick_device()
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        start, running = time.perf_counter(), 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            schedule.step()
            running += loss.item()
        print(
            f"  epoch {epoch}/{epochs}  loss {running / len(loader):.3f}  {time.perf_counter() - start:.0f}s"
        )
    model.cpu().eval()


def main(epochs: int = 10, finetune_epochs: int = 3, amount: float = 0.5, seed: int = 0) -> None:
    torch.manual_seed(seed)
    train_loader, test_loader = get_loaders()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / "cifar_resnet18.pt"

    if checkpoint.exists():
        model = torch.load(checkpoint, weights_only=False)
    else:
        model = cifar_resnet18()
        print(f"training ResNet-18 for {epochs} epochs on {pick_device()}")
        train(model, train_loader, epochs, max_lr=0.1)
        torch.save(model, checkpoint)
    print(f"baseline test accuracy {accuracy(model, test_loader):.2%}\n")

    example = torch.randn(1, 3, 32, 32)
    subset = make_accuracy_fn(test_loader, max_batches=4)  # 2000 images: enough to rank criteria
    # A fresh loader, so the gradients behind the taylor scores are the same batches
    # whether this run trained the model or loaded the cached checkpoint.
    grad_loader, _ = get_loaders()
    grad_fn = make_classification_grad_fn(grad_loader, max_batches=8)
    # Ratios are kept low on purpose. Remove 30% of the channels of every group and every
    # criterion lands near 15%, so the comparison would be between wrecks.
    print(f"accuracy right after pruning, no fine-tuning (baseline {subset(model):.1%}):")
    print(f"{'ratio':<7}" + "".join(f"{name:>9}" for name in CRITERIA))
    for ratio in (0.05, 0.10, 0.20):
        scores = []
        for importance in CRITERIA:
            torch.manual_seed(seed)  # only `random` uses it, but it must not drift
            scores.append(
                subset(prune_structured(model, example, ratio, importance, grad_fn=grad_fn))
            )
        print(f"{ratio:<7.0%}" + "".join(f"{score:>8.1%} " for score in scores))
    print()

    result = optimize(
        model,
        example,
        prune=PruneConfig(amount=amount, importance="taylor"),
        quantize=QuantConfig(mode="onnx_static", calibration=train_loader),
        eval_fn=make_accuracy_fn(test_loader),
        fidelity_inputs=[batch for _, batch in zip(range(4), test_loader, strict=False)],
        finetune_fn=lambda pruned: train(pruned, train_loader, finetune_epochs, max_lr=0.02),
        grad_fn=grad_fn,
        onnx_path=MODEL_DIR / "cifar_resnet18_pruned_int8.onnx",
    )
    print()
    result.report()
    result.save(MODEL_DIR / "cifar_resnet18_pruned.pt")
    result.to_json(MODEL_DIR / "cifar_resnet18_report.json")


if __name__ == "__main__":
    typer.run(main)
