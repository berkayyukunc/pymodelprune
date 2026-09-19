"""FashionMNIST loaders and paths shared by the example scripts."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = DATA_DIR / "models"

# mean and std of the FashionMNIST training set
_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])


def get_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Return (train, test) loaders. Downloads ~30 MB into `data/` on first use."""
    train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=_TRANSFORM)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=_TRANSFORM)
    shuffle_seed = torch.Generator().manual_seed(0)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=shuffle_seed),
        DataLoader(test, batch_size=512),
    )


def model_path(name: str) -> Path:
    return MODEL_DIR / f"fashion_{name}.pt"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_finetune_fn(loader: DataLoader, epochs: int = 1, lr: float = 1e-3):
    """Short recovery training after pruning. Trains on the fastest device, returns on CPU."""

    def finetune(model: torch.nn.Module) -> None:
        device = pick_device()
        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            for inputs, labels in loader:
                optimizer.zero_grad()
                loss_fn(model(inputs.to(device)), labels.to(device)).backward()
                optimizer.step()
        model.cpu().eval()

    return finetune
