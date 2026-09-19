import pytest
import torch
from torch import nn


def make_blob_data(samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """1x8x8 images with a bright 4x4 blob in one of four quadrants; the label is the quadrant."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (samples,), generator=generator)
    images = 0.3 * torch.randn(samples, 1, 8, 8, generator=generator)
    for index, label in enumerate(labels.tolist()):
        row, column = divmod(label, 2)
        images[index, 0, row * 4 : row * 4 + 4, column * 4 : column * 4 + 4] += 1.0
    return images, labels


class BlobNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )  # fmt: skip
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.head(self.features(x))


@pytest.fixture(scope="session")
def blob_batches():
    images, labels = make_blob_data(256, seed=1)
    return [(images[i : i + 64], labels[i : i + 64]) for i in range(0, 256, 64)]


@pytest.fixture(scope="session")
def trained_blobnet():
    torch.manual_seed(0)
    model = BlobNet().train()
    images, labels = make_blob_data(512, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()
    return model.eval()
