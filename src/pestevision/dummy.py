"""Deterministic smoke-test components; these are not experimental results."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_transform(strategy, split):
    return None


def build_model(cfg):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(3, cfg.get("num_classes", 12)))


def build_dataloaders(cfg, transforms=None):
    """Match data.build_dataloaders(cfg, transforms); fixed, disjoint splits."""
    loaders = {}
    for offset, (split, count) in enumerate((("train", 64), ("val", 24), ("test", 24))):
        generator = torch.Generator().manual_seed(cfg.get("seed", 42) + offset)
        images = torch.randn(count, 3, 16, 16, generator=generator)
        labels = torch.randint(cfg.get("num_classes", 12), (count,), generator=generator)
        dataset = TensorDataset(images, labels)
        loaders[split] = DataLoader(
            dataset, batch_size=cfg.get("data", {}).get("batch_size", 16),
            shuffle=split == "train", generator=generator,
        )
    return loaders
