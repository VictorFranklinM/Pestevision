"""Model architectures for Pestevision.

Reproduces the four architectures from "Classification of Agricultural Pests
Through Digital Images Using Deep Learning" (RITA, 2025): AlexNet, ResNet-50,
EfficientNet-B4 and ViT-B/16. All are loaded with torchvision ImageNet
pretrained weights and have only the final classification layer replaced, so
they can be fully fine-tuned (no frozen layers).

Usage (from train.py):
    model = get_model("resnet50", num_classes=12).to(device)
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)

SUPPORTED_MODELS = ("alexnet", "resnet50", "efficientnet_b4", "vit_b_16")


def _alexnet(num_classes: int, pretrained: bool) -> nn.Module:
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)
    in_features = model.classifier[6].in_features  # 4096
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def _resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 2048
    return model


def _efficientnet_b4(num_classes: int, pretrained: bool) -> nn.Module:
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features  # 1792
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _vit_b_16(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features  # 768
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


_BUILDERS: dict[str, Callable[[int, bool], nn.Module]] = {
    "alexnet": _alexnet,
    "resnet50": _resnet50,
    "efficientnet_b4": _efficientnet_b4,
    "vit_b_16": _vit_b_16,
}


def get_model(name: str, num_classes: int = 12, pretrained: bool = True) -> nn.Module:
    """Build a torchvision model with its classification head replaced.

    Args:
        name: "alexnet" | "resnet50" | "efficientnet_b4" | "vit_b_16".
        num_classes: number of output classes (12 for the pests dataset).
        pretrained: load ImageNet weights (default True, as in the paper).
            Set False only for offline smoke tests.

    Returns:
        A standard nn.Module with all parameters trainable, expecting
        224x224 RGB input and returning logits of shape (N, num_classes).
    """
    key = name.lower().strip()
    if key not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. Supported: {', '.join(SUPPORTED_MODELS)}"
        )
    model = _BUILDERS[key](num_classes, pretrained)
    for p in model.parameters():
        p.requires_grad = True
    return model


if __name__ == "__main__":
    import torch

    for arch in SUPPORTED_MODELS:
        net = get_model(arch, num_classes=12)
        net.eval()
        with torch.no_grad():
            out = net(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 12), f"{arch}: got {tuple(out.shape)}"
        print(f"{arch}: OK, output shape {tuple(out.shape)}")
