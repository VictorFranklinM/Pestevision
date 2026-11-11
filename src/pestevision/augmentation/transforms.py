"""Per-split transform factory used by the dataloaders: get_transform(strategy, split).

train -> tensor conversion, then the strategy's augmentation pipeline from
         config/augmentation/<strategy>.yaml (built by AugmentationBuilder), then normalize.
val/test -> deterministic: tensor conversion, plain resize, then normalize.
         (The paper does not say how val/test images are preprocessed; a plain
         resize to 224x224 is our assumption.)

The train image size comes from the `crop: size:` field in the strategy's YAML (224).
`image_size` here only affects val/test.
"""
import torch
from torchvision.transforms import v2

from .builder import AugmentationBuilder

STRATEGIES = ("no_da", "da1", "da2", "da3")
SPLITS = ("train", "val", "test")

# torchvision's pretrained weights were trained on ImageNet-normalized inputs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(strategy: str, split: str, image_size: int = 224, normalize: bool = True):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")

    # Convert PIL -> float tensor in [0, 1] first so every later step (including
    # RandomErasing) works on tensors, in the order the paper describes.
    ops = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if split == "train":
        ops.append(AugmentationBuilder.from_config(strategy).build())
    else:
        ops.append(v2.Resize((image_size, image_size)))
    if normalize:
        ops.append(v2.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    ops.append(v2.ToPureTensor())  # plain torch.Tensor so default_collate batches it cleanly
    return v2.Compose(ops)
