"""Replace each import with its real provider when ready (see README)."""
from .dummy import build_dataloaders
from .dummy import build_model
from .dummy import get_transform


def build_components(cfg):
    transforms = {split: get_transform(cfg.get("strategy", "no_da"), split)
                  for split in ("train", "val", "test")}
    return build_model(cfg), build_dataloaders(cfg, transforms)
