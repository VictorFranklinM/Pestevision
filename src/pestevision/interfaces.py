"""Chooses the providers for training/evaluation.

cfg["provider"] == "real"  -> real data + real model + real augmentation
anything else (default)    -> the dummy smoke-test components in dummy.py
"""
import copy
from pathlib import Path

from . import dummy
from .settings import settings

_SPLITS = ("train", "val", "test")


def _with_absolute_data_paths(cfg):
    """Copy of cfg whose data paths no longer depend on the current directory.

    Works on a copy on purpose: cfg is saved inside checkpoints, and a path
    from one machine should not be stored there.
    """
    out = copy.deepcopy(cfg)
    for key in ("raw_dir", "manifest_path"):
        path = Path(out["data"][key])
        if not path.is_absolute():
            out["data"][key] = str(settings.root_path / path)
    return out


def build_components(cfg):
    provider = cfg.get("provider", "dummy")
    if provider == "dummy":
        transforms = {split: dummy.get_transform(cfg.get("strategy", "no_da"), split)
                      for split in _SPLITS}
        return dummy.build_model(cfg), dummy.build_dataloaders(cfg, transforms)
    if provider != "real":
        raise ValueError(f"provider must be 'real' or 'dummy', got {provider!r}")

    architecture, strategy = cfg.get("architecture"), cfg.get("strategy")
    if architecture is None or strategy is None:
        raise ValueError("provider 'real' needs 'architecture' and 'strategy' in the config")

    # Imported here so dummy runs and the unit tests never need these modules.
    from .augmentation import get_transform
    from .data import build_dataloaders
    from .models import get_model

    image_size = cfg["data"]["image_size"]
    transforms = {split: get_transform(strategy, split, image_size=image_size)
                  for split in _SPLITS}
    model = get_model(architecture, num_classes=cfg.get("num_classes", 12))
    loaders = build_dataloaders(_with_absolute_data_paths(cfg), transforms)
    return model, loaders
