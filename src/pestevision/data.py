"""
Usage:
    from pestevision.data import build_datasets, build_dataloaders

    train_ds, val_ds, test_ds = build_datasets(cfg)

    # or, once the augmentation pipeline exists:
    transforms = {
        "train": get_transform(strategy, "train"),
        "val":   get_transform(strategy, "val"),
        "test":  get_transform(strategy, "test"),
    }
    loaders = build_dataloaders(cfg, transforms)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASS_DISPLAY_NAMES = {
    "ants": "Ants",
    "bees": "Bees",
    "beetle": "Beetles",
    "catterpillar": "Caterpillars",
    "earthworms": "Earthworms",
    "earwig": "Earwigs",
    "grasshopper": "Grasshoppers",
    "moth": "Moths",
    "slug": "Slugs",
    "snail": "Snails",
    "wasp": "Wasps",
    "weevil": "Weevils",
}

def load_config(path: str | Path) -> dict: # yaml file
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_manifest(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _stratified_train_val_split(
    manifest_train: dict[str, list[str]],
    classes: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Per-class deterministic split of the manifest's "train" entries into
    (train, val). Classes are iterated in a fixed sorted order and each
    class's file list is sorted before shuffling, so the result depends
    only on `seed`, never on dict insertion order or filesystem order.
    """
    rng = random.Random(seed)
    train_split: dict[str, list[str]] = {}
    val_split: dict[str, list[str]] = {}

    for cls in classes:
        files = sorted(manifest_train.get(cls, []))
        rng.shuffle(files)
        n_val = round(len(files) * val_fraction)
        val_split[cls] = files[:n_val]
        train_split[cls] = files[n_val:]

    return train_split, val_split


def _default_transform(image_size: int) -> Callable:
    """
    Minimal placeholder transform (paper's own No-DA baseline, Sec 3.3:
    just a resize to 224x224 + tensor conversion). Lets this module be
    used and tested standalone before the real augmentation pipeline
    lands — swap in `transform=get_transform(...)` later, no other
    changes needed.
    """
    from torchvision import transforms as T

    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])


class PestDataset(Dataset):
    """
    split: "train" | "val" | "test"
    transform: callable applied to the PIL image; defaults to a plain
        resize+ToTensor if not given (see _default_transform).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        raw_dir: str | Path,
        split: str,
        val_fraction: float = 0.2,
        seed: int = 42,
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.raw_dir = Path(raw_dir)
        self.split = split
        self.transform = transform or _default_transform(image_size)

        manifest = load_manifest(manifest_path)
        self.classes = sorted(manifest["train"].keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if split == "test":
            class_to_files = manifest["test"]
        else:
            train_split, val_split = _stratified_train_val_split(
                manifest["train"], self.classes, val_fraction, seed
            )
            class_to_files = train_split if split == "train" else val_split

        # Flatten to a list of (path, label) pairs for O(1) __getitem__.
        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            label = self.class_to_idx[cls]
            for fname in class_to_files.get(cls, []):
                self.samples.append((self.raw_dir / cls / fname, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")  # dataset mixes jpg/png; normalize mode
        image = self.transform(image)
        return image, label

    def __repr__(self) -> str:
        return (
            f"PestDataset(split={self.split!r}, "
            f"n_samples={len(self)}, n_classes={len(self.classes)})"
        )


def build_datasets( # change when the data augmentation pipeline is finished
    cfg: dict, transforms: Optional[dict[str, Callable]] = None
) -> tuple[PestDataset, PestDataset, PestDataset]:
    """
    cfg: the loaded base.yaml dict (or equivalent).
    transforms: optional {"train": fn, "val": fn, "test": fn}; falls back
        to the default resize+ToTensor transform per split if omitted.
    """
    data_cfg = cfg["data"]
    seed = cfg["seed"]
    transforms = transforms or {}

    common = dict(
        manifest_path=data_cfg["manifest_path"],
        raw_dir=data_cfg["raw_dir"],
        val_fraction=data_cfg["val_fraction"],
        seed=seed,
        image_size=data_cfg["image_size"],
    )

    train_ds = PestDataset(split="train", transform=transforms.get("train"), **common)
    val_ds = PestDataset(split="val", transform=transforms.get("val"), **common)
    test_ds = PestDataset(split="test", transform=transforms.get("test"), **common)
    return train_ds, val_ds, test_ds


def build_dataloaders(
    cfg: dict,
    transforms: Optional[dict[str, Callable]] = None,
    batch_size: Optional[int] = None,
) -> dict[str, DataLoader]:
    """
    batch_size: override cfg["data"]["batch_size"]; use this to plug in
        the per-architecture values from paper Table 2 at train time.
    """
    train_ds, val_ds, test_ds = build_datasets(cfg, transforms)
    bs = batch_size or cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    return {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers),
    }


if __name__ == "__main__":
    # Quick sanity check: run `python -m pestevision.data` from the repo
    # root (or `python src/pestevision/data.py`) once configs/base.yaml
    # points at your real data/raw and manifest.
    cfg = load_config("configs/base.yaml")
    train_ds, val_ds, test_ds = build_datasets(cfg)

    print(train_ds)
    print(val_ds)
    print(test_ds)

    # Stratification + no-leakage checks
    train_paths = {p for p, _ in train_ds.samples}
    val_paths = {p for p, _ in val_ds.samples}
    overlap = train_paths & val_paths
    print(f"train/val overlap: {len(overlap)} (should be 0)")

    total = len(train_ds) + len(val_ds)
    print(f"train+val = {total}, val fraction = {len(val_ds) / total:.3f}")

    image, label = train_ds[0]
    print(f"sample image shape: {tuple(image.shape)}, label: {label} ({train_ds.classes[label]})")
