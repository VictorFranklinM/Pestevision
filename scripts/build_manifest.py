"""Build (or verify) the seeded train/test manifest from data/raw/.

    uv run python scripts/build_manifest.py --check     # does a fresh build match the committed manifest?
    uv run python scripts/build_manifest.py --seed 7    # write a NEW manifest (refuses to overwrite)

The committed data/split/manifest_seed42.json is the source of truth for every experiment. Use
--check to confirm that this script still reproduces it; only pass --force if you really mean
to replace a manifest, because every result computed from the old one would stop being comparable.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from random import Random

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "split"

DEFAULT_SEED = 42

SPLIT_COUNTS = {
    "ants": (400, 99),
    "bees": (405, 95),
    "beetle": (331, 85),
    "catterpillar": (329, 105),
    "earthworms": (246, 77),
    "earwig": (390, 76),
    "grasshopper": (390, 95),
    "moth": (397, 100),
    "slug": (316, 75),
    "snail": (405, 95),
    "wasp": (392, 106),
    "weevil": (394, 91),
}


def natural_key(s):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', s)]


def build_manifest(seed: int) -> dict:
    rng = Random(seed)

    manifest = {
        "_meta": {
            "seed": seed,
            "test_fraction_source": "paper_table1_predefined_counts",
            "generated_by": "scripts/build_manifest.py",
            "num_classes": len(SPLIT_COUNTS),
        },
        "train": {},
        "test": {}
    }

    for class_name, (train_count, test_count) in SPLIT_COUNTS.items():
        source_class_dir = SOURCE_DIR / class_name

        if not source_class_dir.exists():
            raise FileNotFoundError(f"Could not find class folder: {source_class_dir}")

        # Sorted BEFORE shuffling: iterdir() order differs between operating systems and
        # file systems, and Random(seed).shuffle() only fixes the shuffle, not the input order.
        images = sorted((file for file in source_class_dir.iterdir() if file.is_file()),
                        key=lambda file: file.name)

        expected_total = train_count + test_count

        if len(images) != expected_total:
            raise ValueError(f"{class_name}: expected {expected_total} images, but found {len(images)}")

        rng.shuffle(images)

        train_images = images[:train_count]
        test_images = images[train_count:]

        manifest["train"][class_name] = sorted((img.name for img in train_images), key=natural_key)
        manifest["test"][class_name] = sorted((img.name for img in test_images), key=natural_key)

        print(f" - {class_name}: {len(train_images)} train | {len(test_images)} test")

    return manifest


def manifest_path_for(seed: int) -> Path:
    return OUTPUT_DIR / f"manifest_seed{seed}.json"


def write_manifest(seed: int, force: bool = False) -> Path:
    manifest_path = manifest_path_for(seed)
    if manifest_path.exists() and not force:
        raise SystemExit(f"{manifest_path.name} already exists; refusing to overwrite it. "
                         f"Use --check to compare against it, or --force to replace it.")

    print(f"\nSeed {seed}:")
    manifest = build_manifest(seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"--> {manifest_path}")
    return manifest_path


def check_manifest(seed: int) -> bool:
    """Rebuild in memory and compare the train/test lists with the committed file."""
    path = manifest_path_for(seed)
    if not path.is_file():
        print(f"{path.name}: not found, nothing to compare against")
        return False
    print(f"\nSeed {seed}:")
    committed = json.loads(path.read_text(encoding="utf-8"))
    fresh = build_manifest(seed)
    same = all(committed.get(split) == fresh[split] for split in ("train", "test"))
    print(f"{path.name}: {'MATCH - a fresh build reproduces the committed split' if same else 'MISMATCH'}")
    return same


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or verify seeded train/test manifests from data/raw/."
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help=(
            "Seed to use for the train/test shuffle. Can be passed multiple times "
            f"to handle several manifests in one run. Defaults to {DEFAULT_SEED} "
            "if omitted entirely."
        ),
    )
    parser.add_argument("--check", action="store_true",
                        help="compare a fresh build with the existing manifest; writes nothing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing manifest (invalidates comparisons with old results)")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seed if args.seed else [DEFAULT_SEED]

    if args.check:
        results = [check_manifest(seed) for seed in seeds]
        sys.exit(0 if all(results) else 1)

    written = [write_manifest(seed, args.force) for seed in seeds]

    print(f"\nDone. {len(written)} manifest(s) written to {OUTPUT_DIR}")
    for path in written:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
