import argparse
import json
import re
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
            "generated_by": "scripts/split_dataset.py",
            "num_classes": len(SPLIT_COUNTS),
        }, 
        "train": {}, 
        "test": {}
    }

    for class_name, (train_count, test_count) in SPLIT_COUNTS.items():
        source_class_dir = SOURCE_DIR / class_name

        if not source_class_dir.exists():
            raise FileNotFoundError(f"Could not find class folder: {source_class_dir}")

        images = [file for file in source_class_dir.iterdir() if file.is_file()]

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


def write_manifest(seed: int) -> Path:
    print(f"\nSeed {seed}:")
    manifest = build_manifest(seed)
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / f"manifest_seed{seed}.json"
 
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
 
    print(f"--> {manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more seeded train/test manifests from data/raw/."
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help=(
            "Seed to use for the train/test shuffle. Can be passed multiple times "
            f"to generate several manifests in one run. Defaults to {DEFAULT_SEED} "
            "if omitted entirely."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seed if args.seed else [DEFAULT_SEED]
 
    written = [write_manifest(seed) for seed in seeds]
 
    print(f"\nDone. {len(written)} manifest(s) written to {OUTPUT_DIR}")
    for path in written:
        print(f"  - {path.name}")
 
 
if __name__ == "__main__":
    main()

