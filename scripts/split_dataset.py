import random
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "split"

SEED = 42

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

random.seed(SEED)

train_dir = OUTPUT_DIR / "train"
test_dir = OUTPUT_DIR / "test"

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

for class_name, (train_count, test_count) in SPLIT_COUNTS.items():

    source_class_dir = SOURCE_DIR / class_name

    if not source_class_dir.exists():
        raise FileNotFoundError(
            f"Could not find class folder: {source_class_dir}"
        )

    images = [
        file for file in source_class_dir.iterdir()
        if file.is_file()
    ]

    expected_total = train_count + test_count

    if len(images) != expected_total:
        raise ValueError(
            f"{class_name}: expected {expected_total} images, "
            f"but found {len(images)}"
        )

    random.shuffle(images)

    train_images = images[:train_count]
    test_images = images[train_count:]

    train_class_dir = train_dir / class_name
    test_class_dir = test_dir / class_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        shutil.copy2(
            image,
            train_class_dir / image.name
        )

    for image in test_images:
        shutil.copy2(
            image,
            test_class_dir / image.name
        )

    print(
        f"{class_name}: "
        f"{len(train_images)} train / "
        f"{len(test_images)} test"
    )

print("\nDataset split completed!")
print(f"Output: {OUTPUT_DIR}")