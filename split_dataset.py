import random
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "dataset_split"

SEED = 42

SPLIT_COUNTS = {
    "Ants": (400, 99),
    "Bees": (405, 95),
    "Beetle": (331, 85),
    "Catterpillar": (329, 105),
    "Earthworms": (246, 77),
    "Earwig": (390, 76),
    "Grasshopper": (390, 95),
    "Moth": (397, 100),
    "Slug": (316, 75),
    "Snail": (405, 95),
    "Wasp": (392, 106),
    "Weevil": (394, 91),
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