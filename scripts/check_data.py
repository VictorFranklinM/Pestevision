"""Check that data/raw matches the committed manifest and the paper's Table 1.

    uv run python scripts/check_data.py               # full check, decodes every image (~seconds)
    uv run python scripts/check_data.py --no-decode   # skip decoding

Exit code 0 means everything is fine; 1 means at least one problem was found.
Run it after cloning, after copying the dataset to a new machine, or if training behaves strangely.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Paper Table 1: manifest class folder -> (total, train, test)
TABLE1 = {
    "ants": (499, 400, 99),
    "bees": (500, 405, 95),
    "beetle": (416, 331, 85),
    "catterpillar": (434, 329, 105),
    "earthworms": (323, 246, 77),
    "earwig": (466, 390, 76),
    "grasshopper": (485, 390, 95),
    "moth": (497, 397, 100),
    "slug": (391, 316, 75),
    "snail": (500, 405, 95),
    "wasp": (498, 392, 106),
    "weevil": (485, 394, 91),
}


def preview(items, limit=5):
    items = sorted(items)
    text = ", ".join(items[:limit])
    return text + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=PROJECT_DIR, help="project root (default: this repo)")
    parser.add_argument("--manifest", type=Path, help="default: <root>/data/split/manifest_seed42.json")
    parser.add_argument("--no-decode", action="store_true", help="do not open every image")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw_dir = args.root / "data" / "raw"
    manifest_path = args.manifest or args.root / "data" / "split" / "manifest_seed42.json"
    problems = []

    def check(ok, text, detail=None):
        print(("[ OK ] " if ok else "[FAIL] ") + text)
        if not ok:
            problems.append(text)
            if detail:
                print(f"         {detail}")

    # 1. manifest ---------------------------------------------------------------------------
    if not manifest_path.is_file():
        check(False, f"manifest not found: {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(set(manifest) >= {"train", "test"}, f"manifest has train and test ({manifest_path.name})")
    train, test = manifest.get("train", {}), manifest.get("test", {})
    classes_ok = set(train) == set(TABLE1) == set(test)
    check(classes_ok, "manifest has exactly the 12 classes of Table 1",
          f"manifest classes: {sorted(set(train) | set(test))}")

    # 2. counts vs the paper ------------------------------------------------------------------
    wrong_counts = []
    for cls, (total, n_train, n_test) in TABLE1.items():
        got = (len(train.get(cls, [])) + len(test.get(cls, [])), len(train.get(cls, [])),
               len(test.get(cls, [])))
        if got != (total, n_train, n_test):
            wrong_counts.append(f"{cls}: total/train/test = {got}, paper = {(total, n_train, n_test)}")
    check(not wrong_counts, "per-class total/train/test counts match paper Table 1",
          "; ".join(wrong_counts[:3]))
    n_train_all = sum(len(v) for v in train.values())
    n_test_all = sum(len(v) for v in test.values())
    print(f"         {n_train_all} train + {n_test_all} test = {n_train_all + n_test_all} images "
          f"(paper: 4395 + 1099 = 5494)")

    # 3. duplicates / leakage inside the manifest ---------------------------------------------
    dup, leak = [], []
    for cls in set(train) | set(test):
        tr, te = train.get(cls, []), test.get(cls, [])
        if len(set(tr)) != len(tr) or len(set(te)) != len(te):
            dup.append(cls)
        if set(tr) & set(te):
            leak.append(cls)
    check(not dup and not leak, "no duplicate filenames and no train/test overlap",
          f"duplicates in: {dup}; train/test overlap in: {leak}")

    # 4. files on disk -----------------------------------------------------------------------
    check(raw_dir.is_dir(), f"data/raw exists ({raw_dir})")
    if not raw_dir.is_dir():
        return 1
    missing, extra, listed_existing = [], [], []
    for cls in sorted(set(train) | set(test)):
        class_dir = raw_dir / cls
        listed = set(train.get(cls, [])) | set(test.get(cls, []))
        on_disk = {p.name for p in class_dir.iterdir() if p.is_file()} if class_dir.is_dir() else set()
        missing += [f"{cls}/{n}" for n in listed - on_disk]
        extra += [f"{cls}/{n}" for n in on_disk - listed]
        listed_existing += [class_dir / n for n in listed & on_disk]
    extra_dirs = sorted(p.name for p in raw_dir.iterdir() if p.is_dir() and p.name not in TABLE1)
    check(not missing, "every file listed in the manifest exists (exact spelling and case)",
          f"{len(missing)} missing: {preview(missing)}")
    check(not extra, "no extra files in data/raw that the manifest does not list",
          f"{len(extra)} extra: {preview(extra)}  (e.g. Thumbs.db, .DS_Store, stray copies)")
    check(not extra_dirs, "no extra class folders in data/raw", f"extra folders: {extra_dirs}")

    # 5. decode every image ----------------------------------------------------------------
    if args.no_decode:
        print("[skip] decoding skipped (--no-decode)")
    else:
        broken, modes, sides = [], Counter(), []
        for path in listed_existing:
            try:
                with Image.open(path) as image:
                    image.load()
                    modes[image.mode] += 1
                    sides.append(max(image.size))
            except Exception as error:  # corrupt, truncated or not an image
                broken.append(f"{path.parent.name}/{path.name} ({type(error).__name__})")
        check(not broken, f"all {len(listed_existing)} listed images open and decode",
              f"{len(broken)} broken: {preview(broken)}")
        if sides:
            print(f"         largest side: {min(sides)}-{max(sides)} px (paper: resized to at most 300)")
            print("         color modes: " + ", ".join(f"{m} {n}" for m, n in modes.most_common())
                  + "  (non-RGB images are converted to RGB by the loader)")

    print()
    if problems:
        print(f"{len(problems)} check(s) FAILED. Fix the dataset before training.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
