"""Per-image predictions: which image was classified as what, and how confidently.

The metrics in evaluate.py only keep a confusion matrix, so they cannot say *which* images were
wrong or give the class probabilities needed for precision-recall curves (paper Figure 3).
collect_predictions() runs a model over a loader; write_predictions_csv() saves the result as one
row per image, with the file path, true/predicted class, confidence and all class probabilities.
"""
import csv
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler


@torch.no_grad()
def collect_predictions(model, loader, device="cpu"):
    """Return {"labels": [int], "probs": [[float]]} in the loader's order.

    Predictions are matched to dataset samples by position, so the loader must not shuffle.
    """
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("collect_predictions needs an unshuffled loader (it matches rows to "
                         "dataset samples by position)")
    model.to(device).eval()
    labels, probs = [], []
    for images, batch_labels in loader:
        logits = model(images.to(device))
        probs.extend(logits.float().softmax(1).cpu().tolist())
        labels.extend(batch_labels.tolist())
    return {"labels": labels, "probs": probs}


def _argmax(row):
    return max(range(len(row)), key=row.__getitem__)


def accuracy_from_predictions(predictions):
    labels, probs = predictions["labels"], predictions["probs"]
    return sum(_argmax(row) == label for row, label in zip(probs, labels)) / len(labels)


def write_predictions_csv(path, predictions, dataset=None):
    """Write one row per image. `dataset` (a PestDataset) supplies file paths and class names;
    without it the rows are labelled by index. Raises if the predictions do not line up with the
    dataset's samples, so a wrong file is never written silently."""
    labels, probs = predictions["labels"], predictions["probs"]
    if len(labels) != len(probs) or not labels:
        raise ValueError("predictions are empty or inconsistent")
    num_classes = len(probs[0])

    samples = getattr(dataset, "samples", None)
    if samples is not None:
        if [label for _, label in samples] != labels:
            raise ValueError("predictions do not line up with the dataset's samples "
                             "(different order or size)")
        raw_dir = Path(dataset.raw_dir)
        paths = []
        for sample_path, _ in samples:
            try:
                paths.append(Path(sample_path).relative_to(raw_dir).as_posix())
            except ValueError:
                paths.append(str(sample_path))
    else:
        paths = [f"index_{i}" for i in range(len(labels))]

    classes = list(getattr(dataset, "classes", None) or [str(i) for i in range(num_classes)])
    if len(classes) != num_classes:
        raise ValueError(f"{len(classes)} class names but {num_classes} probability columns")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "true_class", "pred_label", "pred_class",
                         "correct", "confidence"] + [f"prob_{name}" for name in classes])
        for image_path, label, row in zip(paths, labels, probs):
            pred = _argmax(row)
            writer.writerow([image_path, label, classes[label], pred, classes[pred],
                             int(pred == label), format(row[pred], ".6g")]
                            + [format(p, ".6g") for p in row])
