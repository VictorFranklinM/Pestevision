"""Multiclass evaluation; metrics are accumulated over the entire split."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn


def classification_metrics(confusion):
    """Rows=true, columns=predicted. Undefined ratios are zero.

    Macro includes all configured classes; weighted uses true class support.
    Accuracy is overall multiclass accuracy (it has no averaging variant).
    """
    matrix = torch.as_tensor(confusion, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or (matrix < 0).any():
        raise ValueError("Expected a nonnegative square confusion matrix")
    support = matrix.sum(1)
    total = support.sum()
    if not torch.isfinite(matrix).all() or total <= 0:
        raise ValueError("Cannot compute metrics for empty or nonfinite data")
    if (matrix != matrix.trunc()).any():
        raise ValueError("Confusion matrix must contain integer counts")
    tp = matrix.diag()
    precision = tp / matrix.sum(0).clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + matrix.sum(0)).clamp_min(1)
    result = {"accuracy": (tp.sum() / total).item(), "num_samples": int(total),
              "confusion_matrix": matrix.to(torch.int64).tolist()}
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        result[name + "_macro"] = values.mean().item()
        result[name + "_weighted"] = (values * support / total).sum().item()
    return result


def run_epoch(model, loader, num_classes, device="cpu", optimizer=None):
    """Accept (images, integer labels) batches and model logits [N, C]."""
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    model.train(optimizer is not None)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    loss_sum = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if labels.ndim != 1 or labels.dtype != torch.long:
                raise ValueError("Labels must be a 1D int64 tensor")
            if labels.numel() == 0:
                raise ValueError("Batches must contain at least one sample")
            if images.ndim == 0 or images.shape[0] != labels.shape[0]:
                raise ValueError("Images and labels must have matching batch sizes")
            if ((labels < 0) | (labels >= num_classes)).any():
                raise ValueError("Labels outside configured class range")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != (labels.shape[0], num_classes):
                raise ValueError("Model must return logits of shape [batch, num_classes]")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            size = labels.numel()
            loss_sum += loss.detach().item() * size
            count += size
            indices = (labels * num_classes + logits.detach().argmax(1)).cpu()
            confusion += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    if count == 0:
        raise ValueError("Cannot run an epoch on an empty loader")
    result = classification_metrics(confusion)
    result["loss"] = loss_sum / count
    return result


def evaluate(model, loader, num_classes, device="cpu"):
    model.to(device)
    return run_epoch(model, loader, num_classes, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from .interfaces import build_components
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    model, loaders = build_components(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    report = evaluate(model, loaders[args.split], cfg.get("num_classes", 12), args.device)
    report.update(split=args.split, checkpoint_epoch=checkpoint["epoch"])
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
