"""Training scaffold with validation-loss scheduling and best checkpoints."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from .evaluate import run_epoch


@dataclass
class EarlyStopping:
    patience: int = 21
    best: float = math.inf
    bad_epochs: int = 0
    min_delta: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if self.patience < 1:
            raise ValueError("Early stopping patience must be positive")

    def update(self, loss):
        if not math.isfinite(loss):
            raise ValueError("Nonfinite validation loss")
        improved = loss < self.best - self.min_delta
        if improved:
            self.best, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def fit(model, loaders, cfg, output_dir, device="cpu"):
    """Train on train/val only; restore best weights before returning history."""
    settings = cfg.get("training", {})
    epochs = settings.get("max_epochs", 200)
    if epochs < 1:
        raise ValueError("max_epochs must be positive")
    if settings.get("optimizer", "adam").lower() != "adam":
        raise ValueError("Only Adam is supported")
    missing = {"train", "val"} - loaders.keys()
    if missing:
        raise ValueError(f"Missing required loaders: {sorted(missing)}")
    stopping = EarlyStopping(
        patience=settings.get("early_stopping_patience", 21),
        min_delta=settings.get("min_delta", 0.0),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=settings.get("lr_factor", 0.1),
        patience=settings.get("lr_patience", 10), threshold=0.0,
        threshold_mode="abs",
    )
    best_loss = math.inf
    history = []
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train = run_epoch(model, loaders["train"], cfg.get("num_classes", 12), device, optimizer)
        val = run_epoch(model, loaders["val"], cfg.get("num_classes", 12), device)
        scheduler.step(val["loss"])
        _, stop = stopping.update(val["loss"])
        # Always retain the actual minimum, even below the stopping tolerance.
        improved = val["loss"] < best_loss
        if improved:
            best_loss = val["loss"]
            torch.save({"epoch": epoch, "config": cfg, "val_loss": val["loss"],
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()}, checkpoint_path)
        history.append({"epoch": epoch, "lr": lr,
                        "next_lr": optimizer.param_groups[0]["lr"],
                        "train": train, "val": val, "improved": improved,
                        "early_stopped": stop})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(f"epoch={epoch} train_loss={train['loss']:.6f} val_loss={val['loss']:.6f} lr={lr:g}")
        if stop:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML configuration")
    parser.add_argument("--epochs", type=int, help="Override max_epochs (default: 200)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dummy"))
    args = parser.parse_args()
    cfg = {"seed": 42, "num_classes": 12}
    if args.config:
        import yaml
        cfg.update(yaml.safe_load(args.config.read_text()))
    if args.epochs is not None:
        cfg.setdefault("training", {})["max_epochs"] = args.epochs
    torch.manual_seed(cfg.get("seed", 42))
    from .interfaces import build_components
    model, loaders = build_components(cfg)
    fit(model, loaders, cfg, args.output_dir, args.device)


if __name__ == "__main__":
    main()
