"""Run the experiment grid: architectures x augmentation strategies.

Examples (from the repo root):
    uv run python scripts/run_training.py --provider dummy --epochs 2      # plumbing check, seconds
    uv run python scripts/run_training.py --arch resnet50 --strategy no_da --epochs 3
    uv run python scripts/run_training.py                                   # the full 16-run grid

Each run writes outputs/results/<arch>__<strategy>/{best.pt, history.json, results.json}.
outputs/results/summary.csv is rebuilt from every results.json after each run.
A run whose results.json already exists is skipped unless --force is given, so an
interrupted grid can be restarted. (Resuming inside a run is not implemented.)
"""
import argparse
import copy
import csv
import gc
import json
import random
import time
from pathlib import Path

import torch

from pestevision import settings
from pestevision.data import load_config
from pestevision.evaluate import evaluate
from pestevision.interfaces import build_components
from pestevision.train import fit

METRICS = ("accuracy", "precision_macro", "precision_weighted", "recall_macro",
           "recall_weighted", "f1_macro", "f1_weighted")
COLUMNS = (["architecture", "strategy", "lr", "batch_size", "epochs_run", "best_epoch",
            "val_accuracy"] + [f"test_{m}" for m in METRICS])


def parse_args(base, argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", nargs="+", choices=base["architectures"],
                        default=base["architectures"])
    parser.add_argument("--strategy", nargs="+", choices=base["augmentation_strategies"],
                        default=base["augmentation_strategies"])
    parser.add_argument("--provider", choices=("real", "dummy"), default="real")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, help="override training.max_epochs")
    parser.add_argument("--batch-size", type=int, help="override the Table 2 batch size (e.g. if you run out of GPU memory)")
    parser.add_argument("--num-workers", type=int, help="override data.num_workers")
    parser.add_argument("--out-dir", type=Path,
                        default=settings.root_path / base["paths"]["results_dir"])
    parser.add_argument("--force", action="store_true", help="re-run even if results.json exists")
    return parser.parse_args(argv)


def run_one(base, arch, strategy, args):
    run_dir = args.out_dir / f"{arch}__{strategy}"
    result_path = run_dir / "results.json"
    if result_path.exists() and not args.force:
        print(f"[skip] {arch} / {strategy}: results.json already exists")
        return

    hp = base["hyperparameters"][arch]
    cfg = copy.deepcopy(base)  # plain strings/numbers only: it is saved inside best.pt
    cfg.update(provider=args.provider, architecture=arch, strategy=strategy, num_classes=12)
    cfg["training"] = {**base["training"], "lr": hp["lr"]}
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    cfg["data"] = {**base["data"], "batch_size": args.batch_size or hp["batch_size"]}
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    print(f"\n=== {arch} / {strategy} | lr={cfg['training']['lr']} "
          f"batch={cfg['data']['batch_size']} device={args.device} provider={args.provider}")
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    started = time.time()
    model, loaders = build_components(cfg)
    history = fit(model, loaders, cfg, run_dir, args.device)  # restores the best weights
    val = evaluate(model, loaders["val"], cfg["num_classes"], args.device)
    test = evaluate(model, loaders["test"], cfg["num_classes"], args.device)

    results = {
        "architecture": arch, "strategy": strategy, "provider": args.provider,
        "lr": cfg["training"]["lr"], "batch_size": cfg["data"]["batch_size"],
        "epochs_run": len(history),
        "best_epoch": max(h["epoch"] for h in history if h["improved"]),
        "seconds": round(time.time() - started, 1),
        "val": val, "test": test,
    }
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"=== done: val_acc={val['accuracy']:.4f} test_acc={test['accuracy']:.4f}")


def write_summary(base, out_dir):
    arch_order = {a: i for i, a in enumerate(base["architectures"])}
    strat_order = {s: i for i, s in enumerate(base["augmentation_strategies"])}
    rows = []
    for path in out_dir.glob("*/results.json"):
        r = json.loads(path.read_text())
        row = {c: r[c] for c in ("architecture", "strategy", "lr", "batch_size",
                                 "epochs_run", "best_epoch")}
        row["val_accuracy"] = r["val"]["accuracy"]
        row.update({f"test_{m}": r["test"][m] for m in METRICS})
        rows.append(row)
    rows.sort(key=lambda r: (arch_order.get(r["architecture"], 99),
                             strat_order.get(r["strategy"], 99)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv=None):
    base = load_config(settings.config_path / "base.yaml")
    args = parse_args(base, argv)
    failed = []
    for arch in args.arch:
        for strategy in args.strategy:
            try:
                run_one(base, arch, strategy, args)
            except torch.cuda.OutOfMemoryError:
                print(f"[FAILED] {arch} / {strategy}: out of GPU memory. "
                      f"Try --batch-size with a smaller value, or run it on a bigger GPU.")
                failed.append(f"{arch}/{strategy}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_summary(base, args.out_dir)
    print(f"\nSummary: {args.out_dir / 'summary.csv'} ({write_summary(base, args.out_dir)} runs)")
    if failed:
        print("Runs that did not finish:", ", ".join(failed))


if __name__ == "__main__":  # the guard matters on Windows (DataLoader workers re-import this file)
    main()
