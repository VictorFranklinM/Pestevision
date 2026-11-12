"""Tests for per-image predictions. No dataset needed."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pestevision.predictions import (accuracy_from_predictions, collect_predictions,
                                     write_predictions_csv)


def _model_and_data(n=10):
    torch.manual_seed(0)
    images = torch.randn(n, 3, 4, 4)
    labels = torch.arange(n) % 3
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 3))
    return model, TensorDataset(images, labels), images, labels


class PredictionTests(unittest.TestCase):
    def test_order_and_probabilities_match_a_direct_forward_pass(self):
        model, dataset, images, labels = _model_and_data()
        result = collect_predictions(model, DataLoader(dataset, batch_size=4))  # batches 4+4+2
        with torch.no_grad():
            expected = model.eval()(images).softmax(1)
        torch.testing.assert_close(torch.tensor(result["probs"]), expected)
        self.assertEqual(result["labels"], labels.tolist())
        for row in result["probs"]:
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_shuffled_loader_is_rejected(self):
        model, dataset, _, _ = _model_and_data()
        with self.assertRaises(ValueError):
            collect_predictions(model, DataLoader(dataset, batch_size=4, shuffle=True))

    def test_csv_has_one_row_per_image_with_paths_classes_and_probabilities(self):
        samples = [(Path("/raw/ants/a.jpg"), 0), (Path("/raw/bees/b.jpg"), 1),
                   (Path("/raw/bees/c.jpg"), 1)]
        dataset = SimpleNamespace(samples=samples, classes=["ants", "bees"], raw_dir=Path("/raw"))
        predictions = {"labels": [0, 1, 1], "probs": [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]}
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "nested" / "predictions.csv"
            write_predictions_csv(target, predictions, dataset)
            with open(target, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["path"] for r in rows], ["ants/a.jpg", "bees/b.jpg", "bees/c.jpg"])
        self.assertEqual([r["correct"] for r in rows], ["1", "1", "0"])
        self.assertEqual((rows[2]["true_class"], rows[2]["pred_class"]), ("bees", "ants"))
        self.assertAlmostEqual(float(rows[2]["confidence"]), 0.7)
        self.assertAlmostEqual(float(rows[1]["prob_bees"]), 0.8)
        self.assertAlmostEqual(accuracy_from_predictions(predictions), 2 / 3)

    def test_predictions_that_do_not_line_up_with_the_dataset_are_rejected(self):
        dataset = SimpleNamespace(samples=[(Path("/raw/ants/a.jpg"), 0)],
                                  classes=["ants", "bees"], raw_dir=Path("/raw"))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                write_predictions_csv(Path(folder) / "p.csv",
                                      {"labels": [1], "probs": [[0.4, 0.6]]}, dataset)


if __name__ == "__main__":
    unittest.main()
