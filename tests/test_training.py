import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pestevision.dummy import build_dataloaders, build_model
from pestevision.evaluate import classification_metrics, evaluate, run_epoch
from pestevision.train import EarlyStopping, fit


class TrainingTests(unittest.TestCase):
    def test_metrics_imbalanced_and_missing_class(self):
        result = classification_metrics([[2, 1, 0], [0, 1, 0], [0, 0, 0]])
        expected = dict(accuracy=3/4, precision_macro=1/2,
                        precision_weighted=7/8, recall_macro=5/9,
                        recall_weighted=3/4, f1_macro=22/45, f1_weighted=23/30)
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value)
        with self.assertRaises(ValueError):
            classification_metrics(torch.zeros(3, 3))

    def test_invalid_confusion_counts(self):
        for matrix in ([[0.5]], [[-1]], [[float("nan")]], [[1, 2]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                classification_metrics(matrix)

    def test_stopping_tolerance(self):
        stopper = EarlyStopping(patience=2, min_delta=0.1)
        self.assertEqual(stopper.update(1.0), (True, False))
        self.assertEqual(stopper.update(0.95), (False, False))
        self.assertEqual(stopper.update(0.85), (True, False))
        self.assertEqual(stopper.update(0.84), (False, False))
        self.assertEqual(stopper.update(0.83), (False, True))
        for delta in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                EarlyStopping(min_delta=delta)

    def test_invalid_epoch_inputs(self):
        model = nn.Linear(2, 2)
        for images, labels in (
            (torch.empty(0, 2), torch.empty(0, dtype=torch.long)),
            (torch.zeros(2, 2), torch.zeros(1, dtype=torch.long)),
            (torch.zeros(1, 2), torch.tensor([2])),
            (torch.zeros(1, 2), torch.tensor([0.])),
        ):
            with self.subTest(shape=images.shape), self.assertRaises(ValueError):
                run_epoch(model, [(images, labels)], 2)
        with self.assertRaises(ValueError):
            run_epoch(model, [], 0)

    def test_patience_boundaries_and_reset(self):
        stopper = EarlyStopping()
        self.assertEqual(stopper.update(1.0), (True, False))
        for _ in range(20):
            self.assertEqual(stopper.update(1.0), (False, False))
        self.assertEqual(stopper.update(1.0), (False, True))
        self.assertEqual(stopper.update(0.5), (True, False))
        self.assertEqual(stopper.bad_epochs, 0)

    def test_full_loop_schedule_stop_and_best_restoration(self):
        cfg = {"num_classes": 2, "training": {"max_epochs": 40}}
        model = nn.Linear(1, 2)
        def epoch(model, loader, num_classes, device, optimizer=None):
            if optimizer is not None:
                with torch.no_grad():
                    model.weight.add_(1)
            return {"loss": 1.0}
        initial = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory, patch("pestevision.train.run_epoch", side_effect=epoch):
            history = fit(model, {"train": [], "val": []}, cfg, directory)
            self.assertEqual(len(history), 22)
            self.assertEqual(history[10]["next_lr"], 0.001)
            self.assertAlmostEqual(history[11]["next_lr"], 0.0001)
            self.assertTrue(history[-1]["early_stopped"])
            torch.testing.assert_close(model.weight, initial + 1)

    def test_best_checkpoint_includes_improvements_below_tolerance(self):
        cfg = {"num_classes": 2, "training": {
            "max_epochs": 3, "min_delta": 0.1, "early_stopping_patience": 2}}
        model = nn.Linear(1, 2)
        losses = iter([1.0, 0.99, 0.98])
        def epoch(model, loader, num_classes, device, optimizer=None):
            if optimizer is not None:
                with torch.no_grad():
                    model.weight.add_(1)
                return {"loss": 1.0}
            return {"loss": next(losses)}
        initial = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory, patch("pestevision.train.run_epoch", side_effect=epoch):
            history = fit(model, {"train": [], "val": []}, cfg, directory)
            self.assertTrue(history[-1]["early_stopped"])
            self.assertTrue(history[-1]["improved"])
            torch.testing.assert_close(model.weight, initial + 3)

    def test_evaluation_uneven_batches_no_grad(self):
        logits = torch.tensor([[3., 0.], [0., 3.], [2., 0.]])
        labels = torch.tensor([0, 1, 1])
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = evaluate(model, loader, 2)
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertAlmostEqual(result["loss"], nn.functional.cross_entropy(logits, labels).item(), places=6)
        self.assertIsNone(model.weight.grad)
        self.assertFalse(model.training)
        with self.assertRaises(ValueError):
            evaluate(model, [], 2)

    def test_real_gradient_updates_and_dummy_repeatability(self):
        cfg = {"seed": 42, "num_classes": 3}
        loaders = build_dataloaders(cfg)
        again = build_dataloaders(cfg)
        torch.testing.assert_close(loaders["val"].dataset.tensors[0], again["val"].dataset.tensors[0])
        self.assertFalse(torch.equal(loaders["val"].dataset.tensors[0], loaders["test"].dataset.tensors[0]))
        model = build_model(cfg)
        before = model[-1].weight.detach().clone()
        run_epoch(model, loaders["train"], 3, optimizer=torch.optim.Adam(model.parameters()))
        self.assertFalse(torch.equal(before, model[-1].weight))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
