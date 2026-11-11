"""Checks for the transform factory and the Table 2 settings. No dataset needed."""
import unittest

import torch
from PIL import Image

from pestevision import settings
from pestevision.augmentation import get_transform
from pestevision.data import load_config


def _image():
    generator = torch.Generator().manual_seed(0)
    pixels = (torch.rand(90, 120, 3, generator=generator) * 255).byte().numpy()
    return Image.fromarray(pixels)


class GlueTests(unittest.TestCase):
    def test_every_strategy_and_split_gives_a_batchable_tensor(self):
        for strategy in ("no_da", "da1", "da2", "da3"):
            for split in ("train", "val", "test"):
                with self.subTest(strategy=strategy, split=split):
                    out = get_transform(strategy, split)(_image())
                    self.assertIs(type(out), torch.Tensor)
                    self.assertEqual(tuple(out.shape), (3, 224, 224))
                    self.assertEqual(out.dtype, torch.float32)

    def test_val_and_test_transforms_are_deterministic(self):
        transform = get_transform("da1", "val")
        torch.testing.assert_close(transform(_image()), transform(_image()))

    def test_train_transform_is_random(self):
        transform = get_transform("da1", "train")
        self.assertFalse(torch.equal(transform(_image()), transform(_image())))

    def test_unknown_names_are_rejected(self):
        with self.assertRaises(ValueError):
            get_transform("da9", "train")
        with self.assertRaises(ValueError):
            get_transform("da1", "validation")

    def test_table2_hyperparameters(self):
        base = load_config(settings.config_path / "base.yaml")
        expected = {"alexnet": (1e-5, 128), "resnet50": (1e-5, 16),
                    "efficientnet_b4": (1e-4, 16), "vit_b_16": (1e-5, 32)}
        for arch, (lr, batch) in expected.items():
            self.assertEqual(base["hyperparameters"][arch], {"lr": lr, "batch_size": batch})


if __name__ == "__main__":
    unittest.main()
