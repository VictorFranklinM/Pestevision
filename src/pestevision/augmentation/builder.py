from collections.abc import Sequence
from numbers import Number
from pathlib import Path
from typing import Self

import yaml
from torchvision.transforms import v2

from pestevision import settings
from pestevision.augmentation.config import AugmentationConfig, FlipAxis


class AugmentationBuilder:
    def __init__(self):
        self.__pipeline: list[v2.Transform] = []

    def add(self, transform: v2.Transform) -> Self:
        self.__pipeline.append(transform)
        return self

    def crop(self, size: int, scale: tuple[float, float] = (0.08, 1)) -> Self:
        return self.add(v2.RandomResizedCrop(size, scale))

    def flip(self, axis: FlipAxis, probability: float = 0.5) -> Self:
        match axis:
            case "horizontal":
                return self.add(v2.RandomHorizontalFlip(probability))
            case "vertical":
                return self.add(v2.RandomVerticalFlip(probability))
            case _:
                raise ValueError(
                    f"axis must be 'horizontal' or 'vertical', got {axis!r}"
                )

    def rotate(self, degrees: Number) -> Self:
        return self.add(v2.RandomRotation(degrees))

    def color_jitter(
        self,
        brightness: float | None = None,
        contrast: float | None = None,
        saturation: float | None = None,
        hue: float | None = None,
    ) -> Self:
        return self.add(v2.ColorJitter(brightness, contrast, saturation, hue))

    def erase(self, probability: float, scale: Sequence[float]) -> Self:
        return self.add(v2.RandomErasing(probability, scale))

    @classmethod
    def from_config(cls, name_or_path: str | Path) -> Self:
        path = Path(name_or_path)
        if path.suffix not in {".yaml", ".yml"}:
            path = settings.config_path / "augmentation" / f"{path}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"augmentation config not found: {path}")

        with open(path) as f:
            config = AugmentationConfig.model_validate(yaml.safe_load(f) or {})

        builder = cls()
        for step in config.steps:
            getattr(builder, step.type)(**step.model_dump(exclude={"type"}))
        return builder

    def build(self) -> v2.Compose:
        return v2.Compose([*self.__pipeline])
