from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


def _ordered(v: tuple[float, float]) -> tuple[float, float]:
    if v[0] > v[1]:
        raise ValueError(f"min must be <= max, got {v}")
    return v


FlipAxis = Literal["horizontal", "vertical"]
Probability = Annotated[float, Field(ge=0, le=1)]
FractionRange = Annotated[tuple[Probability, Probability], AfterValidator(_ordered)]


class _Step(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CropStep(_Step):
    type: Literal["crop"]
    size: PositiveInt = 224
    scale: FractionRange = (0.8, 1.0)


class FlipStep(_Step):
    type: Literal["flip"]
    axis: FlipAxis = "horizontal"
    probability: Probability = 0.5


class RotateStep(_Step):
    type: Literal["rotate"]
    degrees: Annotated[float, Field(ge=0, le=180)] = 15


class ColorJitterStep(_Step):
    type: Literal["color_jitter"]
    brightness: NonNegativeFloat | None = None
    contrast: NonNegativeFloat | None = None
    saturation: NonNegativeFloat | None = None
    hue: Annotated[float, Field(ge=0, le=0.5)] | None = None


class EraseStep(_Step):
    type: Literal["erase"]
    probability: Probability = 0.5
    scale: FractionRange = (0.02, 0.25)


Step = Annotated[
    CropStep | FlipStep | RotateStep | ColorJitterStep | EraseStep,
    Field(discriminator="type"),
]


class AugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    normalize: bool = True
    steps: list[Step] = Field(min_length=1)

    @field_validator("steps", mode="before")
    @classmethod
    def _unwrap_steps(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        out = []
        for entry in v:
            if isinstance(entry, str):
                out.append({"type": entry})
            elif isinstance(entry, dict) and len(entry) == 1:
                ((name, params),) = entry.items()
                if params is not None and not isinstance(params, dict):
                    raise ValueError(
                        f"parameters of {name!r} must be a mapping, got {params!r}"
                    )
                out.append({"type": name, **(params or {})})
            else:
                out.append(entry)
        return out

    @model_validator(mode="after")
    def _needs_crop(self) -> "AugmentationConfig":
        if not any(isinstance(s, CropStep) for s in self.steps):
            raise ValueError(
                "pipeline needs a crop step, or batched images won't share a size"
            )
        return self
