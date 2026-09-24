from pathlib import Path

from pydantic import BaseModel


def _find_project_root(start: Path, marker: str = "pyproject.toml") -> Path:
    for parent in start.resolve().parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"no {marker} found above {start}; set AUGMENTATION_CONFIG_DIR"
    )


class Settings(BaseModel):
    root_path: Path = _find_project_root()
    config_path: Path = root_path / "config"
    data_path: Path = root_path / "data"


settings = Settings()
