"""Dataset / Tree presets from configs/*.yaml (replaces the case blocks in the .sh files)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset_config(name: str, root: Path = CONFIG_ROOT) -> Dict[str, Any]:
    key = str(name).strip().lower()
    for path in sorted((root / "datasets").glob("*.yaml")):
        cfg = _load_yaml(path)
        names = {str(cfg.get("dataset", path.stem)).lower(), *map(str.lower, cfg.get("aliases", []))}
        if key in names:
            return cfg
    raise ValueError(f"Unknown dataset {name!r}; add configs/datasets/<name>.yaml")


def load_tree_config(root: Path = CONFIG_ROOT) -> Dict[str, Any]:
    return _load_yaml(root / "tree.yaml")
