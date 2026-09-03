"""Paths, secrets and YAML configs. One place, no surprises.

* ``configs/paths.yaml`` — where data / artifacts live (override with env ``CARPARTS_PATHS=/path/to.yaml``).
* ``.env`` (git-ignored) — KAGGLE_USERNAME/KAGGLE_KEY, ROBOFLOW_API_KEY. Loaded once, never printed.
* ``configs/recipes/*.yaml`` — which sources, class map, split policy -> one processed dataset.
* ``configs/train/*.yaml`` — model + hyper-parameters for one hardware budget.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_env() -> None:
    """Load ``<project>/.env`` into ``os.environ`` (existing variables win)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    artifacts: Path
    runs: Path
    reports: Path
    exports: Path

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.runs, self.reports, self.exports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_paths(path: str | Path | None = None) -> Paths:
    cfg_path = Path(path or os.environ.get("CARPARTS_PATHS") or CONFIG_DIR / "paths.yaml")
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {} if cfg_path.exists() else {}
    artifacts = _abs(cfg.get("artifacts", "artifacts"))
    return Paths(
        root=PROJECT_ROOT,
        raw=_abs(cfg.get("raw", "data/raw")),
        processed=_abs(cfg.get("processed", "data/processed")),
        artifacts=artifacts,
        runs=_abs(cfg.get("runs", artifacts / "runs")),
        reports=_abs(cfg.get("reports", artifacts / "data_reports")),
        exports=_abs(cfg.get("exports", artifacts / "exports")),
    )


def _load_named_yaml(kind: str, name_or_path: str | Path) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / kind / f"{name_or_path}.yaml"
    elif not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / kind / p
    if not p.exists():
        available = sorted(x.stem for x in (CONFIG_DIR / kind).glob("*.yaml"))
        raise FileNotFoundError(f"{kind} config {name_or_path!r} not found. Available: {available}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("name", p.stem)
    data["_path"] = str(p)
    return data


def load_recipe(name_or_path: str | Path) -> dict[str, Any]:
    r = _load_named_yaml("recipes", name_or_path)
    r.setdefault("task", "detection")
    if r["task"] not in ("detection", "segmentation", "classification"):
        raise ValueError(f"recipe.task must be detection|segmentation|classification, got {r['task']!r}")
    split = r.setdefault("split", {})
    split.setdefault("strategy", "grouped")
    split.setdefault("fractions", {"train": 0.8, "val": 0.1, "test": 0.1})
    split.setdefault("seed", 2026)
    split.setdefault("dedupe_hash", True)
    r.setdefault("sources", [])
    r.setdefault("classes", None)
    r.setdefault("min_instances_per_class", 0)
    return r


def load_train_config(name_or_path: str | Path) -> dict[str, Any]:
    return _load_named_yaml("train", name_or_path)
