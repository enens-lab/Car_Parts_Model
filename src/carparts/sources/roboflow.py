"""Generic Roboflow Universe source (COCO export). Needs ``ROBOFLOW_API_KEY`` (free account)."""
from __future__ import annotations

import os
from pathlib import Path

from ..constants import TAXONOMY_MAPS
from ..data.coco import CocoDataset
from .base import Source, SourceInfo


class RoboflowSource(Source):
    def __init__(self, name: str, workspace: str, project: str, version: int, info: SourceInfo,
                 taxonomy: str | None = None) -> None:
        self.name = name
        self.workspace, self.project, self.version = workspace, project, version
        self.info = info
        if "ROBOFLOW_API_KEY" not in self.info.requires_env:
            self.info.requires_env.append("ROBOFLOW_API_KEY")
        if taxonomy is not None and taxonomy not in TAXONOMY_MAPS:
            raise ValueError(f"unknown taxonomy {taxonomy!r}; choose from {sorted(TAXONOMY_MAPS)}")
        self.taxonomy = taxonomy  # "engine_bay" | "parts_catalog" -> normalise class names onto a canonical list

    def download(self, raw_root: Path) -> Path:
        out = self.raw_dir(raw_root)
        if (out / "train" / "_annotations.coco.json").exists():
            return out
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError(f"[{self.name}] ROBOFLOW_API_KEY not set (see .env.example)")
        try:
            from roboflow import Roboflow  # optional extra
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install 'carparts[roboflow]' to download Roboflow sources") from e
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{self.name}] downloading {self.workspace}/{self.project}/v{self.version} (coco) -> {out}")
        Roboflow(api_key=key).workspace(self.workspace).project(self.project).version(self.version) \
            .download("coco", location=str(out), overwrite=True)
        return out

    def load(self, raw_root: Path) -> dict[str, CocoDataset]:
        out = self.raw_dir(raw_root)
        result: dict[str, CocoDataset] = {}
        for split in ("train", "valid", "test"):
            ann = out / split / "_annotations.coco.json"
            if not ann.exists():
                continue
            ds = CocoDataset.from_coco_json(ann, image_root=ann.parent, source=self.name)
            if self.taxonomy:
                ds = ds.remap_classes(TAXONOMY_MAPS[self.taxonomy](ds.class_names, drop_unknown=True))
            result["val" if split == "valid" else split] = ds
        if not result:
            raise FileNotFoundError(f"[{self.name}] no _annotations.coco.json under {out}")
        return result
