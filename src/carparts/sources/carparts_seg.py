"""Ultralytics-hosted *carparts-seg* (Gianmarco Russo via Roboflow, CC BY 4.0) — 3,833 exterior photos,
23 classes, polygon masks. No credentials needed (public GitHub release asset, ~133 MB)."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

from ..constants import EXTERIOR_CLASSES
from ..data.coco import CocoDataset
from ..data.yolo import load_yolo_dataset
from .base import Source, SourceInfo

ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"


class CarpartsSegSource(Source):
    name = "carparts_seg"
    info = SourceInfo(
        title="Carparts-Seg (Ultralytics mirror of Roboflow 'car-seg' by Gianmarco Russo)",
        license="CC-BY-4.0",
        url="https://docs.ultralytics.com/datasets/segment/carparts-seg/",
        attribution="Gianmarco Russo, 'car-seg Dataset', Roboflow Universe, Nov 2023 (CC BY 4.0); "
                    "mirrored by Ultralytics at https://github.com/ultralytics/assets/releases.",
        notes="3,156/401/276 train/val/test; 23 classes; Roboflow augmentation copies leak across the public "
              "split (see artifacts/data_reports) — use the grouped split for honest numbers.",
    )

    def raw_dir(self, raw_root: Path) -> Path:
        return raw_root / "carparts-seg"

    def download(self, raw_root: Path) -> Path:
        out = self.raw_dir(raw_root)
        if (out / "images" / "train").exists():
            return out
        raw_root.mkdir(parents=True, exist_ok=True)
        zip_path = raw_root / "carparts-seg.zip"
        if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
            print(f"[carparts_seg] downloading {ZIP_URL} -> {zip_path}")
            urllib.request.urlretrieve(ZIP_URL, zip_path)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            # the archive is flat (images/, labels/, carparts-seg.yaml) — extract into our folder
            z.extractall(out)
        # tolerate a future re-packaging with a top-level folder
        nested = out / "carparts-seg"
        if nested.is_dir() and not (out / "images").exists():
            for child in nested.iterdir():
                shutil.move(str(child), out / child.name)
            nested.rmdir()
        return out

    def class_names(self, raw_root: Path) -> list[str]:
        y = self.raw_dir(raw_root) / "carparts-seg.yaml"
        if y.exists():
            names = yaml.safe_load(y.read_text(encoding="utf-8")).get("names")
            if isinstance(names, dict):
                return [names[i] for i in sorted(names)]
            if isinstance(names, list):
                return list(names)
        return list(EXTERIOR_CLASSES)

    def load(self, raw_root: Path) -> dict[str, CocoDataset]:
        root = self.raw_dir(raw_root)
        return load_yolo_dataset(root, self.class_names(raw_root), source=self.name, splits=["train", "val", "test"])
