"""Kaggle: *Car Engine Bay Images with YOLO Annotations* (Khaled Chawa, MIT) — 1,201 photos, 26 classes,
boxes only. Downloaded with ``kagglehub`` (works anonymously for this public dataset; ``KAGGLE_USERNAME`` +
``KAGGLE_KEY`` or ``KAGGLE_API_TOKEN`` are used when present).

On disk the archive is double-nested (``images/images/N.jpg`` + ``labels/labels/N.txt``), so we locate the
folders that actually hold the files instead of trusting the layout."""
from __future__ import annotations

import os
from pathlib import Path

from ..constants import ENGINE_BAY_CLASSES
from ..data.coco import CocoDataset
from ..data.yolo import IMAGE_EXTS, find_file_dirs, load_yolo_pairs
from .base import Source, SourceInfo

KAGGLE_SLUG = "khaledchawa/car-engine-bay-pictures"


class KaggleEngineBaySource(Source):
    name = "kaggle_engine_bay"
    info = SourceInfo(
        title="Car Engine Bay Images with YOLO Annotations (Kaggle, Khaled Chawa)",
        license="MIT",
        url="https://www.kaggle.com/datasets/khaledchawa/car-engine-bay-pictures",
        attribution="Khaled Chawa, 'Car Engine Bay Images with YOLO Annotations', Kaggle, 2024 (MIT).",
        notes="1,201 engine-bay photos, 26 box classes; two class-name typos fixed in our taxonomy.",
        requires_env=[],  # public dataset — credentials optional
    )

    def is_downloaded(self, raw_root: Path) -> bool:
        d = self.raw_dir(raw_root)
        return d.exists() and any(d.rglob("*.txt"))

    def download(self, raw_root: Path) -> Path:
        out = self.raw_dir(raw_root)
        if self.is_downloaded(raw_root):
            return out
        try:
            import kagglehub  # optional extra
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install 'carparts[kaggle]' (kagglehub) to download the Kaggle source") from e
        out.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("KAGGLEHUB_CACHE", str(out / "_cache"))
        print(f"[kaggle_engine_bay] kagglehub.dataset_download({KAGGLE_SLUG!r}) -> {out}")
        path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        (out / "DOWNLOADED_FROM.txt").write_text(f"{KAGGLE_SLUG}\n{path}\n", encoding="utf-8")
        return out

    def load(self, raw_root: Path) -> dict[str, CocoDataset]:
        root = self.raw_dir(raw_root)
        # kagglehub caches deep: _cache/datasets/<owner>/<slug>/versions/<n>/images/images/*.jpg
        img_dirs = find_file_dirs(root, IMAGE_EXTS, min_files=50, max_depth=16)
        lbl_dirs = find_file_dirs(root, {".txt"}, min_files=50, max_depth=16)
        if not img_dirs or not lbl_dirs:
            raise FileNotFoundError(f"[kaggle_engine_bay] no image/label folders with >=50 files under {root}")
        pairs = {"all": (img_dirs[0], lbl_dirs[0])}
        return load_yolo_pairs(pairs, list(ENGINE_BAY_CLASSES), source=self.name)
