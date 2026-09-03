"""YOLO label parsing (boxes *and* polygons) into the canonical :class:`CocoDataset`.

Handles both layouts in the wild:

* Ultralytics layout   ``<root>/images/<split>/x.jpg`` + ``<root>/labels/<split>/x.txt``
* Roboflow/RF-DETR     ``<root>/<split>/images/x.jpg`` + ``<root>/<split>/labels/x.txt``
* Flat                 ``<root>/images/x.jpg`` + ``<root>/labels/x.txt``  (no splits)

A label line is either ``cls cx cy w h`` (normalized box) or ``cls x1 y1 x2 y2 ... xn yn`` (normalized polygon,
>= 3 points). Roboflow exports name augmented copies ``<stem>_jpg.rf.<md5>.jpg`` — :func:`group_key` folds them
back onto the physical photo so the splitter can keep all copies on one side of the train/test line.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from .coco import Annotation, Category, CocoDataset, CocoImage, polygon_area, polygon_bbox
from .naming import IMAGE_EXTS, group_key  # noqa: F401  (re-exported)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)


def parse_label_file(path: Path, width: int, height: int, num_classes: int | None = None
                     ) -> list[tuple[int, list[float], list[list[float]] | None]]:
    """Return ``[(class_id, bbox_xywh_px, polygons_px | None), ...]`` for one label file."""
    out = []
    if not path.exists():
        return out
    for ln, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError as e:
            raise ValueError(f"{path}:{ln}: unparsable label line {raw!r}") from e
        if num_classes is not None and not 0 <= cls < num_classes:
            raise ValueError(f"{path}:{ln}: class id {cls} outside 0..{num_classes - 1}")
        if len(vals) == 4:  # box
            cx, cy, w, h = vals
            x0 = max(0.0, (cx - w / 2) * width)
            y0 = max(0.0, (cy - h / 2) * height)
            x1 = min(float(width), (cx + w / 2) * width)
            y1 = min(float(height), (cy + h / 2) * height)
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            out.append((cls, [x0, y0, x1 - x0, y1 - y0], None))
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon
            poly = []
            for i in range(0, len(vals), 2):
                poly += [min(max(vals[i], 0.0), 1.0) * width, min(max(vals[i + 1], 0.0), 1.0) * height]
            bbox = polygon_bbox(poly)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            out.append((cls, bbox, [poly]))
        else:
            raise ValueError(f"{path}:{ln}: expected 5 (box) or >=7 even (polygon) fields, got {len(parts)}")
    return out


def _discover(root: Path, splits: Iterable[str] | None) -> list[tuple[str, Path, Path]]:
    """Yield ``(split, images_dir, labels_dir)`` for whichever layout exists under ``root``."""
    found = []
    candidates = list(splits) if splits else ["train", "val", "valid", "test"]
    for s in candidates:
        if (root / "images" / s).is_dir():  # Ultralytics
            found.append((s, root / "images" / s, root / "labels" / s))
        elif (root / s / "images").is_dir():  # Roboflow
            found.append((s, root / s / "images", root / s / "labels"))
    if not found and (root / "images").is_dir():  # flat
        found.append(("all", root / "images", root / "labels"))
    if not found:
        raise FileNotFoundError(f"no YOLO layout found under {root}")
    return found


def write_yolo_dataset(splits: dict[str, CocoDataset], out_root: str | Path, task: str = "detection") -> Path:
    """Write ``{split: CocoDataset}`` as an Ultralytics-style tree + ``data.yaml`` (used only by the optional
    AGPL baseline script). Polygons are written when ``task == "segmentation"``, otherwise boxes."""
    import shutil

    import yaml

    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    names = None
    for split, ds in splits.items():
        names = names or ds.class_names
        img_dir, lbl_dir = out_root / "images" / split, out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        anns = ds.anns_by_image()
        for im in ds.images:
            src = Path(im.orig_path)
            dst = img_dir / f"{im.id:07d}{src.suffix.lower()}"
            try:
                import os
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            lines = []
            for a in anns.get(im.id, []):
                cls = a.category_id - 1
                if task == "segmentation" and a.segmentation:
                    poly = a.segmentation[0]
                    coords = [f"{min(max(v / (im.width if i % 2 == 0 else im.height), 0.0), 1.0):.6f}"
                              for i, v in enumerate(poly)]
                    lines.append(f"{cls} " + " ".join(coords))
                else:
                    x, y, w, h = a.bbox
                    lines.append(f"{cls} {(x + w / 2) / im.width:.6f} {(y + h / 2) / im.height:.6f} "
                                 f"{w / im.width:.6f} {h / im.height:.6f}")
            (lbl_dir / f"{im.id:07d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    data = {"path": str(out_root.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test" if "test" in splits else None,
            "names": {i: n for i, n in enumerate(names or [])}}
    (out_root / "data.yaml").write_text(yaml.safe_dump({k: v for k, v in data.items() if v is not None},
                                                       sort_keys=False), encoding="utf-8")
    return out_root


def load_yolo_pairs(pairs: dict[str, tuple[Path, Path]], class_names: list[str], source: str,
                    strict: bool = True) -> dict[str, CocoDataset]:
    """Convert explicit ``{split: (images_dir, labels_dir)}`` pairs into one :class:`CocoDataset` per split.

    Images without a label file are kept as background images (COCO/RF-DETR handle them). Classes are
    ``class_names[i]`` for YOLO id ``i``; ``strict`` rejects ids outside that range.
    """
    cats = [Category(i + 1, n) for i, n in enumerate(class_names)]
    result: dict[str, CocoDataset] = {}
    img_id = ann_id = 0
    for split, (img_dir, lbl_dir) in pairs.items():
        ds = CocoDataset([Category(c.id, c.name, c.supercategory) for c in cats],
                         info={"description": f"{source}/{split}", "source": source})
        for p in sorted(Path(img_dir).iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            w, h = image_size(p)
            img_id += 1
            ds.images.append(CocoImage(img_id, p.name, w, h, source=source, group=group_key(p.name),
                                       orig_path=str(p.resolve())))
            for cls, bbox, polys in parse_label_file(Path(lbl_dir) / (p.stem + ".txt"), w, h,
                                                     len(class_names) if strict else None):
                ann_id += 1
                area = sum(polygon_area(q) for q in polys) if polys else bbox[2] * bbox[3]
                ds.annotations.append(Annotation(ann_id, img_id, cls + 1, bbox, area, 0, polys))
        result["val" if split == "valid" else split] = ds
    return result


def load_yolo_dataset(root: str | Path, class_names: list[str], source: str,
                      splits: Iterable[str] | None = None, strict: bool = True) -> dict[str, CocoDataset]:
    """Convert a YOLO-format tree (Ultralytics / Roboflow / flat layout) into per-split datasets."""
    pairs = {split: (img_dir, lbl_dir) for split, img_dir, lbl_dir in _discover(Path(root), splits)}
    return load_yolo_pairs(pairs, class_names, source, strict)


def find_file_dirs(start: Path, exts: set[str], min_files: int = 1, max_depth: int = 6) -> list[Path]:
    """Directories under ``start`` that *directly* contain >= ``min_files`` files with the given extensions,
    largest first. Handles odd nestings such as Kaggle's ``images/images/*.jpg``."""
    start = Path(start)
    counts: dict[Path, int] = {}
    for p in start.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and len(p.relative_to(start).parts) <= max_depth:
            counts[p.parent] = counts.get(p.parent, 0) + 1
    return [d for d, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= min_files]
