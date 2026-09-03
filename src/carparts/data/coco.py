"""In-memory COCO dataset model — the single canonical format every source is converted into.

Design rules (they matter for RF-DETR and for merging sources):

* Category ids are contiguous and start at **1** (COCO convention). RF-DETR derives its own contiguous
  label indices from the *train* split's ``categories`` list, so ids only have to be consistent
  across splits — never reuse an id for a different name.
* Every image records ``source`` (which dataset it came from), ``group`` (a leakage-group key: all
  augmented copies / near-duplicates of one physical photo share a group) and ``orig_path`` (where the
  pixels live on disk right now).
* ``bbox`` is COCO ``[x, y, w, h]`` in pixels; ``segmentation`` is a list of polygons
  ``[[x1, y1, x2, y2, ...], ...]`` in pixels, or ``None`` for box-only sources.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .naming import group_key


@dataclass
class Category:
    id: int
    name: str
    supercategory: str = "car_part"


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    source: str = ""
    group: str = ""
    orig_path: str = ""


@dataclass
class Annotation:
    id: int
    image_id: int
    category_id: int
    bbox: list[float]  # x, y, w, h (pixels)
    area: float
    iscrowd: int = 0
    segmentation: list[list[float]] | None = None

    @property
    def has_mask(self) -> bool:
        return bool(self.segmentation)


def polygon_area(poly: list[float]) -> float:
    """Shoelace area of a flat ``[x1, y1, x2, y2, ...]`` polygon (pixels)."""
    n = len(poly) // 2
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[2 * i], poly[2 * i + 1]
        x2, y2 = poly[2 * ((i + 1) % n)], poly[2 * ((i + 1) % n) + 1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_bbox(poly: list[float]) -> list[float]:
    xs, ys = poly[0::2], poly[1::2]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two COCO ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class CocoDataset:
    categories: list[Category] = field(default_factory=list)
    images: list[CocoImage] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def cat_by_name(self) -> dict[str, Category]:
        return {c.name: c for c in self.categories}

    @property
    def cat_by_id(self) -> dict[int, Category]:
        return {c.id: c for c in self.categories}

    @property
    def class_names(self) -> list[str]:
        return [c.name for c in sorted(self.categories, key=lambda c: c.id)]

    def anns_by_image(self) -> dict[int, list[Annotation]]:
        out: dict[int, list[Annotation]] = defaultdict(list)
        for a in self.annotations:
            out[a.image_id].append(a)
        return out

    def has_masks(self) -> bool:
        return any(a.has_mask for a in self.annotations)

    def validate(self) -> None:
        """Raise ``ValueError`` on anything that would silently corrupt training."""
        ids = [c.id for c in self.categories]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate category ids")
        if ids and (min(ids) < 1 or sorted(ids) != list(range(1, len(ids) + 1))):
            raise ValueError(f"category ids must be contiguous from 1, got {sorted(ids)}")
        names = [c.name for c in self.categories]
        if len(set(names)) != len(names):
            raise ValueError("duplicate category names")
        img_ids = [i.id for i in self.images]
        if len(set(img_ids)) != len(img_ids):
            raise ValueError("duplicate image ids")
        img_set, cat_set = set(img_ids), set(ids)
        ann_ids = [a.id for a in self.annotations]
        if len(set(ann_ids)) != len(ann_ids):
            raise ValueError("duplicate annotation ids")
        wh = {i.id: (i.width, i.height) for i in self.images}
        for a in self.annotations:
            if a.image_id not in img_set:
                raise ValueError(f"annotation {a.id} references missing image {a.image_id}")
            if a.category_id not in cat_set:
                raise ValueError(f"annotation {a.id} references missing category {a.category_id}")
            x, y, w, h = a.bbox
            W, H = wh[a.image_id]
            if w <= 0 or h <= 0 or x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                raise ValueError(f"annotation {a.id} has a degenerate/out-of-image bbox {a.bbox} for image {W}x{H}")
            if a.segmentation is not None:
                for poly in a.segmentation:
                    if len(poly) < 6 or len(poly) % 2:
                        raise ValueError(f"annotation {a.id} has a malformed polygon (len={len(poly)})")

    # -------------------------------------------------------------- transforms
    def remap_classes(self, class_map: Mapping[str, str | None] | None = None,
                      keep_order: Iterable[str] | None = None) -> "CocoDataset":
        """Rename / merge / drop classes.

        ``class_map`` maps old name -> new name (merge when several map to one) or -> ``None`` to drop.
        Unlisted classes keep their name. ``keep_order`` optionally fixes the final class order
        (names not in the dataset are appended as empty classes — useful to align to a shared taxonomy).
        Returns a new dataset with contiguous ids from 1.
        """
        class_map = dict(class_map or {})
        old_names = {c.id: c.name for c in self.categories}
        new_name_of_old = {cid: class_map.get(n, n) for cid, n in old_names.items()}
        final_names: list[str] = []
        if keep_order:
            final_names = list(keep_order)
        for cid in sorted(new_name_of_old):
            n = new_name_of_old[cid]
            if n is not None and n not in final_names:
                final_names.append(n)
        new_id = {n: i + 1 for i, n in enumerate(final_names)}
        cats = [Category(new_id[n], n, self.cat_by_name[n].supercategory if n in self.cat_by_name else "car_part")
                for n in final_names]
        anns = []
        for a in self.annotations:
            n = new_name_of_old[a.category_id]
            if n is None:
                continue
            anns.append(Annotation(a.id, a.image_id, new_id[n], list(a.bbox), a.area, a.iscrowd,
                                   None if a.segmentation is None else [list(p) for p in a.segmentation]))
        return CocoDataset(cats, [CocoImage(**asdict(i)) for i in self.images], anns, dict(self.info))

    def subset(self, image_ids: Iterable[int]) -> "CocoDataset":
        keep = set(image_ids)
        imgs = [i for i in self.images if i.id in keep]
        anns = [a for a in self.annotations if a.image_id in keep]
        return CocoDataset(list(self.categories), imgs, anns, dict(self.info))

    def strip_masks(self) -> "CocoDataset":
        """Box-only view (for detection recipes that mix mask and box sources)."""
        ds = self.subset(i.id for i in self.images)
        for a in ds.annotations:
            a.segmentation = None
        return ds

    def dedupe_annotations(self, iou_thr: float = 0.85) -> tuple["CocoDataset", int]:
        """Drop same-image, same-class annotations whose boxes overlap >= ``iou_thr`` (double-labelled objects,
        e.g. a re-annotated Roboflow export that kept the original labels). Mask-bearing / larger
        annotations win. Returns ``(dataset, n_removed)``."""
        keep: list[Annotation] = []
        removed = 0
        for anns in self.anns_by_image().values():
            by_cls: dict[int, list[Annotation]] = defaultdict(list)
            for a in anns:
                by_cls[a.category_id].append(a)
            for group in by_cls.values():
                group.sort(key=lambda a: (not a.has_mask, -a.area))
                kept: list[Annotation] = []
                for a in group:
                    if any(box_iou(a.bbox, k.bbox) >= iou_thr for k in kept):
                        removed += 1
                        continue
                    kept.append(a)
                keep += kept
        keep.sort(key=lambda a: a.id)
        ds = CocoDataset(list(self.categories), [CocoImage(**asdict(i)) for i in self.images],
                         [Annotation(a.id, a.image_id, a.category_id, list(a.bbox), a.area, a.iscrowd,
                                     None if a.segmentation is None else [list(p) for p in a.segmentation])
                          for a in keep], dict(self.info))
        return ds, removed

    @staticmethod
    def merge(datasets: list["CocoDataset"]) -> "CocoDataset":
        """Union of datasets. Classes are matched **by name**; image/annotation ids are re-issued."""
        names: list[str] = []
        for ds in datasets:
            for n in ds.class_names:
                if n not in names:
                    names.append(n)
        new_id = {n: i + 1 for i, n in enumerate(names)}
        out = CocoDataset([Category(new_id[n], n) for n in names])
        img_counter = ann_counter = 0
        for ds in datasets:
            old_names = {c.id: c.name for c in ds.categories}
            id_map: dict[int, int] = {}
            for im in ds.images:
                img_counter += 1
                id_map[im.id] = img_counter
                out.images.append(CocoImage(img_counter, im.file_name, im.width, im.height, im.source, im.group,
                                            im.orig_path))
            for a in ds.annotations:
                ann_counter += 1
                out.annotations.append(Annotation(ann_counter, id_map[a.image_id], new_id[old_names[a.category_id]],
                                                  list(a.bbox), a.area, a.iscrowd,
                                                  None if a.segmentation is None else [list(p) for p in a.segmentation]))
        return out

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict:
        per_class = Counter()
        masks = 0
        for a in self.annotations:
            per_class[self.cat_by_id[a.category_id].name] += 1
            masks += a.has_mask
        with_ann = {a.image_id for a in self.annotations}
        return {
            "images": len(self.images),
            "images_without_annotations": len(self.images) - len(with_ann),
            "annotations": len(self.annotations),
            "annotations_with_masks": masks,
            "classes": len(self.categories),
            "per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
            "sources": dict(Counter(i.source for i in self.images)),
            "groups": len({i.group for i in self.images}),
        }

    # --------------------------------------------------------------------- I/O
    def to_coco_dict(self, file_name_fn=None) -> dict:
        """Standard COCO JSON structure. ``file_name_fn(image) -> str`` lets the writer relocate files."""
        return {
            "info": {"description": self.info.get("description", "carparts canonical COCO"), **self.info},
            "licenses": self.info.get("licenses", []),
            "categories": [asdict(c) for c in sorted(self.categories, key=lambda c: c.id)],
            "images": [
                {"id": i.id, "file_name": file_name_fn(i) if file_name_fn else i.file_name,
                 "width": i.width, "height": i.height,
                 "extra": {"source": i.source, "group": i.group}}
                for i in self.images
            ],
            "annotations": [
                {"id": a.id, "image_id": a.image_id, "category_id": a.category_id,
                 "bbox": [round(v, 2) for v in a.bbox], "area": round(a.area, 2), "iscrowd": a.iscrowd,
                 **({"segmentation": [[round(v, 2) for v in p] for p in a.segmentation]} if a.segmentation else {})}
                for a in self.annotations
            ],
        }

    def save_json(self, path: str | Path, file_name_fn=None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_coco_dict(file_name_fn)), encoding="utf-8")
        return path

    @classmethod
    def from_coco_json(cls, path: str | Path, image_root: str | Path | None = None,
                       source: str = "") -> "CocoDataset":
        """Load a standard COCO json (e.g. a Roboflow export).

        Roboflow exports carry an *unannotated parent* category (id 0, e.g. ``engine-bay-parts``) that the
        real classes point to via ``supercategory``; it is dropped. Every other category is kept — even one
        with zero annotations in this split — so ids stay aligned across splits. Ids are renumbered
        contiguously from 1 preserving order."""
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        image_root = Path(image_root) if image_root else path.parent
        used = {a["category_id"] for a in d.get("annotations", [])}
        cats_raw = sorted(d["categories"], key=lambda c: c["id"])
        parents = {c.get("supercategory") for c in cats_raw} - {None, "none", ""}
        kept = [c for c in cats_raw
                if c["id"] in used or not (c["name"] in parents or (c["id"] == 0 and len(cats_raw) > 1))]
        if not kept:  # degenerate file: keep everything
            kept = cats_raw
        id_map = {c["id"]: i + 1 for i, c in enumerate(kept)}
        ds = cls([Category(id_map[c["id"]], str(c["name"]), str(c.get("supercategory", "car_part")))
                  for c in kept], info={"description": d.get("info", {}).get("description", path.name)})
        for im in d["images"]:
            extra = im.get("extra", {})
            ds.images.append(CocoImage(int(im["id"]), im["file_name"], int(im["width"]), int(im["height"]),
                                       source=extra.get("source", source) or source,
                                       group=extra.get("group", "") or group_key(im["file_name"]),
                                       orig_path=str(image_root / im["file_name"])))
        for a in d.get("annotations", []):
            if a["category_id"] not in id_map:
                continue
            seg = a.get("segmentation")
            polys = [list(map(float, p)) for p in seg] if isinstance(seg, list) and seg and isinstance(seg[0], list) \
                else None
            bbox = list(map(float, a["bbox"]))
            area = float(a.get("area") or (sum(polygon_area(p) for p in polys) if polys else bbox[2] * bbox[3]))
            ds.annotations.append(Annotation(int(a["id"]), int(a["image_id"]), id_map[a["category_id"]], bbox, area,
                                             int(a.get("iscrowd", 0)), polys))
        return ds
