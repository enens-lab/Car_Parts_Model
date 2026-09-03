"""Product-facing inference API.

>>> exterior = CarPartsModel("artifacts/runs/exterior_seg_m/rfdetr/checkpoint_best_total.pth")
>>> pred = exterior.predict("photo.jpg", threshold=0.5)
>>> pred.to_dict()["objects"][0]
{'label': 'hood', 'confidence': 0.93, 'bbox_xyxy': [..], 'polygon': [[x, y], ...], 'area_px': 12345}

``CarPartsPipeline`` runs several models (exterior masks + engine-bay boxes) on the same image and merges
their outputs, tagging each object with the model that produced it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _to_rgb(image: Any) -> np.ndarray:
    """Accept a path, PIL image or HxWx3 uint8 array (RGB) and return an RGB numpy array."""
    from PIL import Image, ImageOps
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {arr.shape}")
    return arr


@dataclass
class Prediction:
    image_hw: tuple[int, int]
    detections: Any                       # supervision.Detections
    class_names: list[str]
    model_tag: str = ""
    source: str = ""

    def __len__(self) -> int:
        return len(self.detections)

    def to_dict(self, include_polygons: bool = True, simplify_px: float = 1.5) -> dict[str, Any]:
        import supervision as sv
        objs = []
        det = self.detections
        masks = det.mask if getattr(det, "mask", None) is not None else None
        for i in range(len(det)):
            x0, y0, x1, y1 = [round(float(v), 1) for v in det.xyxy[i]]
            o: dict[str, Any] = {
                "label": self.class_names[int(det.class_id[i])] if det.class_id is not None else str(i),
                "confidence": round(float(det.confidence[i]), 4) if det.confidence is not None else None,
                "bbox_xyxy": [x0, y0, x1, y1],
            }
            if self.model_tag:
                o["model"] = self.model_tag
            if masks is not None:
                m = masks[i]
                o["area_px"] = int(m.sum())
                if include_polygons:
                    polys = sv.mask_to_polygons(m)
                    if polys:
                        best = max(polys, key=len)
                        if simplify_px > 0:
                            best = _simplify(best, simplify_px)
                        o["polygon"] = [[int(x), int(y)] for x, y in best.tolist()]
            objs.append(o)
        return {"source": self.source, "height": self.image_hw[0], "width": self.image_hw[1], "objects": objs}

    def to_json(self, path: str | Path | None = None, **kw) -> str:
        s = json.dumps(self.to_dict(**kw), indent=1)
        if path:
            Path(path).write_text(s, encoding="utf-8")
        return s


def _simplify(poly: np.ndarray, eps: float) -> np.ndarray:
    try:
        import cv2
        approx = cv2.approxPolyDP(poly.astype(np.int32).reshape(-1, 1, 2), eps, True)
        return approx.reshape(-1, 2)
    except Exception:  # pragma: no cover
        return poly


class CarPartsModel:
    """One fine-tuned RF-DETR checkpoint (detection or segmentation)."""

    def __init__(self, checkpoint: str | Path, tag: str = "", threshold: float = 0.5,
                 optimize: bool = False, half: bool = False) -> None:
        from rfdetr import RFDETR
        self.checkpoint = str(checkpoint)
        self.tag = tag or Path(checkpoint).parent.parent.name
        self.threshold = threshold
        self.model = RFDETR.from_checkpoint(self.checkpoint)
        self.class_names: list[str] = list(self.model.class_names)
        if optimize:
            try:
                self.model.optimize_for_inference(compile=False, dtype="float16" if half else None)
            except Exception as e:  # pragma: no cover - optional path
                print(f"[{self.tag}] optimize_for_inference skipped: {e}")

    @property
    def is_segmentation(self) -> bool:
        return bool(getattr(getattr(self.model, "model_config", None), "segmentation_head", False))

    def predict(self, image: Any, threshold: float | None = None, source: str = "") -> Prediction:
        rgb = _to_rgb(image)
        det = self.model.predict(rgb, threshold=self.threshold if threshold is None else threshold)
        if isinstance(det, list):
            det = det[0]
        return Prediction(rgb.shape[:2], det, self.class_names, self.tag,
                          source or (str(image) if isinstance(image, (str, Path)) else ""))

    def annotate(self, image: Any, pred: Prediction, thickness: int = 2) -> np.ndarray:
        """Return an annotated **RGB** array (boxes, masks when present, labels)."""
        import supervision as sv
        rgb = _to_rgb(image).copy()
        det = pred.detections
        labels = [f"{pred.class_names[int(c)]} {conf:.2f}" for c, conf in zip(det.class_id, det.confidence)]
        if getattr(det, "mask", None) is not None:
            rgb = sv.MaskAnnotator(opacity=0.4).annotate(rgb, det)
        rgb = sv.BoxAnnotator(thickness=thickness).annotate(rgb, det)
        rgb = sv.LabelAnnotator(text_scale=0.5, text_thickness=1).annotate(rgb, det, labels)
        return rgb


@dataclass
class CarPartsPipeline:
    """Run several models on the same image and merge results (e.g. exterior masks + engine-bay boxes)."""
    models: dict[str, CarPartsModel] = field(default_factory=dict)

    @classmethod
    def from_checkpoints(cls, **tag_to_checkpoint: str | Path) -> "CarPartsPipeline":
        return cls({tag: CarPartsModel(ckpt, tag=tag) for tag, ckpt in tag_to_checkpoint.items()})

    def predict(self, image: Any, threshold: float | None = None) -> dict[str, Any]:
        rgb = _to_rgb(image)
        out: dict[str, Any] = {"source": str(image) if isinstance(image, (str, Path)) else "",
                               "height": rgb.shape[0], "width": rgb.shape[1], "objects": []}
        for tag, m in self.models.items():
            out["objects"] += m.predict(rgb, threshold=threshold).to_dict()["objects"]
        out["objects"].sort(key=lambda o: -(o["confidence"] or 0))
        return out
