"""Inference wrapper for the isolated-part classifier (see carparts.train.classifier)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .predictor import _to_rgb


class CarPartsClassifier:
    """Top-k part identification for a photo of a single component."""

    def __init__(self, checkpoint: str | Path, tag: str = "", threshold: float = 0.2, topk: int = 3) -> None:
        import torch
        from ..train.classifier import MEAN, STD, build_classifier

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck.get("kind") != "classifier":
            raise ValueError(f"{checkpoint} is not a classifier checkpoint")
        self.checkpoint, self.tag = str(checkpoint), tag or Path(checkpoint).parent.name
        self.class_names: list[str] = list(ck["class_names"])
        self.resolution = int(ck["resolution"])
        self.threshold, self.topk = threshold, topk
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_classifier(ck["arch"], len(self.class_names), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self._mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(STD).view(1, 3, 1, 1).to(self.device)

    is_segmentation = False

    def _preprocess(self, rgb: np.ndarray):
        import torch
        from PIL import Image
        im = Image.fromarray(rgb)
        s = int(self.resolution * 1.14)
        w, h = im.size
        scale = s / min(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
        w, h = im.size
        l, t = (w - self.resolution) // 2, (h - self.resolution) // 2
        im = im.crop((l, t, l + self.resolution, t + self.resolution))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
        return (x - self._mean) / self._std

    def predict_objects(self, image: Any, threshold: float | None = None, topk: int | None = None) -> list[dict]:
        import torch
        rgb = _to_rgb(image)
        thr = self.threshold if threshold is None else threshold
        k = min(topk or self.topk, len(self.class_names))
        with torch.no_grad():
            probs = torch.softmax(self.model(self._preprocess(rgb)).float(), dim=1)[0]
        vals, idx = probs.topk(k)
        h, w = rgb.shape[:2]
        out = []
        for rank, (p, i) in enumerate(zip(vals.tolist(), idx.tolist())):
            if rank > 0 and p < thr:
                break
            out.append({"label": self.class_names[i], "confidence": round(p, 4), "bbox_xyxy": [0, 0, w, h],
                        "model": self.tag, "kind": "classification", "rank": rank + 1})
        return out

    def annotate(self, image: Any, objects: list[dict]) -> np.ndarray:
        import cv2
        rgb = _to_rgb(image).copy()
        for i, o in enumerate(objects[:3]):
            txt = f"{o['label']} {o['confidence']:.2f}"
            y = 28 + 26 * i
            cv2.rectangle(rgb, (6, y - 20), (12 + 9 * len(txt), y + 6), (20, 20, 20), -1)
            cv2.putText(rgb, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return rgb


def is_classifier_checkpoint(path: str | Path) -> bool:
    return Path(path).name.startswith("classifier")
