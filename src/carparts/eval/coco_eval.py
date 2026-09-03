"""Per-class COCO evaluation of a fine-tuned checkpoint on one processed split.

Runs the model at a low confidence threshold (so the PR curve is populated), converts the
``supervision.Detections`` into COCO result records (boxes + RLE masks) and scores them with pycocotools.
Classes are matched **by name** between the model's ``class_names`` and the split's ``categories`` so a
checkpoint trained on a merged/renamed taxonomy still evaluates correctly.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_DIRS = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def _load_image_rgb(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def _per_class_ap(coco_eval, cat_ids: list[int], names: dict[int, str]) -> dict[str, dict[str, float | None]]:
    """AP@[.5:.95] and AP@.5 per category from the accumulated precision tensor."""
    prec = coco_eval.eval["precision"]  # [T, R, K, A, M]
    iou_thrs = coco_eval.params.iouThrs
    t50 = int(np.argmin(np.abs(iou_thrs - 0.5)))
    out = {}
    for k, cid in enumerate(cat_ids):
        p_all = prec[:, :, k, 0, -1]
        p50 = prec[t50, :, k, 0, -1]
        ap = float(np.mean(p_all[p_all > -1])) if np.any(p_all > -1) else None
        ap50 = float(np.mean(p50[p50 > -1])) if np.any(p50 > -1) else None
        out[names[cid]] = {"AP": ap, "AP50": ap50}
    return out


def evaluate_checkpoint(checkpoint: str | Path, dataset_dir: str | Path, split: str = "test",
                        threshold: float = 0.05, max_images: int | None = None,
                        out_json: str | Path | None = None, model=None) -> dict[str, Any]:
    """Return ``{"summary": {...}, "per_class": {...}, "n_images": N}`` and optionally write it as JSON."""
    from pycocotools import mask as mask_util
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / SPLIT_DIRS[split]
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(ann_file)

    if model is None:
        from rfdetr import RFDETR
        model = RFDETR.from_checkpoint(str(checkpoint))
    model_names = list(model.class_names)

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_file))
    cat_by_name = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}
    id_to_name = {v: k for k, v in cat_by_name.items()}
    label_to_cat = {i: cat_by_name.get(n) for i, n in enumerate(model_names)}
    unmatched = [n for n in model_names if n not in cat_by_name]
    if unmatched:
        print(f"[eval] {len(unmatched)} model classes absent from split categories (ignored): {unmatched[:8]}")

    img_ids = sorted(gt.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]
    results_bbox: list[dict] = []
    results_segm: list[dict] = []
    any_masks = False
    for n, iid in enumerate(img_ids, 1):
        info = gt.loadImgs(iid)[0]
        img = _load_image_rgb(split_dir / info["file_name"])
        det = model.predict(img, threshold=threshold)
        if isinstance(det, list):
            det = det[0]
        h, w = img.shape[:2]
        masks = det.mask if getattr(det, "mask", None) is not None else None
        for j in range(len(det)):
            cid = label_to_cat.get(int(det.class_id[j]))
            if cid is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in det.xyxy[j]]
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
            score = float(det.confidence[j])
            results_bbox.append({"image_id": iid, "category_id": cid, "score": score,
                                 "bbox": [x0, y0, x1 - x0, y1 - y0]})
            if masks is not None:
                any_masks = True
                m = np.asfortranarray(masks[j].astype(np.uint8))
                rle = mask_util.encode(m)
                rle["counts"] = rle["counts"].decode("ascii")
                results_segm.append({"image_id": iid, "category_id": cid, "score": score, "segmentation": rle})
        if n % 50 == 0 or n == len(img_ids):
            print(f"[eval] {n}/{len(img_ids)} images", end="\r")
    print()

    cat_ids = sorted(gt.getCatIds())
    report: dict[str, Any] = {"checkpoint": str(checkpoint), "dataset_dir": str(dataset_dir), "split": split,
                              "n_images": len(img_ids), "threshold": threshold, "summary": {}, "per_class": {}}
    for iou_type, results in (("bbox", results_bbox), ("segm", results_segm)):
        if iou_type == "segm" and not any_masks:
            continue
        if not results:
            report["summary"][iou_type] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(results)
            ev = COCOeval(gt, dt, iou_type)
            ev.params.imgIds = img_ids
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats
        report["summary"][iou_type] = {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
                                       "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
                                       "AR100": float(s[8])}
        report["per_class"][iou_type] = _per_class_ap(ev, cat_ids, id_to_name)

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"split={report['split']} images={report['n_images']} ckpt={Path(report['checkpoint']).name}"]
    for iou_type, s in report["summary"].items():
        lines.append(f"  {iou_type:5s} AP={s['AP']:.3f} AP50={s['AP50']:.3f} AP75={s['AP75']:.3f} AR100={s['AR100']:.3f}")
    pc = report.get("per_class", {})
    if pc:
        first = next(iter(pc))
        names = sorted(pc[first], key=lambda n: -(pc[first][n]["AP"] or 0))
        head = f"  {'class':32s}" + "".join(f"{t + ' AP':>10s}{t + ' AP50':>12s}" for t in pc)
        lines.append(head)
        for n in names:
            row = f"  {n:32s}"
            for t in pc:
                ap, ap50 = pc[t][n]["AP"], pc[t][n]["AP50"]
                row += f"{(f'{ap:.3f}' if ap is not None else '  n/a'):>10s}{(f'{ap50:.3f}' if ap50 is not None else '  n/a'):>12s}"
            lines.append(row)
    return "\n".join(lines)
