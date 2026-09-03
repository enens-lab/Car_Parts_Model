"""OPTIONAL AGPL-3.0 BASELINE — internal benchmarking only.

Ultralytics YOLO (the `ultralytics` package AND its pretrained/derived weights) is licensed AGPL-3.0.
Shipping, hosting, or embedding anything produced by this script in a commercial product requires an
Ultralytics Enterprise License (https://www.ultralytics.com/license). This script exists solely to
compare our Apache-2.0 RF-DETR models against the YOLO family on the *same* leakage-safe split.

  uv sync --extra baseline-agpl
  python scripts/baseline_ultralytics.py --recipe exterior_seg --model yolo26n-seg.pt --epochs 50 --i-accept-agpl
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from carparts.config import load_paths
from carparts.data.coco import CocoDataset
from carparts.data.yolo import write_yolo_dataset

BANNER = "\n".join([
    "=" * 100,
    "  AGPL-3.0 BASELINE. Outputs of this script (weights, exports, metrics used in marketing) are NOT",
    "  cleared for commercial use without an Ultralytics Enterprise License. Re-run with --i-accept-agpl",
    "  to confirm this is an internal benchmark only.",
    "=" * 100,
])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--model", default="yolo26n-seg.pt", help="e.g. yolo26n-seg.pt, yolo26m-seg.pt, yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name")
    ap.add_argument("--i-accept-agpl", action="store_true", dest="accept")
    args = ap.parse_args(argv)

    print(BANNER)
    if not args.accept:
        return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed: uv sync --extra baseline-agpl")

    paths = load_paths().ensure()
    proc = paths.processed / args.recipe
    card = json.loads((proc / "dataset_card.json").read_text(encoding="utf-8"))
    splits = {}
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        ann = proc / folder / "_annotations.coco.json"
        if ann.exists():
            splits[split] = CocoDataset.from_coco_json(ann, image_root=ann.parent)
    yolo_root = write_yolo_dataset(splits, paths.processed / f"{args.recipe}__yolo_agpl", task=card["task"])
    name = args.name or f"{args.recipe}_{Path(args.model).stem}"
    project = paths.runs / "baseline_agpl"

    model = YOLO(args.model)
    model.train(data=str(yolo_root / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=str(project), name=name, exist_ok=True, workers=2, plots=False)
    metrics = model.val(data=str(yolo_root / "data.yaml"), split="test", imgsz=args.imgsz, batch=args.batch,
                        project=str(project), name=f"{name}_test", exist_ok=True, plots=False)
    summary = {"recipe": args.recipe, "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz,
               "license": "AGPL-3.0 (Ultralytics) — internal benchmark only",
               "test": {"box_mAP50-95": float(metrics.box.map), "box_mAP50": float(metrics.box.map50)}}
    if getattr(metrics, "seg", None) is not None and card["task"] == "segmentation":
        summary["test"].update({"mask_mAP50-95": float(metrics.seg.map), "mask_mAP50": float(metrics.seg.map50)})
    (project / name / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
