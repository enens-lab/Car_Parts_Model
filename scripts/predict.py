"""Run one or more fine-tuned models on an image, a folder of images, or a video.

Examples
--------
  python scripts/predict.py --run exterior_seg_m --source photo.jpg --out artifacts/preds
  python scripts/predict.py --run exterior_seg_m --run engine_bay_m --source ./photos --threshold 0.4
  python scripts/predict.py --checkpoint a.pth --source clip.mp4 --out artifacts/preds

Writes <name>.json (structured objects: label, confidence, bbox, polygon) and <name>.jpg (annotated).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from carparts.config import load_paths
from carparts.infer import CarPartsModel, CarPartsPipeline
from carparts.train.rfdetr_trainer import best_checkpoint

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _resolve_checkpoints(args, paths) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for run in args.run or []:
        out[run] = best_checkpoint(paths.runs / run / "rfdetr")
    for ck in args.checkpoint or []:
        out[Path(ck).stem if len(out) else "model"] = Path(ck)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", help="run name (repeatable -> models are combined)")
    ap.add_argument("--checkpoint", action="append", help="checkpoint path (repeatable)")
    ap.add_argument("--source", required=True, help="image, folder or video")
    ap.add_argument("--out", default=None, help="output folder (default artifacts/predictions/<source name>)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-draw", action="store_true")
    args = ap.parse_args(argv)

    paths = load_paths()
    ckpts = _resolve_checkpoints(args, paths)
    if not ckpts:
        ap.error("give at least one --run or --checkpoint")
    models = {tag: CarPartsModel(ck, tag=tag, threshold=args.threshold) for tag, ck in ckpts.items()}
    pipeline = CarPartsPipeline(models)
    src = Path(args.source)
    out = Path(args.out) if args.out else paths.artifacts / "predictions" / src.stem
    out.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    def run_image(path: Path, rgb: np.ndarray | None = None) -> dict:
        rgb = rgb if rgb is not None else cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        result = {"source": str(path), "height": rgb.shape[0], "width": rgb.shape[1], "objects": []}
        canvas = rgb
        for tag, m in models.items():
            pred = m.predict(rgb, source=str(path))
            result["objects"] += pred.to_dict()["objects"]
            if not args.no_draw:
                canvas = m.annotate(canvas, pred)
        (out / f"{path.stem}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        if not args.no_draw:
            cv2.imwrite(str(out / f"{path.stem}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return result

    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        for i, p in enumerate(files, 1):
            r = run_image(p)
            print(f"[{i}/{len(files)}] {p.name}: {len(r['objects'])} objects")
    elif src.suffix.lower() in VID_EXTS:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None if args.no_draw else cv2.VideoWriter(str(out / f"{src.stem}_annotated.mp4"),
                                                            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = 0
        with (out / f"{src.stem}_frames.jsonl").open("w", encoding="utf-8") as jl:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                r = pipeline.predict(rgb, threshold=args.threshold)
                r["frame"] = n
                jl.write(json.dumps(r) + "\n")
                if writer is not None:
                    canvas = rgb
                    for m in models.values():
                        canvas = m.annotate(canvas, m.predict(rgb))
                    writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                n += 1
                if n % 25 == 0:
                    print(f"frame {n}", end="\r")
        cap.release()
        if writer is not None:
            writer.release()
        print(f"\n{n} frames -> {out}")
    else:
        r = run_image(src)
        print(json.dumps(r, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
