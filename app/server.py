"""Localhost demo API + single-page UI for the car-parts models.

  python app/server.py --run poc_exterior_seg                       # one model
  python app/server.py --run exterior_seg_m --run engine_bay_m      # several models, results merged
  python app/server.py --checkpoint path/to/checkpoint_best_total.pth --port 8010

Then open http://127.0.0.1:8000 — drag a photo in, move the threshold slider, read the JSON.

Endpoints
  GET  /health                 models, classes, device, versions
  GET  /classes                {tag: [class names]}
  POST /predict                multipart `file` (+ form `threshold`) -> JSON objects (label, confidence, bbox, polygon)
  POST /predict/image          same input -> annotated JPEG
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import carparts  # noqa: E402,F401  (BLAS env before numpy)

from carparts.config import load_env, load_paths  # noqa: E402

try:  # FastAPI types must be module-level (not string annotations) so request validation can resolve them
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
    from fastapi.responses import HTMLResponse, JSONResponse, Response  # noqa: E402
except ImportError as e:  # pragma: no cover
    raise SystemExit("FastAPI is not installed: uv sync --extra serve") from e

load_env()

INDEX_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def create_app(checkpoints: dict[str, Path], threshold: float = 0.5):
    import numpy as np
    from PIL import Image, ImageOps

    from carparts.infer import CarPartsModel

    t0 = time.time()
    models = {tag: CarPartsModel(ck, tag=tag, threshold=threshold) for tag, ck in checkpoints.items()}
    load_seconds = round(time.time() - t0, 1)

    app = FastAPI(title="carparts demo", version=carparts.__version__)

    def _read(upload: UploadFile) -> np.ndarray:
        data = upload.file.read()
        if not data:
            raise HTTPException(400, "empty upload")
        try:
            with Image.open(io.BytesIO(data)) as im:
                return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"not an image: {e}") from e

    def _run(rgb: np.ndarray, thr: float, draw: bool):
        result = {"height": int(rgb.shape[0]), "width": int(rgb.shape[1]), "threshold": thr, "objects": [],
                  "timings_ms": {}}
        canvas = rgb
        for tag, m in models.items():
            t = time.time()
            pred = m.predict(rgb, threshold=thr)
            result["timings_ms"][tag] = round((time.time() - t) * 1000, 1)
            result["objects"] += pred.to_dict()["objects"]
            if draw:
                canvas = m.annotate(canvas, pred)
        result["objects"].sort(key=lambda o: -(o["confidence"] or 0))
        return result, canvas

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        import torch
        return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu",
                "models": {tag: {"checkpoint": m.checkpoint, "classes": len(m.class_names),
                                 "segmentation": m.is_segmentation} for tag, m in models.items()},
                "load_seconds": load_seconds, "torch": torch.__version__}

    @app.get("/classes")
    def classes():
        return {tag: m.class_names for tag, m in models.items()}

    @app.post("/predict")
    def predict(file: UploadFile = File(...), threshold: float = Form(threshold)):
        result, _ = _run(_read(file), float(threshold), draw=False)
        result["filename"] = file.filename
        return JSONResponse(result)

    @app.post("/predict/image")
    def predict_image(file: UploadFile = File(...), threshold: float = Form(threshold)):
        result, canvas = _run(_read(file), float(threshold), draw=True)
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="JPEG", quality=90)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"X-Objects": str(len(result["objects"])),
                                 "X-Timings": json.dumps(result["timings_ms"])})

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", help="run name under artifacts/runs (repeatable)")
    ap.add_argument("--checkpoint", action="append", help="checkpoint path (repeatable)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    from carparts.train.rfdetr_trainer import best_checkpoint
    paths = load_paths()
    ckpts: dict[str, Path] = {}
    for run in args.run or []:
        ckpts[run] = best_checkpoint(paths.runs / run / "rfdetr")
    for i, ck in enumerate(args.checkpoint or []):
        ckpts[f"model{i}" if ckpts else "model"] = Path(ck)
    if not ckpts:
        ap.error("give --run or --checkpoint")

    import uvicorn
    app = create_app(ckpts, args.threshold)
    print(f"[server] models: {list(ckpts)} -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
