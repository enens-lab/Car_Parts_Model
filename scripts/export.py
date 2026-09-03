"""Export a fine-tuned checkpoint to ONNX (default) / TensorRT / TFLite / CoreML and sanity-check it.

Examples
--------
  python scripts/export.py --run exterior_seg_m                          # -> artifacts/exports/exterior_seg_m/*.onnx
  python scripts/export.py --run exterior_seg_m --verify data/processed/exterior_seg/test/<img>.jpg
  python scripts/export.py --checkpoint path/to.pth --format tensorrt    # needs rfdetr[tensorrt] on the target GPU
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from carparts.config import load_paths
from carparts.infer.classifier import is_classifier_checkpoint
from carparts.train.classifier import export_classifier_onnx
from carparts.train.rfdetr_trainer import load_checkpoint, resolve_run_checkpoint

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def verify_onnx(onnx_path: Path, image: Path | None, resolution: int) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outputs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
    info = {"inputs": inputs, "outputs": outputs}
    if image is not None:
        from PIL import Image
        with Image.open(image) as im:
            rgb = np.asarray(im.convert("RGB").resize((resolution, resolution)), dtype=np.float32) / 255.0
        x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        t0 = time.time()
        outs = sess.run(None, {inputs[0][0]: x})
        info["latency_ms_cpu"] = round((time.time() - t0) * 1000, 1)
        info["output_shapes"] = {o.name: list(v.shape) for o, v in zip(sess.get_outputs(), outs)}
        info["all_finite"] = bool(all(np.isfinite(v).all() for v in outs))
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--checkpoint")
    ap.add_argument("--format", default="onnx", choices=["onnx", "tensorrt", "tflite", "coreml", "executorch"])
    ap.add_argument("--out")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--verify", help="image to push through the exported ONNX graph with onnxruntime")
    args = ap.parse_args(argv)

    paths = load_paths().ensure()
    if args.run:
        ckpt = resolve_run_checkpoint(paths.runs / args.run)
        out = Path(args.out) if args.out else paths.exports / args.run
    elif args.checkpoint:
        ckpt = Path(args.checkpoint)
        out = Path(args.out) if args.out else paths.exports / ckpt.stem
    else:
        ap.error("give --run or --checkpoint")
    out.mkdir(parents=True, exist_ok=True)

    if is_classifier_checkpoint(ckpt):
        if args.format != "onnx":
            ap.error("classifier checkpoints export to ONNX only")
        path = export_classifier_onnx(ckpt, out, opset=args.opset)
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        res = json.loads((out / "preprocess.json").read_text())["center_crop"]
        logits = sess.run(None, {inp.name: np.random.rand(1, 3, res, res).astype(np.float32)})[0]
        print(f"[export] {path.name} ({path.stat().st_size / 1e6:.1f} MB) logits shape {list(logits.shape)} finite={bool(np.isfinite(logits).all())}")
        print(f"-> {out}")
        return 0

    model = load_checkpoint(ckpt)
    resolution = int(getattr(model.model_config, "resolution", 0) or 0)
    print(f"[export] {ckpt.name} ({len(model.class_names)} classes, res {resolution}) -> {out} [{args.format}]")
    model.export(output_dir=str(out), format=args.format, opset_version=args.opset, batch_size=args.batch_size)
    (out / "class_names.json").write_text(json.dumps(model.class_names, indent=1), encoding="utf-8")

    if args.format == "onnx":
        files = sorted(out.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if not files:
            raise SystemExit("export produced no .onnx file")
        info = verify_onnx(files[-1], Path(args.verify) if args.verify else None, resolution)
        print(f"[export] {files[-1].name} ({files[-1].stat().st_size / 1e6:.1f} MB)")
        print(json.dumps(info, indent=1, default=str))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
