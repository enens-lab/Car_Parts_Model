"""End-to-end smoke test on a small photo subset — proves the whole pipeline on this machine in minutes.

  prepare (subsampled exterior_seg) -> train 1 epoch (rfdetr-seg-nano @312) -> per-class eval
  -> predict + annotate one test image -> ONNX export -> onnxruntime forward pass

Numbers are meaningless; only PASS/FAIL per stage matters.

  python scripts/smoke_test.py                 # ~24 photo groups (~150 images)
  python scripts/smoke_test.py --max-groups 60 --epochs 2
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import time
import traceback
from pathlib import Path

from carparts.config import load_paths, load_recipe, load_train_config

RECIPE = "smoke_exterior_seg"
RUN = "smoke"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-groups", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args(argv)

    paths = load_paths().ensure()
    results: dict[str, dict] = {}

    def stage(name, fn):
        t = time.time()
        try:
            out = fn()
            results[name] = {"ok": True, "seconds": round(time.time() - t, 1)}
            print(f"[smoke] {name}: PASS ({results[name]['seconds']}s)")
            return out
        except Exception as e:  # keep going so one report covers everything
            results[name] = {"ok": False, "seconds": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"}
            print(f"[smoke] {name}: FAIL — {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    # 1. data ---------------------------------------------------------------------------------------
    def prepare():
        from carparts.data.recipe import build_recipe
        recipe = load_recipe("exterior_seg")
        recipe["name"] = RECIPE
        recipe["subsample"] = {"max_groups": args.max_groups, "seed": 0}
        return build_recipe(recipe, paths.raw, paths.processed, paths.reports, download=True)

    processed = stage("prepare", prepare)
    if processed is None:
        return _finish(results, paths)

    # 2. train + eval -------------------------------------------------------------------------------
    def train():
        from carparts.train.rfdetr_trainer import train_run
        cfg = load_train_config("smoke")
        return train_run(paths, RECIPE, cfg, RUN, {"epochs": args.epochs})

    summary = stage("train+eval", train)
    if summary is None:
        return _finish(results, paths)

    # 3. predict -------------------------------------------------------------------------------------
    def predict():
        import cv2
        from carparts.infer import CarPartsModel
        m = CarPartsModel(summary["checkpoint"], tag=RUN, threshold=0.3)
        test_dir = paths.processed / RECIPE / "test"
        img = next(p for p in sorted(test_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        pred = m.predict(img)
        d = pred.to_dict()
        out = paths.runs / RUN
        (out / "smoke_prediction.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        cv2.imwrite(str(out / "smoke_prediction.jpg"), cv2.cvtColor(m.annotate(img, pred), cv2.COLOR_RGB2BGR))
        print(f"[smoke] predicted {len(d['objects'])} objects on {img.name} (classes={len(m.class_names)}, "
              f"segmentation={m.is_segmentation})")
        return m

    model = stage("predict", predict)

    # 4. export --------------------------------------------------------------------------------------
    if model is not None and not args.skip_export:
        def export():
            import numpy as np
            import onnxruntime as ort
            out = paths.exports / RUN
            out.mkdir(parents=True, exist_ok=True)
            model.model.export(output_dir=str(out), format="onnx", opset_version=17, batch_size=1)
            files = sorted(out.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
            assert files, "no .onnx produced"
            sess = ort.InferenceSession(str(files[-1]), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            res = int(model.model.model_config.resolution)
            x = np.random.rand(1, 3, res, res).astype(np.float32)
            outs = sess.run(None, {inp.name: x})
            shapes = {o.name: list(v.shape) for o, v in zip(sess.get_outputs(), outs)}
            assert all(np.isfinite(v).all() for v in outs), "non-finite ONNX outputs"
            print(f"[smoke] onnx {files[-1].name} ({files[-1].stat().st_size / 1e6:.1f} MB) outputs={shapes}")
            return files[-1]

        stage("export", export)

    return _finish(results, paths)


def _finish(results: dict, paths) -> int:
    ok = all(r["ok"] for r in results.values())
    (paths.runs / RUN).mkdir(parents=True, exist_ok=True)
    (paths.runs / RUN / "smoke_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n[smoke] " + ("ALL STAGES PASSED" if ok else "SOME STAGES FAILED"))
    for k, v in results.items():
        print(f"   {k:12s} {'PASS' if v['ok'] else 'FAIL':4s} {v['seconds']:>7.1f}s {v.get('error', '')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
