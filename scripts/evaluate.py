"""Per-class COCO evaluation of a run (or any checkpoint) on a processed split.

Examples
--------
  python scripts/evaluate.py --run exterior_seg_m                       # test split of the run's recipe
  python scripts/evaluate.py --run exterior_seg_m --split val
  python scripts/evaluate.py --checkpoint path/to.pth --recipe engine_bay_det --max-images 100
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from carparts.config import load_paths
from carparts.eval.coco_eval import evaluate_checkpoint, format_report
from carparts.train.rfdetr_trainer import best_checkpoint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run name under artifacts/runs")
    ap.add_argument("--checkpoint", help="explicit checkpoint (.pth)")
    ap.add_argument("--recipe", help="recipe whose split to evaluate (default: the run's recipe)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--max-images", type=int)
    args = ap.parse_args(argv)

    paths = load_paths()
    recipe = args.recipe
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    out_json = None
    if args.run:
        run_dir = paths.runs / args.run
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
        recipe = recipe or summary.get("recipe")
        ckpt = ckpt or best_checkpoint(run_dir / "rfdetr")
        out_json = run_dir / f"eval_{args.split}.json"
    if not (recipe and ckpt):
        ap.error("need --run, or --checkpoint together with --recipe")
    report = evaluate_checkpoint(ckpt, paths.processed / recipe, split=args.split, threshold=args.threshold,
                                 max_images=args.max_images, out_json=out_json)
    print(format_report(report))
    if out_json:
        print(f"-> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
