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
from carparts.infer.classifier import is_classifier_checkpoint
from carparts.train.classifier import evaluate_classifier
from carparts.train.rfdetr_trainer import resolve_run_checkpoint


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
        ckpt = ckpt or resolve_run_checkpoint(run_dir)
        out_json = run_dir / f"eval_{args.split}.json"
    if not (recipe and ckpt):
        ap.error("need --run, or --checkpoint together with --recipe")
    if is_classifier_checkpoint(ckpt):
        rep = evaluate_classifier(ckpt, paths.processed / recipe, split=args.split, out_json=out_json)
        print(f"split={rep['split']} n={rep['n']} top1={rep['top1']:.4f} top5={rep['top5']:.4f} macro_acc={rep['macro_acc']:.4f}")
        worst = sorted(((v['acc'] if v['acc'] is not None else 1.0), k, v['n']) for k, v in rep["per_class"].items())[:8]
        print("  weakest classes: " + ", ".join(f"{k} {a:.2f} (n={n})" for a, k, n in worst))
        print("  top confusions: " + ", ".join(f"{c['true']}->{c['pred']} x{c['count']}" for c in rep["top_confusions"][:6]))
        if out_json:
            print(f"-> {out_json}")
        return 0
    report = evaluate_checkpoint(ckpt, paths.processed / recipe, split=args.split, threshold=args.threshold,
                                 max_images=args.max_images, out_json=out_json)
    print(format_report(report))
    if out_json:
        print(f"-> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
