"""Fine-tune RF-DETR on a processed recipe.

Examples
--------
  python scripts/train.py --recipe exterior_seg --config rfdetr_8gb                     # seg-medium @432
  python scripts/train.py --recipe engine_bay_det --config rfdetr_8gb --model rfdetr-medium
  python scripts/train.py --recipe exterior_seg --config rfdetr_8gb --epochs 5 --name quick
  python scripts/train.py --recipe exterior_seg --config rfdetr_8gb --name exterior_m --resume artifacts/runs/exterior_m/rfdetr/last.ckpt

Outputs land in artifacts/runs/<name>/ (summary.json, eval_test.json, rfdetr/ checkpoints + tensorboard).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import datetime as dt

from carparts.config import load_paths, load_train_config
from carparts.train.rfdetr_trainer import MODEL_REGISTRY, train_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--config", default="rfdetr_8gb", help="configs/train/<name>.yaml")
    ap.add_argument("--model", choices=sorted(MODEL_REGISTRY), help="override the config's model")
    ap.add_argument("--name", help="run name (default: <recipe>_<model>_<timestamp>)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--grad-accum", type=int, dest="grad_accum_steps")
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--num-workers", type=int, dest="num_workers")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--devices", help='GPUs for Lightning: an int or "auto" (multi-GPU: launch via torchrun)')
    ap.add_argument("--resume", help="path to a full trainer checkpoint (last.ckpt)")
    args = ap.parse_args(argv)

    cfg = load_train_config(args.config)
    model = args.model or cfg["model"]
    name = args.name or f"{args.recipe}_{model}_{dt.datetime.now():%Y%m%d-%H%M}"
    overrides = {"model": model, "epochs": args.epochs, "batch_size": args.batch_size,
                 "grad_accum_steps": args.grad_accum_steps, "resolution": args.resolution, "lr": args.lr,
                 "num_workers": args.num_workers, "seed": args.seed,
                 "devices": (int(args.devices) if args.devices and args.devices.isdigit() else args.devices)}
    summary = train_run(load_paths().ensure(), args.recipe, cfg, name, overrides, resume=args.resume)
    tm = summary["test_metrics"]
    for k, v in tm.items():
        print(f"[train] test {k}: AP={v['AP']:.3f} AP50={v['AP50']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
