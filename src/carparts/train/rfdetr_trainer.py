"""Thin, opinionated wrapper around ``rfdetr`` fine-tuning.

One *run* = one recipe (processed dataset) x one model variant x one train config. Everything lands in
``artifacts/runs/<run_name>/``::

    rfdetr/                      RF-DETR's own output_dir (checkpoints, tensorboard, results.json)
    summary.json                 what we trained, on what, how long, and the test metrics
    eval_test.json               per-class AP table (see carparts.eval.coco_eval)

Only permissively licensed variants are registered. ``RFDETRXLarge``/``RFDETR2XLarge`` (detection) need
``rfdetr[plus]`` and ship under PML-1.0, so they are deliberately absent.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import Paths


@dataclass(frozen=True)
class ModelSpec:
    cls_name: str
    task: str            # detection | segmentation
    resolution: int      # native default
    multiple: int        # resolution must be a multiple of this
    license: str = "Apache-2.0"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "rfdetr-nano": ModelSpec("RFDETRNano", "detection", 384, 32),
    "rfdetr-small": ModelSpec("RFDETRSmall", "detection", 512, 32),
    "rfdetr-medium": ModelSpec("RFDETRMedium", "detection", 576, 32),
    "rfdetr-large": ModelSpec("RFDETRLarge", "detection", 704, 32),
    "rfdetr-seg-nano": ModelSpec("RFDETRSegNano", "segmentation", 312, 12),
    "rfdetr-seg-small": ModelSpec("RFDETRSegSmall", "segmentation", 384, 24),
    "rfdetr-seg-medium": ModelSpec("RFDETRSegMedium", "segmentation", 432, 24),
    "rfdetr-seg-large": ModelSpec("RFDETRSegLarge", "segmentation", 504, 24),
    "rfdetr-seg-xlarge": ModelSpec("RFDETRSegXLarge", "segmentation", 624, 24),
    "rfdetr-seg-2xlarge": ModelSpec("RFDETRSeg2XLarge", "segmentation", 768, 24),
}

# train() kwargs we forward verbatim from configs/train/*.yaml (everything else is ours or constructor-side)
TRAIN_KEYS = {
    "epochs", "batch_size", "grad_accum_steps", "lr", "lr_encoder", "weight_decay", "use_ema", "early_stopping",
    "early_stopping_patience", "early_stopping_min_delta", "early_stopping_use_ema", "checkpoint_interval",
    "warmup_epochs", "lr_scheduler", "lr_scheduler_kwargs", "num_workers", "seed", "tensorboard", "wandb", "project",
    "run", "log_per_class_metrics", "eval_interval", "drop_path", "multi_scale", "expanded_scales", "run_test",
    "eval_batch_size", "progress_bar", "best_model_metric", "skip_best_epochs", "devices",
}
CONSTRUCTOR_KEYS = {"resolution", "gradient_checkpointing", "pretrain_weights", "device"}


def model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from None


def resolve_resolution(spec: ModelSpec, requested: int | None) -> int:
    res = int(requested) if requested else spec.resolution
    if res % spec.multiple:
        raise ValueError(f"resolution {res} must be a multiple of {spec.multiple} for {spec.cls_name}")
    return res


def build_model(name: str, resolution: int | None = None, gradient_checkpointing: bool = False,
                pretrain_weights: str | None = None, **extra: Any):
    """Instantiate an RF-DETR variant (downloads the COCO-pretrained weights on first use)."""
    import rfdetr  # heavy import kept local

    spec = model_spec(name)
    kwargs: dict[str, Any] = {"resolution": resolve_resolution(spec, resolution)}
    if gradient_checkpointing:
        kwargs["gradient_checkpointing"] = True
    if pretrain_weights:
        kwargs["pretrain_weights"] = pretrain_weights
    kwargs.update(extra)
    return getattr(rfdetr, spec.cls_name)(**kwargs)


def load_checkpoint(path: str | Path):
    """Load a fine-tuned checkpoint with its class names (variant is inferred from the file)."""
    from rfdetr import RFDETR
    return RFDETR.from_checkpoint(str(path))


def best_checkpoint(rfdetr_out: Path) -> Path:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        p = rfdetr_out / name
        if p.exists():
            return p
    cands = sorted(rfdetr_out.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime)
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"no checkpoint found under {rfdetr_out}")


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("torch", "rfdetr", "supervision", "pytorch_lightning"):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # pragma: no cover
            out[m] = "n/a"
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        pass
    return out


def train_run(paths: Paths, recipe_name: str, train_cfg: dict[str, Any], run_name: str,
              overrides: dict[str, Any] | None = None, resume: str | None = None) -> dict[str, Any]:
    """Fine-tune, then evaluate the best checkpoint on the recipe's **test** split. Returns the summary."""
    from ..eval.coco_eval import evaluate_checkpoint

    cfg = {**train_cfg, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    cfg.pop("_path", None)
    cfg.pop("name", None)
    model_name = cfg.pop("model")
    spec = model_spec(model_name)

    dataset_dir = paths.processed / recipe_name
    card_path = dataset_dir / "dataset_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"processed recipe missing: {dataset_dir} — run scripts/prepare_data.py --recipe {recipe_name}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if spec.task == "segmentation" and card["task"] != "segmentation":
        raise ValueError(f"{model_name} needs masks but recipe {recipe_name!r} is detection-only")
    class_names: list[str] = card["classes"]

    run_dir = paths.runs / run_name
    out_dir = run_dir / "rfdetr"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctor = {k: cfg.pop(k) for k in list(cfg) if k in CONSTRUCTOR_KEYS}
    train_kwargs = {k: v for k, v in cfg.items() if k in TRAIN_KEYS and v is not None}
    unknown = sorted(k for k in cfg if k not in TRAIN_KEYS and k not in CONSTRUCTOR_KEYS)
    if unknown:
        print(f"[train] ignoring unknown config keys: {unknown}")

    model = build_model(model_name, **ctor)
    resolution = resolve_resolution(spec, ctor.get("resolution"))
    eff_batch = (train_kwargs.get("batch_size", 4) if isinstance(train_kwargs.get("batch_size", 4), int) else "auto")
    print(f"[train] run={run_name} recipe={recipe_name} model={model_name}@{resolution} classes={len(class_names)} "
          f"batch={eff_batch}x{train_kwargs.get('grad_accum_steps', 1)} epochs={train_kwargs.get('epochs')}")

    started = _dt.datetime.now()
    t0 = time.time()
    model.train(dataset_dir=str(dataset_dir), dataset_file="roboflow", output_dir=str(out_dir),
                class_names=class_names, resume=resume, **train_kwargs)
    train_seconds = time.time() - t0

    ckpt = best_checkpoint(out_dir)
    print(f"[train] done in {train_seconds/60:.1f} min -> {ckpt.name}; evaluating on test split ...")
    metrics = evaluate_checkpoint(ckpt, dataset_dir, split="test", out_json=run_dir / "eval_test.json")

    summary = {
        "run": run_name, "recipe": recipe_name, "model": model_name, "model_spec": asdict(spec),
        "resolution": resolution, "constructor": ctor, "train_config": train_kwargs,
        "class_names": class_names, "num_classes": len(class_names),
        "dataset_card": {"path": str(card_path), "created": card.get("created"),
                         "split": card.get("split_report", {}).get("strategy"),
                         "images": {s: v["images"] for s, v in card.get("stats", {}).items()}},
        "checkpoint": str(ckpt), "train_seconds": round(train_seconds, 1),
        "started": started.isoformat(timespec="seconds"), "finished": _dt.datetime.now().isoformat(timespec="seconds"),
        "test_metrics": metrics.get("summary", {}), "versions": _versions(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[train] summary -> {run_dir / 'summary.json'}")
    return summary
