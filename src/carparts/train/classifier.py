"""Image classifier for isolated-part identification ("what component is this photo of?").

Catalog-style datasets (one part per photo, whole-image boxes) are classification data; a detector adds nothing
there. This trainer fine-tunes a torchvision ConvNeXt (BSD-3 code + weights) on the ImageFolder layout that
``classification`` recipes produce, with the same run conventions as the RF-DETR track:
``artifacts/runs/<run>/{classifier_best.pth, summary.json, eval_test.json}``.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import time
from pathlib import Path
from typing import Any

from ..config import Paths

ARCHS = {  # torchvision constructors; all BSD-3 with ImageNet weights
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "convnext_small": ("convnext_small", "ConvNeXt_Small_Weights"),
    "convnext_base": ("convnext_base", "ConvNeXt_Base_Weights"),
    "efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNet_V2_S_Weights"),
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_classifier(arch: str, num_classes: int, pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models

    if arch not in ARCHS:
        raise KeyError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
    fn_name, w_name = ARCHS[arch]
    weights = getattr(models, w_name).DEFAULT if pretrained else None
    model = getattr(models, fn_name)(weights=weights)
    if arch.startswith("convnext"):
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:  # efficientnet
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _transforms(resolution: int, train: bool):
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.RandomResizedCrop(resolution, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize(int(resolution * 1.14)), T.CenterCrop(resolution), T.ToTensor(), T.Normalize(MEAN, STD)])


def _loaders(dataset_dir: Path, resolution: int, batch_size: int, num_workers: int):
    import torch
    from torchvision.datasets import ImageFolder

    train = ImageFolder(dataset_dir / "train", _transforms(resolution, True))
    valid = ImageFolder(dataset_dir / "valid", _transforms(resolution, False))
    test = ImageFolder(dataset_dir / "test", _transforms(resolution, False))
    if not (train.classes == valid.classes == test.classes):
        raise ValueError("class folders differ between splits — rebuild the recipe")
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    return (train, valid, test,
            torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True, **kw),
            torch.utils.data.DataLoader(valid, batch_size * 2, shuffle=False, **kw),
            torch.utils.data.DataLoader(test, batch_size * 2, shuffle=False, **kw))


def _amp_dtype(device):
    import torch
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device.type == "cuda" else None


def _predict(model, loader, device):
    import torch
    model.eval()
    preds, targets, probs = [], [], []
    dtype = _amp_dtype(device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(x)
            p = torch.softmax(logits.float(), dim=1)
            probs.append(p.cpu())
            preds.append(p.argmax(1).cpu())
            targets.append(y)
    return torch.cat(preds), torch.cat(targets), torch.cat(probs)


def _metrics(preds, targets, probs, class_names: list[str]) -> dict[str, Any]:
    import torch
    n = len(targets)
    top1 = float((preds == targets).float().mean()) if n else 0.0
    top5 = float((probs.topk(min(5, probs.shape[1]), dim=1).indices == targets[:, None]).any(1).float().mean()) if n else 0.0
    per_class = {}
    for i, name in enumerate(class_names):
        m = targets == i
        per_class[name] = {"n": int(m.sum()), "acc": float((preds[m] == i).float().mean()) if m.any() else None}
    conf = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist()):
        conf[t, p] += 1
    off = conf.clone()
    off.fill_diagonal_(0)
    top_conf = []
    for _ in range(10):
        v, idx = off.flatten().max(0)
        if v.item() == 0:
            break
        t, p = divmod(int(idx), len(class_names))
        top_conf.append({"true": class_names[t], "pred": class_names[p], "count": int(v)})
        off[t, p] = 0
    return {"n": n, "top1": top1, "top5": top5, "macro_acc": float(sum(v["acc"] for v in per_class.values() if v["acc"] is not None) /
                                                             max(1, sum(1 for v in per_class.values() if v["acc"] is not None))),
            "per_class": per_class, "top_confusions": top_conf}


def train_classifier(paths: Paths, recipe_name: str, cfg: dict[str, Any], run_name: str,
                     overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    cfg = {**cfg, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    arch = cfg.get("model", "convnext_tiny")
    epochs, bs = int(cfg.get("epochs", 15)), int(cfg.get("batch_size", 64))
    lr, wd = float(cfg.get("lr", 3e-4)), float(cfg.get("weight_decay", 0.05))
    res = int(cfg.get("resolution") or 224)
    workers, seed = int(cfg.get("num_workers", 0)), int(cfg.get("seed", 2026))
    smoothing = float(cfg.get("label_smoothing", 0.1))
    warmup = float(cfg.get("warmup_epochs", 1.0))

    dataset_dir = paths.processed / recipe_name
    card_path = dataset_dir / "dataset_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"processed recipe missing: {dataset_dir} — run scripts/prepare_data.py --recipe {recipe_name}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card["task"] != "classification":
        raise ValueError(f"recipe {recipe_name!r} is {card['task']}, not classification")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, _, _, tl, vl, tel = _loaders(dataset_dir, res, bs, workers)
    class_names = list(train_ds.classes)
    model = build_classifier(arch, len(class_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(tl))
    total = steps_per_epoch * epochs
    warm = int(warmup * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    dtype = _amp_dtype(device)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    run_dir = paths.runs / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "classifier_best.pth"
    print(f"[cls] run={run_name} recipe={recipe_name} arch={arch}@{res} classes={len(class_names)} "
          f"train={len(train_ds)} batch={bs} epochs={epochs} device={device}", flush=True)
    best, history, t0 = -1.0, [], time.time()
    started = _dt.datetime.now()
    for epoch in range(epochs):
        model.train()
        te, losses = time.time(), []
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                loss = F.cross_entropy(model(x), y, label_smoothing=smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        preds, targets, probs = _predict(model, vl, device)
        val = _metrics(preds, targets, probs, class_names)
        history.append({"epoch": epoch + 1, "loss": sum(losses) / max(1, len(losses)), "val_top1": val["top1"],
                        "val_top5": val["top5"], "seconds": round(time.time() - te, 1)})
        print(f"[cls] epoch {epoch + 1}/{epochs} loss={history[-1]['loss']:.3f} val_top1={val['top1']:.4f} "
              f"val_top5={val['top5']:.4f} [{history[-1]['seconds']}s]", flush=True)
        if val["top1"] > best:
            best = val["top1"]
            torch.save({"kind": "classifier", "arch": arch, "resolution": res, "class_names": class_names,
                        "state_dict": model.state_dict(), "epoch": epoch + 1, "val_top1": best}, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    preds, targets, probs = _predict(model, tel, device)
    test = _metrics(preds, targets, probs, class_names)
    (run_dir / "eval_test.json").write_text(json.dumps({"checkpoint": str(ckpt_path), "split": "test", **test}, indent=1),
                                            encoding="utf-8")
    summary = {
        "kind": "classifier", "run": run_name, "recipe": recipe_name, "model": arch, "resolution": res,
        "train_config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "class_names": class_names,
        "num_classes": len(class_names), "checkpoint": str(ckpt_path), "best_val_top1": best,
        "history": history, "train_seconds": round(time.time() - t0, 1),
        "started": started.isoformat(timespec="seconds"), "finished": _dt.datetime.now().isoformat(timespec="seconds"),
        "test_metrics": {"top1": test["top1"], "top5": test["top5"], "macro_acc": test["macro_acc"], "n": test["n"]},
        "dataset_card": {"path": str(card_path), "created": card.get("created"),
                         "images": {s: v["images"] for s, v in card.get("stats", {}).items()}},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[cls] test top1={test['top1']:.4f} top5={test['top5']:.4f} macro_acc={test['macro_acc']:.4f} -> {run_dir}")
    return summary


def evaluate_classifier(checkpoint: str | Path, dataset_dir: str | Path, split: str = "test", batch_size: int = 128,
                        out_json: str | Path | None = None) -> dict[str, Any]:
    import torch
    from torchvision.datasets import ImageFolder

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(ck["arch"], len(ck["class_names"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    folder = {"test": "test", "val": "valid", "valid": "valid", "train": "train"}[split]
    ds = ImageFolder(Path(dataset_dir) / folder, _transforms(ck["resolution"], False))
    if ds.classes != ck["class_names"]:
        raise ValueError("split class folders do not match the checkpoint's classes")
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False)
    preds, targets, probs = _predict(model, loader, device)
    rep = {"checkpoint": str(checkpoint), "split": split, **_metrics(preds, targets, probs, ck["class_names"])}
    if out_json:
        Path(out_json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def export_classifier_onnx(checkpoint: str | Path, out_dir: str | Path, opset: int = 17) -> Path:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_classifier(ck["arch"], len(ck["class_names"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ck['arch']}_classifier.onnx"
    dummy = torch.zeros(1, 3, ck["resolution"], ck["resolution"])
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["logits"], opset_version=opset,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    (out_dir / "class_names.json").write_text(json.dumps(ck["class_names"], indent=1), encoding="utf-8")
    (out_dir / "preprocess.json").write_text(json.dumps({"resize": int(ck["resolution"] * 1.14), "center_crop": ck["resolution"],
                                                          "mean": MEAN, "std": STD, "layout": "NCHW RGB float32"}), encoding="utf-8")
    return path
