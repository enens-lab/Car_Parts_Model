"""Recipe -> processed dataset.

A *recipe* (``configs/recipes/*.yaml``) says which sources to pull, how to rename/merge classes, and how
to split. The result is a Roboflow/RF-DETR-shaped folder::

    data/processed/<recipe>/
        train/_annotations.coco.json + images        (folder names are what RF-DETR expects)
        valid/_annotations.coco.json + images
        test/_annotations.coco.json  + images
        dataset_card.json                            (sources, licenses, attribution, class list, split report)

Images are hard-linked when possible (same NTFS volume), copied otherwise, so a recipe costs no extra disk.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ..constants import TAXONOMY_MAPS
from ..sources import registry
from .coco import CocoDataset
from .splits import grouped_split, merge_duplicate_groups, report

SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def _safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _resolve_class_map(spec: Any, ds: CocoDataset) -> dict[str, str | None] | None:
    if spec in (None, {}, ""):
        return None
    if isinstance(spec, str) and spec in TAXONOMY_MAPS:
        return TAXONOMY_MAPS[spec](ds.class_names, drop_unknown=True)
    if isinstance(spec, dict):
        return {str(k): (None if v is None else str(v)) for k, v in spec.items()}
    raise ValueError(f"unsupported class_map spec: {spec!r}")


def load_sources(recipe: dict[str, Any], raw_root: Path, download: bool = True,
                 skip_unavailable: bool = True) -> tuple[dict[str, dict[str, CocoDataset]], list[str]]:
    """Return ``{source_name: {split: ds}}`` for every enabled source, plus the names that were skipped."""
    loaded: dict[str, dict[str, CocoDataset]] = {}
    skipped: list[str] = []
    for entry in recipe["sources"]:
        name = entry["name"] if isinstance(entry, dict) else str(entry)
        src = registry.get(name)
        missing = src.missing_env()
        if missing and not src.is_downloaded(raw_root):
            msg = f"[{name}] skipped — missing env {missing} (see .env.example)"
            if not skip_unavailable:
                raise RuntimeError(msg)
            print(msg)
            skipped.append(name)
            continue
        try:
            if download:
                src.download(raw_root)
            splits = src.load(raw_root)
        except Exception as e:  # network / credentials / layout problems
            if not skip_unavailable:
                raise
            print(f"[{name}] skipped — {type(e).__name__}: {e}")
            skipped.append(name)
            continue
        cmap_spec = entry.get("class_map") if isinstance(entry, dict) else None
        for split, ds in splits.items():
            cmap = _resolve_class_map(cmap_spec, ds)
            if cmap:
                ds = ds.remap_classes(cmap)
            splits[split] = ds
        loaded[name] = splits
    return loaded, skipped


def assemble(recipe: dict[str, Any], loaded: dict[str, dict[str, CocoDataset]]) -> tuple[dict[str, CocoDataset], dict]:
    """Merge sources and split. Returns ``({split: ds}, split_report_dict)``."""
    strategy = recipe["split"]["strategy"]
    seed = int(recipe["split"]["seed"])
    fractions = {k: float(v) for k, v in recipe["split"]["fractions"].items()}
    task = recipe["task"]

    per_source = []
    removed_total = 0
    for name, splits in loaded.items():
        for split, ds in splits.items():
            ds = ds if task == "segmentation" else ds.strip_masks()
            if recipe.get("dedupe_annotations", True):
                ds, removed = ds.dedupe_annotations(float(recipe.get("dedupe_iou", 0.85)))
                removed_total += removed
                if removed:
                    print(f"[recipe] {name}/{split}: removed {removed} duplicate same-class annotations (IoU>=0.85)")
            per_source.append((name, split, ds))

    max_ham = int(recipe["split"].get("max_hamming", 4))
    if strategy == "original":
        # keep each source's own split; sources without splits ("all") get a grouped split of their own
        buckets: dict[str, list[CocoDataset]] = {"train": [], "val": [], "test": []}
        for name, split, ds in per_source:
            if split in buckets:
                buckets[split].append(ds)
            else:
                groups = merge_duplicate_groups(ds, recipe["split"]["dedupe_hash"], max_ham)
                ids = grouped_split(ds, fractions, seed, groups)
                for s, v in ids.items():
                    buckets[s].append(ds.subset(v))
        merged = {s: CocoDataset.merge(v) if v else CocoDataset() for s, v in buckets.items()}
        # align class lists across splits (merge() numbers classes by first appearance)
        all_names = CocoDataset.merge([d for d in merged.values() if d.images]).class_names
        merged = {s: d.remap_classes(keep_order=all_names) for s, d in merged.items()}
        full = CocoDataset.merge([d for d in merged.values() if d.images])
        split_ids = {}
        offset = 0
        for s in ("train", "val", "test"):
            n = len(merged[s].images)
            split_ids[s] = list(range(offset + 1, offset + n + 1))
            offset += n
        rep = report(full, split_ids, "original", seed)
    elif strategy == "grouped":
        full = CocoDataset.merge([ds for _, _, ds in per_source])
        groups = merge_duplicate_groups(full, recipe["split"]["dedupe_hash"], max_ham)
        sub = recipe.get("subsample")
        if sub and sub.get("max_groups"):
            # deterministic subset of photo-groups (smoke tests / quick experiments)
            import random as _random
            all_groups = sorted(set(groups.values()))
            keep = set(_random.Random(int(sub.get("seed", 0))).sample(all_groups,
                                                                     min(int(sub["max_groups"]), len(all_groups))))
            full = full.subset(iid for iid, g in groups.items() if g in keep)
            groups = {iid: g for iid, g in groups.items() if g in keep}
        split_ids = grouped_split(full, fractions, seed, groups)
        merged = {s: full.subset(ids) for s, ids in split_ids.items()}
        rep = report(full, split_ids, "grouped", seed, groups)
    else:
        raise ValueError(f"unknown split strategy {strategy!r}")

    # optional class subset / fixed class order / rare-class pruning (applied identically to every split)
    if recipe.get("keep_classes"):
        keep = set(recipe["keep_classes"])
        drop = {n: None for d in merged.values() for n in d.class_names if n not in keep}
        merged = {s: d.remap_classes(drop) for s, d in merged.items()}
        # drop images that lost every annotation (pure background would dominate a subset recipe)
        for s, d in merged.items():
            with_ann = {a.image_id for a in d.annotations}
            merged[s] = d.subset(with_ann)
    if recipe.get("classes"):
        merged = {s: d.remap_classes(keep_order=recipe["classes"]) for s, d in merged.items()}
    min_inst = int(recipe.get("min_instances_per_class") or 0)
    if min_inst > 0:
        counts = full.stats()["per_class"]
        drop = {n: None for n, c in counts.items() if c < min_inst}
        if drop:
            print(f"[recipe] dropping rare classes (<{min_inst} instances): {sorted(drop)}")
            merged = {s: d.remap_classes(drop) for s, d in merged.items()}
    for d in merged.values():
        d.validate()
    out_report = rep.__dict__
    out_report["duplicate_annotations_removed"] = removed_total
    return merged, out_report


def write_processed(recipe: dict[str, Any], splits: dict[str, CocoDataset], out_root: Path, split_report: dict,
                    skipped_sources: list[str]) -> Path:
    out = out_root / recipe["name"]
    if out.exists():
        shutil.rmtree(out)
    used_sources = sorted({im.source for d in splits.values() for im in d.images})
    cls_dropped = 0
    for split, ds in splits.items():
        d = out / SPLIT_DIRS[split]
        d.mkdir(parents=True, exist_ok=True)
        if recipe["task"] == "classification":
            # torchvision ImageFolder layout: <split>/<class>/<image>; one label per image
            cats = ds.cat_by_id
            anns = ds.anns_by_image()
            for im in ds.images:
                labels = {cats[a.category_id].name for a in anns.get(im.id, [])}
                if len(labels) != 1:
                    cls_dropped += 1
                    continue
                cdir = d / _safe(labels.pop())
                cdir.mkdir(exist_ok=True)
                _link_or_copy(Path(im.orig_path), cdir / _safe(f"{im.source}__{im.file_name}"))
            for c in ds.categories:  # keep every class folder so ImageFolder indices match across splits
                (d / _safe(c.name)).mkdir(exist_ok=True)
            continue
        new_names: dict[int, str] = {}
        for im in ds.images:
            new_names[im.id] = _safe(f"{im.source}__{im.file_name}")
            _link_or_copy(Path(im.orig_path), d / new_names[im.id])
        ds.info.update({"description": f"carparts recipe {recipe['name']} / {split}",
                        "licenses": [registry.get(s).info.to_dict() for s in used_sources]})
        ds.save_json(d / "_annotations.coco.json", file_name_fn=lambda im: new_names[im.id])
    if cls_dropped:
        print(f"[recipe] classification layout: dropped {cls_dropped} images without exactly one class")
        split_report = {**split_report, "classification_images_dropped": cls_dropped}
    card = {
        "recipe": {k: v for k, v in recipe.items() if not k.startswith("_")},
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "task": recipe["task"],
        "classes": splits["train"].class_names,
        "sources": {s: registry.get(s).info.to_dict() for s in used_sources},
        "skipped_sources": skipped_sources,
        "split_report": split_report,
        "stats": {s: d.stats() for s, d in splits.items()},
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    (out / "NOTICE.md").write_text(_notice(card), encoding="utf-8")
    return out


def _notice(card: dict) -> str:
    lines = [f"# Data attribution — recipe `{card['recipe']['name']}`", "",
             "This training set was assembled from the following sources. Keep this notice with any model, "
             "dataset or product derived from it.", ""]
    for name, info in card["sources"].items():
        lines += [f"## {info['title']}", f"- License: **{info['license']}**", f"- URL: {info['url']}",
                  f"- Attribution: {info['attribution']}", f"- Notes: {info['notes']}", ""]
    return "\n".join(lines)


def card_hash(processed_dir: Path) -> str:
    p = processed_dir / "dataset_card.json"
    return hashlib.sha1(p.read_bytes()).hexdigest()[:10] if p.exists() else "n/a"


def build_recipe(recipe: dict[str, Any], raw_root: Path, processed_root: Path, reports_root: Path,
                 download: bool = True, strict: bool = False) -> Path:
    loaded, skipped = load_sources(recipe, raw_root, download=download, skip_unavailable=not strict)
    if not loaded:
        raise RuntimeError(f"recipe {recipe['name']!r}: no sources could be loaded (skipped: {skipped})")
    splits, rep = assemble(recipe, loaded)
    out = write_processed(recipe, splits, processed_root, rep, skipped)
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / f"{recipe['name']}_split_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    src_stats = {name: {split: ds.stats() for split, ds in sp.items()} for name, sp in loaded.items()}
    (reports_root / f"{recipe['name']}_sources.json").write_text(
        json.dumps({"sources": src_stats, "skipped": skipped}, indent=1), encoding="utf-8")
    print(f"[recipe {recipe['name']}] -> {out}")
    for s, d in splits.items():
        st = d.stats()
        print(f"   {s:5s} images={st['images']:5d} annotations={st['annotations']:6d} classes={st['classes']}")
    print(f"   split: {rep['strategy']} | cross-split groups={rep['cross_split_groups']} | "
          f"eval images sharing a photo with train={rep['leaked_eval_images']}")
    return out
