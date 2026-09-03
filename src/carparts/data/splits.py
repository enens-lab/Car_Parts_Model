"""Leakage-safe train/val/test splitting.

Why this exists: Roboflow exports ship *augmented copies* of the same photo, and the public carparts-seg
split lets copies of one photo land in both train and test. Any mAP measured on that split is optimistic.
We therefore split by **group** (physical photo, see :func:`carparts.data.yolo.group_key`, unioned with
exact/near-duplicate detection via a perceptual hash) so every copy of a photo is on one side of the line.
The public split is still available (``strategy="original"``) for apples-to-apples comparison with
published numbers.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .coco import CocoDataset


# ----------------------------------------------------------------------------- duplicate detection
# Measured on our data (Kaggle engine bay vs its Roboflow 640x640 re-export): 89 % of copies have Hamming
# distance 0, 99.9 % <= 2, 100 % <= 4; distinct Kaggle photos rarely come within 4 bits of each other.
DEFAULT_MAX_HAMMING = 4
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash_bits(path: str | Path, size: int = 8) -> int:
    """64-bit difference hash (row-wise gradient signs on a 9x8 grayscale thumbnail)."""
    with Image.open(path) as im:
        g = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        px = np.frombuffer(g.tobytes(), dtype=np.uint8).reshape(size, size + 1).astype(np.int16)
    bits = (px[:, :-1] > px[:, 1:]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def dhash(path: str | Path, size: int = 8) -> str:
    """Hex string form of :func:`dhash_bits` (identical for exact copies and re-encodes)."""
    return f"{dhash_bits(path, size):016x}"


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def merge_duplicate_groups(ds: CocoDataset, use_hash: bool = True,
                           max_distance: int = DEFAULT_MAX_HAMMING) -> dict[int, str]:
    """Return ``image_id -> group`` where filename-groups whose images are perceptual near-duplicates
    (dhash Hamming distance <= ``max_distance``) are unioned. Vectorised pairwise comparison in blocks."""
    groups = {im.id: (im.group or Path(im.file_name).stem) for im in ds.images}
    if not use_hash or len(ds.images) < 2:
        return groups
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids = [im.id for im in ds.images if im.orig_path and Path(im.orig_path).exists()]
    if len(ids) < 2:
        return {iid: find(g) for iid, g in groups.items()}
    path_of = {im.id: im.orig_path for im in ds.images}
    hashes = np.array([dhash_bits(path_of[i]) for i in ids], dtype=np.uint64)
    n = len(ids)
    block = 128
    for start in range(0, n, block):
        x = hashes[start:start + block, None] ^ hashes[None, :]                   # (b, n) uint64
        dist = _POPCOUNT[x.view(np.uint8)].reshape(x.shape[0], n, 8).sum(-1)     # (b, n) popcount
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if c > i:
                union(groups[ids[i]], groups[ids[c]])
    return {iid: find(g) for iid, g in groups.items()}


# ------------------------------------------------------------------------------------ splitting
@dataclass
class SplitReport:
    strategy: str
    seed: int
    counts: dict[str, int]
    groups: dict[str, int]
    cross_split_groups: int
    leaked_eval_images: int
    per_class: dict[str, dict[str, int]]

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")


def grouped_split(ds: CocoDataset, fractions: dict[str, float], seed: int = 2026,
                  groups: dict[int, str] | None = None) -> dict[str, list[int]]:
    """Deterministic group-aware split. Greedy: big groups first, each goes to the split that is
    furthest below its target image count (with class-presence balancing as a tiebreaker)."""
    assert abs(sum(fractions.values()) - 1) < 1e-6, "fractions must sum to 1"
    groups = groups or {im.id: im.group for im in ds.images}
    members: dict[str, list[int]] = defaultdict(list)
    for iid, g in groups.items():
        members[g].append(iid)
    anns = ds.anns_by_image()
    n_total = len(ds.images)
    targets = {s: f * n_total for s, f in fractions.items()}
    assigned: dict[str, list[int]] = {s: [] for s in fractions}
    class_counts: dict[str, Counter] = {s: Counter() for s in fractions}
    # big groups first; ties broken by a seed-keyed hash so a different seed gives a different (reproducible) split
    order = sorted(members, key=lambda g: (-len(members[g]), hashlib.md5(f"{seed}:{g}".encode()).hexdigest()))
    for g in order:
        ids = members[g]
        g_classes = Counter(a.category_id for iid in ids for a in anns.get(iid, []))

        def deficit(s: str) -> tuple[float, float]:
            need = (targets[s] - len(assigned[s])) / max(targets[s], 1)
            rare = sum(1 for c in g_classes if class_counts[s][c] == 0)  # prefer giving unseen classes
            return (need, rare)

        best = max(fractions, key=deficit)
        assigned[best] += ids
        class_counts[best].update(g_classes)
    return {s: sorted(v) for s, v in assigned.items()}


def report(ds: CocoDataset, split_ids: dict[str, list[int]], strategy: str, seed: int,
           groups: dict[int, str] | None = None) -> SplitReport:
    groups = groups or {im.id: im.group for im in ds.images}
    cat = {c.id: c.name for c in ds.categories}
    anns = ds.anns_by_image()
    split_of = {iid: s for s, ids in split_ids.items() for iid in ids}
    groups_in: dict[str, set[str]] = {s: {groups[i] for i in ids} for s, ids in split_ids.items()}
    train_groups = groups_in.get("train", set())
    cross = set()
    for a, ga in groups_in.items():
        for b, gb in groups_in.items():
            if a < b:
                cross |= ga & gb
    leaked = sum(1 for iid, s in split_of.items() if s != "train" and groups[iid] in train_groups)
    per_class = {cat[c]: {s: 0 for s in split_ids} for c in cat}
    for iid, s in split_of.items():
        for a in anns.get(iid, []):
            per_class[cat[a.category_id]][s] += 1
    return SplitReport(strategy, seed, {s: len(v) for s, v in split_ids.items()},
                       {s: len(g) for s, g in groups_in.items()}, len(cross), leaked, per_class)
