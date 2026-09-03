import random

from PIL import Image, ImageDraw

from carparts.data.coco import Annotation, Category, CocoDataset, CocoImage
from carparts.data.splits import dhash, grouped_split, merge_duplicate_groups, report


def _synthetic(n_groups=60, copies=5, seed=0):
    rng = random.Random(seed)
    ds = CocoDataset([Category(i + 1, f"c{i}") for i in range(4)])
    iid = aid = 0
    for g in range(n_groups):
        for _ in range(rng.randint(1, copies)):
            iid += 1
            ds.images.append(CocoImage(iid, f"g{g}_jpg.rf.{iid:032x}.jpg", 64, 64, "s", f"g{g}", ""))
            for _ in range(rng.randint(0, 3)):
                aid += 1
                ds.annotations.append(Annotation(aid, iid, rng.randint(1, 4), [0, 0, 10, 10], 100))
    return ds


def test_grouped_split_has_no_leakage_and_roughly_right_sizes():
    ds = _synthetic()
    ids = grouped_split(ds, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    assert sum(len(v) for v in ids.values()) == len(ds.images)
    assert not (set(ids["train"]) & set(ids["val"])) and not (set(ids["val"]) & set(ids["test"]))
    rep = report(ds, ids, "grouped", 1)
    assert rep.cross_split_groups == 0 and rep.leaked_eval_images == 0
    n = len(ds.images)
    assert abs(rep.counts["train"] / n - 0.8) < 0.08
    assert rep.counts["val"] > 0 and rep.counts["test"] > 0
    assert set(rep.per_class) == {"c0", "c1", "c2", "c3"}


def test_grouped_split_is_deterministic_per_seed():
    ds = _synthetic()
    a = grouped_split(ds, {"train": 0.8, "val": 0.2}, seed=3)
    b = grouped_split(ds, {"train": 0.8, "val": 0.2}, seed=3)
    c = grouped_split(ds, {"train": 0.8, "val": 0.2}, seed=4)
    assert a == b and a != c


def test_original_split_report_detects_leakage():
    ds = _synthetic(n_groups=10, copies=4)
    # naive split: every other image -> copies of the same photo land on both sides
    ids = {"train": [i.id for i in ds.images if i.id % 2], "test": [i.id for i in ds.images if not i.id % 2]}
    rep = report(ds, ids, "original", 0)
    assert rep.cross_split_groups > 0 and rep.leaked_eval_images > 0


def _photo(invert=False):
    from PIL import ImageOps
    im = Image.radial_gradient("L").resize((128, 96)).convert("RGB")
    if invert:
        im = ImageOps.invert(im)
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, 60, 40], fill=(255, 0, 0))
    d.ellipse([70, 50, 120, 90], fill=(0, 0, 255))
    return im


def test_dhash_merges_resized_copies(tmp_path):
    from carparts.data.splits import dhash_bits, hamming

    base = _photo()
    base.save(tmp_path / "a.png")
    base.resize((256, 192), Image.Resampling.LANCZOS).save(tmp_path / "b.png")  # same photo, re-sized
    base.save(tmp_path / "a2.jpg", quality=92)                                    # same photo, re-encoded
    _photo(invert=True).save(tmp_path / "c.png")
    a, b, a2, c = (dhash_bits(tmp_path / n) for n in ("a.png", "b.png", "a2.jpg", "c.png"))
    assert hamming(a, b) <= 4 and hamming(a, a2) <= 4
    assert hamming(a, c) > 20
    assert dhash(tmp_path / "a.png") == f"{a:016x}"

    ds = CocoDataset([Category(1, "x")])
    ds.images = [CocoImage(1, "a.png", 128, 96, "s", "a", str(tmp_path / "a.png")),
                 CocoImage(2, "b.png", 256, 192, "s", "b", str(tmp_path / "b.png")),
                 CocoImage(3, "a2.jpg", 128, 96, "s", "a2", str(tmp_path / "a2.jpg")),
                 CocoImage(4, "c.png", 128, 96, "s", "c", str(tmp_path / "c.png"))]
    groups = merge_duplicate_groups(ds)
    assert groups[1] == groups[2] == groups[3] and groups[1] != groups[4]
    assert len(set(merge_duplicate_groups(ds, use_hash=False).values())) == 4


def test_dedupe_annotations():
    from carparts.data.coco import Annotation, box_iou
    assert abs(box_iou([0, 0, 10, 10], [5, 0, 10, 10]) - 1 / 3) < 1e-9
    ds = CocoDataset([Category(1, "x"), Category(2, "y")])
    ds.images = [CocoImage(1, "i.jpg", 100, 100, "s", "g", "")]
    ds.annotations = [Annotation(1, 1, 1, [0, 0, 50, 50], 2500),
                      Annotation(2, 1, 1, [1, 1, 50, 50], 2500),          # same object, same class -> dropped
                      Annotation(3, 1, 2, [0, 0, 50, 50], 2500),          # same box, other class -> kept
                      Annotation(4, 1, 1, [60, 60, 20, 20], 400)]
    out, removed = ds.dedupe_annotations()
    assert removed == 1 and [a.id for a in out.annotations] == [1, 3, 4]
