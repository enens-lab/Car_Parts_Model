import json

from PIL import Image, ImageDraw

from carparts.data.coco import CocoDataset
from carparts.data.recipe import build_recipe
from carparts.data.yolo import load_yolo_dataset
from carparts.sources.base import Source, SourceInfo, registry


class _FakeSource(Source):
    name = "test_fake_source"
    info = SourceInfo(title="fake", license="MIT", url="http://example.invalid", attribution="fake attribution")

    def __init__(self, root):
        self.root = root

    def raw_dir(self, raw_root):
        return self.root

    def download(self, raw_root):
        return self.root

    def load(self, raw_root):
        return load_yolo_dataset(self.root, ["a", "b"], source=self.name, splits=["train"])


def _build_raw(root, n_groups=12, copies=3):
    """12 'photos' x 3 augmented copies. Geometry differs per photo so the perceptual hash does not fold
    distinct photos together; copies of one photo differ by a 1-px jitter like real augmentation."""
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for g in range(n_groups):
        for c in range(copies):
            name = f"p{g}_jpg.rf.{g:016x}{c:016x}"
            im = Image.new("RGB", (80, 60), (g * 20 % 255, 50, c * 80 % 255))
            x0, y0 = 4 + 5 * g, 5 + (g % 3) * 10
            ImageDraw.Draw(im).rectangle([x0 + c, y0, x0 + 20, y0 + 15], fill=(200, 0, 0))
            im.save(root / "images" / "train" / f"{name}.jpg")
            (root / "labels" / "train" / f"{name}.txt").write_text(
                f"{g % 2} 0.125 0.166 0.5 0.166 0.5 0.666 0.125 0.666\n")


def _recipe(task):
    return {"name": f"t_recipe_{task}", "task": task, "sources": [{"name": "test_fake_source"}],
            "split": {"strategy": "grouped", "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
                      "seed": 1, "dedupe_hash": True},
            "classes": None, "min_instances_per_class": 0}


def test_build_segmentation_recipe_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    _build_raw(raw)
    registry.register(_FakeSource(raw), replace=True)

    out = build_recipe(_recipe("segmentation"), tmp_path / "rawroot", tmp_path / "processed", tmp_path / "reports",
                       download=False)
    for split in ("train", "valid", "test"):
        ann = out / split / "_annotations.coco.json"
        assert ann.exists()
        ds = CocoDataset.from_coco_json(ann, image_root=ann.parent)
        ds.validate()
        assert ds.class_names == ["a", "b"]
        assert ds.images and all((ann.parent / im.file_name).exists() for im in ds.images)
        assert all(a.segmentation for a in ds.annotations)
    card = json.loads((out / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["classes"] == ["a", "b"]
    assert card["split_report"]["cross_split_groups"] == 0 and card["split_report"]["leaked_eval_images"] == 0
    assert sum(card["split_report"]["counts"].values()) == 36
    assert "MIT" in (out / "NOTICE.md").read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "t_recipe_segmentation_split_report.json").exists()
    assert (tmp_path / "reports" / "t_recipe_segmentation_sources.json").exists()


def test_detection_recipe_strips_masks_and_subsamples(tmp_path):
    raw = tmp_path / "raw"
    _build_raw(raw)
    registry.register(_FakeSource(raw), replace=True)
    recipe = _recipe("detection")
    recipe["subsample"] = {"max_groups": 5, "seed": 0}
    recipe["split"]["dedupe_hash"] = False  # count by filename groups only
    out = build_recipe(recipe, tmp_path / "rawroot", tmp_path / "processed", tmp_path / "reports", download=False)
    total = 0
    for split in ("train", "valid", "test"):
        ds = CocoDataset.from_coco_json(out / split / "_annotations.coco.json")
        total += len(ds.images)
        assert all(a.segmentation is None for a in ds.annotations)
    assert total == 15  # 5 groups x 3 copies


def test_class_map_merge_via_recipe(tmp_path):
    raw = tmp_path / "raw"
    _build_raw(raw)
    registry.register(_FakeSource(raw), replace=True)
    recipe = _recipe("detection")
    recipe["sources"] = [{"name": "test_fake_source", "class_map": {"a": "ab", "b": "ab"}}]
    out = build_recipe(recipe, tmp_path / "rawroot", tmp_path / "processed", tmp_path / "reports", download=False)
    ds = CocoDataset.from_coco_json(out / "train" / "_annotations.coco.json")
    assert ds.class_names == ["ab"]
