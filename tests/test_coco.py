import json

import pytest

from carparts.data.coco import Annotation, Category, CocoDataset, CocoImage, polygon_area


def _ds():
    ds = CocoDataset([Category(1, "a"), Category(2, "b"), Category(3, "c")])
    ds.images = [CocoImage(1, "i1.jpg", 100, 100, "s", "g1", "/x/i1.jpg"),
                 CocoImage(2, "i2.jpg", 50, 50, "s", "g2", "/x/i2.jpg")]
    ds.annotations = [
        Annotation(1, 1, 1, [0, 0, 10, 10], 100),
        Annotation(2, 1, 2, [5, 5, 10, 10], 100, segmentation=[[5, 5, 15, 5, 15, 15, 5, 15]]),
        Annotation(3, 2, 3, [0, 0, 5, 5], 25),
    ]
    return ds


def test_polygon_area():
    assert polygon_area([0, 0, 10, 0, 10, 10, 0, 10]) == 100
    assert polygon_area([0, 0, 1, 1]) == 0


def test_validate_catches_corruption():
    _ds().validate()
    bad = _ds()
    bad.categories[2].id = 5
    with pytest.raises(ValueError):
        bad.validate()
    bad = _ds()
    bad.annotations[0].bbox = [0, 0, 0, 10]
    with pytest.raises(ValueError):
        bad.validate()
    bad = _ds()
    bad.annotations[0].image_id = 99
    with pytest.raises(ValueError):
        bad.validate()


def test_remap_merge_and_drop():
    ds = _ds().remap_classes({"a": "ab", "b": "ab", "c": None})
    assert ds.class_names == ["ab"]
    assert len(ds.annotations) == 2 and {a.category_id for a in ds.annotations} == {1}
    ds.validate()


def test_remap_keep_order_adds_empty_classes():
    ds = _ds().remap_classes(keep_order=["z", "c", "a", "b"])
    assert ds.class_names == ["z", "c", "a", "b"]
    assert {a.category_id for a in ds.annotations} == {2, 3, 4}
    ds.validate()


def test_merge_by_name_and_strip_masks():
    m = CocoDataset.merge([_ds(), _ds().remap_classes({"a": "q"})])
    assert m.class_names == ["a", "b", "c", "q"]
    assert len(m.images) == 4 and len({i.id for i in m.images}) == 4
    assert len(m.annotations) == 6 and len({a.id for a in m.annotations}) == 6
    m.validate()
    assert m.has_masks() and not m.strip_masks().has_masks()
    assert m.stats()["per_class"]["b"] == 2


def test_json_roundtrip_keeps_groups_and_empty_classes(tmp_path):
    ds = _ds().remap_classes(keep_order=["a", "b", "c", "never_seen"])
    p = ds.save_json(tmp_path / "ann.json")
    back = CocoDataset.from_coco_json(p, image_root=tmp_path, source="s")
    assert back.class_names == ["a", "b", "c", "never_seen"]  # zero-annotation class survives the round trip
    assert len(back.annotations) == 3
    assert back.annotations[1].segmentation == [[5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0]]
    assert back.images[0].group == "g1"
    back.validate()


def test_from_coco_json_drops_roboflow_parent(tmp_path):
    d = {
        "categories": [{"id": 0, "name": "parts", "supercategory": "none"},
                       {"id": 1, "name": "hood", "supercategory": "parts"},
                       {"id": 2, "name": "wheel", "supercategory": "parts"}],
        "images": [{"id": 7, "file_name": "x_jpg.rf.0123456789abcdef.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 7, "category_id": 1, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}],
    }
    p = tmp_path / "_annotations.coco.json"
    p.write_text(json.dumps(d))
    ds = CocoDataset.from_coco_json(p)
    assert ds.class_names == ["hood", "wheel"]
    assert ds.annotations[0].category_id == 1
    assert ds.images[0].group == "x"
    ds.validate()
