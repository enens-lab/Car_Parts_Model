import pytest
from PIL import Image

from carparts.data.yolo import (find_file_dirs, group_key, load_yolo_dataset, load_yolo_pairs, parse_label_file,
                                write_yolo_dataset)


def _img(path, w=64, h=48):
    Image.new("RGB", (w, h), (10, 20, 30)).save(path)


def test_group_key_folds_roboflow_copies():
    assert group_key("car87_jpg.rf.27fc37a7f231e296a3b36ba606e9261c.jpg") == "car87"
    assert group_key("new_10_png_jpg.rf.a67300788847ef2767d469b1e75729e8.jpg") == "new_10"
    assert group_key("0.jpg") == "0"
    assert group_key("IMG_1234.JPG") == "IMG_1234"


def test_parse_label_file_boxes_and_polygons(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("1 0.5 0.5 0.5 0.5\n2 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0\n")
    out = parse_label_file(f, 100, 50, num_classes=3)
    assert len(out) == 2
    cls, bbox, polys = out[0]
    assert cls == 1 and polys is None and bbox == [25.0, 12.5, 50.0, 25.0]
    cls, bbox, polys = out[1]
    assert cls == 2 and bbox == [0.0, 0.0, 100.0, 50.0] and len(polys[0]) == 8
    with pytest.raises(ValueError):
        parse_label_file(f, 100, 50, num_classes=2)
    assert parse_label_file(tmp_path / "missing.txt", 10, 10) == []


def test_load_ultralytics_layout(tmp_path):
    root = tmp_path / "ds"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    _img(root / "images" / "train" / "a_jpg.rf.0123456789abcdef.jpg")
    (root / "labels" / "train" / "a_jpg.rf.0123456789abcdef.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    _img(root / "images" / "train" / "bg.jpg")  # no label file -> background image
    _img(root / "images" / "val" / "b.jpg")
    (root / "labels" / "val" / "b.txt").write_text("1 0.1 0.1 0.2 0.2 0.3 0.3 0.1 0.3\n")

    res = load_yolo_dataset(root, ["x", "y"], source="t")
    assert set(res) == {"train", "val"}
    tr, va = res["train"], res["val"]
    assert len(tr.images) == 2 and len(tr.annotations) == 1
    assert tr.images[0].group == "a" and tr.images[0].width == 64 and tr.images[0].height == 48
    assert tr.annotations[0].category_id == 1 and tr.annotations[0].segmentation is None
    assert va.annotations[0].category_id == 2 and va.annotations[0].segmentation is not None
    tr.validate()
    va.validate()
    assert tr.stats()["images_without_annotations"] == 1


def test_find_file_dirs_and_yolo_roundtrip(tmp_path):
    img_dir = tmp_path / "k" / "images" / "images"
    lbl_dir = tmp_path / "k" / "labels" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(3):
        _img(img_dir / f"{i}.jpg")
        (lbl_dir / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert find_file_dirs(tmp_path / "k", {".jpg"}, min_files=2) == [img_dir]
    assert find_file_dirs(tmp_path / "k", {".txt"}, min_files=2) == [lbl_dir]

    ds = load_yolo_pairs({"all": (img_dir, lbl_dir)}, ["a"], "t")["all"]
    assert len(ds.images) == 3 and len(ds.annotations) == 3
    out = write_yolo_dataset({"train": ds, "val": ds}, tmp_path / "yolo_out")
    assert (out / "data.yaml").exists()
    assert len(list((out / "labels" / "train").glob("*.txt"))) == 3
    assert (out / "labels" / "train" / "0000001.txt").read_text().startswith("0 0.5")
