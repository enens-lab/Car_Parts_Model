# Licensing & commercial-use review

**Goal:** everything that ends up in the shipped product — code, pretrained weights, training data — must be
usable commercially without copyleft obligations. This is an engineering summary, not legal advice; have
counsel confirm before launch.

## TL;DR

| Layer | Choice | License | Commercial use | What we must do |
|---|---|---|---|---|
| Detector / segmenter | **RF-DETR** (`rfdetr` package, ICLR 2026) | Apache-2.0 | ✅ | keep LICENSE + NOTICE, state modifications |
| Pretrained weights | RF-DETR Nano/Small/Medium/Large (det) and **all Seg variants** | Apache-2.0 | ✅ | attribution in NOTICE |
| Backbone inside RF-DETR | DINOv2 (Meta) | Apache-2.0 | ✅ | attribution in NOTICE |
| Training stack | PyTorch, Lightning, torchvision, transformers, supervision, pycocotools, faster-coco-eval, onnx, onnxruntime, OpenCV, numpy, Pillow, PyYAML | BSD / Apache-2.0 / MIT | ✅ | keep notices |
| Exterior data | **carparts-seg** (Gianmarco Russo via Roboflow; Ultralytics mirror) | CC BY 4.0 | ✅ | attribute, link license, note changes |
| Engine-bay data | **Kaggle "Car Engine Bay Images with YOLO Annotations"** (Khaled Chawa) | MIT | ✅ | keep copyright/permission notice |
| Engine-bay data | **Roboflow "engine-bay-parts"** (Stephens Workspace) | MIT | ✅ | keep notice |
| Isolated-part data | **Roboflow "car parts"** (Used auto parts classification, 50 classes) | Public Domain | ✅ | none (attribution kept anyway) |
| Isolated-part data | Kaggle "50 Types of Car Parts" (gpiosenka) — alternative/classifier source | Apache 2.0 | ✅ | keep notice |
| Engine internals | Roboflow "Engine Parts" (engineparts), "engine parts" (project-tevws) | CC BY 4.0 | ✅ | attribute |
| Optional extras | Roboflow "Car Engine Bay" (Razeen), "Engine Parts Detector", "car under the hood" | CC BY 4.0 | ✅ | attribute |

### Deliberately NOT in the product

| Component | License | Why excluded |
|---|---|---|
| **Ultralytics YOLO** (`ultralytics`, YOLOv8/11/26, incl. pretrained + fine-tuned weights) | **AGPL-3.0** | Commercial use requires an Ultralytics Enterprise License; AGPL would otherwise require releasing the full source of any product/service that uses the models. Kept only as an *internal benchmark* behind `scripts/baseline_ultralytics.py --i-accept-agpl` in a separate extra (`uv sync --extra baseline-agpl`). |
| RF-DETR **XLarge / 2XLarge detection** weights (`rfdetr[plus]`) | PML-1.0 (Roboflow Platform Model License) | Not permissive — not registered in `MODEL_REGISTRY`. (The Seg XL/2XL weights *are* Apache-2.0 and are registered.) |
| Roboflow `car-parts-jv0or/car-parts-detection-owvwe` (crankshaft, clutch plate, piston …) | **Private** (per the project's license field) | Not usable; the Public Domain 50-class set covers the same parts. |
| DEIMv2 (DINOv3-based) | custom "DEIMv2 License" (commercial → contact authors); DINOv3 weights are gated and require a "Built with DINOv3" notice | Friction + uncertainty; RF-DETR matches its accuracy class with a clean license. |
| RT-DETRv4 | Apache-2.0 | Fine license, but detection-only (no masks) and no fine-tuning toolchain comparable to `rfdetr`. Viable fallback for the box detectors. |
| `carparts-seg.yaml` from the Ultralytics repo | AGPL-3.0 (file header) | We only read it locally to confirm class order; class names are duplicated in `carparts/constants.py`, and the file is never shipped. |

## Attribution text (ship this)

Every processed dataset carries an auto-generated `data/processed/<recipe>/NOTICE.md` listing the exact sources
that went in. Copy it into the product's third-party notices together with:

```
This product uses RF-DETR (c) 2025 Roboflow, Inc., licensed under the Apache License 2.0, and DINOv2
(c) Meta Platforms, Inc., licensed under the Apache License 2.0.
Training data: "car-seg" by Gianmarco Russo (Roboflow Universe, CC BY 4.0) — modified (re-split, re-labelled
taxonomy); "Car Engine Bay Images with YOLO Annotations" by Khaled Chawa (Kaggle, MIT); "engine-bay-parts"
by Stephens Workspace (Roboflow Universe, MIT).
```

## Notes & open questions for counsel

* **CC BY 4.0 and trained weights.** Whether a model trained on CC-BY data is a "derivative work" is unsettled.
  We attribute regardless; that satisfies the license in either reading. CC BY also forbids implying endorsement
  by the dataset authors.
* **Roboflow Universe uploads.** Licenses are self-declared by uploaders. Both engine-bay sets we use were
  labelled MIT / CC BY 4.0 by their owners; the "engine-bay-parts" set looks like a re-annotation of the Kaggle
  photos (same 26 numeric ids), so its provenance ultimately traces to the MIT Kaggle set. If provenance matters
  for your risk appetite, drop `rf_engine_bay_parts_stephens` from the recipe (one YAML line).
* **Photos of people/plates.** Exterior photos may contain license plates or faces. Our models only output car
  parts, but the *training data* is stored locally — don't redistribute the raw datasets.
* **RF-DETR weights download** from `storage.googleapis.com/rfdetr` at first use — vendor them into an internal
  artifact store for reproducible builds (`pretrain_weights=` accepts a local path).
