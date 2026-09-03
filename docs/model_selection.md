# Model selection

Requirement: real-time detection **and instance segmentation** of car parts, fine-tunable on a few thousand
images, exportable to ONNX/TensorRT, and — the hard constraint — **commercially usable without copyleft**.

## Candidates (state of the art as of Sept 2026)

| Model | Task | COCO AP (det / mask) | Latency (T4, TRT fp16) | License | Verdict |
|---|---|---|---|---|---|
| **RF-DETR** N/S/M/L (Roboflow, ICLR 2026) | det | 48.4 / 53.0 / 54.7 / 56.5 | 2.3 – 6.8 ms | Apache-2.0 | ✅ **chosen** for boxes |
| **RF-DETR-Seg** N/S/M/L/XL/2XL | det + masks | mask 40.3 / 43.1 / 45.3 / 47.1 / 48.8 / 49.9 | 3.4 – 21.8 ms | Apache-2.0 | ✅ **chosen** for masks |
| RF-DETR XL/2XL (det) | det | 58.6 / 60.1 | 11.5 / 17.2 ms | PML-1.0 | ❌ not permissive |
| Ultralytics YOLO26-seg n…x | det + masks | mask 33.9 … 47.0 | 2.1 – 16.4 ms | AGPL-3.0 | ❌ Enterprise license needed; benchmark only |
| RT-DETRv4 S/M/L/X (ECCV 2026) | det | 49.8 / 53.7 / 55.4 / 57.0 | 3.7 – 12.9 ms | Apache-2.0 | ⚠️ fallback for boxes; no masks |
| DEIMv2 (DINOv3) Atto…X | det | 23.8 … 57.8 | 1.1 – 13.8 ms | custom, commercial → contact | ❌ |
| D-FINE / DEIM | det | ≤ 56.5 | ~8–13 ms | Apache-2.0 | superseded by RF-DETR / RT-DETRv4 |
| Mask2Former, SAM 2 | masks | high | not real-time | MIT / Apache-2.0 | future: pseudo-masks for engine-bay boxes |

Why RF-DETR wins here: one codebase covers both tasks with the same fine-tuning API (`model.train(dataset_dir=…)`),
a frozen-ish DINOv2 backbone converges fast on small datasets (RF-DETR was designed around the RF100-VL
fine-tuning benchmark), export to ONNX / TensorRT / TFLite / CoreML is built in, and every weight we touch is
Apache-2.0. The accuracy gap to the best AGPL/PML options is a few AP points at most.

## What we deploy

| Recipe | Model | Why |
|---|---|---|
| `exterior_seg` (23 classes, masks) | `rfdetr-seg-medium` on 8 GB, `rfdetr-seg-xlarge` on the 4×48 GB box | pixel masks for exterior parts (damage/repair workflows) |
| `engine_bay_det` (26 classes, boxes) | `rfdetr-medium` / `rfdetr-large` | box-only labels exist for the engine bay |
| `unified_det` (49 classes, boxes) | `rfdetr-large` | one model when deployment simplicity beats exterior masks |

## Lessons carried over from the knee-MRI pipeline (`enens-lab/knee_abnormality_detection`)

That repository is an MRI *classification* system (DINOv3 ViT + per-target attention head, noisy-student
distillation, 5-fold grouped CV, rank-mean ensembles), so its architecture does not transfer — but its
engineering discipline does, and this repo copies it deliberately:

1. **One deterministic data contract.** Every source is converted into one canonical COCO representation
   (`carparts.data.coco`) by code that is also what inference reads. No hand-edited label files.
2. **Leakage-safe, frozen splits.** The knee repo groups folds by report hash; we group by *physical photo*
   (Roboflow augmentation copies + perceptual-hash duplicates). The public carparts-seg split leaks 96–98 % of its
   eval images into train — see `docs/datasets.md`. Split reports are committed under `artifacts/data_reports/`.
3. **`summary.json` per run** with config, data card hash, versions and test metrics; weights git-ignored.
4. **Honest evaluation.** Test split is touched once per run by `evaluate_checkpoint`; per-class AP tables make
   weak classes visible instead of hiding behind a single mAP.
5. **HANDOFF.md** as the living mission log for whoever runs the GPU box next.

Ideas from the knee repo worth trying once the baselines exist: multi-seed / multi-resolution ensembles
(cheap +0.5–1 AP), pseudo-labelling unlabelled engine-bay photos with the best model (noisy student), and a
long-schedule check (their 9+16-epoch result beat the frozen 6+8 schedule by 3× seed noise).
