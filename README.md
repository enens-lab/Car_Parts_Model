# carparts — commercial-grade car-parts detection & segmentation

Two model families from one canonical data pipeline, built for a **commercial product**:

| Model | Task | Classes | Data |
|---|---|---|---|
| `exterior_seg` | instance segmentation | 23 exterior body parts (bumpers, doors, lights, hood, mirrors, wheels, …) | carparts-seg (CC BY 4.0) |
| `engine_bay_det` | detection | 26 engine-bay components (battery, fuse box, reservoirs, dipsticks, filters, …) | Kaggle engine bay (MIT) + Roboflow re-labelling (MIT) |
| `unified_det` | detection | 49 = both taxonomies, boxes only | all of the above |

Detector: **RF-DETR** (Roboflow, ICLR 2026) — Apache-2.0 code *and* weights, DINOv2 backbone, real-time, native
ONNX/TensorRT export. Ultralytics YOLO was rejected for the product because it is AGPL-3.0 (it survives only as an
opt-in internal benchmark). Full reasoning: [docs/model_selection.md](docs/model_selection.md),
[docs/licensing.md](docs/licensing.md).

## Quick start

```bash
# 1. environment (Python 3.12 + CUDA 12.6 wheels; CPU-only machines: comment out [tool.uv.sources] in pyproject)
uv sync --extra serve --extra kaggle --extra roboflow --extra dev
cp .env.example .env            # optional: ROBOFLOW_API_KEY (Kaggle downloads work anonymously)

# 2. data: download sources, build leakage-safe COCO datasets (data/processed/<recipe>/)
uv run python scripts/prepare_data.py --all

# 3. prove the pipeline on your GPU in a few minutes
uv run python scripts/smoke_test.py

# 4. train / evaluate / predict / export
uv run python scripts/train.py    --recipe exterior_seg   --config rfdetr_8gb --name exterior_seg_m
uv run python scripts/train.py    --recipe engine_bay_det --config rfdetr_8gb --model rfdetr-medium --name engine_bay_m
uv run python scripts/evaluate.py --run exterior_seg_m                       # per-class AP table (test split)
uv run python scripts/predict.py  --run exterior_seg_m --run engine_bay_m --source photo.jpg
uv run python scripts/export.py   --run exterior_seg_m --format onnx --verify photo.jpg

# 5. localhost demo (drag-and-drop UI + JSON API)
uv run python app/server.py --run exterior_seg_m --run engine_bay_m     # -> http://127.0.0.1:8000
```

On the shared 4×RTX 6000 Ada box: `bash scripts/remote_setup.sh` (sets `CUDA_VISIBLE_DEVICES=3`), then the same
commands with `--config rfdetr_48gb`. **GPU 0 is forbidden there and GPU 3 is the default** — the package refuses to
start on GPU 0 (see [HANDOFF.md](HANDOFF.md)).

## What makes this pipeline trustworthy

* **Leakage-safe splits.** The public carparts-seg split leaks 96–98 % of its eval images into train (Roboflow
  augmentation copies of only 585 photos). We split by *physical photo* (filename lineage + perceptual-hash
  near-duplicates, Hamming ≤ 4) and freeze the seed. Details and numbers: [docs/datasets.md](docs/datasets.md).
* **One canonical format.** Every source → `CocoDataset` → recipe → `data/processed/<recipe>/{train,valid,test}` with a
  `dataset_card.json` (sources, licenses, split report, stats) and an auto-generated `NOTICE.md` for attribution.
* **Honest evaluation.** `scripts/evaluate.py` computes per-class box *and* mask AP with pycocotools, independent of the
  trainer's own metrics; every run writes `summary.json` (config, data card, versions, test metrics).
* **License hygiene by construction.** Only Apache/MIT/BSD/CC-BY components are in the default environment; AGPL code
  lives in a separate extra and a script that refuses to run without `--i-accept-agpl`.

## Repository map

```
configs/
  paths.yaml                 where data/artifacts live
  recipes/*.yaml             sources + class map + split policy -> one processed dataset
  train/*.yaml               model + hyper-parameters per hardware budget (smoke, poc_8gb, rfdetr_8gb, rfdetr_24gb, rfdetr_48gb, rfdetr_3x48gb)
src/carparts/
  constants.py               canonical taxonomies (23 exterior, 26 engine bay) + name normalisation
  gpu_policy.py              shared-box rule: GPU 0 forbidden, GPU 3 default (no-op on < 4-GPU machines)
  data/coco.py               canonical in-memory COCO model: merge, remap, dedupe, validate, save/load
  data/yolo.py               YOLO box/polygon parsing (Ultralytics / Roboflow / flat layouts), YOLO writer
  data/splits.py             grouped split, perceptual-hash duplicate detection, split reports
  data/recipe.py             recipe -> processed dataset + dataset card + NOTICE
  sources/                   downloadable sources with license/attribution metadata (registry)
  train/rfdetr_trainer.py    RF-DETR fine-tuning wrapper, model registry (permissive variants only), summaries
  eval/coco_eval.py          per-class COCO AP (bbox + segm) for any checkpoint/split
  infer/predictor.py         CarPartsModel / CarPartsPipeline -> JSON objects (label, confidence, bbox, polygon)
scripts/                     prepare_data, train, evaluate, predict, export, smoke_test, baseline_ultralytics (AGPL), remote_setup.sh
app/                         FastAPI demo server + single-page UI
tests/                       unit tests (no GPU, no downloads)
artifacts/
  runs/<name>/               summary.json, eval_test.json (+ git-ignored checkpoints/tensorboard)
  data_reports/              split & source reports per recipe (committed)
docs/                        licensing, model selection, datasets
```

## Results

See [HANDOFF.md](HANDOFF.md) for the live scoreboard. Proof-of-concept runs on the 8 GB laptop GPU exist only to
validate the pipeline; production numbers come from the 4×48 GB box.

## License

Project code: proprietary (Innova Electronics). Third-party components and datasets: see
[docs/licensing.md](docs/licensing.md) and the per-recipe `NOTICE.md`.
